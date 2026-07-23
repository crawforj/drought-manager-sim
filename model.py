"""OLS and GBM demand forecast models for WTP finished water.

OLS (statsmodels): interpretable coefficients, exact 80% prediction intervals,
  leave-one-out cross-validation RMSE. Best when transparency matters.

GBM (sklearn GradientBoostingRegressor, quantile mode): captures nonlinear
  temperature-demand curve and feature interactions; 80% prediction intervals
  via separate quantile models; 5-fold time-series CV RMSE.

Both models return DemandForecastResult and are directly comparable in the
report. Use config.yaml model.type ("ols", "gbm", "compare") and model.primary
("ols" or "gbm") to control which drives the operational forecast.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

log = logging.getLogger(__name__)

# GBM hyperparameters — tuned for ~1,000-2,000 row / ~10 feature datasets.
# Shallow trees + low learning rate + subsampling resist overfitting.
_GBM_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42,
)


@dataclass
class DemandForecastResult:
    issue_date: date
    forecasts: pd.DataFrame        # columns: date, horizon_days, point_mgd, pi_low, pi_high
    r_squared: float
    rmse_mgd: float                # LOO RMSE (OLS) or 5-fold CV RMSE (GBM)
    mape_30d: float | None         # rolling 30-day MAPE; None if < MAPE_MIN_ROWS_FULL paired rows in log
    n_train: int
    feature_names: list[str]
    coefficients: dict[str, float]   # OLS: regression params; GBM: feature importances
    predictor_values: dict[str, float]  # feature values for the first forecast day
    model_label: str = "ols"           # "ols" or "gbm"
    manual_offset_mgd: float = 0.0
    auto_bias_mgd: float = 0.0
    # Provisional MAPE (2026-07-20): same real data, lower confidence bar
    # (MAPE_MIN_ROWS_PROVISIONAL). Only set when mape_30d is None -- display
    # ONLY, never fed into model comparison/selection (see _compute_mape_30d
    # docstring for why). No fabricated data behind either number.
    mape_30d_provisional: float | None = None
    mape_30d_provisional_n: int = 0


# ── Feature engineering ──────────────────────────────────────────────────────

def _cyclic_encode(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    theta = 2 * math.pi * series / period
    return np.sin(theta), np.cos(theta)


def _build_calendar(index: pd.DatetimeIndex, holidays: set[date]) -> pd.DataFrame:
    month_sin, month_cos = _cyclic_encode(pd.Series(index.month, index=index), 12)
    doy_sin, doy_cos = _cyclic_encode(pd.Series(index.dayofyear, index=index), 365)
    return pd.DataFrame({
        "is_weekend": (index.dayofweek >= 5).astype(int),
        "is_holiday": pd.array([int(d.date() in holidays) for d in index], dtype=int),
        "is_irrigation_season": pd.array([int(d.month in (5, 6, 7, 8, 9)) for d in index], dtype=int),
        "month_sin": month_sin.values,
        "month_cos": month_cos.values,
        "doy_sin": doy_sin.values,
        "doy_cos": doy_cos.values,
    }, index=index)


def build_features(
    weather_df: pd.DataFrame,
    production_df: pd.DataFrame,
    holidays: set[date],
) -> pd.DataFrame:
    """Merge historical weather + production; compute all predictor columns."""
    df = weather_df[["tmax_f", "tmin_f", "precip_in"]].copy()
    df = df.join(production_df[["production_mgd", "maintenance"]], how="inner")
    df = df.sort_index()

    dt_index = pd.DatetimeIndex([pd.Timestamp(d) for d in df.index])
    cal = _build_calendar(dt_index, holidays)
    cal.index = df.index

    df = pd.concat([df, cal], axis=1)
    # Date-aware, not positional: shift(1) alone shifts by ROW position, so
    # any gap in the merged weather+production history (a missing weather
    # day, a missing production day dropped by the inner join above) pairs
    # "yesterday's high" with whatever the previous ROW happens to be --
    # possibly several calendar days earlier -- silently mispairing the lag
    # feature for both training and (via _get_last_obs_tmax) prediction.
    # `.where(is_contiguous)` forces a real gap to NaN instead.
    idx_ts = pd.to_datetime(pd.Series(df.index, index=df.index))
    is_contiguous = (idx_ts - idx_ts.shift(1)) == pd.Timedelta(days=1)
    df["tmax_lag1_f"] = df["tmax_f"].shift(1).where(is_contiguous)
    df["tmax_f_sq"] = df["tmax_f"] ** 2   # quadratic term for OLS nonlinear temp response
    return df


def _forecast_features(
    nws_df: pd.DataFrame,
    target_dates: list[date],
    holidays: set[date],
    last_observed_tmax: float | None,
) -> pd.DataFrame:
    """Build predictor rows for future dates using NWS forecast data."""
    rows = []
    for i, d in enumerate(target_dates):
        if d not in nws_df.index:
            log.warning("NWS forecast missing for %s -- skipping this horizon", d)
            continue

        nws_row = nws_df.loc[d]
        tmax = float(nws_row["tmax_f"])

        if i == 0:
            tmax_lag1 = last_observed_tmax
            if tmax_lag1 is None:
                log.warning("No last-observed tmax available (empty weather history) -- "
                           "skipping %s (horizon 1, needs tmax_lag1_f)", d)
                continue
        else:
            prev = target_dates[i - 1]
            if prev not in nws_df.index:
                # Previously fell through to tmax_lag1=None -> NaN written
                # into this row's tmax_lag1_f, silently: OLS's predict()
                # doesn't raise on NaN (ships "nan MGD" to operators), GBM's
                # predict() does raise (aborting the whole run, discarding
                # an already-good OLS result in compare/gbm mode). Treating
                # an unresolvable lag exactly like `d` itself missing from
                # nws_df (skip this horizon, loud warning) avoids NaN ever
                # reaching either model.
                log.warning("NWS forecast missing for %s (needed as tmax_lag1 for %s) "
                           "-- skipping %s", prev, d, d)
                continue
            tmax_lag1 = float(nws_df.loc[prev, "tmax_f"])

        ts = pd.Timestamp(d)
        month_sin = math.sin(2 * math.pi * ts.month / 12)
        month_cos = math.cos(2 * math.pi * ts.month / 12)
        doy_sin = math.sin(2 * math.pi * ts.dayofyear / 365)
        doy_cos = math.cos(2 * math.pi * ts.dayofyear / 365)

        rows.append({
            "date": d,
            "tmax_f": tmax,
            "tmax_f_sq": tmax ** 2,
            "tmax_lag1_f": float(tmax_lag1),
            "precip_in": float(nws_row.get("precip_in", 0.0)),
            "is_weekend": int(ts.dayofweek >= 5),
            "is_holiday": int(d in holidays),
            "is_irrigation_season": int(ts.month in (5, 6, 7, 8, 9)),
            "month_sin": month_sin,
            "month_cos": month_cos,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ── Forecast log helpers ─────────────────────────────────────────────────────

def _read_forecast_log(p: Path, model_label: str | None = None) -> pd.DataFrame:
    """Read forecast_log.csv, optionally filtering to a specific model."""
    df = pd.read_csv(p, parse_dates=["forecast_date"])
    df["forecast_date"] = df["forecast_date"].dt.date
    if model_label is not None and "model" in df.columns:
        df = df[df["model"].fillna("ols") == model_label]
    return df


# Forecasts logged before this date were computed against a mis-configured
# weather station (actually a mountain site well away from the service
# area, with a substantial cool bias). Their forecast-vs-actual residuals
# reflect that bug, not the
# corrected model's real skill, so auto-bias calibration must not look back
# past this date -- otherwise it "corrects" today's already-fixed forecast
# for an error that no longer exists, double-counting the fix. Remove once
# enough post-fix history accumulates to make this irrelevant on its own
# (roughly whenever demand_adjustment.auto_bias_days of clean data exists).
_BIAS_CALIBRATION_FLOOR = date(2026, 7, 16)

# Hard cap on the auto-bias magnitude, roughly one LOO-RMSE of the OLS model
# (~1.0 MGD). The trailing-mean correction is BLIND -- it cannot distinguish
# genuine behavioral drift from an upstream bug, and once nearly shipped an
# implausible multi-MGD "correction" computed from residuals of the
# pre-station-fix model. A correction bigger than the model's own
# day-to-day error bar is a symptom to investigate, never an adjustment to
# silently apply: cap correction magnitude so a single bad week/bug can't
# swing the forecast by several MGD. Clamped values are logged loudly.
_AUTO_BIAS_CAP_MGD = 1.0


def _compute_auto_bias(
    forecast_log_path: Path | str,
    days_back: int,
    model_label: str | None = None,
) -> float:
    """Mean (actual - forecast) over the past days_back days from forecast_log.

    A negative return means the model has been over-forecasting (common during
    drought restrictions). Returns 0.0 if insufficient data or log not found.
    model_label filters to rows from that model; None uses all rows.
    """
    if days_back <= 0:
        return 0.0
    p = Path(forecast_log_path)
    if not p.exists():
        return 0.0
    try:
        df = _read_forecast_log(p, model_label)
        cutoff = max(date.today() - timedelta(days=days_back), _BIAS_CALIBRATION_FLOOR)
        recent = df[
            (df["forecast_date"] >= cutoff)
            & (df["forecast_date"] < date.today())
            & (df["horizon_days"] == 1)
        ].dropna(subset=["forecast_mgd", "actual_mgd"])
        # Defensive dedup -- see _compute_mape_30d()'s docstring (same
        # historical forecast_log.csv, same 2026-07-20 finding).
        recent = recent.drop_duplicates(subset=["forecast_date", "issue_date", "horizon_days"], keep="last")
        if len(recent) < 3:
            return 0.0
        bias = float((recent["actual_mgd"] - recent["forecast_mgd"]).mean())
        if abs(bias) > _AUTO_BIAS_CAP_MGD:
            log.warning(
                "Auto bias (%s) computed at %.3f MGD -- EXCEEDS the %.1f MGD cap and was "
                "clamped. A drift this large is a symptom (data bug? regime change?), "
                "not a routine correction -- investigate before trusting this week's "
                "forecasts.",
                model_label or "all", bias, _AUTO_BIAS_CAP_MGD,
            )
            bias = _AUTO_BIAS_CAP_MGD if bias > 0 else -_AUTO_BIAS_CAP_MGD
        log.info(
            "Auto bias correction (%s): %.3f MGD (mean over %d days)",
            model_label or "all", bias, len(recent),
        )
        return bias
    except Exception as e:
        log.warning("Could not compute auto bias: %s", e)
        return 0.0


MAPE_MIN_ROWS_FULL = 5          # "trustworthy" threshold -- drives model comparison/selection
MAPE_MIN_ROWS_PROVISIONAL = 3   # same floor _compute_auto_bias() already uses for its own minimum


def _compute_mape_30d(
    forecast_log_path: Path | str,
    model_label: str | None = None,
    min_rows: int = MAPE_MIN_ROWS_FULL,
) -> tuple[float | None, int]:
    """Rolling 30-day MAPE from forecast_log.csv, and the row count it was
    computed from (0 if None). Requires at least min_rows rows with both
    forecast_mgd and actual_mgd. model_label filters to rows from that
    model; None uses all rows.

    Two protections, added after this once reported ~50% while the most
    recent forecasts were visibly accurate (a few percent error):
      - Same _BIAS_CALIBRATION_FLOOR _compute_auto_bias() already uses --
        forecasts issued before the weather-station fix reflect that bug,
        not the corrected model's real skill, and were still inside the
        30-day window dragging the average up.
      - Defensive dedup on (forecast_date, issue_date, horizon_days) even
        though append_forecasts() now dedups on write -- this function
        must stay correct against OLDER log history written
        before that fix existed, which still has real duplicate rows
        (the same bad forecast logged up to 4x from repeated manual
        re-runs, each counted separately in a naive mean).
      - Rows with actual_mgd <= 0 are excluded before the MAPE mean (2026-07-23):
        a genuine zero-flow reading (real for some BEACON site meters, which
        have no sanity_min_mgd floor unlike the primary WTP -- see
        production.py) makes % error divide-by-zero -> inf and silently
        poisons the mean, which model_selection.select() then compares
        directly to pick the operational model.

    min_rows is lowered by callers to MAPE_MIN_ROWS_PROVISIONAL only for a
    clearly-labeled "provisional, small sample" display value (2026-07-20,
    user request) -- NEVER for the model comparison/selection logic, which
    must keep using the strict (min_rows=5) result so a 3-4-day noise blip
    can't flip which model is recommended. This deliberately does NOT
    fabricate any forecast/actual pairs to hit a row-count target -- every
    row behind either number is real; only the confidence bar differs.
    """
    p = Path(forecast_log_path)
    if not p.exists():
        return None, 0
    try:
        log_df = _read_forecast_log(p, model_label)
        if "forecast_mgd" not in log_df.columns or "actual_mgd" not in log_df.columns:
            return None, 0
        cutoff = max(date.today() - timedelta(days=30), _BIAS_CALIBRATION_FLOOR)
        recent = log_df[
            (log_df["forecast_date"] >= cutoff) & (log_df["horizon_days"] == 1)
        ].dropna(subset=["forecast_mgd", "actual_mgd"])
        recent = recent.drop_duplicates(subset=["forecast_date", "issue_date", "horizon_days"], keep="last")
        # actual_mgd == 0 is a valid reading (e.g. a BEACON site meter on a
        # genuine zero-flow day) but makes % error divide-by-zero -> inf,
        # which would silently poison mean-MAPE and anything comparing it
        # (model_selection.select()). production.py's sanity_min_mgd floor
        # keeps this from happening for the primary WTP forecast, but
        # per-site forecasts (run_multisite_forecast.py) have no such floor,
        # so guard here instead. Only the MAPE metric excludes these rows --
        # the row count / min_rows gate is evaluated on the same excluded
        # set, since a row that can't contribute to the metric shouldn't
        # count toward "enough rows to trust the metric" either.
        recent = recent[recent["actual_mgd"] > 0]
        if len(recent) < min_rows:
            return None, 0
        mape = float(
            (abs(recent["forecast_mgd"] - recent["actual_mgd"]) / recent["actual_mgd"]).mean() * 100
        )
        return mape, len(recent)
    except Exception as e:
        log.warning("Could not compute 30-day MAPE: %s", e)
        return None, 0


def _compute_mape_pair(
    forecast_log_path: Path | str | None,
    model_label: str,
) -> tuple[float | None, float | None, int]:
    """(strict_mape, provisional_mape, provisional_n). strict_mape is None
    below MAPE_MIN_ROWS_FULL real rows; provisional_mape/n are only
    populated when strict_mape is None AND at least MAPE_MIN_ROWS_PROVISIONAL
    real rows exist -- display-only fallback, see _compute_mape_30d docstring.
    """
    if not forecast_log_path:
        return None, None, 0
    mape, _ = _compute_mape_30d(forecast_log_path, model_label=model_label)
    if mape is not None:
        return mape, None, 0
    prov_mape, prov_n = _compute_mape_30d(
        forecast_log_path, model_label=model_label, min_rows=MAPE_MIN_ROWS_PROVISIONAL)
    return None, prov_mape, prov_n


# ── Shared forecast assembly ─────────────────────────────────────────────────

def _get_last_obs_tmax(weather_hist_df: pd.DataFrame) -> float | None:
    if not weather_hist_df.empty and "tmax_f" in weather_hist_df.columns:
        recent = weather_hist_df["tmax_f"].dropna()
        if not recent.empty:
            return float(recent.iloc[-1])
    return None


def _apply_demand_adjustment(
    forecasts_df: pd.DataFrame,
    manual_offset_mgd: float,
    auto_bias: float,
    model_label: str,
    sanity_min_mgd: float | None = None,
    sanity_max_mgd: float | None = None,
) -> None:
    """Apply demand adjustment in-place.

    sanity_min_mgd/sanity_max_mgd (None by default -- no clamp, unchanged
    behavior for any caller that doesn't pass them, e.g. fit_and_forecast_site()'s
    multi-site BEACON paths, which have no single valid bound across sites of
    very different scale): raw production_mgd is already bounds-checked at
    ingestion (production.py's [sanity_min_mgd, sanity_max_mgd], config.yaml-
    documented as 1-12 MGD for Maple Grove WTP), but a manual offset or a
    runaway auto-bias correction added AFTER that check was not -- a
    misconfigured manual_offset_mgd or several consecutive days of one-sided
    MAPE error could ship a physically implausible forecast (negative MGD,
    or far above plant capacity) with no flag, when a caller does pass the
    same bounds used at ingestion.
    """
    total = manual_offset_mgd + auto_bias
    if total != 0.0 and not forecasts_df.empty:
        log.info(
            "Demand adjustment (%s): manual=%.3f, auto_bias=%.3f, total=%.3f MGD",
            model_label, manual_offset_mgd, auto_bias, total,
        )
        forecasts_df["point_mgd"] += total
        forecasts_df["pi_low"] += total
        forecasts_df["pi_high"] += total

    if sanity_min_mgd is not None and sanity_max_mgd is not None and not forecasts_df.empty:
        out_of_range = (forecasts_df["point_mgd"] < sanity_min_mgd) | (forecasts_df["point_mgd"] > sanity_max_mgd)
        if out_of_range.any():
            log.warning(
                "Demand adjustment (%s): %d forecast day(s) outside [%.2f, %.2f] MGD "
                "sanity bounds after adjustment -- clamping to the bound",
                model_label, int(out_of_range.sum()), sanity_min_mgd, sanity_max_mgd,
            )
        forecasts_df["point_mgd"] = forecasts_df["point_mgd"].clip(sanity_min_mgd, sanity_max_mgd)
        forecasts_df["pi_low"] = forecasts_df["pi_low"].clip(sanity_min_mgd, sanity_max_mgd)
        forecasts_df["pi_high"] = forecasts_df["pi_high"].clip(sanity_min_mgd, sanity_max_mgd)


def _predictor_values_for(X_future: pd.DataFrame, first_date: date, feature_cols: list[str]) -> dict[str, float]:
    if first_date is None or X_future.empty or first_date not in X_future.index:
        return {}
    return {
        col: float(X_future.loc[first_date, col]) if col in X_future.columns and pd.notna(X_future.loc[first_date, col]) else float("nan")
        for col in feature_cols
    }


# ── OLS model ────────────────────────────────────────────────────────────────

def fit_and_forecast_ols(
    production_df: pd.DataFrame,
    weather_hist_df: pd.DataFrame,
    nws_df: pd.DataFrame,
    feature_cols: list[str],
    forecast_horizons: list[int],
    issue_date: date,
    holidays: set[date],
    run_loo: bool = True,
    pi_level: float = 0.80,
    forecast_log_path: str | Path | None = None,
    manual_offset_mgd: float = 0.0,
    auto_bias_days: int = 0,
    sanity_min_mgd: float | None = None,
    sanity_max_mgd: float | None = None,
) -> DemandForecastResult:
    """Fit OLS on historical data and forecast demand for each horizon."""
    all_features = build_features(weather_hist_df, production_df, holidays)
    train_mask = (all_features["maintenance"] == 0) & all_features["production_mgd"].notna()
    train = all_features.loc[train_mask].dropna(subset=feature_cols + ["production_mgd"])

    if len(train) < 30:
        raise RuntimeError(
            f"Only {len(train)} usable training rows -- need at least 30. "
            "Ensure production_history.csv has at least a few months of non-maintenance data "
            "with matching weather history from NCEI."
        )

    X_train = sm.add_constant(train[feature_cols], has_constant="add")
    y_train = train["production_mgd"]
    ols_model = sm.OLS(y_train, X_train).fit()
    log.info("OLS fit: R2=%.3f, n_train=%d", ols_model.rsquared, len(train))

    rmse = float("nan")
    if run_loo:
        loo_resid = []
        for i in range(len(train)):
            Xi = X_train.drop(X_train.index[i])
            yi = y_train.drop(y_train.index[i])
            mi = sm.OLS(yi, Xi).fit()
            pred = mi.predict(X_train.iloc[[i]]).iloc[0]
            loo_resid.append(float(y_train.iloc[i]) - pred)
        rmse = float(np.sqrt(np.mean(np.square(loo_resid))))
        log.info("LOO RMSE: %.3f MGD", rmse)

    alpha = 1 - pi_level
    last_obs_tmax = _get_last_obs_tmax(weather_hist_df)
    target_dates = [issue_date + timedelta(days=h) for h in forecast_horizons]
    X_future = _forecast_features(nws_df, target_dates, holidays, last_obs_tmax)

    forecast_rows = []
    for h, d in zip(forecast_horizons, target_dates):
        if X_future.empty or d not in X_future.index:
            log.warning("No forecast features for %s (horizon=%d) -- skipping", d, h)
            continue
        row_feats = X_future.loc[[d], [c for c in feature_cols if c in X_future.columns]].copy()
        for col in feature_cols:
            if col not in row_feats.columns:
                row_feats[col] = float(train[col].mean()) if col in train.columns else 0.0
        X_row = sm.add_constant(row_feats[feature_cols], has_constant="add")
        pred_obj = ols_model.get_prediction(X_row)
        summary = pred_obj.summary_frame(alpha=alpha)
        forecast_rows.append({
            "date": d,
            "horizon_days": h,
            "point_mgd": float(summary["mean"].iloc[0]),
            "pi_low": float(summary["obs_ci_lower"].iloc[0]),
            "pi_high": float(summary["obs_ci_upper"].iloc[0]),
        })

    forecasts_df = pd.DataFrame(forecast_rows)
    first_date = target_dates[0] if target_dates else None

    mape_30d, mape_prov, mape_prov_n = _compute_mape_pair(forecast_log_path, "ols")
    auto_bias = _compute_auto_bias(forecast_log_path, auto_bias_days, model_label="ols") if forecast_log_path else 0.0
    _apply_demand_adjustment(forecasts_df, manual_offset_mgd, auto_bias, "ols",
                             sanity_min_mgd, sanity_max_mgd)

    return DemandForecastResult(
        issue_date=issue_date,
        forecasts=forecasts_df,
        r_squared=float(ols_model.rsquared),
        rmse_mgd=rmse,
        mape_30d=mape_30d,
        n_train=len(train),
        feature_names=feature_cols,
        coefficients={k: float(v) for k, v in ols_model.params.items()},
        predictor_values=_predictor_values_for(X_future, first_date, feature_cols),
        model_label="ols",
        manual_offset_mgd=manual_offset_mgd,
        auto_bias_mgd=auto_bias,
        mape_30d_provisional=mape_prov,
        mape_30d_provisional_n=mape_prov_n,
    )


# backward-compat alias
fit_and_forecast = fit_and_forecast_ols


# ── Climatology model (Tier B: sparse/new sites) ────────────────────────────

def fit_and_forecast_climatology(
    history_df: pd.DataFrame,
    forecast_horizons: list[int],
    issue_date: date,
    pi_level: float = 0.80,
    doy_window_days: int = 7,
    forecast_log_path: str | Path | None = None,
    manual_offset_mgd: float = 0.0,
    auto_bias_days: int = 0,
    sanity_min_mgd: float | None = None,
    sanity_max_mgd: float | None = None,
) -> DemandForecastResult:
    """Day-of-year climatology forecast: median + percentile band from a site's
    own history. Honest and cheap for sites without enough history to fit OLS
    -- every new BEACON entry point starts here and only "graduates" to
    fit_and_forecast_ols() once it has enough days (see
    fit_and_forecast_site()'s tier threshold).

    history_df: indexed by date, single column "value_mgd".

    With only a handful of days of history (typical for a freshly onboarded
    meter), the day-of-year window degenerates to "most of what we have" and
    the percentile band will be wide/noisy -- that's an honest reflection of
    genuine uncertainty, not a bug. It tightens automatically as history grows.
    """
    values = history_df["value_mgd"].dropna()
    if len(values) == 0:
        raise RuntimeError("No usable history for climatology forecast.")

    alpha = 1 - pi_level
    lo_q, hi_q = alpha / 2, 1 - alpha / 2

    doy_index = pd.DatetimeIndex([pd.Timestamp(d) for d in values.index]).dayofyear

    forecast_rows = []
    for h in forecast_horizons:
        target = issue_date + timedelta(days=h)
        target_doy = pd.Timestamp(target).dayofyear
        # Circular day-of-year distance (handles year wraparound)
        dist = np.minimum(
            np.abs(doy_index - target_doy),
            365 - np.abs(doy_index - target_doy),
        )
        window_vals = values[dist <= doy_window_days]
        # Fall back to all available history if the window is too sparse to
        # be meaningful (always true today, given days-not-years of data).
        if len(window_vals) < 5:
            window_vals = values

        forecast_rows.append({
            "date": target,
            "horizon_days": h,
            "point_mgd": float(window_vals.median()),
            "pi_low": float(window_vals.quantile(lo_q)),
            "pi_high": float(window_vals.quantile(hi_q)),
        })

    forecasts_df = pd.DataFrame(forecast_rows)

    mape_30d, mape_prov, mape_prov_n = _compute_mape_pair(forecast_log_path, "climatology")
    auto_bias = _compute_auto_bias(forecast_log_path, auto_bias_days, model_label="climatology") if forecast_log_path else 0.0
    _apply_demand_adjustment(forecasts_df, manual_offset_mgd, auto_bias, "climatology",
                             sanity_min_mgd, sanity_max_mgd)

    return DemandForecastResult(
        issue_date=issue_date,
        forecasts=forecasts_df,
        r_squared=float("nan"),
        rmse_mgd=float(values.std()) if len(values) > 1 else float("nan"),
        mape_30d=mape_30d,
        n_train=len(values),
        feature_names=[],
        coefficients={},
        predictor_values={},
        model_label="climatology",
        manual_offset_mgd=manual_offset_mgd,
        auto_bias_mgd=auto_bias,
        mape_30d_provisional=mape_prov,
        mape_30d_provisional_n=mape_prov_n,
    )


# ── Per-site tiered orchestrator ────────────────────────────────────────────

def fit_and_forecast_site(
    history_df: pd.DataFrame,
    forecast_horizons: list[int],
    issue_date: date,
    weather_hist_df: pd.DataFrame | None = None,
    nws_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    holidays: set[date] | None = None,
    pi_level: float = 0.80,
    tier_a_min_days: int = 30,
    forecast_log_path: str | Path | None = None,
    manual_offset_mgd: float = 0.0,
    auto_bias_days: int = 0,
) -> DemandForecastResult:
    """Pick Tier A (OLS, weather-driven) or Tier B (climatology) automatically
    based on how much clean history a site has. This is the entry point
    multi-site orchestration should call per site rather than choosing a
    tier manually.

    history_df: indexed by date, single column "value_mgd" (e.g. one site's
    slice of entry_point_history.parquet, reshaped -- see run_multisite_forecast.py).
    weather_hist_df/nws_df/feature_cols/holidays are required only for Tier A;
    a site without enough history to attempt Tier A never touches them.
    """
    usable_days = history_df["value_mgd"].notna().sum()

    if usable_days >= tier_a_min_days and weather_hist_df is not None and nws_df is not None:
        production_df = pd.DataFrame({
            "production_mgd": history_df["value_mgd"],
            "maintenance": 0,
        })
        try:
            return fit_and_forecast_ols(
                production_df=production_df,
                weather_hist_df=weather_hist_df,
                nws_df=nws_df,
                feature_cols=feature_cols or [],
                forecast_horizons=forecast_horizons,
                issue_date=issue_date,
                holidays=holidays or set(),
                pi_level=pi_level,
                forecast_log_path=forecast_log_path,
                manual_offset_mgd=manual_offset_mgd,
                auto_bias_days=auto_bias_days,
            )
        except RuntimeError as e:
            log.warning("Tier A (OLS) failed (%s) -- falling back to climatology", e)

    return fit_and_forecast_climatology(
        history_df=history_df,
        forecast_horizons=forecast_horizons,
        issue_date=issue_date,
        pi_level=pi_level,
        forecast_log_path=forecast_log_path,
        manual_offset_mgd=manual_offset_mgd,
        auto_bias_days=auto_bias_days,
    )


# ── GBM model ────────────────────────────────────────────────────────────────

def fit_and_forecast_gbm(
    production_df: pd.DataFrame,
    weather_hist_df: pd.DataFrame,
    nws_df: pd.DataFrame,
    feature_cols: list[str],
    forecast_horizons: list[int],
    issue_date: date,
    holidays: set[date],
    run_cv: bool = True,
    pi_level: float = 0.80,
    forecast_log_path: str | Path | None = None,
    manual_offset_mgd: float = 0.0,
    auto_bias_days: int = 0,
    sanity_min_mgd: float | None = None,
    sanity_max_mgd: float | None = None,
) -> DemandForecastResult:
    """Fit gradient-boosted quantile regression and forecast demand.

    Three separate GBM models are trained: median (point), lower quantile, and
    upper quantile. The point forecast is the median (50th percentile), which is
    more robust to outliers than the mean used by OLS.

    Prediction intervals are formed from the lower/upper quantile models.
    Quantile crossing (where low > point) is corrected by clamping.

    CV RMSE uses 5-fold time-series cross-validation on the point model.
    """
    all_features = build_features(weather_hist_df, production_df, holidays)
    train_mask = (all_features["maintenance"] == 0) & all_features["production_mgd"].notna()
    train = all_features.loc[train_mask].dropna(subset=feature_cols + ["production_mgd"])

    if len(train) < 30:
        raise RuntimeError(
            f"Only {len(train)} usable GBM training rows -- need at least 30."
        )

    X_train = train[feature_cols].values
    y_train = train["production_mgd"].values

    alpha_low = (1 - pi_level) / 2    # 0.10 for 80% PI
    alpha_high = 1 - alpha_low         # 0.90 for 80% PI

    gbm_point = GradientBoostingRegressor(loss="quantile", alpha=0.5, **_GBM_PARAMS)
    gbm_low   = GradientBoostingRegressor(loss="quantile", alpha=alpha_low, **_GBM_PARAMS)
    gbm_high  = GradientBoostingRegressor(loss="quantile", alpha=alpha_high, **_GBM_PARAMS)

    gbm_point.fit(X_train, y_train)
    gbm_low.fit(X_train, y_train)
    gbm_high.fit(X_train, y_train)

    y_pred_train = gbm_point.predict(X_train)
    ss_res = np.sum((y_train - y_pred_train) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    log.info("GBM fit: R2=%.3f, n_train=%d", r_squared, len(train))

    rmse = float("nan")
    if run_cv:
        tscv = TimeSeriesSplit(n_splits=5)
        cv_resids: list[float] = []
        for train_idx, val_idx in tscv.split(X_train):
            cv_gbm = GradientBoostingRegressor(loss="quantile", alpha=0.5, **_GBM_PARAMS)
            cv_gbm.fit(X_train[train_idx], y_train[train_idx])
            preds = cv_gbm.predict(X_train[val_idx])
            cv_resids.extend((y_train[val_idx] - preds).tolist())
        rmse = float(np.sqrt(np.mean(np.square(cv_resids))))
        log.info("GBM 5-fold CV RMSE: %.3f MGD", rmse)

    importances = {
        col: float(gbm_point.feature_importances_[i])
        for i, col in enumerate(feature_cols)
    }

    last_obs_tmax = _get_last_obs_tmax(weather_hist_df)
    target_dates = [issue_date + timedelta(days=h) for h in forecast_horizons]
    X_future = _forecast_features(nws_df, target_dates, holidays, last_obs_tmax)

    forecast_rows = []
    for h, d in zip(forecast_horizons, target_dates):
        if X_future.empty or d not in X_future.index:
            log.warning("No GBM forecast features for %s (horizon=%d) -- skipping", d, h)
            continue
        row_feats = X_future.loc[[d], [c for c in feature_cols if c in X_future.columns]].copy()
        for col in feature_cols:
            if col not in row_feats.columns:
                row_feats[col] = float(train[col].mean()) if col in train.columns else 0.0
        X_row = row_feats[feature_cols].values

        point  = float(gbm_point.predict(X_row)[0])
        pi_low  = float(gbm_low.predict(X_row)[0])
        pi_high = float(gbm_high.predict(X_row)[0])
        # Correct quantile crossing
        pi_low  = min(pi_low, point)
        pi_high = max(pi_high, point)

        forecast_rows.append({
            "date": d,
            "horizon_days": h,
            "point_mgd": point,
            "pi_low": pi_low,
            "pi_high": pi_high,
        })

    forecasts_df = pd.DataFrame(forecast_rows)
    first_date = target_dates[0] if target_dates else None

    mape_30d, mape_prov, mape_prov_n = _compute_mape_pair(forecast_log_path, "gbm")
    auto_bias = _compute_auto_bias(forecast_log_path, auto_bias_days, model_label="gbm") if forecast_log_path else 0.0
    _apply_demand_adjustment(forecasts_df, manual_offset_mgd, auto_bias, "gbm",
                             sanity_min_mgd, sanity_max_mgd)

    return DemandForecastResult(
        issue_date=issue_date,
        forecasts=forecasts_df,
        r_squared=r_squared,
        rmse_mgd=rmse,
        mape_30d=mape_30d,
        n_train=len(train),
        feature_names=feature_cols,
        coefficients=importances,
        predictor_values=_predictor_values_for(X_future, first_date, feature_cols),
        model_label="gbm",
        manual_offset_mgd=manual_offset_mgd,
        auto_bias_mgd=auto_bias,
        mape_30d_provisional=mape_prov,
        mape_30d_provisional_n=mape_prov_n,
    )
