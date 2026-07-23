"""CC-DIDS baseline: SARIMA with weather covariates, 1-7 day horizons.

SARIMAX (statsmodels) on a contiguous daily DatetimeIndex with weekly
seasonality (m=7) and exogenous [tmax_f, precip_in, et0_in]. Gaps in the
demand series are left as NaN — the Kalman filter handles missing endog
natively, which keeps AR terms and the weekly phase honest across the
series' multi-month holes (naive gap-dropping would stitch April to
November as consecutive days). Annual seasonality is carried by the exog
terms (ET0/tmax encode the irrigation season).

Anchoring: the demand series ends 1-3 days before the issue date (AMI
ingestion lag). The model forecasts every day from its last observed day
through issue_date + n, then only the days after issue_date are reported,
labeled with their true calendar horizon.

Validation (comparable to the SVM's out-of-fold RMSE): parameters are fit
with the last VALIDATION_DAYS excluded, then filtered over the full series,
so the tail's one-step-ahead errors are out-of-sample w.r.t. estimation.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .covariates import COVARIATE_COLS

log = logging.getLogger(__name__)

ORDER = (2, 0, 1)
SEASONAL_ORDER = (1, 0, 1, 7)
FALLBACK_ORDER = (1, 0, 0)
FALLBACK_SEASONAL = (1, 0, 0, 7)
VALIDATION_DAYS = 60
SHORT_PROBE = 7      # days used to sanity-probe a fit's forecasts


@dataclass
class MemberResult:
    """One ensemble member's forecast — shared by all three model modules."""
    member: str
    issue_date: date
    forecasts: pd.DataFrame          # date, horizon_days, point_mgd, pi_low, pi_high
    cv_rmse_mgd: float               # comparable across members -> ensemble weights
    n_train: int
    diagnostics: dict = field(default_factory=dict)


# Fit attempts, in order. enforce_stationarity=False fits fastest but can
# land on explosive AR roots whose forecasts diverge to astronomical values —
# caught live by the 2026 backtest, where several February/March issue dates
# produced 1e47-MGD "forecasts". The stationarity-enforced retries fix that.
_FIT_CHAIN = [
    (ORDER, SEASONAL_ORDER, False),
    (ORDER, SEASONAL_ORDER, True),
    (FALLBACK_ORDER, FALLBACK_SEASONAL, True),
]


def _forecast_sane(res, exog_f: pd.DataFrame, y_max: float) -> tuple[bool, str]:
    """(ok, reason) — whether a short probe forecast stays plausible."""
    try:
        probe = res.get_forecast(steps=len(exog_f), exog=exog_f).predicted_mean
    except Exception as exc:      # noqa: BLE001
        return False, f"probe forecast raised {type(exc).__name__}: {exc}"
    if not np.isfinite(probe).all():
        return False, "probe forecast contains non-finite values"
    if probe.max() > 3.0 * y_max or probe.min() < -0.5 * y_max:
        return False, (f"probe forecast implausible (range {probe.min():.1f}"
                       f"..{probe.max():.1f} vs series max {y_max:.1f})")
    return True, ""


def _fit(endog: pd.Series, exog: pd.DataFrame,
         exog_probe: pd.DataFrame | None = None):
    """Fit down the chain until a fit succeeds AND forecasts sanely."""
    y_max = float(endog.max())
    failures = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order, seasonal, enforce in _FIT_CHAIN:
            try:
                model = SARIMAX(endog, exog=exog, order=order,
                                seasonal_order=seasonal, trend="c",
                                enforce_stationarity=enforce,
                                enforce_invertibility=enforce)
                res = model.fit(disp=False, maxiter=200)
            except Exception as exc:      # noqa: BLE001 — statespace failure -> next
                failures.append(f"{order}x{seasonal}: fit raised {exc}")
                continue
            if exog_probe is None:
                ok, reason = True, ""
            else:
                ok, reason = _forecast_sane(res, exog_probe, y_max)
            if ok:
                if (order, seasonal, enforce) != _FIT_CHAIN[0]:
                    log.warning("SARIMA fit degraded to order=%s seasonal=%s "
                                "enforce=%s", order, seasonal, enforce)
                return res, (order, seasonal)
            failures.append(f"{order}x{seasonal}: {reason}")
            log.warning("SARIMA order=%s rejected: %s", (order, seasonal), reason)
    raise RuntimeError("SARIMA: no fit in chain produced sane forecasts — "
                       + "; ".join(failures))


def fit_and_forecast_sarima(y: pd.Series, exog_hist: pd.DataFrame,
                            exog_ahead: pd.DataFrame, issue_date: date,
                            pi_level: float = 0.80) -> MemberResult:
    """y: daily MGD indexed by date (gaps allowed); exog_hist must be NaN-free
    over y's full calendar span; exog_ahead must be NaN-free covariates for
    every day from the day after y's last observation through the last
    horizon to report (issue_date + max horizon)."""
    y = y.dropna()
    t_last: date = max(y.index)
    idx = pd.date_range(min(y.index), t_last, freq="D")
    endog = pd.Series([y.get(d.date(), np.nan) for d in idx], index=idx, name="y")
    exog = pd.DataFrame(
        [exog_hist.loc[d.date(), COVARIATE_COLS] for d in idx],
        index=idx).astype(float)
    n_obs = int(endog.notna().sum())

    ahead_days = sorted(exog_ahead.index)
    assert ahead_days[0] == t_last + timedelta(days=1), \
        "exog_ahead must start the day after the last observation"
    exog_f = exog_ahead.loc[ahead_days, COVARIATE_COLS].astype(float)

    # fail loud and early on NaN covariates — a NaN here used to surface as
    # an unactionable "no fit produced sane forecasts" after 3 wasted fits
    for name, frame in (("exog_hist", exog), ("exog_ahead", exog_f)):
        if frame.isna().any().any():
            bad = list(frame.index[frame.isna().any(axis=1)])[:5]
            raise ValueError(
                f"SARIMA {name} contains NaN covariates (first bad days: "
                f"{bad}) — covariate_frame should be NaN-free; check the "
                "weather cache / climatology fallback")

    # honest validation: params from the head, one-step errors on the tail
    cut = len(endog) - VALIDATION_DAYS
    res_train, _ = _fit(endog.iloc[:cut], exog.iloc[:cut],
                        exog_probe=exog.iloc[cut:cut + SHORT_PROBE])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_model = SARIMAX(endog, exog=exog, order=res_train.model.order,
                             seasonal_order=res_train.model.seasonal_order,
                             trend="c", enforce_stationarity=False,
                             enforce_invertibility=False)
        res_filtered = full_model.filter(res_train.params)
    pred = res_filtered.get_prediction(start=cut)
    err = (endog.iloc[cut:] - pred.predicted_mean).dropna()
    cv_rmse = float(np.sqrt(np.mean(err ** 2)))

    res, order_used = _fit(endog, exog, exog_probe=exog_f)

    fc = res.get_forecast(steps=len(ahead_days), exog=exog_f)
    ci = fc.conf_int(alpha=1.0 - pi_level)

    out = pd.DataFrame({
        "date": ahead_days,
        "point_mgd": fc.predicted_mean.values,
        "pi_low": ci.iloc[:, 0].values,
        "pi_high": ci.iloc[:, 1].values,
    })
    out["horizon_days"] = [(d - issue_date).days for d in out["date"]]
    out = out[out["horizon_days"] >= 1].reset_index(drop=True)
    out["point_mgd"] = out["point_mgd"].clip(lower=0.0)
    out["pi_low"] = out["pi_low"].clip(lower=0.0)

    return MemberResult(
        member="sarima",
        issue_date=issue_date,
        forecasts=out[["date", "horizon_days", "point_mgd", "pi_low", "pi_high"]],
        cv_rmse_mgd=cv_rmse,
        n_train=n_obs,
        diagnostics={
            "order": str(order_used),
            "aic": float(res.aic),
            "anchor_last_obs": t_last.isoformat(),
            "anchor_lag_days": (issue_date - t_last).days,
            "missing_endog_days": int(endog.isna().sum()),
            "exog": COVARIATE_COLS,
        },
    )
