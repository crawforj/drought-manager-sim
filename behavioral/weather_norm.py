"""Weather normalization for Module 3 ITS analysis.

Fits an OLS demand~weather+calendar model on the PRE-intervention period only
(the clean, restriction-free 2020->2026-03-31 record — first drought
declaration since 2012, per user), then predicts over the full range. The
residual series (observed − weather-expected) is what the ITS interruption
test runs on: any systematic post-intervention departure that weather and
calendar can't explain.

Reuses model.build_features() — the same feature stack the operational
forecast uses (tmax + tmax², lag, precip, weekend/holiday, month/doy
harmonics, irrigation season) — rather than reimplementing it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
import statsmodels.api as sm

import model as model_mod

log = logging.getLogger(__name__)


@dataclass
class WeatherNormResult:
    frame: pd.DataFrame       # index=date; columns: actual_mgd, expected_mgd, residual_mgd, in_train
    r_squared: float
    n_train: int
    feature_cols: list[str]
    # The fitted statsmodels OLS results object -- exposed 2026-07-17 (additive,
    # drought_sim needs to predict expected demand for synthetic/future weather
    # via predict_expected() below; no existing caller reads this).
    ols_result: object | None = None


def fit(
    production_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    holidays: set[date],
    feature_cols: list[str],
    train_end: date,
    level_shifts: list[tuple[str, date]] = (),
) -> WeatherNormResult:
    """Fit on non-maintenance, pre-`train_end` days; predict everywhere possible.

    level_shifts: known NON-drought structural breaks in the series — each
    (name, date) becomes a 0/1 dummy (1 from that date onward) in the weather
    model, so the customer-base change it represents is absorbed into
    expected_mgd instead of leaking into the residuals the ITS runs on.
    Added 2026-07-16 for the ~April 2024 Red Zone reassignment (~410 services
    moved Denver Water -> Maple Grove): without it, BOTH systems' residuals
    carried a mid-series level step that (a) collapsed the Denver Water
    weather model to R²=0.005 and flipped its "drought effect" to +40%, and
    (b) made 6 of 8 Maple Grove placebo tests significant, clustered around
    the reassignment date. The break date must lie in the PRE-intervention
    period for its coefficient to be estimable; a dummy whose date falls
    after train_end would be all-zeros in training and silently useless.
    """
    feats = model_mod.build_features(weather_df, production_df, holidays)

    # Precipitation MEMORY, Module 3 only (not the operational forecast's
    # feature list): daily precip_in can't explain multi-week irrigation
    # suppression after a wet spell — the residuals then carry weather signal
    # the ITS can misread as behavioral. Added 2026-07-16 after placebo tests
    # kept failing on 2023 (the famously wet Front Range year: Apr/May 2023
    # placebos at -0.8 MGD, p<0.001) even once the Red Zone level shift was
    # controlled; a trailing-30d precip total lets the model see "it has been
    # a wet month," not just "it rained today."
    feats = feats.copy()
    feats["precip_30d"] = feats["precip_in"].fillna(0).rolling(30, min_periods=10).sum()
    # Antecedent precipitation index (standard hydrology soil-moisture proxy):
    # exponentially-decaying precip memory, k=0.92 (~
    # 30-day half-life-ish). ewm(adjust=False) gives s_t = (1-k)p_t + k*s_{t-1},
    # proportional to the classic API -- proportionality is all a regression
    # needs. Complements the flat 30d sum: the sum weights a rain event the
    # same on day 1 and day 29; API lets its influence fade the way soil
    # moisture actually does.
    feats["precip_api"] = feats["precip_in"].fillna(0).ewm(alpha=0.08, adjust=False).mean()

    usable = feats.dropna(subset=feature_cols + ["production_mgd", "precip_30d"])
    usable = usable[usable["maintenance"] == 0]

    all_cols = list(feature_cols) + ["precip_30d", "precip_api"]
    for name, shift_date in level_shifts:
        col = f"shift_{name}"
        usable = usable.copy()
        usable[col] = (usable.index >= shift_date).astype(int)
        all_cols.append(col)
        if shift_date > train_end:
            log.warning("level shift %r (%s) is after train_end (%s) — "
                        "coefficient not estimable, dummy will do nothing",
                        name, shift_date, train_end)

    train = usable[usable.index <= train_end]
    if len(train) < 365:
        raise RuntimeError(
            f"Only {len(train)} pre-intervention training days — need at least a full "
            "year for a stable seasonal weather model."
        )

    X_train = sm.add_constant(train[all_cols], has_constant="add")
    ols = sm.OLS(train["production_mgd"], X_train).fit()
    log.info("Weather model: R²=%.3f on %d pre-intervention days (through %s)",
             ols.rsquared, len(train), train_end)

    X_all = sm.add_constant(usable[all_cols], has_constant="add")
    expected = ols.predict(X_all)

    frame = pd.DataFrame({
        "actual_mgd": usable["production_mgd"],
        "expected_mgd": expected,
        "residual_mgd": usable["production_mgd"] - expected,
        "in_train": usable.index <= train_end,
    })
    return WeatherNormResult(
        frame=frame,
        r_squared=float(ols.rsquared),
        n_train=len(train),
        feature_cols=all_cols,
        ols_result=ols,
    )


def predict_expected(
    norm: WeatherNormResult,
    weather_df: pd.DataFrame,
    holidays: set[date],
    target_dates: list[date],
    level_shifts: list[tuple[str, date]] = (),
) -> pd.Series:
    """Expected (weather-normal, pre-intervention-behavior) demand for arbitrary
    dates whose weather is supplied -- including synthetic/resampled weather the
    system never actually experienced. Added 2026-07-17 for drought_sim's
    scenario baselines; purely additive, no existing caller affected.

    weather_df must cover target_dates PLUS >=30 leading days (the precip-memory
    features need history: precip_30d rolls 30 days, precip_api decays in).
    level_shifts must be the same list fit() was called with -- the dummies are
    recomputed here with the same date rule (>= shift_date), which for any
    target date after the shift is simply 1.
    """
    if norm.ols_result is None:
        raise RuntimeError("predict_expected: WeatherNormResult has no ols_result -- "
                           "was it produced by an older fit()?")

    # Stub production frame: build_features() inner-joins production columns,
    # so give it NaN production / maintenance=0 rows for every weather date.
    stub = pd.DataFrame({"production_mgd": float("nan"), "maintenance": 0},
                        index=weather_df.index)
    feats = model_mod.build_features(weather_df, stub, holidays).copy()
    feats["precip_30d"] = feats["precip_in"].fillna(0).rolling(30, min_periods=10).sum()
    feats["precip_api"] = feats["precip_in"].fillna(0).ewm(alpha=0.08, adjust=False).mean()
    for name, shift_date in level_shifts:
        feats[f"shift_{name}"] = (feats.index >= shift_date).astype(int)

    missing = [c for c in norm.feature_cols if c not in feats.columns]
    if missing:
        raise RuntimeError(f"predict_expected: feature column(s) {missing} not derivable "
                           "from the supplied weather -- check feature_cols/level_shifts")

    rows = feats.loc[[d for d in target_dates if d in feats.index], norm.feature_cols]
    dropped = len(target_dates) - len(rows)
    if dropped:
        log.warning("predict_expected: %d target date(s) missing from supplied weather", dropped)
    rows = rows.dropna()

    X = sm.add_constant(rows, has_constant="add")
    return pd.Series(norm.ols_result.predict(X), index=rows.index, name="expected_mgd")
