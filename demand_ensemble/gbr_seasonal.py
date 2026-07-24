"""Seasonal extension: gradient-boosted regression, 8-30 day outlooks.

Conditions->demand regression (not autoregressive — demand lags are stale by
day 8): calendar encoding + target-day weather (NWS <=7d, climatology beyond)
+ drought-stage dummies. Quantile-loss models give the 10/90 band, following
the existing model.py GBM convention.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from .covariates import COVARIATE_COLS, calendar_features
from .datasets import StageSpan, stage_on
from .sarima_baseline import MemberResult

log = logging.getLogger(__name__)

_GBR_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   min_samples_leaf=10, subsample=0.8, random_state=42)
# every stage the plan defines, so dummy columns are stable across retrains
STAGES = ["yellow", "orange", "red", "brown", "black"]


def _features(dates: list[date], cov: pd.DataFrame, spans: list[StageSpan],
              holidays: set | None = None) -> pd.DataFrame:
    df = calendar_features(dates, holidays)
    for c in COVARIATE_COLS:
        df[c] = [float(cov.loc[d, c]) if d in cov.index else np.nan for d in dates]
    for s in STAGES:
        df[f"stage_{s}"] = [1.0 if stage_on(d, spans) == s else 0.0 for d in dates]
    return df


def fit_and_forecast_gbr(y: pd.Series, cov_hist: pd.DataFrame,
                         cov_future: pd.DataFrame, spans: list[StageSpan],
                         issue_date: date, horizons: range = range(8, 31),
                         pi_level: float = 0.80,
                         holidays: set | None = None) -> MemberResult:
    y = y.dropna()
    X = _features(list(y.index), cov_hist, spans, holidays).dropna()
    yv = y.loc[X.index]
    alpha = (1.0 - pi_level) / 2.0

    mid = GradientBoostingRegressor(loss="quantile", alpha=0.5, **_GBR_PARAMS)
    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha, **_GBR_PARAMS)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha, **_GBR_PARAMS)

    resids = []
    for tr_idx, te_idx in TimeSeriesSplit(n_splits=5).split(X):
        mid.fit(X.iloc[tr_idx], yv.iloc[tr_idx])
        resids.extend(yv.iloc[te_idx] - mid.predict(X.iloc[te_idx]))
    cv_rmse = float(np.sqrt(np.mean(np.asarray(resids) ** 2)))

    mid.fit(X, yv)
    lo.fit(X, yv)
    hi.fit(X, yv)

    fut = cov_future[cov_future["horizon_days"].isin(list(horizons))]
    fut_dates = list(fut.index)
    Xf = _features(fut_dates, fut, spans, holidays)
    out = pd.DataFrame({
        "date": fut_dates,
        "horizon_days": fut["horizon_days"].astype(int).values,
        "point_mgd": np.clip(mid.predict(Xf), 0.0, None),
        "pi_low": np.clip(lo.predict(Xf), 0.0, None),
        "pi_high": hi.predict(Xf),
    })
    # quantile crossings happen with separately-fit models; enforce ordering
    out["pi_low"] = np.minimum(out["pi_low"], out["point_mgd"])
    out["pi_high"] = np.maximum(out["pi_high"], out["point_mgd"])

    return MemberResult(
        member="gbr",
        issue_date=issue_date,
        forecasts=out,
        cv_rmse_mgd=cv_rmse,
        n_train=len(yv),
        diagnostics={
            "params": _GBR_PARAMS,
            # stages with enough training days for the dummy to be learnable;
            # the drought adjustment must NOT re-correct these (double count)
            "stages_learned": [s for s in STAGES
                               if X[f"stage_{s}"].sum() >= 14],
            "feature_importances": dict(zip(
                X.columns, [round(float(v), 4) for v in mid.feature_importances_])),
        },
    )
