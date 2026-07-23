"""CC-DIDS primary: SVM regression on AMI-derived daily totals, 1-7 day.

Direct multi-horizon SVR (RBF): one model per *effective* horizon k predicts
y[t+k] from features known at time t (demand lags/rolling means, calendar)
plus the target-day weather covariates. All shifting happens on a contiguous
daily calendar reindex of the demand series, so lags and targets are true
calendar offsets — rows touching the series' internal gaps carry NaNs and
drop out of training instead of silently pairing across a gap.

Anchoring: the last observed demand day t_last lags the issue date by the
AMI ingestion delay, so the model for calendar day issue_date+h is the one
with effective horizon k = (issue_date+h) - t_last (k > h). Prediction
intervals are empirical residual quantiles from time-series cross-validation.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .covariates import COVARIATE_COLS, calendar_features
from .sarima_baseline import MemberResult

log = logging.getLogger(__name__)

_SVR_PARAMS = dict(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale")
LAGS = [1, 2, 3, 7, 14]
CV_SPLITS = 5


def _daily_reindex(y: pd.Series) -> pd.Series:
    """Contiguous daily calendar (date-object index), gaps as NaN."""
    y = y.dropna()
    idx = [d.date() for d in pd.date_range(min(y.index), max(y.index), freq="D")]
    return pd.Series([y.get(d, np.nan) for d in idx], index=idx, name="y")


def _issue_time_features(y_full: pd.Series,
                         holidays: set | None = None) -> pd.DataFrame:
    """Features known at each calendar day t (index = t); NaN inside/after gaps."""
    df = pd.DataFrame(index=y_full.index)
    for lag in LAGS:
        df[f"y_lag{lag}"] = y_full.shift(lag)
    df["y_roll7"] = y_full.shift(1).rolling(7).mean()
    return pd.concat(
        [df, calendar_features(list(y_full.index), holidays)], axis=1)


def fit_and_forecast_svm(y: pd.Series, cov_full: pd.DataFrame,
                         cov_future: pd.DataFrame, issue_date: date,
                         pi_level: float = 0.80,
                         holidays: set | None = None) -> MemberResult:
    """cov_full: NaN-free covariates covering y's full calendar span;
    cov_future: covariates w/ horizon_days for issue_date+1..+7."""
    y_full = _daily_reindex(y)
    t_last = max(d for d in y_full.index if not pd.isna(y_full[d]))
    base = _issue_time_features(y_full, holidays)

    # Prediction-row features must be NaN-free even when a series gap sits
    # within LAGS days of the anchor (crashed the live run + 2 backtest weeks
    # before this guard): fall back to forward-filled lags, which substitute
    # the nearest earlier observation for a missing lag day.
    pred_row = base.loc[[t_last]]
    if pred_row.isna().any().any():
        filled = _issue_time_features(y_full.ffill(), holidays)
        n_bad = int(pred_row.isna().sum().sum())
        log.warning("SVM anchor %s has %d NaN lag feature(s) (series gap "
                    "within %d days) — using forward-filled lags for the "
                    "prediction row", t_last, n_bad, max(LAGS))
        pred_row = filled.loc[[t_last]]
    # weather on the same contiguous calendar, so positional shift == calendar
    wx = cov_full.loc[list(y_full.index), COVARIATE_COLS].astype(float)
    alpha = (1.0 - pi_level) / 2.0

    rows, rmses = [], []
    for h in sorted(cov_future["horizon_days"].astype(int)):
        target_day = issue_date + timedelta(days=h)
        k = (target_day - t_last).days          # effective horizon from t_last
        train = pd.concat(
            [base,
             wx.shift(-k).add_suffix("_tgt"),   # weather at t+k, keyed to t
             y_full.shift(-k).rename("y_target")],
            axis=1).dropna()
        X, yv = train.drop(columns="y_target"), train["y_target"]

        model = make_pipeline(StandardScaler(), SVR(**_SVR_PARAMS))
        resids = []
        for tr_idx, te_idx in TimeSeriesSplit(n_splits=CV_SPLITS).split(X):
            model.fit(X.iloc[tr_idx], yv.iloc[tr_idx])
            resids.extend(yv.iloc[te_idx] - model.predict(X.iloc[te_idx]))
        resids = np.asarray(resids)
        rmses.append(float(np.sqrt(np.mean(resids ** 2))))

        model.fit(X, yv)
        x_new = pred_row.copy()
        fut = cov_future[cov_future["horizon_days"] == h]
        for c in COVARIATE_COLS:
            x_new[f"{c}_tgt"] = float(fut[c].iloc[0])
        point = float(model.predict(x_new[X.columns])[0])
        rows.append({
            "date": target_day,
            "horizon_days": h,
            "point_mgd": max(point, 0.0),
            "pi_low": max(point + float(np.quantile(resids, alpha)), 0.0),
            "pi_high": point + float(np.quantile(resids, 1.0 - alpha)),
        })

    out = pd.DataFrame(rows)
    return MemberResult(
        member="svm",
        issue_date=issue_date,
        forecasts=out,
        cv_rmse_mgd=float(np.mean(rmses)),
        n_train=int(y_full.notna().sum()),
        diagnostics={"params": _SVR_PARAMS, "lags": LAGS,
                     "anchor_last_obs": t_last.isoformat(),
                     "anchor_lag_days": (issue_date - t_last).days,
                     "per_horizon_cv_rmse": [round(r, 4) for r in rmses]},
    )
