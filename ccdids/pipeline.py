"""The one CC-DIDS forecast pipeline — shared by the live runner and backtest.

Before this module, run_ccdids_ensemble.py and _backtest_ccdids.py each
carried their own copy of the covariate-assembly + three-member-fit +
combine + drought-adjustment sequence, so any change to the live pipeline
silently stopped being what the backtest measured (and what the PI
calibration factors were derived from). Both entry points now call
run_pipeline(); they differ only in what data they pass in (live vs
truncated-to-issue-date) and what they do with the result (report/log vs
score). PI calibration is deliberately NOT applied here — the live runner
applies it after, and the backtest must score raw member intervals before
recomputing factors.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from . import covariates, datasets, drought_adjustment
from .ensemble import EnsembleResult, combine
from .gbr_seasonal import fit_and_forecast_gbr
from .sarima_baseline import fit_and_forecast_sarima
from .svm_primary import fit_and_forecast_svm

log = logging.getLogger(__name__)

HORIZON_DAYS = 30
SHORT_H = 7


def run_pipeline(y: pd.Series, weather_hist: pd.DataFrame,
                 nws_df: pd.DataFrame | None, issue_date: date,
                 cfg: dict, coeffs: dict,
                 horizon_days: int = HORIZON_DAYS,
                 short_h: int = SHORT_H) -> EnsembleResult:
    """Fit all members and return the combined, drought-adjusted forecast.

    y: demand series indexed by date (gaps allowed); weather_hist: GHCND
    frame; nws_df: NWS forecast frame or None (climatology fallback).
    """
    lat = cfg["location"]["lat"]
    t_last = max(y.index)
    clim = covariates.climatology(weather_hist, lat)
    span_days = [d.date() for d in pd.date_range(min(y.index), t_last, freq="D")]
    cov_full = covariates.covariate_frame(span_days, weather_hist, nws_df, lat,
                                         clim=clim)
    cov_future = covariates.future_covariates(issue_date, horizon_days,
                                             weather_hist, nws_df, lat, clim=clim)
    # SARIMA forecasts every day from its anchor (t_last) forward, so its
    # exog must bridge the AMI ingestion lag between t_last and issue_date
    bridge_days = [d.date() for d in pd.date_range(
        t_last + timedelta(days=1), issue_date + timedelta(days=short_h), freq="D")]
    cov_ahead = covariates.covariate_frame(bridge_days, weather_hist, nws_df, lat,
                                          clim=clim)

    spans = datasets.stage_history(cfg)
    holidays = covariates.parse_holidays(cfg)

    sarima = fit_and_forecast_sarima(y, cov_full, cov_ahead, issue_date)
    svm = fit_and_forecast_svm(y, cov_full, cov_future.head(short_h), issue_date,
                               holidays=holidays)
    gbr = fit_and_forecast_gbr(y, cov_full, cov_future, spans, issue_date,
                               holidays=holidays)

    result = combine(sarima, svm, gbr)
    adjusted = drought_adjustment.apply_adjustment(
        result.forecasts, t_last, spans, coeffs,
        gbr_learned_stages=set(gbr.diagnostics["stages_learned"]))
    result.forecasts = adjusted
    result.training_regime = adjusted.attrs.get("training_regime", "none")
    result.diagnostics = {
        "anchor_last_obs": t_last.isoformat(),
        "anchor_lag_days": (issue_date - t_last).days,
        "nws_days": int((cov_future["source"] == "nws").sum()),
    }
    return result
