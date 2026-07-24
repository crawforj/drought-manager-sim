"""Regression tests for the demand forecast ensemble, grown from real bugs found live
code-review verification repros. Everything runs on synthetic data — no
network, no repo state/ — so CI can run them on a bare checkout.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from demand_ensemble.calibration import apply as cal_apply
from demand_ensemble.covariates import calendar_features, parse_holidays
from demand_ensemble.datasets import (StageSpan, _ami_system_daily, _interp_short_gaps,
                             stage_history, stage_on)
from demand_ensemble.drought_adjustment import apply_adjustment

COEFFS = {
    "none":   {"f_p5": 0.08, "f_p50": 0.12, "f_p95": 0.17},
    "orange": {"f_p5": 0.13, "f_p50": 0.18, "f_p95": 0.22},
    "red":    {"f_p5": 0.14, "f_p50": 0.20, "f_p95": 0.25},
}


# ── gap handling ─────────────────────────────────────────────────────────────

def test_interp_fills_short_gaps_only():
    df = pd.DataFrame({
        "long": [1.0] + [np.nan] * 10 + [100.0],
        "short": [1.0, np.nan, np.nan, 4.0] + [1.0] * 8,
    })
    out = _interp_short_gaps(df, 3)
    assert out["long"].isna().sum() == 10          # long gap untouched
    assert out["short"].isna().sum() == 0
    assert out["short"][1] == pytest.approx(2.0)   # linear fill


def _write_store(tmp_path, rows):
    (tmp_path / "state").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(tmp_path / "state" / "entry_point_history.parquet")


def _cfg(accounts):
    return {"sites": [{"kind": "beacon", "beacon_account": a, "role": "dw_master"}
                      for a in accounts]}


def test_all_nan_account_dropped_not_poisoning(tmp_path):
    days = [d.date() for d in pd.date_range(dt.date.today() - dt.timedelta(days=30),
                                            dt.date.today(), freq="D")]
    rows = [{"beacon_account": "A1", "date": d, "mgd": 1.0} for d in days]
    rows += [{"beacon_account": "A2", "date": d, "mgd": np.nan} for d in days]
    _write_store(tmp_path, rows)
    total = _ami_system_daily(tmp_path, _cfg(["A1", "A2"]))
    assert total.notna().sum() == len(days)        # aggregate survives
    assert total.dropna().iloc[-1] == pytest.approx(1.0)


def test_empty_store_raises_clear_error(tmp_path):
    days = [d.date() for d in pd.date_range(dt.date.today() - dt.timedelta(days=5),
                                            dt.date.today(), freq="D")]
    _write_store(tmp_path, [{"beacon_account": "A1", "date": d, "mgd": np.nan}
                            for d in days])
    with pytest.raises(RuntimeError, match="no valid reads"):
        _ami_system_daily(tmp_path, _cfg(["A1"]))


def test_stale_store_refuses_to_run(tmp_path):
    old = [d.date() for d in pd.date_range(dt.date.today() - dt.timedelta(days=60),
                                           dt.date.today() - dt.timedelta(days=40),
                                           freq="D")]
    _write_store(tmp_path, [{"beacon_account": "A1", "date": d, "mgd": 1.0}
                            for d in old])
    with pytest.raises(RuntimeError, match="stale"):
        _ami_system_daily(tmp_path, _cfg(["A1"]))


# ── stage timeline ───────────────────────────────────────────────────────────

def test_open_span_shadowing_rejected():
    cfg = {"demand_ensemble": {"stage_history": [
        {"stage": "orange", "start": "2026-04-01", "end": None},
        {"stage": "red", "start": "2026-08-01", "end": None}]}}
    with pytest.raises(ValueError, match="open-ended"):
        stage_history(cfg)


def test_overlapping_spans_rejected():
    cfg = {"demand_ensemble": {"stage_history": [
        {"stage": "orange", "start": "2026-04-01", "end": "2026-08-01"},
        {"stage": "red", "start": "2026-08-01", "end": None}]}}
    with pytest.raises(ValueError, match="overlap"):
        stage_history(cfg)


def test_valid_timeline_and_stage_on():
    cfg = {"demand_ensemble": {"stage_history": [
        {"stage": "orange", "start": "2026-04-01", "end": "2026-07-31"},
        {"stage": "red", "start": "2026-08-01", "end": None}]}}
    spans = stage_history(cfg)
    assert stage_on(dt.date(2026, 3, 1), spans) == "none"
    assert stage_on(dt.date(2026, 5, 1), spans) == "orange"
    assert stage_on(dt.date(2026, 8, 15), spans) == "red"


# ── calendar / holidays ──────────────────────────────────────────────────────

def test_holiday_feature():
    hol = parse_holidays({"holidays": {"repeating": ["07-04"],
                                       "fixed_dates": ["2026-05-25"]}})
    cf = calendar_features([dt.date(2026, 7, 4), dt.date(2026, 7, 5),
                            dt.date(2026, 5, 25)], hol)
    assert list(cf["is_holiday"]) == [1.0, 0.0, 1.0]
    assert {"dow_sin", "doy_cos", "is_weekend"} <= set(cf.columns)


# ── drought adjustment ───────────────────────────────────────────────────────

def _fc(rows):
    return pd.DataFrame(rows)


def test_no_correction_when_stage_matches_regime():
    spans = [StageSpan("orange", dt.date(2026, 4, 1), None)]
    fc = _fc([{"date": dt.date(2026, 7, 20), "horizon_days": 1,
               "point_mgd": 10.0, "pi_low": 8.0, "pi_high": 12.0,
               "source": "sarima+svm"}])
    out = apply_adjustment(fc, dt.date(2026, 7, 19), spans, COEFFS)
    assert out["correction_mgd"].iloc[0] == pytest.approx(0.0)


def test_rebound_correction_after_stage_lifts():
    spans = [StageSpan("orange", dt.date(2026, 4, 1), dt.date(2026, 7, 20))]
    fc = _fc([{"date": dt.date(2026, 7, 25), "horizon_days": 6,
               "point_mgd": 10.0, "pi_low": 8.0, "pi_high": 12.0,
               "source": "sarima+svm"}])
    out = apply_adjustment(fc, dt.date(2026, 7, 19), spans, COEFFS)
    # regime orange, day none: -(f_none - f_orange)*10 = +0.6
    assert out["correction_mgd"].iloc[0] == pytest.approx(0.6)


def test_gbr_learned_stage_not_double_counted():
    spans = [StageSpan("orange", dt.date(2026, 4, 1), None)]
    fc = _fc([{"date": dt.date(2026, 8, 1), "horizon_days": 12,
               "point_mgd": 10.0, "pi_low": 8.0, "pi_high": 12.0,
               "source": "gbr"}])
    out = apply_adjustment(fc, dt.date(2026, 7, 19), spans, COEFFS,
                           gbr_learned_stages={"orange"})
    assert out["correction_mgd"].iloc[0] == pytest.approx(0.0)


def test_gbr_unlearned_stage_corrected_vs_none():
    spans = [StageSpan("red", dt.date(2026, 7, 25), None)]
    fc = _fc([{"date": dt.date(2026, 8, 1), "horizon_days": 12,
               "point_mgd": 10.0, "pi_low": 8.0, "pi_high": 12.0,
               "source": "gbr"}])
    out = apply_adjustment(fc, dt.date(2026, 7, 19), spans, COEFFS,
                           gbr_learned_stages={"orange"})
    # -(f_red - f_none)*10 = -0.8
    assert out["correction_mgd"].iloc[0] == pytest.approx(-0.8)


# ── calibration ──────────────────────────────────────────────────────────────

def test_calibration_scales_halfwidths():
    factors = {"pi_level": 0.8, "factors": {"ensemble:1-7": 0.5,
                                            "ensemble:8-14": 2.0}}
    fc = _fc([
        {"date": dt.date(2026, 7, 20), "horizon_days": 1,
         "point_mgd": 10.0, "pi_low": 6.0, "pi_high": 14.0},
        {"date": dt.date(2026, 7, 29), "horizon_days": 10,
         "point_mgd": 10.0, "pi_low": 9.0, "pi_high": 11.0},
    ])
    out = cal_apply(fc, factors, member="ensemble")
    assert out["pi_low"].iloc[0] == pytest.approx(8.0)    # shrunk x0.5
    assert out["pi_high"].iloc[0] == pytest.approx(12.0)
    assert out["pi_low"].iloc[1] == pytest.approx(8.0)    # widened x2
    assert out["pi_high"].iloc[1] == pytest.approx(12.0)


# ── svm survives a gap near the anchor (slowest test, ~20 s) ─────────────────

@pytest.mark.slow
def test_svm_gap_adjacent_anchor():
    from demand_ensemble.svm_primary import fit_and_forecast_svm
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-06-01", "2026-07-10", freq="D")
    y = pd.Series(10 + 3 * np.sin(2 * np.pi * np.arange(len(days)) / 365)
                  + rng.normal(0, .5, len(days)),
                  index=[d.date() for d in days])
    y = y.drop([d.date() for d in pd.date_range("2026-06-28", "2026-07-05")])
    cov = pd.DataFrame({"tmax_f": 80.0, "precip_in": 0.0, "et0_in": 0.2},
                       index=[d.date() for d in
                              pd.date_range("2024-06-01", "2026-08-15", freq="D")])
    fut = cov.loc[[d.date() for d in
                   pd.date_range("2026-07-14", "2026-07-20")]].copy()
    fut["horizon_days"] = range(1, 8)
    res = fit_and_forecast_svm(y, cov, fut, dt.date(2026, 7, 13))
    assert len(res.forecasts) == 7
    assert res.forecasts["point_mgd"].notna().all()
