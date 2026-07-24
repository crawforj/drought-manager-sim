"""Empirical prediction-interval calibration for the demand forecast ensemble.

A backtest once showed miscalibrated intervals in both directions:
SARIMA/ensemble 80% bands covered 100% at 1-7d (too wide), GBR quantile
bands covered 37-47% at 8-30d (too narrow). This module computes per-band
scale factors from the backtest's standardized errors and applies them to
live forecasts.

Standardized error u = (actual - point) / halfwidth, using the upper
halfwidth when the error is positive and the lower when negative. The
scale factor for nominal coverage q is quantile(|u|, q): multiplying both
halfwidths by it makes the interval empirically cover q of the backtest
errors. Factors live in state/pi_calibration.json, written after each
backtest run — so they are always derived from a *previous* evaluation,
and each fresh backtest scores the raw (uncalibrated) member intervals
before recomputing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CALIBRATION_FILE = "pi_calibration.json"
BANDS = [(1, 7), (8, 14), (15, 30)]


def compute_factors(results_csv: Path, pi_level: float = 0.80) -> dict:
    """Per (member, horizon band) interval scale factors from backtest rows."""
    df = pd.read_csv(results_csv)
    df = df.dropna(subset=["pi_low", "pi_high", "actual_mgd"])
    out = {"pi_level": pi_level, "source_rows": len(df), "factors": {}}
    for member in df["member"].unique():
        for lo, hi in BANDS:
            s = df[(df["member"] == member)
                   & df["horizon_days"].between(lo, hi)].copy()
            if len(s) < 30:
                continue
            err = s["actual_mgd"] - s["point_mgd"]
            hw_hi = (s["pi_high"] - s["point_mgd"]).clip(lower=1e-6)
            hw_lo = (s["point_mgd"] - s["pi_low"]).clip(lower=1e-6)
            u = np.where(err >= 0, err / hw_hi, -err / hw_lo)
            factor = float(np.quantile(np.abs(u), pi_level))
            out["factors"][f"{member}:{lo}-{hi}"] = round(factor, 3)
    return out


def load_factors(proj_root: Path) -> dict | None:
    p = proj_root / "state" / CALIBRATION_FILE
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def apply(forecasts: pd.DataFrame, factors: dict | None,
          member: str = "ensemble") -> pd.DataFrame:
    """Scale pi_low/pi_high halfwidths by the stored per-band factor."""
    if not factors:
        return forecasts
    out = forecasts.copy()
    applied = []
    for lo, hi in BANDS:
        f = factors.get("factors", {}).get(f"{member}:{lo}-{hi}")
        if f is None:
            continue
        m = out["horizon_days"].between(lo, hi)
        out.loc[m, "pi_high"] = out.loc[m, "point_mgd"] + \
            f * (out.loc[m, "pi_high"] - out.loc[m, "point_mgd"])
        out.loc[m, "pi_low"] = (out.loc[m, "point_mgd"] - f *
                                (out.loc[m, "point_mgd"] - out.loc[m, "pi_low"])).clip(lower=0.0)
        applied.append(f"{lo}-{hi}d x{f}")
    if applied:
        log.info("PI calibration applied (%s): %s", member, ", ".join(applied))
    return out
