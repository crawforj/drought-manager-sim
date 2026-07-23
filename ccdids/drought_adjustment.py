"""CC-DIDS drought-stage adjustment: behavioral elasticities as additive
corrections.

Applies a regime-delta contract:

    correction(day) = -(f[stage(day)] - f[stage_train]) * point_forecast(day)

where f[stage] is the fractional demand reduction for a stage from
state/drought_sim_coefficients.json (Module 3 outputs; orange is
"measured-here" from the ITS, others literature-anchored/assumption) and
stage_train is the stage prevailing over the trailing training window. The
correction is exactly zero while the forecast-day stage matches the regime
the statistical models were just trained on — it activates only across stage
transitions inside the forecast window, where a purely statistical model is
blind. p5/p95 coefficient spread widens the interval on corrected days.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .datasets import StageSpan, stage_on

log = logging.getLogger(__name__)

TRAIN_REGIME_WINDOW_DAYS = 28


def load_stage_coefficients(proj_root: Path) -> dict[str, dict]:
    """{stage: {f_p5, f_p50, f_p95, tier}} from drought_sim_coefficients.json."""
    p = proj_root / "state" / "drought_sim_coefficients.json"
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)["stages"]


def training_regime(t_last: date, spans: list[StageSpan]) -> str:
    """Modal stage over the trailing training window ending at `t_last`,
    the anchor the models were actually fit through -- NOT `issue_date`.
    SARIMA/SVM's demand series ends 1-3 days before issue_date (AMI
    ingestion lag; see sarima_baseline.py's module docstring / t_last), so
    anchoring this window at issue_date instead can attribute a stage
    transition that falls inside that lag gap to the wrong regime."""
    days = [t_last - timedelta(days=k) for k in range(TRAIN_REGIME_WINDOW_DAYS)]
    return Counter(stage_on(d, spans) for d in days).most_common(1)[0][0]


def apply_adjustment(forecasts: pd.DataFrame, t_last: date,
                     spans: list[StageSpan], coeffs: dict[str, dict],
                     gbr_learned_stages: set[str] | None = None) -> pd.DataFrame:
    """Add stage / correction_mgd columns and shift point + interval.

    forecasts: date, horizon_days, point_mgd, pi_low, pi_high, source
    (combined ensemble output). Returns a copy with the adjustment applied.

    Per-row reference regime: SARIMA/SVM rows are corrected relative to the
    trailing training regime (they carry no stage awareness). GBR rows
    already encode learned stages as dummy features, so correcting those
    again would double count — GBR rows are corrected only for stages the
    GBR could NOT learn (no training days), relative to "none" (an unlearned
    stage's dummy is all-zeros in training, i.e. the model predicts as if
    unstaged).

    `t_last` (the last day of real observed data the models were fit
    through -- see pipeline.py) anchors the trailing regime window, not
    `issue_date` (previously used here): a stage transition falling inside
    the AMI-ingestion-lag gap between t_last and issue_date was otherwise
    attributed to whichever regime issue_date happened to fall in, not the
    regime the models actually trained under.
    """
    out = forecasts.copy()
    gbr_learned = gbr_learned_stages or set()
    regime = training_regime(t_last, spans)

    stages, corrections, lo_shift, hi_shift = [], [], [], []
    for _, row in out.iterrows():
        s = stage_on(row["date"], spans)
        stages.append(s)
        is_gbr = row.get("source") == "gbr"
        if is_gbr and (s in gbr_learned or s == "none"):
            ref = None                       # GBR already models this stage
        else:
            ref = "none" if is_gbr else regime
        if ref is None or s == ref:
            corrections.append(0.0)
            lo_shift.append(0.0)
            hi_shift.append(0.0)
            continue
        c = coeffs.get(s, coeffs["none"])
        f_ref = coeffs.get(ref, coeffs["none"])["f_p50"]
        delta50 = c["f_p50"] - f_ref
        corrections.append(-delta50 * row["point_mgd"])
        if abs(delta50) > 1e-12:
            # coefficient uncertainty (p5..p95) widens the band asymmetrically
            deltas = [c["f_p5"] - f_ref, c["f_p95"] - f_ref]
            shifts = [-d * row["point_mgd"] for d in deltas]
            lo_shift.append(min(shifts))
            hi_shift.append(max(shifts))
        else:
            lo_shift.append(0.0)
            hi_shift.append(0.0)

    out["stage"] = stages
    out["correction_mgd"] = corrections
    out["point_mgd"] = (out["point_mgd"] + out["correction_mgd"]).clip(lower=0.0)
    out["pi_low"] = (out["pi_low"] + lo_shift).clip(lower=0.0)
    out["pi_high"] = out["pi_high"] + hi_shift
    out.attrs["training_regime"] = regime

    n_active = sum(1 for c in corrections if abs(c) > 1e-9)
    log.info("Drought adjustment: training regime '%s'; corrections active on "
             "%d/%d forecast day(s)", regime, n_active, len(out))
    return out
