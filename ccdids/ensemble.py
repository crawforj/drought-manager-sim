"""Ensemble combination for the CC-DIDS multi-model demand forecast.

Horizons 1-7: inverse-CV-RMSE weighted blend of SARIMA (baseline) and SVM
(primary). Horizons 8-30: GBR seasonal extension alone. The drought-stage
adjustment is applied to the combined series afterward (proposal Module 2b:
"additive corrections to baseline forecasts when stages are active").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .sarima_baseline import MemberResult

log = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    issue_date: date
    forecasts: pd.DataFrame       # date, horizon_days, point_mgd, pi_low, pi_high,
                                  # member sources, stage, correction_mgd (after adj.)
    weights: dict[str, float]
    members: dict[str, MemberResult] = field(default_factory=dict)
    training_regime: str = "none"
    diagnostics: dict = field(default_factory=dict)   # pipeline-level metadata


def combine(sarima: MemberResult, svm: MemberResult,
            gbr: MemberResult) -> EnsembleResult:
    w_raw = {"sarima": 1.0 / max(sarima.cv_rmse_mgd, 1e-6),
             "svm": 1.0 / max(svm.cv_rmse_mgd, 1e-6)}
    total = sum(w_raw.values())
    w = {k: v / total for k, v in w_raw.items()}
    log.info("Blend weights (inverse CV-RMSE): sarima=%.3f svm=%.3f "
             "(rmse %.3f vs %.3f MGD)", w["sarima"], w["svm"],
             sarima.cv_rmse_mgd, svm.cv_rmse_mgd)

    s = sarima.forecasts.set_index("horizon_days")
    v = svm.forecasts.set_index("horizon_days")
    short_h = sorted(set(s.index) & set(v.index))
    rows = []
    for h in short_h:
        rows.append({
            "date": s.loc[h, "date"],
            "horizon_days": int(h),
            "point_mgd": w["sarima"] * s.loc[h, "point_mgd"] + w["svm"] * v.loc[h, "point_mgd"],
            "pi_low": w["sarima"] * s.loc[h, "pi_low"] + w["svm"] * v.loc[h, "pi_low"],
            "pi_high": w["sarima"] * s.loc[h, "pi_high"] + w["svm"] * v.loc[h, "pi_high"],
            "sarima_mgd": s.loc[h, "point_mgd"],
            "svm_mgd": v.loc[h, "point_mgd"],
            "gbr_mgd": None,
            "source": "sarima+svm",
        })
    for _, r in gbr.forecasts.iterrows():
        rows.append({
            "date": r["date"],
            "horizon_days": int(r["horizon_days"]),
            "point_mgd": r["point_mgd"],
            "pi_low": r["pi_low"],
            "pi_high": r["pi_high"],
            "sarima_mgd": None,
            "svm_mgd": None,
            "gbr_mgd": r["point_mgd"],
            "source": "gbr",
        })

    fc = pd.DataFrame(rows).sort_values("horizon_days").reset_index(drop=True)
    return EnsembleResult(
        issue_date=sarima.issue_date,
        forecasts=fc,
        weights=w,
        members={"sarima": sarima, "svm": svm, "gbr": gbr},
    )
