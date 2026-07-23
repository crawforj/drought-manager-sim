"""Auto-selection between the OLS and GBM demand models based on rolling accuracy.

User request 2026-07-21: rather than a human periodically checking 30-day MAPE
and manually flipping config.yaml's model.primary (and separately toggling
email.enabled off while a model is underperforming, as happened 2026-07-20),
run_forecast.py picks whichever model is currently more accurate each run and
persists that choice here so render_dashboard.py's forecast panel and
run_forecast.py's email always agree on which model backs "today's number".

Only ever compares REAL (non-provisional) mape_30d values -- model.py's own
docstring is explicit that provisional MAPE (fewer than MAPE_MIN_ROWS_FULL
paired forecast/actual rows) is display-only and must not feed model
comparison/selection. If either side lacks a real mape_30d yet (early in the
project, or after a long data gap), there isn't enough signal to compare, so
selection falls back to config.yaml's model.primary rather than guessing.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

STATE_FILENAME = "state/model_primary.json"


def select(result_ols, result_gbm, configured_default: str) -> tuple[str, str]:
    """Return (chosen_model, reason). chosen_model is "ols" or "gbm"."""
    if result_ols is None:
        return "gbm", "OLS did not run this cycle"
    if result_gbm is None:
        return "ols", "GBM did not run this cycle"

    ols_mape, gbm_mape = result_ols.mape_30d, result_gbm.mape_30d
    if ols_mape is None or gbm_mape is None:
        return (
            configured_default,
            "not enough paired forecast/actual history yet for a real 30-day MAPE "
            "on both models -- using configured default rather than comparing "
            "provisional numbers",
        )
    if ols_mape == gbm_mape:
        return configured_default, f"MAPE tied at {ols_mape:.1f}% -- keeping configured default"
    if ols_mape < gbm_mape:
        return "ols", f"OLS 30-day MAPE {ols_mape:.1f}% beats GBM's {gbm_mape:.1f}%"
    return "gbm", f"GBM 30-day MAPE {gbm_mape:.1f}% beats OLS's {ols_mape:.1f}%"


def write(project_root: Path, chosen_model: str, reason: str,
          result_ols, result_gbm, issue_date) -> None:
    path = project_root / STATE_FILENAME
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "model": chosen_model,
        "reason": reason,
        "issue_date": str(issue_date),
        "mape_ols": result_ols.mape_30d if result_ols is not None else None,
        "mape_gbm": result_gbm.mape_30d if result_gbm is not None else None,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")


def read_primary(project_root: Path, configured_default: str) -> str:
    """For consumers (render_dashboard.py) that don't refit models themselves.

    Falls back to configured_default if run_forecast.py hasn't run yet today
    (or ever) -- same missing-state-file fallback pattern used throughout
    this project (e.g. customer_pilot.parquet when the full backfill isn't
    ready).
    """
    path = project_root / STATE_FILENAME
    if not path.exists():
        return configured_default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("model", configured_default)
    except Exception:
        return configured_default
