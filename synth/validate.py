"""Validate the shipped synthetic dataset against its calibration targets.

Runs the real analysis code in this repo (behavioral.did,
behavioral.segmentation, nrw, demand_ensemble) against the data actually
committed here, and compares to the aggregate targets the data was
generated to reproduce (see synth/generate_panel.py's baked-in calibration
constants, sourced from a real utility's own data during this project's
development). Used by .github/workflows/refresh-data.yml after each
scheduled data refresh to regenerate the README's validation table --
also safe to run manually any time.

Unlike the private development pipeline this was built from, this script
uses ONLY data shipped in this repo: no debug/internal files, no real API
credentials (weather is served from the shipped state/weather_cache.csv
for this repo's fixed date range). Tier exposure is classified via the
usage-volume PROXY method (behavioral.did.classify_tier_exposure(
source="proxy")) rather than a ground-truth column, since that's what
anyone cloning this repo actually has available.

Run: python -m synth.validate
Writes: synth/validation_results.json
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from statsmodels.tsa.stattools import acf

import customer_store
import nrw
import weather_history as wh_mod
from behavioral import did as did_mod
from behavioral import segmentation as seg_mod
from behavioral import weather_norm
from demand_ensemble.datasets import load_target_series

REPO_ROOT = Path(__file__).parent.parent
PARALLEL_TRENDS_CUTOFF = date(2026, 5, 31)
PLACEBO_DATES = [date(2025, 4, 1), date(2025, 9, 1)]
ACF_LAGS = [1, 7, 14]

# Aggregate targets from the real utility's own data, extracted during this
# project's development (see README's "What this is" for the methodology).
# Never a raw per-account value.
TARGETS = {
    "tier1_step_pct": -10.167,
    "tier23_step_pct": -9.739,
    "did_effect_mgd": -0.2765,
    "did_p_value": 0.0329,
    "parallel_trends_ok": False,
    "segmentation_k": 3,
    "segmentation_silhouette": 0.3397,
    "top_consumers_volume_share_pct": 25.47,
    "vacant_share_pct": 3.21,
    "nrw_correlation_system_a": 0.9884,
    "nrw_correlation_system_b": 0.9387,
    "demand_mean_mgd": 10.7292,
    "demand_std_mgd": 6.8277,
    "demand_acf": {"1": 0.6399, "7": 0.6209, "14": 0.6050},
}


def _group_series(daily: pd.DataFrame, accounts: set, min_frac: float = 0.9) -> pd.DataFrame:
    sub = daily[daily["account_id"].isin(accounts)].dropna(subset=["gallons"])
    per_day = sub.groupby("date").agg(gallons=("gallons", "sum"), n=("account_id", "nunique"))
    per_day.loc[per_day["n"] < min_frac * len(accounts), "gallons"] = float("nan")
    return pd.DataFrame({"production_mgd": per_day["gallons"] / 1e6, "maintenance": 0})


def validate_price_elasticity(cfg: dict, daily: pd.DataFrame, weather: pd.DataFrame,
                              holidays: set) -> dict:
    feature_cols = cfg["model"]["features"]
    tier_2026 = cfg["pricing"]["tiers"][1]
    breakpoints = tier_2026["proxy_breakpoints_gal_per_month"]
    step_date = tier_2026["effective_date"]
    if isinstance(step_date, str):
        step_date = date.fromisoformat(step_date)

    labels = did_mod.classify_tier_exposure(
        daily, date(2026, 1, 1), date(2026, 4, 30), breakpoints, source="proxy")
    tier1 = set(labels[labels == "tier1_only"].index)
    tier23 = set(labels[labels == "tier2_3_exposed"].index)

    tier1_series = _group_series(daily, tier1)
    tier23_series = _group_series(daily, tier23)
    tier1_norm = weather_norm.fit(tier1_series, weather, holidays, feature_cols, date(2026, 3, 31))
    tier23_norm = weather_norm.fit(tier23_series, weather, holidays, feature_cols, date(2026, 3, 31))

    passed, _ = did_mod.check_parallel_trends(tier1_norm.frame, tier23_norm.frame, PARALLEL_TRENDS_CUTOFF)
    result = did_mod.estimate_did(
        tier1_norm.frame, tier23_norm.frame, step_date,
        parallel_trends_ok=passed, placebo_dates=PLACEBO_DATES, is_proxy_classification=True,
    )
    post1 = tier1_norm.frame[tier1_norm.frame.index >= step_date]
    post23 = tier23_norm.frame[tier23_norm.frame.index >= step_date]
    tier1_pct = result.tier1_effect_mgd / post1["expected_mgd"].mean() * 100 if len(post1) else float("nan")
    tier23_pct = result.tier23_effect_mgd / post23["expected_mgd"].mean() * 100 if len(post23) else float("nan")

    return {
        "n_tier1": len(tier1), "n_tier23": len(tier23),
        "tier1_step_pct": round(tier1_pct, 3), "tier23_step_pct": round(tier23_pct, 3),
        "did_effect_mgd": round(result.did_effect_mgd, 4),
        "did_p_value": round(result.did_p_value, 4),
        "parallel_trends_ok": passed,
        "placebo_clean": result.placebo_clean,
        "targets": {k: TARGETS[k] for k in
                    ("tier1_step_pct", "tier23_step_pct", "did_effect_mgd",
                     "did_p_value", "parallel_trends_ok")},
    }


def validate_segmentation(daily: pd.DataFrame) -> dict:
    res = seg_mod.segment(daily)
    total = len(res.assignments)
    top_share = (res.assignments["segment"] == "top_consumers").sum() / total * 100
    vacant_share = (res.assignments["segment"] == "vacant_intermittent").sum() / total * 100
    return {
        "k": res.k, "silhouette": round(res.silhouette, 4),
        "top_consumers_volume_share_pct": round(
            float(res.profiles.loc["top_consumers", "volume_share_pct"]), 2),
        "vacant_share_pct": round(vacant_share, 2),
        "targets": {k: TARGETS[k] for k in
                    ("segmentation_k", "segmentation_silhouette",
                     "top_consumers_volume_share_pct", "vacant_share_pct")},
    }


def validate_nrw(cfg: dict, daily: pd.DataFrame) -> dict:
    production_df = pd.read_csv(REPO_ROOT / cfg["production"]["history_csv"], parse_dates=["date"])
    production_df["date"] = production_df["date"].dt.date
    production_df = production_df.set_index("date")
    entry_history = pd.read_parquet(REPO_ROOT / "state" / "entry_point_history.parquet")
    dw_accounts = {s["beacon_account"] for s in cfg["sites"] if s.get("role") == "dw_master"}
    inputs = nrw.daily_system_inputs(production_df, entry_history, dw_accounts)

    zone_map = pd.read_parquet(REPO_ROOT / "state" / "meter_zone_map.parquet")
    account_system = zone_map.set_index("account_id_new")["system"]
    consumption = nrw.daily_consumption(daily, account_system)
    result = nrw.compute_nrw(inputs, consumption)

    out = {}
    for system, g in result.frame.groupby("system"):
        both = g.dropna(subset=["input_mgd", "consumption_mgd"])
        corr = float(both["input_mgd"].corr(both["consumption_mgd"])) if len(both) > 30 else None
        out[system] = {"correlation": round(corr, 4) if corr is not None else None,
                       "target": TARGETS.get(f"nrw_correlation_{system}")}
    return out


def validate_demand_ensemble(cfg: dict) -> dict:
    demand = load_target_series(REPO_ROOT, cfg, target="system").dropna()
    vals = acf(demand, nlags=max(ACF_LAGS), fft=True)
    return {
        "demand_mean_mgd": round(float(demand.mean()), 4),
        "demand_std_mgd": round(float(demand.std()), 4),
        "demand_acf": {str(lag): round(float(vals[lag]), 4) for lag in ACF_LAGS},
        "targets": {"demand_mean_mgd": TARGETS["demand_mean_mgd"],
                    "demand_std_mgd": TARGETS["demand_std_mgd"],
                    "demand_acf": TARGETS["demand_acf"]},
    }


def main():
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))

    print("Loading synthetic panel...")
    daily = customer_store.read_range(REPO_ROOT)
    daily = customer_store.exclude_estimated(daily)
    daily = daily[~customer_store.flag_catchup_spikes(daily)]
    daily = customer_store.exclude_negative_gallons(daily)

    print("Fetching weather (shipped cache, no token needed for this repo's date range)...")
    weather = wh_mod.fetch_ghcnd(
        cfg["location"]["ncei_station"], "2022-06-01", date(2026, 7, 14).isoformat(),
        REPO_ROOT / "state" / "cache", cfg["api"].get("ncei_token", ""),
        30, cfg["api"]["cache_ttl_hours"]["weather_history"])
    holidays = set()
    for mmdd in cfg["holidays"].get("repeating", []) or []:
        m, d = (int(x) for x in str(mmdd).split("-"))
        for year in range(2022, 2028):
            holidays.add(date(year, m, d))

    results = {}
    print("Validating price elasticity...")
    results["price_elasticity"] = validate_price_elasticity(cfg, daily, weather, holidays)
    print("Validating segmentation...")
    results["segmentation"] = validate_segmentation(daily)
    print("Validating NRW...")
    results["nrw"] = validate_nrw(cfg, daily)
    print("Validating demand ensemble...")
    results["demand_ensemble"] = validate_demand_ensemble(cfg)

    out_path = Path(__file__).parent / "validation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
