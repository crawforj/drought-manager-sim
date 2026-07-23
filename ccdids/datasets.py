"""Target-series and drought-stage-timeline assembly for the CC-DIDS ensemble.

System-wide daily demand (MGD) = system_b's WTP finished water (SCADA) +
BEACON master-meter imports (roles dw_master + emergency). Roles
dw_submeter, mg_internal, and retired are excluded to avoid double
counting — see the sites-block comments in config.yaml.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

AMI_INCLUDE_ROLES = {"dw_master", "emergency"}


GAP_INTERP_LIMIT_DAYS = 3     # per-account gaps up to this are interpolated
DEAD_ACCOUNT_LOOKBACK_DAYS = 90   # no reads in this window -> zero-fill tail
STALE_STORE_MAX_AGE_DAYS = 10     # newest read older than this -> refuse to run


def _interp_short_gaps(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Interpolate ONLY interior NaN runs of length <= limit.

    pandas interpolate(limit=n) fills the first n values of arbitrarily long
    gaps — for a 184-day hole that fabricates 3 values on a line toward an
    endpoint months away. This fills short gaps fully and leaves long gaps
    entirely NaN.
    """
    out = df.interpolate(limit_area="inside")
    for c in df.columns:
        isna = df[c].isna()
        run_id = (~isna).cumsum()
        run_len = isna.groupby(run_id).transform("sum")
        long_gap = isna & (run_len > limit)
        out.loc[long_gap, c] = np.nan
    return out


def _ami_system_daily(proj_root: Path, cfg: dict) -> pd.Series:
    """Sum of included master-meter accounts, daily MGD, NaN on any-gap days.

    Accounts with no reads in the trailing lookback window (dead meters,
    e.g. changeouts not yet mapped) keep their full recorded history and are
    zero-filled only AFTER their last read, so one dead account can't blank
    the whole recent series — and its historical flow isn't silently erased
    from the past either. Per-account gaps up to GAP_INTERP_LIMIT_DAYS are
    interpolated; a day still missing any account's value is left NaN rather
    than summed short — a partial sum would masquerade as a demand drop.

    Raises RuntimeError if the whole store has gone stale (newest read older
    than STALE_STORE_MAX_AGE_DAYS) — a zero-filled "recent" series would
    otherwise silently degrade the system target to SCADA-only.
    """
    state_file = proj_root / cfg.get("beacon", {}).get(
        "state_file", "state/entry_point_history.parquet")
    hist = pd.read_parquet(state_file)
    include = {
        s["beacon_account"] for s in cfg.get("sites", [])
        if s.get("kind") == "beacon" and s.get("role") in AMI_INCLUDE_ROLES
    }
    sub = hist[hist["beacon_account"].isin(include)]
    wide = sub.pivot_table(index="date", columns="beacon_account",
                           values="mgd", aggfunc="first", dropna=False)
    wide.index = pd.to_datetime(wide.index).date
    wide = wide.sort_index()

    last_read = {c: wide[c].last_valid_index() for c in wide.columns}
    # an account with ZERO valid reads contributes nothing knowable — drop it
    # (leaving it would NaN every day via min_count; there is no history to
    # erase, unlike the dead-tail case below)
    no_reads = sorted(c for c, d in last_read.items() if d is None)
    if no_reads:
        log.warning("AMI aggregate: dropping %d account(s) with zero valid "
                    "reads in the store: %s", len(no_reads), ", ".join(no_reads))
        wide = wide.drop(columns=no_reads)
        for c in no_reads:
            del last_read[c]
    if not last_read:
        raise RuntimeError(
            "entry_point_history has no valid reads for any included "
            "account — store empty or corrupt; run ingest_beacon.py")
    newest = max(last_read.values())
    if newest < date.today() - timedelta(days=STALE_STORE_MAX_AGE_DAYS):
        raise RuntimeError(
            f"entry_point_history is stale (newest read {newest}) — run "
            "ingest_beacon.py before forecasting")

    cutoff = date.today() - timedelta(days=DEAD_ACCOUNT_LOOKBACK_DAYS)
    dead = [c for c, d in last_read.items() if d < cutoff]
    for c in dead:
        wide.loc[wide.index > last_read[c], c] = 0.0
    if dead:
        log.warning("AMI aggregate: %d account(s) with no reads since %s "
                    "zero-filled after their last read: %s", len(dead),
                    cutoff, ", ".join(sorted(dead)))

    wide = _interp_short_gaps(wide, GAP_INTERP_LIMIT_DAYS)
    total = wide.sum(axis=1, min_count=len(wide.columns))
    log.info("AMI system aggregate: %d accounts (%d dead-tailed), %d days, "
             "%d complete", len(wide.columns), len(dead), len(total),
             total.notna().sum())
    return total.rename("ami_mgd")


def _production_daily(proj_root: Path, cfg: dict) -> pd.Series:
    """SCADA finished-water MGD; future placeholder rows and NaNs dropped."""
    csv = proj_root / cfg["production"]["history_csv"]
    df = pd.read_csv(csv, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df = df[(df["date"] <= date.today()) & df["production_mgd"].notna()]
    return df.set_index("date")["production_mgd"].rename("scada_mgd")


def load_target_series(proj_root: Path, cfg: dict, target: str = "system") -> pd.Series:
    """Daily demand series the whole ensemble trains against.

    target="system": SCADA production + AMI imports (full system demand)
    target="ami":    AMI master-meter aggregate only (proposal 2b wording)
    """
    ami = _ami_system_daily(proj_root, cfg)
    if target == "ami":
        return ami.dropna().rename("demand_mgd")
    scada = _production_daily(proj_root, cfg)
    joined = pd.concat([scada, ami], axis=1)
    total = (joined["scada_mgd"] + joined["ami_mgd"]).dropna().rename("demand_mgd")
    log.info("System demand series: %d days (%s .. %s)",
             len(total), min(total.index), max(total.index))
    return total


def zone_daily_totals(proj_root: Path) -> dict[str, pd.Series] | None:
    """Pressure-zone daily totals (KGAL->MGD) from the customer store.

    Requires state/meter_zone_map.parquet (meter_id, zone) — the meter->zone
    GIS join that does not exist yet (no coordinate source in the customer
    data; see plan doc). Returns None, with a log line, until it lands.
    """
    zone_map_file = proj_root / "state" / "meter_zone_map.parquet"
    if not zone_map_file.exists():
        log.info("zone_daily_totals: %s not found — pressure-zone models "
                 "skipped (system-wide only)", zone_map_file.name)
        return None
    zone_map = pd.read_parquet(zone_map_file)
    parts = sorted((proj_root / "state" / "customer_history").glob("*.parquet"))
    if not parts:
        return None
    frames = [pd.read_parquet(p) for p in parts]
    cust = pd.concat(frames, ignore_index=True)
    cust = cust.merge(zone_map[["meter_id", "zone"]], on="meter_id", how="inner")
    daily = (cust.groupby(["zone", "date"])["gallons"].sum() / 1e6).rename("mgd")
    return {z: daily.loc[z] for z in daily.index.get_level_values(0).unique()}


# ── Drought-stage timeline ───────────────────────────────────────────────────

@dataclass
class StageSpan:
    stage: str
    start: date
    end: date | None   # None = still active


def stage_history(cfg: dict) -> list[StageSpan]:
    """Stage activations from the ccdids.stage_history config block.

    Validates the timeline hard: spans must not overlap, and only the LAST
    span may be open-ended (end: null). Without this, appending a new stage
    while forgetting to close the previous open span silently shadows the
    new stage for every day (stage_on returns the first match) — corrupting
    GBR dummies, the training regime, and the elasticity correction exactly
    when a stage changes.
    """
    spans = []
    for item in cfg.get("ccdids", {}).get("stage_history", []):
        start = pd.to_datetime(item["start"]).date()
        end = pd.to_datetime(item["end"]).date() if item.get("end") else None
        if end is not None and end < start:
            raise ValueError(f"ccdids.stage_history: span '{item['stage']}' "
                             f"ends {end} before it starts {start}")
        spans.append(StageSpan(stage=str(item["stage"]), start=start, end=end))
    spans = sorted(spans, key=lambda s: s.start)
    for prev, nxt in zip(spans, spans[1:]):
        if prev.end is None:
            raise ValueError(
                f"ccdids.stage_history: span '{prev.stage}' (start {prev.start}) "
                f"is open-ended (end: null) but '{nxt.stage}' starts "
                f"{nxt.start} — close the earlier span with an end date")
        if prev.end >= nxt.start:
            raise ValueError(
                f"ccdids.stage_history: spans '{prev.stage}' (ends {prev.end}) "
                f"and '{nxt.stage}' (starts {nxt.start}) overlap")
    return spans


def stage_on(day: date, spans: list[StageSpan]) -> str:
    """Active stage name on a given day; "none" outside all spans."""
    for s in spans:
        if s.start <= day and (s.end is None or day <= s.end):
            return s.stage
    return "none"
