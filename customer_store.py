"""Monthly-partitioned local store for customer-meter daily consumption.

state/customer_history/YYYY-MM.parquet — one file per calendar month, tidy
rows: account_id, meter_id, date, gallons, class_code, estimated.

Partitioned because the full population is ~23k rows/day (~8.4M rows/year):
a single file would work today but partitioning keeps upserts cheap (a 3-day
lookback touches at most 2 partitions) and lets analytics read only the
months they need. Same new-data-wins merge semantics as the master-meter
store (ingest_beacon.py) — re-running any ingest/backfill is always safe.

Local-disk for now, same as storage.py; swaps to Blob with the Phase 2
migration alongside everything else.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

import data_quality as dq

log = logging.getLogger(__name__)

STORE_DIR_NAME = "state/customer_history"
_COLUMNS = ["account_id", "meter_id", "date", "gallons", "class_code", "estimated"]


def _store_dir(proj_root: Path) -> Path:
    d = proj_root / STORE_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


# If any backfill log was written within this window, assume a backfill is
# still running. Shared guard (moved here from ingest_customers.py same day it
# was built, 2026-07-17) for anything that shouldn't run concurrently with a
# backfill: the daily ingest (BEACON allows ~one concurrent export -- see M24)
# and the weekly statistical-refresh precomputes (which read the whole store
# while the backfill is rewriting partitions). Log mtime beats a lockfile:
# the backfill logs at least once per ~20-min batch, so a crashed backfill
# stops refreshing its log and this guard self-clears, with no stale-lockfile
# cleanup to forget.
BACKFILL_ACTIVE_WINDOW_MINUTES = 45


def backfill_active(proj_root: Path) -> str | None:
    """Name + age of the freshest live-looking backfill log, else None."""
    import time
    for logfile in sorted(proj_root.glob("backfill_customers*.log")):
        age_minutes = (time.time() - logfile.stat().st_mtime) / 60
        if age_minutes < BACKFILL_ACTIVE_WINDOW_MINUTES:
            return f"{logfile.name} (written {age_minutes:.0f} min ago)"
    return None


def aggregate_to_daily(tidy: pd.DataFrame) -> pd.DataFrame:
    """Collapse sub-daily interval rows to one row per (account_id, date).

    BEACON's v1 EDS endpoint returns per-~6-hour-interval Flow, not a daily
    total (confirmed real values like 328.8 gal / 93.6 gal for two
    consecutive intervals on one meter -- see beacon_customers.py's module
    docstring). This module's own docstring has always promised "daily
    consumption," but nothing previously enforced that: multiple interval
    rows sharing the same account/date were written to the store as-is.
    Every downstream daily analysis assumed one row per account-day anyway
    (revision_sweep.py's diff_and_log merge, flag_catchup_spikes's rolling
    30-*day* window below, nrw.py) and was silently wrong as a result.
    Aggregating once, here, at the single write path into the store, fixes
    every one of those call sites at once instead of requiring each to
    re-aggregate defensively.

    gallons: summed (a missing/unparseable interval, NaN after
    pd.to_numeric upstream, is excluded by pandas' default skipna sum rather
    than poisoning the whole day to NaN). meter_id/class_code: last value of
    the day (population is one row per account-day going forward; a
    same-day meter swap is the rare case revision_sweep.py's own endpoint-
    transition tracking exists to catch, not something this rollup needs to
    resolve). estimated: "1" if ANY interval that day was BEACON-flagged
    estimated -- a daily total partly built from an estimated interval is
    itself not a fully trustworthy real total.
    """
    if tidy.empty:
        return tidy
    grouped = (
        tidy.sort_values(["account_id", "date"])
        .groupby(["account_id", "date"], as_index=False)
        .agg(
            meter_id=("meter_id", "last"),
            gallons=("gallons", "sum"),
            class_code=("class_code", "last"),
            estimated=("estimated", lambda s: "1" if (s == "1").any() else (s.iloc[-1] if len(s) else "")),
        )
    )
    return grouped[_COLUMNS]


def upsert(proj_root: Path, tidy: pd.DataFrame) -> dict:
    """Merge tidy rows into the monthly partitions (new data wins on
    (account_id, date) collision). Returns a per-partition summary dict."""
    store = _store_dir(proj_root)
    tidy = tidy[_COLUMNS].copy()
    # NOT isinstance(d, date): pandas NaT duck-types as a date instance (True!)
    # despite NaT.year silently returning float('nan') instead of raising --
    # confirmed live 2026-07-16, this let NaT rows through and crashed
    # _month_key's f-string formatting. pd.isna() is the correct check.
    bad_date = tidy["date"].map(pd.isna)
    if bad_date.any():
        log.warning("upsert: dropping %d row(s) with unparseable/missing date", int(bad_date.sum()))
        tidy = tidy[~bad_date]
    pre_agg_rows = len(tidy)
    tidy = aggregate_to_daily(tidy)
    if len(tidy) != pre_agg_rows:
        log.info("upsert: aggregated %d interval row(s) into %d account-day row(s)",
                 pre_agg_rows, len(tidy))
    summary = {}
    for key, chunk in tidy.groupby(tidy["date"].map(_month_key)):
        path = store / f"{key}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            collisions = set(zip(chunk["account_id"], chunk["date"]))
            existing = existing[
                ~existing.set_index(["account_id", "date"]).index.isin(collisions)
            ]
            merged = pd.concat([existing, chunk], ignore_index=True)
        else:
            merged = chunk
        merged = merged.sort_values(["account_id", "date"]).reset_index(drop=True)
        merged.to_parquet(path)
        summary[key] = {"added_or_updated": len(chunk), "partition_total": len(merged)}
        log.info("partition %s: +%d rows (now %d)", key, len(chunk), len(merged))
    return summary


def read_range(proj_root: Path, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Concatenate partitions intersecting [start, end] (inclusive; None = open)."""
    store = _store_dir(proj_root)
    frames = []
    for path in sorted(store.glob("*.parquet")):
        key = path.stem  # YYYY-MM
        year, month = int(key[:4]), int(key[5:7])
        first = date(year, month, 1)
        last = date(year + (month == 12), (month % 12) + 1, 1)
        if (start is None or last > start) and (end is None or first <= end):
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df


def exclude_estimated(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where BEACON flagged the read as estimated (Estimated Flag ==
    "1") -- these run ~6x lower at the mean / ~44x lower at the median than real
    reads (measured 2026-07-16 against customer_pilot.parquet + the full backfill)
    and must not silently anchor consumption totals, NRW balance, or segmentation.

    Treated as missing, not low-confidence-but-kept: composes for free with every
    existing completeness/staleness guard downstream (min_fleet_fraction in nrw.py,
    MIN_VALID_DAYS in segmentation.py) -- a day that's mostly-fabricated for a given
    account naturally starts looking incomplete once those rows are gone, no new
    logic needed there. Only affects the DataFrame returned here, never mutates the
    stored parquet rows -- if revision_sweep.py later heals a row to a real value,
    the next call to this function picks up the corrected value automatically.

    Not baked into read_range() itself: revision_sweep.py legitimately wants the
    raw estimated flag to study mutability, and the pilot path
    (state/customer_pilot.parquet) bypasses read_range() entirely, so callers must
    apply this explicitly at each load site rather than relying on a hidden default.
    """
    if "estimated" not in daily.columns:
        return daily
    excluded = daily["estimated"] == "1"
    if excluded.any():
        log.info("exclude_estimated: dropping %d/%d row(s) flagged estimated by BEACON (%.2f%%)",
                 int(excluded.sum()), len(daily), 100 * excluded.sum() / len(daily))
    return daily[~excluded]


def exclude_negative_gallons(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with negative gallons -- register rollover/replacement
    artifacts (a cumulative-register reset or changeout can produce a huge
    negative "delta" for one day, observed in real fleets at a scale of
    tens of millions of gallons in a single row). Physically impossible,
    not just low-confidence, so dropped rather than clamped to zero
    (clamping would still let the row's presence imply a real reading
    happened).

    Thin wrapper around data_quality.apply_bounds -- the same primitive
    segmentation.py's build_features() already applies inline for its own
    feature computation (kept there, not migrated, per M13). This is the
    first time it's applied on the NRW/mass-balance consumption path; before
    this, nrw.py had no negative-gallon filter at all. Note: dropping a
    negative row makes a day's total consumption go UP (removing a value
    that was dragging the sum down), so NRW_mgd (input - consumption) goes
    DOWN -- this fix biases NRW% down, not up. Measured live 2026-07-17:
    -39.5 MG / 23,581 rows over the NRW panel's 550-day window (~0.07 MGD/day
    avg, median row -0.1 gal but one confirmed at -29,998,927 gal) -- too
    small to move the headline number at 1-decimal precision, and it moves
    NRW% further from the ~20% ops expects, not toward it, so it's not a
    candidate explanation for the M14 too-low reading. Corrects a real gap
    regardless.

    Not baked into read_range() for the same reason exclude_estimated() isn't
    -- callers apply it explicitly at each load site.
    """
    return dq.apply_bounds(daily, "gallons", min_value=0.0, action="drop", label="customer gallons")


def flag_catchup_spikes(daily: pd.DataFrame, spike_factor: float = 5.0,
                        min_gap_days: int = 3) -> pd.Series:
    """Boolean mask marking probable post-outage catch-up reads.

    When a dead endpoint is repaired/replaced, the first read after the gap
    can report the accumulated consumption of the whole outage as one giant
    "day" — poison for daily analytics if taken literally. Flag rows where:
      - the account had a reporting gap of >= min_gap_days immediately before, AND
      - the value exceeds spike_factor x that account's trailing 30-day median.

    Callers should exclude or redistribute flagged rows in event-window
    analyses (Module 3 / NRW). Kept as a mask, not a mutation — the store
    always holds what BEACON reported.

    Returns a Series reindexed to match `daily`'s original row order (not the
    account_id/date-sorted order used internally) -- `df[~flag_catchup_spikes(df)]`
    must align cleanly at the call site without triggering pandas' boolean-key
    reindex warning, which relies on the index being a set match, not an
    order match.
    """
    original_index = daily.index
    sorted_daily = daily.sort_values(["account_id", "date"])
    flags = pd.Series(False, index=sorted_daily.index)
    for _, g in sorted_daily.groupby("account_id"):
        dates = pd.to_datetime(pd.Series(g["date"].values))
        gap_before = dates.diff().dt.days.fillna(1).values
        med = g["gallons"].rolling(30, min_periods=7).median().shift(1).values
        spike = (gap_before >= min_gap_days) & (g["gallons"].values > spike_factor * pd.Series(med).fillna(float("inf")).values)
        flags.loc[g.index] = spike
    return flags.reindex(original_index)


def summary(proj_root: Path) -> str:
    store = _store_dir(proj_root)
    parts = sorted(store.glob("*.parquet"))
    if not parts:
        return "customer_history: empty"
    total = 0
    lines = []
    for p in parts:
        n = len(pd.read_parquet(p, columns=["date"]))
        total += n
        lines.append(f"  {p.stem}: {n:,} rows")
    return f"customer_history: {total:,} rows across {len(parts)} partition(s)\n" + "\n".join(lines)
