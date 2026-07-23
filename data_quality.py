"""Shared data-validation primitives — consolidates bounds/freshness/
interpretability-gate checks that would otherwise get reimplemented ad hoc
at each call site (nrw.py, behavioral/segmentation.py, and others).

Built after a day that found several distinct data-integrity bugs, each
fixed with a bespoke one-off check. The synchronized-gap detector below is
the one primitive that would have caught the most recent of them (a
multi-day gap across an entire master-meter fleet) automatically instead
of needing a person's suspicion to surface it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Structural validation ────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Consolidates reservoir.py's and calls_flows.py's near-duplicate
    dataclasses (the latter added n_rows) into one shared shape. Default
    n_rows=0 keeps existing 4-positional-arg call sites working unchanged."""
    ok: bool
    reason: str
    first_date: date | None = None
    last_date: date | None = None
    n_rows: int = 0


# ── Value-bounds checks ──────────────────────────────────────────────────────

def out_of_bounds_mask(
    values: pd.Series,
    min_value: float | None = None,
    max_value: float | None = None,
    *,
    eligible: pd.Series | None = None,
) -> pd.Series:
    """Boolean mask, True where `values` is non-null and outside
    [min_value, max_value] (either bound optional, e.g. min_value=0.0,
    max_value=None for a >=0 check). `eligible`, if given, restricts flagging
    to a subset of rows (mirrors production.py's restriction to
    maintenance==0 rows -- maintenance days are expected to be off-nominal).
    """
    below = values < min_value if min_value is not None else pd.Series(False, index=values.index)
    above = values > max_value if max_value is not None else pd.Series(False, index=values.index)
    mask = values.notna() & (below | above)
    if eligible is not None:
        mask &= eligible
    return mask


def apply_bounds(
    df: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
    *,
    eligible: pd.Series | None = None,
    action: str = "nan",
    label: str = "",
) -> pd.DataFrame:
    """Generalizes production.py's sanity-bound block, DW_METER_SANITY_MAX_MGD's
    single-value check, segmentation.py's negative-gallon drop, and
    supply_usgs.py's flow.where(flow>=0) into one function.

    action="nan": mask the value, keep the row (production.py's convention --
        the historical chart shows a gap rather than silently losing the day).
    action="drop": remove the row entirely (segmentation.py's/supply_usgs.py's
        convention -- used when the row itself, not just the value, is
        untrustworthy).
    """
    mask = out_of_bounds_mask(df[column], min_value, max_value, eligible=eligible)
    if mask.any():
        n = int(mask.sum())
        log.warning("%s: %d value(s) outside [%s, %s] -> %s (sample index: %s%s)",
                    label or column, n, min_value, max_value, action,
                    df.index[mask][:5].tolist(), " ..." if n > 5 else "")
        if action == "drop":
            df = df.loc[~mask].copy()
        else:
            df = df.copy()
            df.loc[mask, column] = float("nan")
    return df


# ── Freshness / staleness ────────────────────────────────────────────────────

@dataclass
class FreshnessResult:
    tier: str                  # "ok" | "warn" | "bad"
    age_days: int | None
    last_date: date | None
    message: str


def check_freshness(
    last_valid_date: date | None,
    *,
    warn_days: int,
    bad_days: int | None = None,
    today: date | None = None,
    label: str = "",
) -> FreshnessResult:
    """Generalizes nrw.py's _STALENESS_TOLERANCE_DAYS, beacon.py's staleness
    concerns, and morning_standup._freshness()'s per-source hardcoded
    thresholds. Deliberately UI-agnostic (returns a tier string, not a
    color) -- callers map tier to their own color scheme.

    bad_days=None means binary ok/bad at warn_days (matches the existing
    master-meter check style); bad_days set gives a three-tier ok/warn/bad.
    """
    today = today or date.today()
    if last_valid_date is None:
        return FreshnessResult("bad", None, None, f"{label}: no valid data".strip(": "))
    age = (today - last_valid_date).days
    if bad_days is not None:
        tier = "bad" if age > bad_days else ("warn" if age > warn_days else "ok")
    else:
        tier = "bad" if age > warn_days else "ok"
    return FreshnessResult(tier, age, last_valid_date, f"{last_valid_date} ({age} d old)")


# ── Interpretability / group-size gates ──────────────────────────────────────

def interpretability_gate(r_squared: float, min_r2: float = 0.30, *, label: str = "") -> tuple[bool, str]:
    """Generalizes MIN_WEATHER_R2 (run_module3_its.py, run_segment_its.py).
    Returns (passed, message); callers still choose what to DO on failure --
    this only centralizes the threshold+message so a future ITS variant
    can't silently omit it (the exact way run_segment_its.py did, until the
    2026-07-17 per-segment gate fix)."""
    passed = r_squared >= min_r2
    return passed, f"{label} weather R²={r_squared:.3f} {'>=' if passed else '<'} {min_r2} gate".strip()


def group_size_floor(n: int, min_n: int = 10, *, label: str = "") -> tuple[bool, str]:
    """Generalizes MIN_SEGMENT_ACCOUNTS (run_segment_its.py) -- a generic
    'is this group too small to trust an aggregate statistic from' floor."""
    passed = n >= min_n
    prefix = f"{label}: " if label else ""
    return passed, f"{prefix}{n} member(s), {'>=' if passed else '<'} {min_n}-member noise floor"


# ── Synchronized-gap detector ────────────────────────────────────────────────

@dataclass
class SyncGapFinding:
    group: str
    start_date: date
    end_date: date
    missing_frac_mean: float
    n_active_mean: float
    n_missing_mean: float
    severity: str    # "warn" (single anomalous day) | "bad" (>= min_consecutive_for_bad days)


def detect_synchronized_gaps(
    long_df: pd.DataFrame,
    *,
    member_col: str,
    date_col: str,
    value_col: str,
    group: str = "",
    min_active_members: int = 5,
    min_fraction: float = 0.5,
    z_threshold: float = 4.0,
    background_window_days: int = 180,
    min_consecutive_for_bad: int = 2,
    analysis_end: date | None = None,
) -> list[SyncGapFinding]:
    """Flags date ranges where an unusually large fraction of a group's
    members are simultaneously missing data -- distinct from, and much
    rarer than, ordinary independent per-member gaps (one meter down for a
    week while the rest are fine). Found live: this is exactly the shape
    of a real bug (an entire master-meter fleet NaN on the same several
    consecutive days, caused by an ingestion lookback window aging out
    before the gap self-healed) that nothing was previously checking for.

    IMPORTANT: "missing" means "no non-null value_col reading exists" for
    that (member, date) -- this covers BOTH representations that occur in
    this codebase: a date with no row at all (e.g. beacon.py's
    daily_by_meter(), where a day BEACON never received a ping for produces
    no row, not even a NaN one) and a date with a present-but-NaN row (e.g.
    an invalid diff). Both must be treated identically or the detector
    silently misses half of what it's meant to catch.

    Algorithm:
    1. Restrict to [analysis_end - background_window_days, analysis_end]
       (default analysis_end = the data's own max date) to bound cost.
    2. Each member's ACTIVE SPAN = (min(date), max(date)) over all its rows
       (present or NaN) in the window. Dates outside a member's own span
       aren't "missing" -- not yet onboarded / already retired -- this is
       what stops a legitimate meter changeout from ever registering as a
       false gap.
    3. Per calendar date: n_active, n_present (non-null), n_missing =
       n_active - n_present, missing_frac = n_missing/n_active (skipped if
       n_active < min_active_members).
    4. Background = trailing background_window_days, excluding the date
       itself (skipped if <30 background days available). Robust stats:
       med = median, robust_std = 1.4826 * MAD.
    5. Flag a date iff BOTH: missing_frac >= min_fraction (absolute floor --
       half or more of the active group down at once; makes a single-meter
       week-long outage, ~1/24 = 4%, mathematically incapable of tripping
       this) AND missing_frac >= med + z_threshold * max(robust_std, 1e-6)
       (relative outlier vs. THIS group's own normal background rate --
       lets a naturally gappier group have a higher normal floor without
       desensitizing the test).
    6. Merge consecutive flagged dates into one finding per contiguous run.
       severity="bad" if the run is >= min_consecutive_for_bad days, else
       "warn" (a single anomalous day -- still worth surfacing, less
       alarming than a multi-day systemic-looking gap).
    """
    if long_df.empty:
        return []

    df = long_df[[member_col, date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    end = pd.Timestamp(analysis_end) if analysis_end else df[date_col].max()
    start = end - pd.Timedelta(days=background_window_days + 30)  # pad for the trailing background lookback
    df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    if df.empty:
        return []

    # Per-member active span within the window.
    spans = df.groupby(member_col)[date_col].agg(["min", "max"])

    all_dates = pd.date_range(start, end, freq="D")
    n_active = pd.Series(0, index=all_dates)
    n_present = pd.Series(0, index=all_dates)

    present = df.dropna(subset=[value_col])
    present_counts = present.groupby(date_col).size()
    n_present.loc[n_present.index.intersection(present_counts.index)] = present_counts

    for _, row in spans.iterrows():
        member_dates = all_dates[(all_dates >= row["min"]) & (all_dates <= row["max"])]
        n_active.loc[member_dates] += 1

    n_missing = (n_active - n_present).clip(lower=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        missing_frac = (n_missing / n_active).where(n_active >= min_active_members)

    flagged = pd.Series(False, index=all_dates)
    for d in all_dates:
        frac = missing_frac.loc[d]
        if pd.isna(frac):
            continue
        bg_start = d - pd.Timedelta(days=background_window_days)
        bg = missing_frac.loc[bg_start:d - pd.Timedelta(days=1)].dropna()
        if len(bg) < 30:
            continue
        med = bg.median()
        robust_std = 1.4826 * (bg - med).abs().median()
        if frac >= min_fraction and frac >= med + z_threshold * max(robust_std, 1e-6):
            flagged.loc[d] = True

    if not flagged.any():
        return []

    findings: list[SyncGapFinding] = []
    flagged_dates = flagged[flagged].index.sort_values()
    run_start = flagged_dates[0]
    prev = flagged_dates[0]
    for d in flagged_dates[1:]:
        if (d - prev).days > 1:
            findings.append(_make_finding(group, run_start, prev, missing_frac, n_active, n_missing,
                                          min_consecutive_for_bad))
            run_start = d
        prev = d
    findings.append(_make_finding(group, run_start, prev, missing_frac, n_active, n_missing,
                                  min_consecutive_for_bad))
    return findings


def _make_finding(group, run_start, run_end, missing_frac, n_active, n_missing,
                  min_consecutive_for_bad) -> SyncGapFinding:
    window = pd.date_range(run_start, run_end, freq="D")
    n_days = len(window)
    return SyncGapFinding(
        group=group,
        start_date=run_start.date(),
        end_date=run_end.date(),
        missing_frac_mean=float(missing_frac.loc[window].mean()),
        n_active_mean=float(n_active.loc[window].mean()),
        n_missing_mean=float(n_missing.loc[window].mean()),
        severity="bad" if n_days >= min_consecutive_for_bad else "warn",
    )


def _self_test() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

    def make_meters(n=24, background_nan_rate=0.02, synced_gap_dates=None, synced_gap_frac=1.0,
                     single_outage_meter=None, single_outage_dates=None):
        rows = []
        gap_set = set(synced_gap_dates) if synced_gap_dates is not None else set()
        outage_set = set(single_outage_dates) if single_outage_dates is not None else set()
        n_synced = int(round(n * synced_gap_frac))
        for i in range(n):
            meter = f"m{i}"
            for d in dates:
                is_nan = rng.random() < background_nan_rate
                if d in gap_set and i < n_synced:
                    is_nan = True
                if meter == single_outage_meter and d in outage_set:
                    is_nan = True
                rows.append({"meter": meter, "date": d, "mgd": np.nan if is_nan else rng.uniform(0.1, 1.0)})
        return pd.DataFrame(rows)

    # Positive case: 24 meters, 3 consecutive days fully synchronized-NaN.
    gap_dates = pd.date_range("2025-07-10", "2025-07-12", freq="D")
    df_pos = make_meters(synced_gap_dates=gap_dates, synced_gap_frac=1.0)
    findings = detect_synchronized_gaps(df_pos, member_col="meter", date_col="date", value_col="mgd",
                                        group="test", analysis_end=date(2025, 12, 31))
    assert len(findings) == 1, f"expected 1 finding, got {findings}"
    f = findings[0]
    assert f.start_date == date(2025, 7, 10) and f.end_date == date(2025, 7, 12), f
    assert f.severity == "bad", f
    print(f"POSITIVE case: {f}")

    # Negative control: one meter down 7 straight days, others clean.
    outage_dates = pd.date_range("2025-07-10", "2025-07-16", freq="D")
    df_neg = make_meters(single_outage_meter="m0", single_outage_dates=outage_dates)
    findings_neg = detect_synchronized_gaps(df_neg, member_col="meter", date_col="date", value_col="mgd",
                                            group="test", analysis_end=date(2025, 12, 31))
    assert findings_neg == [], f"expected no findings, got {findings_neg}"
    print("NEGATIVE control: no findings (correct)")

    # Boundary: 13/24 (54%) down for exactly 1 day.
    df_bound = make_meters(synced_gap_dates=[pd.Timestamp("2025-07-10")], synced_gap_frac=13 / 24)
    findings_bound = detect_synchronized_gaps(df_bound, member_col="meter", date_col="date", value_col="mgd",
                                               group="test", analysis_end=date(2025, 12, 31))
    assert len(findings_bound) == 1, findings_bound
    assert findings_bound[0].severity == "warn", findings_bound
    print(f"BOUNDARY case: {findings_bound[0]}")

    print("\nSELF-TEST PASSED.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    _self_test()
