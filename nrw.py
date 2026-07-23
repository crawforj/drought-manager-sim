"""Non-revenue water (NRW) monitoring — daily water balance per hydraulic system.

    NRW = system input − metered customer consumption − unbilled authorized

Inputs side: system_a's input is the sum of its master meters
(entry_point_history.parquet); system_b's input is finished-water
production (production_history.csv). `python nrw.py` runs the self-test on
synthetic data.

Daily balance, 7-day rolling NRW%, trend-based alerting (7d NRW% >
trailing-90d mean + margin for >=3 consecutive days) — the AWWA M36
ratio-of-sums convention, see compute_nrw()'s docstring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ALERT_MARGIN_PCT_POINTS = 5.0
ALERT_MIN_CONSECUTIVE_DAYS = 3
_TRAILING_BASELINE_DAYS = 90
# A single master meter reading beyond this is a register/changeout
# artifact, not real flow -- excluded from the input sum rather than
# corrupting the day's balance.
DW_METER_SANITY_MAX_MGD = 10.0


def daily_system_inputs(
    production_df: pd.DataFrame,
    entry_history: pd.DataFrame,
    dw_master_accounts: set,
) -> pd.DataFrame:
    """Tidy per-system daily input volumes: system, date, input_mgd.

    system_b = WTP finished production (SCADA import).
    system_a = sum of role=dw_master meters only (engineering-verified list —
    excludes retired meters, downstream sub-meters, and emergency
    interconnects); days missing more than 2 meters' valid reads become NaN,
    not a low sum (partial-fleet sums are noise, not data).
    """
    frames = []

    mg = production_df["production_mgd"].dropna()
    frames.append(pd.DataFrame({"system": "system_b", "date": mg.index, "input_mgd": mg.values}))

    dw = entry_history[entry_history["beacon_account"].isin(dw_master_accounts)].dropna(subset=["mgd"])
    import data_quality as dq
    n_artifact = int(dq.out_of_bounds_mask(dw["mgd"], 0.0, DW_METER_SANITY_MAX_MGD).sum())
    if n_artifact:
        log.warning("daily_system_inputs: %d DW meter-day(s) beyond %.0f MGD sanity bound -- "
                    "excluding from the input sum (register/changeout artifact)",
                    n_artifact, DW_METER_SANITY_MAX_MGD)
        dw = dq.apply_bounds(dw, "mgd", 0.0, DW_METER_SANITY_MAX_MGD, action="drop")
        # Drops just the offending meter-day, not the whole day (unlike
        # build_dw_series()'s whole-day-NaN, chosen there because ANY spike
        # ruins that series' weather-normalization use). Here the existing
        # full_fleet_n-2 completeness check below still NaNs the day if
        # dropping pushes too many meters missing -- a single bad meter
        # doesn't need to cost the whole day's balance.
    per_day = dw.groupby("date").agg(total=("mgd", "sum"), n=("mgd", "size"))
    if not per_day.empty:
        full_fleet_n = int(per_day["n"].quantile(0.95))
        per_day.loc[per_day["n"] < full_fleet_n - 2, "total"] = float("nan")
        frames.append(pd.DataFrame({
            "system": "system_a", "date": per_day.index, "input_mgd": per_day["total"].values,
        }))

    return pd.concat(frames, ignore_index=True)


def load_account_system_map(proj_root: Path) -> pd.Series | None:
    """account_id -> "system_a"/"system_b" Series for daily_consumption(),
    built from state/meter_zone_map.parquet. Returns None if the parquet
    doesn't exist yet.

    customer_history's own `account_id` column can mix two account-numbering
    eras across different backfill partitions, so both `account_id_old` and
    `account_id_new` are mapped to the same system, letting a single
    `.map()` call work regardless of which era a given row came from.
    Unclassified-zone accounts are deliberately left OUT of the map rather
    than given a third "unclassified" system value, so they fall into
    daily_consumption's normal "unmapped, excluded" path instead of
    silently forming a bucket compute_nrw would just drop anyway (no
    matching "unclassified" row exists on the inputs side).
    """
    path = proj_root / "state" / "meter_zone_map.parquet"
    if not path.exists():
        return None
    zm = pd.read_parquet(path)
    zm = zm[zm["system"] != "unclassified"]
    pairs = pd.concat([
        zm[["account_id_new", "system"]].rename(columns={"account_id_new": "account_id"}),
        zm[["account_id_old", "system"]].rename(columns={"account_id_old": "account_id"}),
    ], ignore_index=True)
    pairs = pairs.dropna(subset=["account_id"])
    pairs = pairs[pairs["account_id"] != ""]
    # A handful of accounts have 2+ meters (e.g. duplex/multi-unit premises)
    # that could in principle straddle a zone boundary and disagree on
    # system -- drop_duplicates(keep="first") picks one deterministically
    # rather than raising; systems are consistent for the ~23k population
    # for anything this would plausibly affect.
    pairs = pairs.drop_duplicates(subset="account_id", keep="first")
    return pairs.set_index("account_id")["system"]


def resolve_unmapped_via_meter_id(
    account_system: pd.Series,
    customer_daily: pd.DataFrame,
    proj_root: Path,
) -> pd.Series:
    """Extend account_system with a meter_id fallback, for accounts whose
    account_id doesn't match meter_zone_map.parquet's account_id_new/
    account_id_old but whose meter_id (present on every customer_store row)
    DOES appear in that same file's own meter_id column -- a small,
    unconditional recovery pass for accounts a GIS export update would
    otherwise leave permanently unmapped.
    """
    path = proj_root / "state" / "meter_zone_map.parquet"
    if not path.exists():
        return account_system
    zm = pd.read_parquet(path)
    if "meter_id" not in zm.columns:
        return account_system
    zm = zm[["meter_id", "system"]]
    zm = zm[zm["system"] != "unclassified"]
    meter_system = zm.drop_duplicates(subset="meter_id", keep="last").set_index("meter_id")["system"]

    still_unmapped = customer_daily.loc[
        ~customer_daily["account_id"].isin(account_system.index), ["account_id", "meter_id"]
    ].drop_duplicates(subset="account_id")
    via_meter = still_unmapped.set_index("account_id")["meter_id"].map(meter_system).dropna()
    if via_meter.empty:
        return account_system
    log.info("resolve_unmapped_via_meter_id: recovered %d account(s) via meter_id fallback "
             "that account_id_new/account_id_old missed", len(via_meter))
    return pd.concat([account_system, via_meter])


def daily_consumption(
    customer_daily: pd.DataFrame,
    account_system: pd.Series,
    min_fleet_fraction: float = 0.95,
) -> pd.DataFrame:
    """Tidy per-system daily consumption: system, date, consumption_mgd.

    customer_daily: account_id, date, gallons (one row per account-day).
    account_system: Series mapping account_id -> "system_b"/"system_a"
    (from the verified zone->system table; accounts not in the map are
    excluded and counted in the log — infrastructure/PRV/unknown meters
    must not silently deflate consumption).

    Days where fewer than min_fleet_fraction of that system's mapped accounts
    reported become NaN — a partially-reported day would read as fake loss.

    Completeness is judged against the fleet size AS OF EACH DAY (the count
    of distinct accounts that have appeared in the data by that date, not
    a single global count across the whole window). Found 2026-07-20
    building run_module3_its.build_dw_customer_series(): the Denver-mapped
    fleet grows for real over a multi-year window (~11,950 accounts/day
    reporting Jan 2023 -> ~14,700+ by mid-2026, smooth, no mid-series drops
    -- genuine new connections, not a data gap), so comparing every
    historical day against the FINAL fleet size wrongly NaN'd out most of
    2023-2024 for having a legitimately smaller population that year. Over
    a short window (the dashboard's usual case) the two approaches barely
    differ, since the fleet is roughly stable either way -- this fix is
    free there and necessary for long windows.
    """
    df = customer_daily.dropna(subset=["gallons"]).copy()
    # Same non-billed/master-meter exclusion as utility_wide_frames(): a
    # handful of master-meter-class accounts can carry a real zone
    # assignment and, if not excluded here, get summed as "customer
    # consumption" -- inflating a system's apparent usage well past its own
    # production and producing a physically-impossible negative NRW%.
    df = df[~df["class_code"].fillna("").str.startswith(("0", "MM"))]
    df["system"] = df["account_id"].map(account_system)
    unmapped = df["system"].isna().sum()
    if unmapped:
        log.info("daily_consumption: %d rows from unmapped accounts excluded", unmapped)
    df = df.dropna(subset=["system"])

    out = []
    for system, g in df.groupby("system"):
        # Eligible fleet size as of each date: cumulative count of accounts
        # whose FIRST appearance in the data is on or before that date.
        # searchsorted on the sorted array of first-seen dates gives this
        # in one vectorized pass, no look-ahead (a day's eligible fleet
        # never counts an account that first appears later).
        first_seen = np.sort(g.groupby("account_id")["date"].min().to_numpy())
        per_day = g.groupby("date").agg(gallons=("gallons", "sum"), n=("account_id", "nunique"))
        eligible_fleet = np.searchsorted(first_seen, per_day.index.to_numpy(), side="right")
        per_day.loc[per_day["n"] < min_fleet_fraction * eligible_fleet, "gallons"] = float("nan")
        out.append(pd.DataFrame({
            "system": system, "date": per_day.index,
            "consumption_mgd": per_day["gallons"].values / 1e6,
        }))
    return pd.concat(out, ignore_index=True)


def allocate_unmapped_consumption(
    consumption_sys: pd.DataFrame,
    customer_daily: pd.DataFrame,
    account_system: pd.Series,
) -> pd.DataFrame:
    """Pro-rate no-confirmed-zone billed volume across systems by each
    system's same-day share of KNOWN (mapped) consumption, so the per-system
    split sums back to the utility-wide total instead of dropping unmapped
    gallons off both ledgers as apparent loss.

    This is the fix for a gap surfaced in the dashboard's per-system NRW
    caveat: utility_wide_frames()
    correctly nets unmapped consumption out of its one total (it doesn't care
    which system an account belongs to), but daily_consumption() above simply
    drops unmapped rows, so that same real consumption vanished from BOTH
    per-system ledgers at once and inflated both systems' apparent NRW% by
    the same gallons. Pro-rata by same-day known-consumption share is an
    ASSUMPTION, not a measurement -- we don't actually know which system an
    unmapped account is on until it gets a confirmed zone (see
    import_meter_zones.py) -- but it is the AWWA M36 convention for
    apportioning an unattributed volume across known cost centers, and it is
    the only allocation that keeps sum(per-system consumption) == utility-
    wide consumption exactly, which is the property that matters here: it
    stops the double-counted "missing twice" artifact even though the
    per-system split it produces is still an estimate for this slice.

    customer_daily: same cleaned (non-estimated/non-spike/non-negative)
    frame daily_consumption() was called with, so the unmapped population is
    filtered the same way the mapped one was.
    """
    df = customer_daily.dropna(subset=["gallons"]).copy()
    df = df[~df["class_code"].fillna("").str.startswith(("0", "MM"))]
    mapped = df["account_id"].map(account_system)
    unmapped_daily_mgd = df.loc[mapped.isna()].groupby("date")["gallons"].sum() / 1e6
    if unmapped_daily_mgd.empty:
        return consumption_sys

    pivot = consumption_sys.pivot(index="date", columns="system", values="consumption_mgd")
    day_total = pivot.sum(axis=1, min_count=1)
    share = pivot.div(day_total, axis=0)
    share_long = share.stack().rename("share").reset_index()

    out = consumption_sys.merge(share_long, on=["date", "system"], how="left")
    unmapped_mgd = out["date"].map(unmapped_daily_mgd).fillna(0.0)
    out["consumption_mgd"] = out["consumption_mgd"] + out["share"].fillna(0.0) * unmapped_mgd
    return out.drop(columns="share")


_STALENESS_TOLERANCE_DAYS = 14


def utility_wide_frames(
    customer_daily: pd.DataFrame,
    inputs: pd.DataFrame,
    min_fleet_fraction: float = 0.90,
    total_accounts_estimate: int | None = None,
    min_population_fraction: float = 0.90,
    today: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Zone-free utility-wide balance inputs: aggregate BOTH systems' inputs
    and ALL billed customers to a single 'utility' system, so NRW% works
    before the zone->system mapping exists (the per-system split activates
    later once a GIS zone join is available).

    Guards, all essential:
      - Non-billed meters excluded (class codes starting '0' or 'MM');
        master meters must never sit on the consumption side.
      - Per-day completeness: the store self-calibrates full population as
        the max daily billed-account count *seen so far*, and days below
        min_fleet_fraction of that get NaN'd. This alone is NOT enough while
        a backfill is still running: self-calibrated population tracks
        whatever fraction of the fleet has been backfilled, not the true
        fleet size, so a "complete" day at 50% backfill still only
        represents half of real consumption -- a fake ~50% "loss" that has
        nothing to do with actual non-revenue water.
      - True-population completeness: when total_accounts_estimate is given,
        the self-calibrated population itself must reach
        min_population_fraction of it, or this returns None so the caller
        falls back to the pilot-scaled estimate (nrw.pilot_scaled_frames),
        which applies real scaling and stays current during the backfill.
      - Staleness: the most recent passing day must be within
        _STALENESS_TOLERANCE_DAYS of `today`, or this returns None. A stalled
        backfill must not let an old day silently stand in for "current."
      - Input completeness: utility input needs BOTH systems present that
        day (min_count=2) — with WTP production stale past its last import,
        the balance simply ends there until a fresh SCADA export.

    Returns (inputs_utility, consumption_utility) or None if no day passes.
    """
    billed = customer_daily.dropna(subset=["gallons"]).copy()
    billed = billed[~billed["class_code"].fillna("").str.startswith(("0", "MM"))]
    if billed.empty:
        return None

    per_day = billed.groupby("date").agg(gallons=("gallons", "sum"),
                                         n=("account_id", "nunique"))
    full_pop = int(per_day["n"].max())
    if total_accounts_estimate and full_pop < min_population_fraction * total_accounts_estimate:
        log.info("utility-wide NRW: self-calibrated population %d is only %.0f%% of the "
                 "estimated %d-account fleet -- backfill not mature enough to trust yet, "
                 "falling back to pilot-scaled estimate",
                 full_pop, 100 * full_pop / total_accounts_estimate, total_accounts_estimate)
        return None

    per_day.loc[per_day["n"] < min_fleet_fraction * full_pop, "gallons"] = float("nan")
    consumption = pd.DataFrame({
        "system": "utility", "date": per_day.index,
        "consumption_mgd": per_day["gallons"].values / 1e6,
    })
    if consumption["consumption_mgd"].notna().sum() == 0:
        return None

    last_valid_date = consumption.loc[consumption["consumption_mgd"].notna(), "date"].max()
    ref_today = today or date.today()
    if (ref_today - last_valid_date).days > _STALENESS_TOLERANCE_DAYS:
        log.info("utility-wide NRW: most recent complete-population day (%s) is more than "
                 "%d days old -- stale, falling back to pilot-scaled estimate",
                 last_valid_date, _STALENESS_TOLERANCE_DAYS)
        return None

    inp = inputs.groupby("date")["input_mgd"].sum(min_count=2)
    inputs_util = pd.DataFrame({"system": "utility", "date": inp.index,
                                "input_mgd": inp.values})
    log.info("utility-wide NRW: %d complete consumption day(s), full population=%d billed accounts",
             int(consumption["consumption_mgd"].notna().sum()), full_pop)
    return inputs_util, consumption


def pilot_scaled_frames(
    pilot_daily: pd.DataFrame,
    inputs: pd.DataFrame,
    total_accounts_estimate: int,
    min_fleet_fraction: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, float] | None:
    """Scale a small customer PILOT sample up to a utility-wide consumption
    estimate, as a stand-in for utility_wide_frames() while a full customer
    backfill is running (state/customer_history stays empty/partial for
    days-to-weeks in that window).

    This exists so the NRW panel shows SOMETHING rather than nothing. It
    rests on a real assumption, not a measured fact: that the pilot (first
    ~500 meters by account ID, not a random sample -- see run_segment_its.py)
    is representative of the full fleet's per-account consumption. Every
    number this produces must be labeled a pilot-scaled ESTIMATE, never
    presented as the real balance -- swap to utility_wide_frames() the moment
    customer_history has complete-population days.

    Returns (inputs_utility, consumption_scaled, scale_factor) or None.
    """
    billed = pilot_daily.dropna(subset=["gallons"]).copy()
    if "class_code" in billed.columns:
        billed = billed[~billed["class_code"].fillna("").str.startswith(("0", "MM"))]
    if billed.empty:
        return None

    per_day = billed.groupby("date").agg(gallons=("gallons", "sum"), n=("account_id", "nunique"))
    pilot_full_pop = int(per_day["n"].max())
    if pilot_full_pop == 0:
        return None
    per_day.loc[per_day["n"] < min_fleet_fraction * pilot_full_pop, "gallons"] = float("nan")

    scale_factor = total_accounts_estimate / pilot_full_pop
    consumption = pd.DataFrame({
        "system": "utility", "date": per_day.index,
        "consumption_mgd": per_day["gallons"].values * scale_factor / 1e6,
    })
    if consumption["consumption_mgd"].notna().sum() == 0:
        return None

    inp = inputs.groupby("date")["input_mgd"].sum(min_count=2)
    inputs_util = pd.DataFrame({"system": "utility", "date": inp.index, "input_mgd": inp.values})
    log.info("pilot-scaled NRW: %d pilot accounts -> x%.1f scale to ~%d accounts",
             pilot_full_pop, scale_factor, total_accounts_estimate)
    return inputs_util, consumption, scale_factor


# Generic AWWA M6 aged-fleet bound -- a placeholder until a real per-meter
# error model is calibrated against actual make/model/test data. Customer
# meters typically UNDER-register with age, which inflates apparent NRW
# (unread water looks like loss); master meters are billing-grade and
# treated as accurate, so all the uncertainty is assigned to the
# customer-meter side.
GENERIC_METER_BIAS_LOW_PCT = 0.0    # best case: fleet reads at spec
GENERIC_METER_BIAS_HIGH_PCT = 5.0   # AWWA M6 aged-fleet under-registration ceiling


def apply_generic_meter_bias_band(nrw_pct: float) -> tuple[float, float]:
    """(low, high) REAL-NRW% bound implied by generic customer-meter under-
    registration alone. A placeholder until a hierarchical per-meter error
    model is built and calibrated against real meter test data.
    """
    return (nrw_pct - GENERIC_METER_BIAS_HIGH_PCT, nrw_pct - GENERIC_METER_BIAS_LOW_PCT)


@dataclass
class NRWResult:
    frame: pd.DataFrame        # system, date, input_mgd, consumption_mgd, unbilled_mgd,
                               # nrw_mgd, nrw_pct, nrw_mgd_7d, nrw_pct_7d, nrw_mgd_30d,
                               # nrw_pct_30d, baseline_90d, alert
    active_alerts: list[str]   # systems currently in alert state


def compute_nrw(
    inputs: pd.DataFrame,
    consumption: pd.DataFrame,
    unbilled_mgd: float | dict = 0.0,
) -> NRWResult:
    """Join inputs and consumption per system-day and compute the balance.

    unbilled_mgd: scalar or {system: mgd} daily allowance for authorized
    unbilled uses (flushing etc.) — starts at 0 per the plan, refine with ops.
    """
    df = inputs.merge(consumption, on=["system", "date"], how="inner")
    df = df.sort_values(["system", "date"]).reset_index(drop=True)

    if isinstance(unbilled_mgd, dict):
        df["unbilled_mgd"] = df["system"].map(unbilled_mgd).fillna(0.0)
    else:
        df["unbilled_mgd"] = float(unbilled_mgd)

    df["nrw_mgd"] = df["input_mgd"] - df["consumption_mgd"] - df["unbilled_mgd"]
    df["nrw_pct"] = df["nrw_mgd"] / df["input_mgd"] * 100

    pieces = []
    active_alerts = []
    for system, g in df.groupby("system"):
        g = g.sort_values("date").copy()
        # Windowed ratio, not mean-of-daily-ratios: sum(nrw_mgd)/sum(input_mgd)
        # over the trailing window, rather than averaging each day's noisy
        # nrw_pct. Found 2026-07-17: mean-of-ratios and AMI read-timing scatter
        # (individual meters posting a day early/late, catching up after a
        # missed interval) combine to swing daily nrw_pct wildly -- including
        # physically-impossible negative "losses" on ~40% of days -- even
        # though input and consumption track each other at r=0.90 overall. A
        # ratio-of-sums is the AWWA M36 water-audit convention specifically
        # because it's robust to that kind of day-level timing noise; a mean
        # of ratios is not (it weights every day equally regardless of how
        # small/noisy that day's denominator was). input_mgd is masked to the
        # same valid days as nrw_mgd first, so the numerator and denominator
        # sums are always computed over an identical set of dates -- otherwise
        # a day with valid input but NaN consumption (or vice versa) would
        # silently pull the ratio's denominator out of sync with its numerator.
        input_valid = g["input_mgd"].where(g["nrw_mgd"].notna())
        g["nrw_mgd_7d"] = g["nrw_mgd"].rolling(7, min_periods=4).mean()
        g["nrw_pct_7d"] = (g["nrw_mgd"].rolling(7, min_periods=4).sum()
                           / input_valid.rolling(7, min_periods=4).sum() * 100)
        g["nrw_mgd_30d"] = g["nrw_mgd"].rolling(30, min_periods=14).mean()
        g["nrw_pct_30d"] = (g["nrw_mgd"].rolling(30, min_periods=14).sum()
                            / input_valid.rolling(30, min_periods=14).sum() * 100)
        g["baseline_90d"] = g["nrw_pct_7d"].shift(1).rolling(_TRAILING_BASELINE_DAYS, min_periods=30).mean()
        over = (g["nrw_pct_7d"] > g["baseline_90d"] + ALERT_MARGIN_PCT_POINTS)
        # alert = margin exceeded for >= N consecutive days
        streak = over.groupby((~over).cumsum()).cumsum()
        g["alert"] = streak >= ALERT_MIN_CONSECUTIVE_DAYS
        if bool(g["alert"].iloc[-1]) if len(g) else False:
            active_alerts.append(system)
        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)
    return NRWResult(frame=out, active_alerts=active_alerts)


# ── Self-test: synthetic loss-step scenario ─────────────────────────────────

def _self_test() -> None:
    import numpy as np
    rng = np.random.default_rng(11)
    dates = pd.date_range("2025-06-01", "2026-06-30", freq="D").date

    # True demand ~8 MGD seasonal; baseline losses 10% of input; a main break
    # adds +1.5 MGD of loss starting 2026-05-01.
    doy = pd.DatetimeIndex(pd.to_datetime(list(dates))).dayofyear.values
    demand = 7 + 2.5 * np.clip(np.sin((doy - 105) / 365 * 2 * np.pi), -0.3, None) + rng.normal(0, 0.15, len(dates))
    base_loss = demand * 0.11
    break_loss = np.where(pd.to_datetime(list(dates)) >= "2026-05-01", 1.5, 0.0)
    inputs = pd.DataFrame({"system": "system_a", "date": dates,
                           "input_mgd": demand + base_loss + break_loss})
    consumption = pd.DataFrame({"system": "system_a", "date": dates,
                                "consumption_mgd": demand})

    res = compute_nrw(inputs, consumption)
    g = res.frame
    pre = g[pd.to_datetime(g["date"]) < "2026-04-15"]
    assert not pre["alert"].any(), "False alert in the stable pre-break period!"
    post = g[pd.to_datetime(g["date"]) >= "2026-05-10"]
    assert post["alert"].any(), "Break not detected within 10 days!"
    first_alert = g[g["alert"]]["date"].min()
    print(f"Baseline NRW% (pre-break): {pre['nrw_pct_7d'].mean():.1f}%")
    print(f"Break injected 2026-05-01 -> first alert {first_alert} "
          f"({(pd.Timestamp(first_alert) - pd.Timestamp('2026-05-01')).days} days to detect)")
    print(f"Active alerts at series end: {res.active_alerts}")

    # ── daily_consumption(): fleet completeness judged as-of each day ──────
    # 3 accounts report from day 1; 2 more join (first appear) on day 50 --
    # organic growth, mimicking the real Denver-mapped-fleet pattern found
    # 2026-07-20. Day 60 has a genuine gap (only 2 of the eligible 5 report).
    dc_dates = pd.date_range("2025-01-01", "2025-04-01", freq="D").date
    rows = []
    for i, d in enumerate(dc_dates):
        accounts = ["a1", "a2", "a3"] if i < 49 else ["a1", "a2", "a3", "a4", "a5"]
        if i == 60:
            accounts = ["a1", "a2"]   # genuine incomplete-reporting day
        for acct in accounts:
            rows.append({"account_id": acct, "date": d, "gallons": 100.0, "class_code": "RES"})
    dc_daily = pd.DataFrame(rows)
    dc_system = pd.Series("system_a", index=["a1", "a2", "a3", "a4", "a5"])

    dc = daily_consumption(dc_daily, dc_system)
    dc_by_date = dc.set_index("date")["consumption_mgd"]
    early_day = dc_dates[10]     # fleet-as-of-day = 3, all 3 report -> should NOT be NaN
    growth_day = dc_dates[55]    # fleet-as-of-day = 5, all 5 report -> should NOT be NaN
    gap_day = dc_dates[60]       # fleet-as-of-day = 5, only 2 report -> SHOULD be NaN
    assert pd.notna(dc_by_date[early_day]), \
        "daily_consumption wrongly NaN'd a day with 100% of its AS-OF-THAT-DATE fleet reporting"
    assert pd.notna(dc_by_date[growth_day]), \
        "daily_consumption wrongly NaN'd a fully-reported day after fleet growth"
    assert pd.isna(dc_by_date[gap_day]), \
        "daily_consumption failed to catch a genuine incomplete-reporting day"
    print(f"\ndaily_consumption fleet-as-of-date check: early/growth days kept, "
          f"genuine gap day ({gap_day}) correctly NaN'd.")

    # ── allocate_unmapped_consumption(): pro-rata split keeps the mass
    # balance tight -- sum(per-system) must equal mapped + unmapped total ──
    alloc_dates = pd.date_range("2025-01-01", "2025-01-10", freq="D").date
    alloc_rows = []
    for d in alloc_dates:
        # system_a: d1, d2 @ 100 gal/day each = 200 gal/day known
        for acct in ("d1", "d2"):
            alloc_rows.append({"account_id": acct, "date": d, "gallons": 100.0, "class_code": "RES"})
        # system_b: m1 @ 300 gal/day known
        alloc_rows.append({"account_id": "m1", "date": d, "gallons": 300.0, "class_code": "RES"})
        # u1: no confirmed zone -- 50 gal/day, must not vanish from both ledgers
        alloc_rows.append({"account_id": "u1", "date": d, "gallons": 50.0, "class_code": "RES"})
    alloc_daily = pd.DataFrame(alloc_rows)
    alloc_system = pd.Series({"d1": "system_a", "d2": "system_a", "m1": "system_b"})

    consumption_mapped = daily_consumption(alloc_daily, alloc_system, min_fleet_fraction=0.0)
    consumption_allocated = allocate_unmapped_consumption(consumption_mapped, alloc_daily, alloc_system)

    by_sys = consumption_allocated.set_index(["date", "system"])["consumption_mgd"]
    denver_day1 = by_sys[(alloc_dates[0], "system_a")]
    maple_day1 = by_sys[(alloc_dates[0], "system_b")]
    # known shares: system_a 200/500 = 0.4, system_b 300/500 = 0.6 of the 50 gal/day unmapped
    assert abs(denver_day1 * 1e6 - (200.0 + 0.4 * 50.0)) < 1e-6, \
        f"system_a allocation wrong: expected {200 + 0.4*50} gal, got {denver_day1*1e6}"
    assert abs(maple_day1 * 1e6 - (300.0 + 0.6 * 50.0)) < 1e-6, \
        f"system_b allocation wrong: expected {300 + 0.6*50} gal, got {maple_day1*1e6}"
    total_known_and_unmapped = 200.0 + 300.0 + 50.0
    assert abs((denver_day1 + maple_day1) * 1e6 - total_known_and_unmapped) < 1e-6, \
        "allocate_unmapped_consumption must not lose or double-count gallons: " \
        "sum(per-system) should equal mapped + unmapped total"
    print(f"\nallocate_unmapped_consumption: unmapped 50 gal/day split 0.4/0.6 by known share, "
          f"system_a+system_b == mapped+unmapped total ({total_known_and_unmapped:.0f} gal/day) -- tight.")

    # ── resolve_unmapped_via_meter_id(): meter_id fallback for accounts
    # whose account_id doesn't match account_id_new/account_id_old ─────────
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "state").mkdir()
        # zone map: d1 keyed by account_id AND meter_id "mtr-d1"; a second
        # meter "mtr-rescue" is geolocated with a system but its billing
        # account number ("acctX", below) never appears in account_id_new/
        # account_id_old -- only via meter_id can it be resolved.
        zm = pd.DataFrame({
            "meter_id": ["mtr-d1", "mtr-rescue"],
            "account_id_new": ["d1", "acct-not-billing-id"],
            "account_id_old": [None, None],
            "system": ["system_a", "system_b"],
        })
        zm.to_parquet(root / "state" / "meter_zone_map.parquet")

        base_system = load_account_system_map(root)
        assert "acctX" not in base_system.index, \
            "fixture invariant broken: acctX must NOT be directly resolvable"

        billing = pd.DataFrame({
            "account_id": ["d1", "acctX", "acctX", "never-surveyed"],
            "meter_id": ["mtr-d1", "mtr-rescue", "mtr-rescue", "mtr-nowhere"],
            "date": [dates[0]] * 4, "gallons": [10.0] * 4, "class_code": ["RES"] * 4,
        })
        extended = resolve_unmapped_via_meter_id(base_system, billing, root)
        assert extended.get("acctX") == "system_b", \
            "acctX should be recovered via its meter_id even though its account_id never matched"
        assert "never-surveyed" not in extended.index, \
            "an account whose meter_id ALSO isn't in the GIS export must stay unmapped, not be invented"
        assert extended["d1"] == "system_a", "direct account_id match must still work unchanged"
        print(f"\nresolve_unmapped_via_meter_id: recovered acctX via meter_id fallback; "
              f"never-surveyed correctly left unmapped.")

    print("SELF-TEST PASSED: no false alerts pre-break; break detected; "
          "daily_consumption judges completeness against the fleet as of each date; "
          "allocate_unmapped_consumption keeps the per-system split mass-balance-tight; "
          "resolve_unmapped_via_meter_id recovers meter_id-only matches without inventing new ones.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    _self_test()
