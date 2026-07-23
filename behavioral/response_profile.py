"""Module 3 segment enrichment: behavior labels, account-type caveats, and a
raw year-over-year responder/non-responder breakdown per segment.

Complements run_segment_its.py's weather-normalized ITS (the rigorous causal
read) with two things ITS doesn't answer on its own:
  1. What does each segment actually LOOK like behaviorally (human-readable
     label from the shape features, not just "seg_0"/"seg_1")?
  2. Which accounts, within each segment, are and are NOT reducing at all —
     a simple raw year-over-year (2025 vs 2026, same calendar window) mean-gpd
     comparison. This is NOT weather-adjusted (call it out every time it's
     shown) — it exists to answer "who isn't responding," which an aggregate
     ITS coefficient can't show. The system/segment ITS panels remain the
     rigorous causal estimate.

Account TYPE (residential/commercial/irrigation) is explicitly NOT available
from the billing system's class_code, which encodes billing cycle + pressure
zone, not customer type. Until a GIS service-type export is available, the
behavioral segment IS the closest thing to an account-type label this
pipeline has.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from scipy import stats as scipy_stats

from behavioral import segmentation as seg_mod

REDUCER_THRESHOLD = -0.10   # >=10% below year-ago in the matched window
INCREASER_THRESHOLD = 0.10  # >=10% above year-ago
MIN_WINDOW_COVERAGE = 0.6   # need >=60% of days present in BOTH windows


def _base_label(segment: str, profile_row: pd.Series) -> str:
    """Human-readable behavior label from a profiles-table row, ignoring
    collisions with sibling segments -- see label_segments() for that."""
    if segment == "top_consumers":
        return "Top consumers (mega-accounts, likely commercial/irrigation-heavy)"
    if segment == "vacant_intermittent":
        return "Vacant / intermittent (>50% zero-use days)"
    swr = profile_row.get("summer_winter_ratio", 1.0)
    if swr >= 5.0:
        return "Heavy irrigator (strongly summer-peaked)"
    if swr >= 2.0:
        return "Moderate seasonal (some outdoor use)"
    return "Low-base / indoor-only (flat year-round)"


def label_segments(profiles: pd.DataFrame) -> dict[str, str]:
    """Human-readable behavior labels for every segment in one profiles table.

    The base label only looks at summer_winter_ratio, but the clustering
    itself (behavioral.segmentation.segment) uses four features -- so once k
    grows past 3 (2026-07-16: k now auto-selects up to 4, see segmentation.py),
    two genuinely different clusters can land in the same summer_winter_ratio
    band and get identical labels. When that happens within a single
    profiles table, this disambiguates the colliding segments by
    irrigation_share, the next most label-relevant clustering feature.
    """
    labels = {seg: _base_label(seg, row) for seg, row in profiles.iterrows()}

    band_members: dict[str, list[str]] = {}
    for seg, base in labels.items():
        if seg not in ("top_consumers", "vacant_intermittent"):
            band_members.setdefault(base, []).append(seg)

    for base, members in band_members.items():
        if len(members) < 2:
            continue
        ranked = sorted(members, key=lambda s: profiles.loc[s, "irrigation_share"])
        if len(ranked) == 2:
            qualifiers = ["lower irrigation share", "higher irrigation share"]
        else:
            qualifiers = [f"irrigation share rank {i + 1}/{len(ranked)}" for i in range(len(ranked))]
        for seg, qual in zip(ranked, qualifiers):
            labels[seg] = f"{base}, {qual}"

    return labels


def yoy_account_change(
    daily: pd.DataFrame,
    pre_start: date, pre_end: date,
    post_start: date, post_end: date,
    min_coverage: float = MIN_WINDOW_COVERAGE,
) -> pd.DataFrame:
    """Per-account raw mean-gpd change between two matched calendar windows.

    daily: account_id, date, gallons. Returns one row per account with
    pre_mean_gpd, post_mean_gpd, pct_change, coverage_ok, response (str).
    Windows should be the same length and, ideally, the same calendar dates a
    year apart so seasonal weather roughly cancels — this is a raw comparison,
    not a weather-normalized one.
    """
    df = daily.dropna(subset=["gallons"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    pre_days = (pre_end - pre_start).days + 1
    post_days = (post_end - post_start).days + 1

    pre = df[(df["date"] >= pre_start) & (df["date"] <= pre_end)]
    post = df[(df["date"] >= post_start) & (df["date"] <= post_end)]

    pre_agg = pre.groupby("account_id")["gallons"].agg(pre_mean_gpd="mean", pre_n="count")
    post_agg = post.groupby("account_id")["gallons"].agg(post_mean_gpd="mean", post_n="count")
    out = pre_agg.join(post_agg, how="inner")

    out["coverage_ok"] = (
        (out["pre_n"] >= min_coverage * pre_days) &
        (out["post_n"] >= min_coverage * post_days)
    )
    out["pct_change"] = (out["post_mean_gpd"] - out["pre_mean_gpd"]) / out["pre_mean_gpd"].replace(0, float("nan"))

    def classify(row):
        if not row["coverage_ok"] or pd.isna(row["pct_change"]):
            return "insufficient_data"
        if row["pct_change"] <= REDUCER_THRESHOLD:
            return "reducer"
        if row["pct_change"] >= INCREASER_THRESHOLD:
            return "increaser"
        return "flat"

    out["response"] = out.apply(classify, axis=1)
    return out


@dataclass
class SignTestResult:
    """Classic sign test (Arbuthnot 1710, the oldest inferential test in
    common use): among accounts that moved at all -- ties ('flat' accounts)
    excluded, the standard sign-test convention -- is the count of reducers
    vs. increasers different from a 50/50 coin flip? No weather model, no
    machine learning, no cross-fitting: a single binomial test, small enough
    to check by hand for modest n. Deliberately simple and fully auditable --
    added 2026-07-17 after the user asked whether a simpler tool got
    overlooked while building up DiD/DML/BSTS. Complements the
    weather-normalized ITS (which answers "is the LEVEL down more than
    weather explains") with the plainest possible answer to a related,
    narrower question: are more individual accounts choosing to use less
    than to use more, more often than chance alone would produce?
    """
    n_reducer: int
    n_increaser: int
    n_moved: int
    p_value: float
    significant: bool


def sign_test_reduction(n_reducer: int, n_increaser: int) -> SignTestResult:
    n_moved = n_reducer + n_increaser
    if n_moved == 0:
        return SignTestResult(n_reducer, n_increaser, 0, float("nan"), False)
    result = scipy_stats.binomtest(n_reducer, n_moved, p=0.5, alternative="two-sided")
    return SignTestResult(n_reducer=n_reducer, n_increaser=n_increaser, n_moved=n_moved,
                          p_value=float(result.pvalue), significant=result.pvalue < 0.05)


def segment_response_table(
    assignments: pd.DataFrame, profiles: pd.DataFrame, yoy: pd.DataFrame,
) -> list[dict]:
    """Join segmentation + YoY response into one per-segment reporting table."""
    joined = assignments[["segment"]].join(yoy, how="left")
    labels = label_segments(profiles)
    rows = []
    for segment, g in joined.groupby("segment"):
        prof = profiles.loc[segment]
        valid = g[g["response"] != "insufficient_data"]
        n_valid = len(valid)
        counts = valid["response"].value_counts()
        pre_total = valid["pre_mean_gpd"].sum()
        post_total = valid["post_mean_gpd"].sum()
        fleet_pct_change = (post_total - pre_total) / pre_total * 100 if pre_total else float("nan")
        sign_test = sign_test_reduction(int(counts.get("reducer", 0)), int(counts.get("increaser", 0)))
        rows.append({
            "segment": segment,
            "label": labels[segment],
            "n_accounts": int(prof["n_accounts"]),
            "volume_share_pct": round(float(prof["volume_share_pct"]), 1),
            "summer_winter_ratio": round(float(prof["summer_winter_ratio"]), 2),
            "irrigation_share": round(float(prof["irrigation_share"]), 2),
            "n_yoy_valid": n_valid,
            "pct_reducer": round(100 * counts.get("reducer", 0) / n_valid, 1) if n_valid else float("nan"),
            "pct_flat": round(100 * counts.get("flat", 0) / n_valid, 1) if n_valid else float("nan"),
            "pct_increaser": round(100 * counts.get("increaser", 0) / n_valid, 1) if n_valid else float("nan"),
            "fleet_pct_change_yoy": round(fleet_pct_change, 1) if pre_total else float("nan"),
            "sign_test_p": sign_test.p_value,
            "sign_test_significant": sign_test.significant,
        })
    rows.sort(key=lambda r: r["pct_reducer"] if not pd.isna(r["pct_reducer"]) else -1, reverse=True)
    return rows


def fleet_sign_test(yoy: pd.DataFrame) -> SignTestResult:
    """Same sign test as segment_response_table(), pooled across the whole
    fleet -- the single simplest, most transparent "is the drought response
    real" number on the dashboard: literally a coin-flip test on which
    direction each account moved."""
    valid = yoy[yoy["response"] != "insufficient_data"]
    counts = valid["response"].value_counts()
    return sign_test_reduction(int(counts.get("reducer", 0)), int(counts.get("increaser", 0)))


@dataclass
class FeatureCorrelationResult:
    """Fully transparent complement to behavioral/dml.py's CATE ranking: one
    bivariate Spearman (rank) correlation between a single behavioral feature
    and each account's actual YoY %% change, plus the underlying (x, y) pairs
    for a scatter chart. Spearman, not Pearson -- robust to this population's
    extreme right-skew (same reason nrw.py uses ratio-of-sums over
    mean-of-ratios), and doesn't assume a linear relationship, just a
    monotonic one. No model, no cross-fitting, no machine learning: every
    point on the resulting chart is one real account, directly checkable by
    eye rather than by trusting a random forest's feature-importance output.

    binned_medians resolves a correlation-vs-CATE-ranking tension found
    live: the overall Spearman was near zero (r=-0.046) while the CATE
    ranked this feature dominant -- because the
    real relationship is THRESHOLD-shaped, not monotonic: medians sit flat
    (~-1%%) below summer_winter_ratio 5, then break to -4.7%% (ratio 5-8) and
    -8.5%% (8+). A rank correlation dilutes toward zero when ~70%% of the
    population is in the flat region; a forest doesn't. Medians, never
    means: near-zero-baseline accounts make pct_change means astronomically
    large and meaningless (observed live: bin means of +8818%%).
    """
    feature: str
    feature_label: str
    r: float
    p_value: float
    n: int
    points: pd.DataFrame        # account_id, feature value, pct_change (x100 already)
    binned_medians: list[dict]  # [{bin_label, n, median_pct}] -- threshold-shape check


# Bin edges around the feature's own meaning: 1.0 = flat year-round use,
# rising = more summer-skewed. Chosen for summer_winter_ratio; a different
# feature passed to feature_response_correlation() gets quartile bins instead.
_SWR_BIN_EDGES = [0, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, float("inf")]


def feature_response_correlation(
    assignments: pd.DataFrame, yoy: pd.DataFrame, feature: str,
) -> FeatureCorrelationResult:
    joined = assignments[[feature]].join(yoy[["pct_change", "response"]], how="inner")
    joined = joined[joined["response"] != "insufficient_data"].dropna()
    r, p = scipy_stats.spearmanr(joined[feature], joined["pct_change"])
    points = joined.reset_index().rename(columns={"index": "account_id"})
    points["pct_change_pct"] = points["pct_change"] * 100

    if feature == "summer_winter_ratio":
        bins = pd.cut(joined[feature], _SWR_BIN_EDGES, right=False)
    else:
        bins = pd.qcut(joined[feature], 6, duplicates="drop")
    binned = []
    for b, g in joined.groupby(bins, observed=True):
        binned.append({
            "bin_label": str(b),
            "n": int(len(g)),
            "median_pct": round(float(g["pct_change"].median() * 100), 1),
        })

    return FeatureCorrelationResult(
        feature=feature, feature_label=seg_mod.feature_label(feature),
        r=float(r), p_value=float(p), n=len(joined), points=points,
        binned_medians=binned,
    )
