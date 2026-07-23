"""Customer behavioral segmentation.

k-means clustering on per-customer daily consumption profiles, reducing each
account's daily series to a small set of behaviorally meaningful features
rather than clustering raw time series (robust to gaps, cheap at large
account counts, and the features themselves are interpretable for a report).

Expected archetypes: low-base year-round users, summer-dominant irrigators,
and steady commercial — with the top ~1% of accounts by volume typically
holding a wildly disproportionate share of total consumption, which argues
for a mandatory "top-consumers" stratum handled OUTSIDE the clustering (see
segment(), which splits mega-users off first so they can't drag cluster
centroids around).

Run `python -m behavioral.segmentation` for the self-test (fully synthetic
fixture, no real data).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

_SUMMER_MONTHS = (6, 7, 8, 9)
_WINTER_MONTHS = (12, 1, 2, 3)
MIN_VALID_DAYS = 180           # accounts with less usable history are excluded
MEGA_USER_QUANTILE = 0.99      # top-1% by volume stratified out pre-clustering


@dataclass
class SegmentationResult:
    assignments: pd.DataFrame   # account_id, segment, plus the feature columns
    profiles: pd.DataFrame      # per-segment feature means + counts + volume share
    k: int
    silhouette: float
    n_excluded: int             # accounts below MIN_VALID_DAYS


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-account behavioral features from tidy daily data.

    daily: columns account_id, date, gallons (NaN allowed = missing day).
    Returns one row per account (index=account_id).
    """
    df = daily.dropna(subset=["gallons"]).copy()
    # Negative daily consumption is physically impossible -- register
    # rollover/replacement producing a negative interval diff, the mirror
    # image of the positive-spike register artifacts found in the Denver
    # Water master-meter series 2026-07-17 (run_module3_its.py). Found live
    # same day: 24,314 rows across 2,326 accounts, most tiny (-0.1 to -0.2,
    # likely rounding noise) but a handful catastrophic (min -29,998,930 gal
    # in one reading), enough to drag 5 accounts' multi-year MEAN negative
    # and produce log1p(negative) -> NaN, corrupting those accounts'
    # clustering features. Dropped like any other invalid row, not clamped
    # to zero -- a day this is true for isn't "zero use," it's unknown.
    n_before = len(df)
    df = df[df["gallons"] >= 0]
    if n_before > len(df):
        log.info("build_features: dropping %d row(s) with negative gallons "
                 "(register rollover/replacement artifacts)", n_before - len(df))
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    rows = {}
    for account, g in df.groupby("account_id"):
        n = len(g)
        if n < MIN_VALID_DAYS:
            continue
        vals = g["gallons"]
        summer = g[g["month"].isin(_SUMMER_MONTHS)]["gallons"]
        winter = g[g["month"].isin(_WINTER_MONTHS)]["gallons"]
        weekend = g[g["is_weekend"]]["gallons"]
        weekday = g[~g["is_weekend"]]["gallons"]

        base = float(vals.quantile(0.10))               # indoor floor
        mean = float(vals.mean())
        rows[account] = {
            "log_mean_gpd": np.log1p(mean),
            # winter.mean() > 0 (any nonzero reading) was too weak a guard --
            # found live 2026-07-20: an account with 450 winter days of $0
            # usage and a single 0.1-gal blip has winter.mean()=0.0002, so a
            # normal summer mean divides out to a 27-million-x ratio. Reuse
            # the same "essentially zero" threshold zero_day_frac already
            # applies per-reading (<=1.0 gal) rather than inventing a new one.
            "summer_winter_ratio": float(summer.mean() / winter.mean())
                if len(summer) > 20 and len(winter) > 20 and winter.mean() > 1.0 else 1.0,
            "weekend_weekday_ratio": float(weekend.mean() / weekday.mean())
                if len(weekend) > 10 and weekday.mean() > 0 else 1.0,
            "irrigation_share": float(max(mean - base, 0.0) / mean) if mean > 0 else 0.0,
            "cv": float(vals.std() / mean) if mean > 0 else 0.0,
            "zero_day_frac": float((vals <= 1.0).mean()),
            "mean_gpd": mean,      # kept for stratification/reporting, not clustering
            "n_days": n,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("account_id")


# Shape-only — see segment() docstring for the real-data rationale.
_CLUSTER_FEATURES = ["summer_winter_ratio", "weekend_weekday_ratio",
                     "irrigation_share", "cv"]
VACANT_ZERO_DAY_FRAC = 0.5   # >50% zero-days -> vacant/intermittent stratum

# Plain-language names/descriptions for the raw feature columns above, for any
# report or dashboard panel that surfaces a feature name directly (e.g. the
# CATE ranking in behavioral/dml.py) rather than a full segment label
# (behavioral.response_profile.label_segments already handles those). Added
# 2026-07-17 after direct user feedback that "Summer Winter Ratio" (a
# mechanical title-case of the column name) wasn't understandable on its own.
FEATURE_LABELS = {
    "summer_winter_ratio": "Summer-vs-winter usage ratio",
    "weekend_weekday_ratio": "Weekend-vs-weekday usage ratio",
    "irrigation_share": "Estimated outdoor-watering share",
    "cv": "Day-to-day usage variability",
    "log_mean_gpd": "Overall usage level",
    "zero_day_frac": "Share of zero-use days",
    "mean_gpd": "Average daily usage",
}
FEATURE_DESCRIPTIONS = {
    "summer_winter_ratio": "How many times more water an account uses in summer "
        "(Jun-Sep) than winter (Dec-Mar). 1.0 = flat year-round use; 8.0 = uses "
        "8x more water in summer, a strong outdoor-irrigation signal.",
    "weekend_weekday_ratio": "How much more (or less) an account uses on "
        "Saturday/Sunday vs. weekdays -- a signature of some commercial and "
        "timer-driven irrigation patterns.",
    "irrigation_share": "Estimated share of an account's average use that sits "
        "above its own lowest-use ('indoor floor') days -- a rough proxy for how "
        "much of their water goes to outdoor watering vs. baseline indoor use.",
    "cv": "How much an account's day-to-day usage swings around its own average, "
        "regardless of overall volume -- a steady user has a low value, a "
        "start-stop user (e.g. an irrigation timer cycling on/off) has a high one.",
    "log_mean_gpd": "Overall size of the account's usage (log-scaled so huge and "
        "typical accounts don't distort the comparison).",
    "zero_day_frac": "Share of days with essentially no recorded use -- high values "
        "suggest a vacant property or an intermittent/seasonal user.",
}


def feature_label(key: str) -> str:
    """Plain-language name for a raw feature column, falling back to a
    mechanical title-case if the key isn't in FEATURE_LABELS."""
    return FEATURE_LABELS.get(key, key.replace("_", " ").title())


SILHOUETTE_TOLERANCE = 0.02   # accept up to this much silhouette loss for a finer k


def segment(daily: pd.DataFrame, k_range: range = range(2, 9),
            random_state: int = 42,
            silhouette_tolerance: float = SILHOUETTE_TOLERANCE) -> SegmentationResult:
    """Stratify mega-users and vacant accounts out, cluster the rest on usage
    SHAPE, pick k by silhouette.

    SHAPE-ONLY metric (real-data lesson, 2026-07-16 pilot): size features in
    the distance metric drown out behavior — the pilot's size-inclusive "best"
    clustering was a trivial vacant-vs-everyone split (silhouette 0.74 but
    meaningless), while shape-only k=3 recovered the proposal's expected
    low-base vs heavy-irrigator structure among SAME-SIZED accounts (230 at
    7.8x summer ratio vs 250 at 2.4x). Size lives in the strata and in
    reporting, never in the metric; vacancy gets its own stratum for the same
    reason.

    k selection (updated 2026-07-16): raw argmax-silhouette tends to pick the
    coarsest split available (k=2 narrowly beat k=3/k=4 on live data, 0.310 vs
    0.293/0.294) even though the proposal's own domain theory (Drought Mgmt
    Plan Tables 4/5 + proposal Sec 3b) expects at least three meaningful
    archetypes: low-base year-round, summer-dominant irrigators, and steady
    commercial. Picking the bare silhouette argmax under-segments the report
    for a marginal, likely-noise difference in score. Instead: among all k
    whose silhouette is within `silhouette_tolerance` of the best score, take
    the LARGEST k — more operationally useful granularity at a small, bounded
    cost in cluster separation, not an arbitrary override of the metric.
    """
    feats = build_features(daily)
    n_excluded = daily["account_id"].nunique() - len(feats)

    # interpolation="lower": the cut lands ON an observed value, so ties and
    # small-n cases (like the self-test) don't split a mega-user cohort across
    # the boundary via interpolation between two mega values.
    mega_cut = feats["mean_gpd"].quantile(MEGA_USER_QUANTILE, interpolation="lower")
    mega = feats[feats["mean_gpd"] >= mega_cut]
    rest = feats[feats["mean_gpd"] < mega_cut]
    vacant = rest[rest["zero_day_frac"] > VACANT_ZERO_DAY_FRAC]
    rest = rest[rest["zero_day_frac"] <= VACANT_ZERO_DAY_FRAC]

    X = StandardScaler().fit_transform(rest[_CLUSTER_FEATURES])

    candidates = []   # (k, score, labels)
    for k in k_range:
        if k >= len(rest):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        score = silhouette_score(X, km.labels_)
        log.info("k=%d silhouette=%.3f", k, score)
        candidates.append((k, score, km.labels_))
    if not candidates:
        raise RuntimeError("Too few accounts to cluster.")

    best_score = max(c[1] for c in candidates)
    within_tolerance = [c for c in candidates if c[1] >= best_score - silhouette_tolerance]
    k, silhouette, labels = max(within_tolerance, key=lambda c: c[0])
    log.info("k selected: %d (silhouette=%.3f, best available=%.3f, tolerance=%.2f)",
             k, silhouette, best_score, silhouette_tolerance)
    assignments = rest.copy()
    assignments["segment"] = [f"seg_{l}" for l in labels]
    mega_out = mega.copy()
    mega_out["segment"] = "top_consumers"
    vacant_out = vacant.copy()
    vacant_out["segment"] = "vacant_intermittent"
    assignments = pd.concat([assignments, mega_out, vacant_out])

    total_volume = (assignments["mean_gpd"] * assignments["n_days"]).sum()
    profiles = (
        assignments.groupby("segment")
        .agg(n_accounts=("mean_gpd", "size"),
             mean_gpd=("mean_gpd", "mean"),
             summer_winter_ratio=("summer_winter_ratio", "mean"),
             irrigation_share=("irrigation_share", "mean"),
             zero_day_frac=("zero_day_frac", "mean"))
    )
    profiles["volume_share_pct"] = (
        assignments.groupby("segment").apply(
            lambda g: (g["mean_gpd"] * g["n_days"]).sum() / total_volume * 100,
            include_groups=False)
    )

    return SegmentationResult(assignments=assignments, profiles=profiles,
                              k=k, silhouette=silhouette, n_excluded=n_excluded)


def load_account_premise_type(proj_root: Path) -> pd.Series | None:
    """account_id -> premise_ty (e.g. "Residential 1 Unit", "Irrigation Only",
    "Commercial") from state/meter_zone_map.parquet -- real GIS service-type
    labels, preferred over class-code guessing.

    Used only to LABEL and VALIDATE segments after clustering (see
    premise_type_profile below), never as a clustering feature: segment()'s
    metric is deliberately shape-only (summer/winter ratio, weekend ratio,
    irrigation share, CV) after the 2026-07-16 pilot lesson that including
    size/type features in the distance metric produces trivial splits (e.g.
    a size-driven "best" clustering that was really just vacant-vs-everyone)
    instead of the behavioral structure the metric is meant to find.

    Same dual-account-ID-era join as nrw.load_account_system_map():
    customer_history's account_id mixes an old dash-format and a new
    10-digit format across different backfill eras.
    """
    path = proj_root / "state" / "meter_zone_map.parquet"
    if not path.exists():
        return None
    zm = pd.read_parquet(path)
    zm = zm[zm["premise_ty"].notna() & (zm["premise_ty"] != "")]
    pairs = pd.concat([
        zm[["account_id_new", "premise_ty"]].rename(columns={"account_id_new": "account_id"}),
        zm[["account_id_old", "premise_ty"]].rename(columns={"account_id_old": "account_id"}),
    ], ignore_index=True)
    pairs = pairs.dropna(subset=["account_id"])
    pairs = pairs[pairs["account_id"] != ""]
    pairs = pairs.drop_duplicates(subset="account_id", keep="first")
    return pairs.set_index("account_id")["premise_ty"]


def premise_type_profile(assignments: pd.DataFrame, proj_root: Path) -> pd.DataFrame:
    """Per-segment premise-type composition -- validates/enriches the
    behavioral segments in real service-type terms (e.g. "seg_1 is 78%
    Residential 1 Unit, 14% Irrigation Only...") without ever feeding
    premise type into the clustering itself.

    assignments: SegmentationResult.assignments (index=account_id, has a
    "segment" column). Returns one row per (segment, premise_ty) with a
    share_pct column, sorted by segment then descending share. Accounts
    with no confirmed premise type are reported as "(unmapped)" rather
    than silently dropped -- fail-open-with-a-flag, same as everywhere
    else in this project (nrw.py, the zone-scope filter in
    mass_balance_audit.py).
    """
    premise = load_account_premise_type(proj_root)
    if premise is None:
        return pd.DataFrame(columns=["segment", "premise_ty", "n_accounts", "share_pct"])
    df = assignments[["segment"]].copy()
    df["premise_ty"] = df.index.map(premise).fillna("(unmapped)")
    counts = df.groupby(["segment", "premise_ty"], observed=True).size().rename("n_accounts").reset_index()
    seg_totals = counts.groupby("segment")["n_accounts"].transform("sum")
    counts["share_pct"] = counts["n_accounts"] / seg_totals * 100
    return counts.sort_values(["segment", "share_pct"], ascending=[True, False]).reset_index(drop=True)


@dataclass
class StabilityResult:
    mean_ari: float
    min_ari: float
    n_boot: int
    subsample_frac: float
    k: int

    @property
    def verdict(self) -> str:
        """Qualitative band (Ben-Hur et al.-style reading of ARI levels)."""
        if self.mean_ari >= 0.8:
            return "stable"
        if self.mean_ari >= 0.6:
            return "moderately stable"
        return "unstable"


def stability(features: pd.DataFrame, k: int, *, n_boot: int = 20,
              subsample_frac: float = 0.8, random_state: int = 42) -> StabilityResult:
    """Subsampling stability check for the k-means partition (Ben-Hur,
    Elisseeff & Guyon 2002): does roughly the same grouping come back when
    the clustering is re-run on random 80%% subsets of the same accounts, or
    is the partition an artifact of this exact data draw?

    features: the CLUSTERED population's rows only (caller excludes the
    top_consumers/vacant_intermittent strata, same as segment() does), with
    at least the _CLUSTER_FEATURES columns. Reference labels come from one
    full-data fit; each bootstrap refits on a subsample and agreement is
    measured by Adjusted Rand Index on the overlap. ARI is 1.0 for identical
    partitions and ~0 for chance-level agreement -- and is label-permutation
    invariant, so "cluster 2 became cluster 0" doesn't count against it.

    Scaling is fit once on the full data (the subsample re-fit sees the same
    feature space, so instability measured here is about the PARTITION, not
    about scaler jitter).
    """
    X = StandardScaler().fit_transform(features[_CLUSTER_FEATURES])
    ref = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)

    rng = np.random.default_rng(random_state)
    n = len(X)
    n_sub = max(int(n * subsample_frac), k + 1)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n_sub, replace=False)
        boot = KMeans(n_clusters=k, n_init=10,
                      random_state=int(rng.integers(1 << 31))).fit(X[idx])
        aris.append(adjusted_rand_score(ref.labels_[idx], boot.labels_))

    result = StabilityResult(mean_ari=float(np.mean(aris)), min_ari=float(np.min(aris)),
                             n_boot=n_boot, subsample_frac=subsample_frac, k=k)
    log.info("stability: k=%d, %d resamples at %.0f%%: mean ARI=%.3f min=%.3f (%s)",
             k, n_boot, subsample_frac * 100, result.mean_ari, result.min_ari, result.verdict)
    return result


# ── Self-test on synthetic archetypes ────────────────────────────────────────

def _synthetic_daily(n_per_type: int = 60, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Three known archetypes + a couple of mega-users; returns (daily, truth)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", "2026-06-30", freq="D")
    doy = dates.dayofyear.values
    summer_curve = np.clip(np.sin((doy - 105) / 365 * 2 * np.pi), 0, None)

    rows, truth = [], {}
    acct = 0
    for _ in range(n_per_type):   # low-base year-round
        acct += 1
        use = rng.normal(140, 25, len(dates)).clip(20)
        rows.append(pd.DataFrame({"account_id": f"A{acct}", "date": dates, "gallons": use}))
        truth[f"A{acct}"] = "low_base"
    for _ in range(n_per_type):   # summer irrigators
        acct += 1
        use = rng.normal(150, 30, len(dates)).clip(20) + summer_curve * rng.normal(900, 150)
        rows.append(pd.DataFrame({"account_id": f"A{acct}", "date": dates, "gallons": use}))
        truth[f"A{acct}"] = "irrigator"
    for _ in range(n_per_type):   # commercial: steady high, weekend dip
        acct += 1
        weekend = (dates.dayofweek >= 5).astype(float)
        use = rng.normal(2200, 250, len(dates)).clip(200) * (1 - 0.5 * weekend)
        rows.append(pd.DataFrame({"account_id": f"A{acct}", "date": dates, "gallons": use}))
        truth[f"A{acct}"] = "commercial"
    for _ in range(3):            # mega-users
        acct += 1
        use = rng.normal(60000, 5000, len(dates)).clip(1000)
        rows.append(pd.DataFrame({"account_id": f"A{acct}", "date": dates, "gallons": use}))
        truth[f"A{acct}"] = "mega"
    return pd.concat(rows, ignore_index=True), truth


def _self_test() -> None:
    daily, truth = _synthetic_daily()
    res = segment(daily)
    print(f"k={res.k}  silhouette={res.silhouette:.3f}  excluded={res.n_excluded}")
    print(res.profiles.round(2).to_string())

    # Purity: each true archetype should map dominantly to one segment.
    joined = res.assignments.copy()
    joined["truth"] = [truth[a] for a in joined.index]
    purity = (
        joined.groupby("truth")["segment"]
        .agg(lambda s: s.value_counts(normalize=True).iloc[0])
    )
    print("\nPer-archetype purity (fraction in dominant segment):")
    print(purity.round(3).to_string())
    assert (purity > 0.9).all(), "Archetype recovery below 90% purity!"
    assert (joined[joined["truth"] == "mega"]["segment"] == "top_consumers").all(), \
        "Mega-users not stratified into top_consumers!"

    # ── stability(): structured data must be stable, pure noise must not be ──
    clustered = res.assignments[~res.assignments["segment"].isin(["top_consumers", "vacant_intermittent"])]
    stab = stability(clustered, k=res.k, n_boot=10)
    print(f"\nstability (structured archetypes): mean ARI={stab.mean_ari:.3f} ({stab.verdict})")
    assert stab.mean_ari > 0.8, f"expected stable clustering on separable archetypes, got ARI {stab.mean_ari:.3f}"

    rng = np.random.default_rng(5)
    noise = pd.DataFrame({c: rng.uniform(0, 1, 400) for c in _CLUSTER_FEATURES})
    stab_noise = stability(noise, k=res.k, n_boot=10)
    print(f"stability (pure-noise negative control): mean ARI={stab_noise.mean_ari:.3f} ({stab_noise.verdict})")
    assert stab_noise.mean_ari < stab.mean_ari - 0.1, \
        (f"noise control ARI {stab_noise.mean_ari:.3f} not clearly below structured "
         f"{stab.mean_ari:.3f} -- stability check may not discriminate")

    # ── premise_type_profile(): real service-type labels validate the segments ──
    import tempfile

    assert load_account_premise_type(Path(tempfile.mkdtemp())) is None, \
        "expected None when meter_zone_map.parquet doesn't exist"
    assert premise_type_profile(joined, Path(tempfile.mkdtemp())).empty, \
        "expected an empty frame (not a crash) when the zone map is missing"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "state").mkdir()
        # Premise type correlated with the KNOWN archetype (irrigators are
        # "Irrigation Only", commercial is "Commercial", low-base is
        # residential) -- lets the assertions check the composition is
        # actually right, not just that the function runs. Half the
        # low_base accounts are left OUT of the zone map entirely, to
        # exercise the "(unmapped)" fallback on a real, non-trivial split.
        archetype_premise = {"irrigator": "Irrigation Only", "commercial": "Commercial",
                             "low_base": "Residential 1 Unit", "mega": "Master Meter Conn."}
        rows = []
        for i, (acct, arche) in enumerate(truth.items()):
            if arche == "low_base" and i % 2 == 0:
                continue   # left unmapped on purpose
            rows.append({"account_id_new": acct, "account_id_old": None,
                        "premise_ty": archetype_premise[arche]})
        pd.DataFrame(rows).to_parquet(root / "state" / "meter_zone_map.parquet")

        profile = premise_type_profile(joined, root)
        print("\npremise_type_profile (first 8 rows):")
        print(profile.head(8).to_string())

        # The segment(s) that are >=90% pure "irrigator" by truth should
        # also read as >=90% "Irrigation Only" by premise type -- the two
        # independent labelings (behavioral cluster, real GIS service type)
        # should agree, since this synthetic fixture ties them together.
        irrigator_segments = joined.loc[joined["truth"] == "irrigator", "segment"].unique()
        for seg in irrigator_segments:
            seg_rows = profile[profile["segment"] == seg]
            top = seg_rows.iloc[0]
            assert top["premise_ty"] == "Irrigation Only" and top["share_pct"] >= 90, (
                f"segment {seg} (irrigator-truth) should read >=90% Irrigation Only "
                f"by premise type, got {top['premise_ty']} at {top['share_pct']:.1f}%")

        assert "(unmapped)" in profile["premise_ty"].values, \
            "expected the deliberately-unmapped low_base accounts to show up as (unmapped)"
        assert profile.groupby("segment")["share_pct"].sum().round(4).eq(100.0).all(), \
            "each segment's premise-type shares should sum to 100%"

    print("\nSELF-TEST PASSED: archetypes recovered, mega-users stratified, "
          "stability check discriminates structure from noise, premise-type "
          "profile agrees with behavioral clustering and handles unmapped "
          "accounts without crashing.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    _self_test()
