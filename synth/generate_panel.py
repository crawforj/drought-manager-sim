"""Synthetic account-level panel + system aggregate series generator.

Consumes the fictional geography from synth/geography.py plus a set of
calibration constants extracted by hand from a real utility's own data
during this project's development -- baked in below as constants, not read
from any external file at runtime, so this module has no dependency on any
private data source and is safe to run as-is. Every constant here is an
aggregate statistic (a mean, a correlation, a cluster centroid/covariance,
a bin median), never a raw per-account value -- see the README for the
full calibrate-then-generate methodology.

Produces, at the repo root (the same layout the rest of this repo reads):
  - state/customer_history/YYYY-MM.parquet  (customer_store.py schema)
  - state/entry_point_history.parquet        (nrw.py / demand-ensemble AMI input)
  - production_history.csv                   (SCADA finished-water)
  - state/meter_zone_map.parquet             (account -> zone/system)

Run: python -m synth.generate_panel [--n-accounts 9000] [--seed 20260723]
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OUT_DIR = Path(__file__).parent.parent
GEO_DIR = Path(__file__).parent / "geo"
WEATHER_CACHE = OUT_DIR / "state" / "weather_cache.csv"

# Real, public NOAA GHCND station (Wheat Ridge, CO). Reusing real public
# weather data (not any utility's proprietary data) is what makes the
# "Colorado Front Range" narrative genuine without needing to fabricate a
# plausible climate.
NCEI_STATION = "USC00058995"

START_DATE = date(2023, 1, 1)
END_DATE = date(2026, 7, 14)

# ── Phase 1 calibration constants ────────────────────────────────────────────
# (extracted by hand from a real utility's own data during development; see
# the README for the full calibrate-then-generate methodology)

SEG_SHARES = {"seg_0": 0.572, "seg_1": 0.3856, "top_consumers": 0.0101, "vacant": 0.0321}

CLUSTERS = {
    "seg_0": {
        "mean": {"summer_winter_ratio": 6.103, "weekend_weekday_ratio": 1.0481,
                 "irrigation_share": 0.8792, "cv": 1.5708},
        "cov": [[93.1075, 0.1162, 0.1574, 0.7949],
                [0.1162, 0.1149, -0.002, 0.0199],
                [0.1574, -0.002, 0.0078, 0.0083],
                [0.7949, 0.0199, 0.0083, 0.4081]],
        "mean_gpd": 306.37,
    },
    "seg_1": {
        "mean": {"summer_winter_ratio": 1.8986, "weekend_weekday_ratio": 1.0882,
                 "irrigation_share": 0.5909, "cv": 0.8134},
        "cov": [[0.895, 0.0029, 0.0606, 0.1468],
                [0.0029, 0.0337, 0.0013, 0.0054],
                [0.0606, 0.0013, 0.0156, 0.0215],
                [0.1468, 0.0054, 0.0215, 0.0816]],
        "mean_gpd": 351.51,
    },
}
FEATURE_ORDER = ["summer_winter_ratio", "weekend_weekday_ratio", "irrigation_share", "cv"]
TOP_CONSUMERS_VOLUME_SHARE_PCT = 25.47
VACANT_GPD_SHAPE, VACANT_GPD_SCALE = 1.0, 5.0   # gamma(shape, scale), mean ~5 gpd

# n_tier1=14040, n_tier23=7923 (real, classified accounts only -- "unknown"
# folded proportionally into tier1 for simplicity)
TIER23_FRAC = 7923 / (14040 + 7923)

RESTRICTIONS_DATE = date(2026, 4, 1)   # real, public drought-declaration date
PRICE_STEP_DATE = date(2026, 6, 1)     # real, public rate-schedule date
# Both groups' OWN weather-normalized ITS step at PRICE_STEP_DATE, as a
# percent of their own weather-expected demand -- dimensionless, so it can
# be applied per-account regardless of account size. The DiD (price-only)
# effect is the tier23-vs-tier1 DIFFERENCE; see how these three combine
# below in apply_drought_response().
TIER1_STEP_PCT = -10.167
TIER23_STEP_PCT = -9.739

# Threshold-shaped (NOT monotonic) relationship between summer_winter_ratio
# and raw YoY %% change, from behavioral/response_profile.py's
# feature_response_correlation() binned_medians -- (lo, hi, median_pct).
RESPONSE_BINS = [(0.0, 1.0, 0.5), (1.0, 1.5, -0.2), (1.5, 2.0, -0.9),
                 (2.0, 3.0, 0.1), (3.0, 5.0, -1.5), (5.0, 8.0, -4.8),
                 (8.0, float("inf"), -7.0)]

DOW_SEASONALITY = {0: 1.0029, 1: 0.9744, 2: 1.0845, 3: 0.969, 4: 0.9943, 5: 0.9826, 6: 0.9924}

DEMAND_MEAN_MGD = 10.7292   # real aggregate system demand, used only to normalize the weather shock below
# OLS coefficients from regressing real aggregate demand on
# tmax_f/tmax_f_sq/precip_in/et0_in (a Hargreaves-Samani reference ET term),
# fit against a real utility's own data -- feeds weather_shock() below.
WEATHER_OLS = {"const": 9.631294, "tmax_f": -0.263516, "tmax_f_sq": 0.00307,
              "precip_in": 1.783409, "et0_in": 27.256649}

# NRW: system_a ("Ridgeline") = denver-analog (master-meter-summed input);
# system_b ("Cottonwood") = maple_grove-analog (WTP SCADA production input).
NRW = {
    "system_a": {"mean_nrw_pct": 11.83, "correlation": 0.9884},
    "system_b": {"mean_nrw_pct": 13.21, "correlation": 0.9387},
}

MISSING_FRAC = 0.005          # matches make_synthetic_panel.py's convention
NEGATIVE_ARTIFACT_FRAC = 0.0005
ESTIMATED_FRAC = 0.023        # matches the real ~2.3% BEACON "estimated" rate found in Phase 1


def _load_geography():
    zone_summary = pd.read_json(GEO_DIR / "zone_summary.json", typ="series")
    zones = pd.DataFrame(zone_summary["zones"])
    with open(GEO_DIR / "sites.yaml", encoding="utf-8") as f:
        sites = pd.DataFrame(yaml.safe_load(f))
    return zones, sites


def _load_weather(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Real, public GHCND weather (tmax_f/tmin_f/precip_in) for NCEI_STATION.

    Ships a static cache (state/weather_cache.csv, real public NOAA data,
    safe to commit) so regenerating the panel with the same date range
    needs no API token at all. A token (free, see the error message below)
    is only needed to fetch beyond the shipped cache's date range -- set it
    via the NCEI_TOKEN environment variable.
    """
    if WEATHER_CACHE.exists():
        cached = pd.read_csv(WEATHER_CACHE, index_col="date", parse_dates=["date"])
        if cached.index.min() <= pd.Timestamp(dates.min()) and cached.index.max() >= pd.Timestamp(dates.max()):
            return cached

    import weather_history as wh_mod
    token = os.environ.get("NCEI_TOKEN", "")

    try:
        weather = wh_mod.fetch_ghcnd(
            NCEI_STATION, dates.min().date().isoformat(), dates.max().date().isoformat(),
            OUT_DIR / "state" / "cache", token, 30, 24)
    except Exception as e:
        raise RuntimeError(
            f"No cached weather at {WEATHER_CACHE} and the live GHCND fetch failed "
            f"({e}) -- get a free token at https://www.ncdc.noaa.gov/cdo-web/token "
            "and set NCEI_TOKEN, or narrow --start/--end to fit the shipped cache.") from e
    weather.index = pd.to_datetime(weather.index)
    weather = weather[["tmax_f", "tmin_f", "precip_in"]]
    OUT_DIR.mkdir(exist_ok=True)
    weather.to_csv(WEATHER_CACHE, index_label="date")
    return weather


def weather_shock(dates: pd.DatetimeIndex, weather: pd.DataFrame) -> np.ndarray:
    """Fractional, day-level demand adjustment shared identically across
    every account -- the piece the pure per-account idiosyncratic noise in
    build_daily_series() structurally can't produce, because independent
    per-account noise averages toward zero once you sum thousands of
    accounts (law of large numbers). Real day-to-day system volatility
    comes largely from weather hitting every account at once (a heat wave
    doesn't average out); this reintroduces that as a shared multiplier.

    Uses the calibrated WEATHER_OLS coefficients to predict a real-scale
    demand series from real weather, then keeps only its HIGH-FREQUENCY
    residual (a 21-day centered rolling mean removes the smooth seasonal
    trend, which is already baked into every account's own seasonal curve
    via _solve_seasonal_params -- adding it back in here would double-count
    seasonality and distort the already-validated summer_winter_ratio).
    The residual is normalized by DEMAND_MEAN_MGD into a dimensionless
    fraction, so it applies correctly regardless of the synthetic
    population's size.
    """
    from demand_ensemble.covariates import hargreaves_et0_in
    lat_deg = 39.7   # generic Front Range latitude; et0 is not lat-sensitive enough to matter here
    doy = pd.Series(weather.index.dayofyear, index=weather.index)
    et0 = hargreaves_et0_in(weather["tmax_f"], weather["tmin_f"], doy, lat_deg)

    w = WEATHER_OLS
    pred = (w["const"] + w["tmax_f"] * weather["tmax_f"] + w["tmax_f_sq"] * weather["tmax_f"] ** 2
           + w["precip_in"] * weather["precip_in"].fillna(0.0) + w["et0_in"] * et0)
    pred = pred.reindex(dates).interpolate(limit_direction="both")

    trend = pred.rolling(21, center=True, min_periods=5).mean()
    residual = (pred - trend).fillna(0.0)
    return (residual / DEMAND_MEAN_MGD).to_numpy()


def _lognormal_params(mean: float, var: float) -> tuple[float, float]:
    sigma2 = np.log(1.0 + var / mean ** 2)
    return np.log(mean) - sigma2 / 2, np.sqrt(sigma2)


def _beta_params(mean: float, var: float) -> tuple[float, float]:
    var = min(var, mean * (1 - mean) * 0.98)   # stay inside the feasible region
    nu = mean * (1 - mean) / var - 1
    return mean * nu, (1 - mean) * nu


def _draw_features(rng: np.random.Generator, cluster: dict, n: int) -> np.ndarray:
    """Independent-marginal draws matched to each feature's real mean/variance
    (diagonal of the calibrated covariance matrix), NOT a joint multivariate
    normal on the raw features.

    Found live: summer_winter_ratio's real distribution is heavily right-
    skewed (std ~9.65 on a mean of ~6.1 for seg_0) -- drawing it from a plain
    MVN and clipping the left tail at a positive floor systematically INFLATES
    the realized mean (E[max(X, floor)] >= E[X] always, and with variance this
    large a large share of the unclipped mass sits below the floor). Realized
    seg_0 mean came out 8.16 vs the calibrated 6.10 before this fix. Lognormal
    (for the positive, heavy-tailed swr/weekend_weekday_ratio/cv) and Beta
    (for the bounded-[0,1] irrigation_share), each moment-matched to the real
    mean/variance, are exactly positive/bounded by construction -- no clipping,
    no bias. Trade-off: this drops the real cross-feature correlations (the
    covariance matrix's off-diagonal terms) in favor of correct marginals;
    still using the calibrated variance, not an arbitrary spread. See
    calibration notes above.
    """
    mean = {f: cluster["mean"][f] for f in FEATURE_ORDER}
    var = {f: cluster["cov"][i][i] for i, f in enumerate(FEATURE_ORDER)}

    draws = np.empty((n, len(FEATURE_ORDER)))
    for i, f in enumerate(("summer_winter_ratio", "weekend_weekday_ratio", "cv")):
        mu, sigma = _lognormal_params(mean[f], var[f])
        draws[:, FEATURE_ORDER.index(f)] = rng.lognormal(mu, sigma, n)
    a, b = _beta_params(mean["irrigation_share"], var["irrigation_share"])
    draws[:, FEATURE_ORDER.index("irrigation_share")] = rng.beta(a, b, n)

    draws[:, 0] = np.clip(draws[:, 0], 0.3, 60.0)     # summer_winter_ratio: keep a sane ceiling
    draws[:, 1] = np.clip(draws[:, 1], 0.2, 3.0)      # weekend_weekday_ratio
    draws[:, 3] = np.clip(draws[:, 3], 0.05, 3.0)     # cv
    return draws


def build_accounts(n_accounts: int, zones: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    names = list(SEG_SHARES)
    shares = np.array([SEG_SHARES[n] for n in names])
    strata = rng.choice(names, size=n_accounts, p=shares / shares.sum())

    swr = np.empty(n_accounts); wwr = np.empty(n_accounts)
    irr = np.empty(n_accounts); cv = np.empty(n_accounts)
    mean_gpd = np.empty(n_accounts)

    for seg in ("seg_0", "seg_1"):
        idx = np.where(strata == seg)[0]
        if len(idx) == 0:
            continue
        feats = _draw_features(rng, CLUSTERS[seg], len(idx))
        swr[idx], wwr[idx], irr[idx], cv[idx] = feats.T
        med = CLUSTERS[seg]["mean_gpd"] / np.exp(0.4 ** 2 / 2)
        mean_gpd[idx] = rng.lognormal(np.log(med), 0.4, len(idx))

    idx = np.where(strata == "vacant")[0]
    swr[idx], wwr[idx], irr[idx], cv[idx] = 1.0, 1.0, 0.1, 1.2
    mean_gpd[idx] = rng.gamma(VACANT_GPD_SHAPE, VACANT_GPD_SCALE, len(idx))

    idx = np.where(strata == "top_consumers")[0]
    swr[idx] = rng.uniform(1.2, 2.0, len(idx))    # steady commercial-ish, mild seasonality
    wwr[idx] = rng.uniform(0.6, 1.0, len(idx))    # weekday-heavy (commercial)
    irr[idx] = rng.uniform(0.2, 0.5, len(idx))
    cv[idx] = rng.uniform(0.3, 0.7, len(idx))
    non_top_sum = mean_gpd[strata != "top_consumers"].sum()
    # Volume-share back-calculation: solve the average top-consumer mean_gpd
    # so this stratum's realized share of total system volume matches the
    # real calibrated 25.47%, given everyone else's already-drawn volume.
    share = TOP_CONSUMERS_VOLUME_SHARE_PCT / 100
    target_top_sum = share / (1 - share) * non_top_sum
    draw = rng.lognormal(np.log(1.0), 0.5, len(idx))
    mean_gpd[idx] = draw * (target_top_sum / draw.sum())

    zone_choice = rng.choice(zones["name"].values, size=n_accounts,
                             p=zones["acreage"].values / zones["acreage"].values.sum())
    zone_lookup = zones.set_index("name")["system"]
    system = zone_lookup.loc[zone_choice].values

    # Real tier exposure is a USAGE breakpoint (accounts above tier1_max
    # gal/month), not independent of volume -- real tier23_q0_mgd (6.17 MGD
    # / 7923 accounts =~ 780 gpd/account) is ~3.3x real tier1_q0_mgd (3.35
    # MGD / 14040 accounts =~ 239 gpd/account). Found live: assigning tier
    # independent of mean_gpd made both groups' aggregate baseline volume
    # similar, so a similar %-of-expected step produced similar absolute-
    # MGD effects in both groups -- the DiD (their MGD difference) came out
    # positive/near-zero instead of the real -0.28 MGD, because the real
    # DiD's magnitude depends on tier23 being the bigger-volume group, not
    # just on the two groups' step percentages differing. Rank by mean_gpd
    # and take the top TIER23_FRAC as tier2_3_exposed instead.
    threshold = np.quantile(mean_gpd, 1 - TIER23_FRAC)
    tier = np.where(mean_gpd >= threshold, "tier2_3_exposed", "tier1_only")
    # Vacant accounts shouldn't carry a meaningful tier-price response --
    # near-zero usage means near-zero exposure to a per-kgal marginal rate.
    tier[strata == "vacant"] = "tier1_only"

    account_id = np.array([str(rng.integers(1_000_000_000, 9_999_999_999)) for _ in range(n_accounts)])
    meter_id = np.array([str(rng.integers(10_000_000, 99_999_999)) for _ in range(n_accounts)])

    return pd.DataFrame({
        "account_id": account_id, "meter_id": meter_id, "stratum": strata,
        "zone": zone_choice, "system": system, "tier": tier,
        "summer_winter_ratio": swr, "weekend_weekday_ratio": wwr,
        "irrigation_share": irr, "cv": cv, "mean_gpd": mean_gpd,
    })


def _seasonal_shape(doy: np.ndarray) -> np.ndarray:
    """Irrigation-season-shaped curve, peak ~day 195 (mid-July), smooth
    trough in mid-January -- a raised cosine, exactly representable as a
    linear combination of a single sin/cos-of-day-of-year harmonic (it IS
    one: (1+cos(x))/2), so the weather model's doy_sin/doy_cos/month_sin/
    month_cos features can explain it almost perfectly, leaving nothing of
    the deterministic seasonal shape itself in the residuals.

    Found live, in two stages:
    1. An earlier `clip(sin(...), 0, None) ** 1.5` version was nonlinear
       enough that the linear calendar features couldn't fully explain it
       -- weather-model R^2 came out ~0.76 vs. the real pipeline's
       ~0.93-0.94, and the injected June-1 step got swamped/sign-flipped
       by the leftover unexplained seasonal signal.
    2. Switching to a plain `clip(sin(...), 0, None)` fixed R^2 (~0.92) but
       the CLIP itself is a kink a single harmonic still can't represent
       exactly -- residual diagnostics showed a recurring ~-0.09 to -0.13
       MGD bias every year's April-May "green-up" transition (present even
       in years with no injected step, e.g. 2025), large enough to hide the
       injected effect inside seasonal-misfit noise. In the REAL system,
       actual daily temperature data explains that transition (irrigation
       starts when it warms up); this synthetic demand has no true weather
       dependence to lean on, so 100% of that kink's residual was
       unexplained. A raised cosine has no kink at all -- smooth, always
       non-negative, and algebraically a single harmonic -- removing the
       residual noise source rather than trying to out-model it.
    """
    return (1.0 + np.cos(2 * np.pi * (doy - 195) / 365.25)) / 2.0


def _solve_seasonal_params(mean_gpd: np.ndarray, swr: np.ndarray,
                            mean_shape: float, s_summer: float, s_winter: float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Per-account (indoor_floor, seasonal_amplitude) so that:
      mean(daily) == mean_gpd, and summer_mean/winter_mean == swr,
    using the SAME summer(Jun-Sep)/winter(Dec-Mar) month definitions
    behavioral/segmentation.py's build_features() uses, so recomputing
    summer_winter_ratio on the generated series recovers close to the
    target (round-trip fidelity for Phase 4 validation).
    """
    denom = (s_summer - mean_shape) - swr * (s_winter - mean_shape)
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    amp = mean_gpd * (swr - 1.0) / denom
    floor = mean_gpd - amp * mean_shape
    # Extreme swr draws can push floor negative (all-irrigation account) --
    # clip to a small positive floor and accept the resulting mean drift
    # rather than fabricate an unphysical negative baseline.
    floor = np.clip(floor, 1.0, None)
    return floor, amp


def build_daily_series(accounts: pd.DataFrame, dates: pd.DatetimeIndex,
                       rng: np.random.Generator) -> np.ndarray:
    """Returns (n_accounts, n_days) gallons matrix."""
    doy = np.asarray(dates.dayofyear)
    month = np.asarray(dates.month)
    is_weekend = np.asarray(dates.dayofweek) >= 5
    shape = _seasonal_shape(doy)

    summer_mask = np.isin(month, [6, 7, 8, 9])
    winter_mask = np.isin(month, [12, 1, 2, 3])
    mean_shape, s_summer, s_winter = shape.mean(), shape[summer_mask].mean(), shape[winter_mask].mean()

    mean_gpd = accounts["mean_gpd"].values
    swr = accounts["summer_winter_ratio"].values
    wwr = accounts["weekend_weekday_ratio"].values
    cv = accounts["cv"].values
    n, d = len(accounts), len(dates)

    floor, amp = _solve_seasonal_params(mean_gpd, swr, mean_shape, s_summer, s_winter)
    seasonal = floor[:, None] + amp[:, None] * shape[None, :]
    seasonal = np.clip(seasonal, 0.5, None)

    dow_raw = np.where(is_weekend[None, :], wwr[:, None], 1.0)
    dow_norm = (5 / 7) * 1.0 + (2 / 7) * wwr
    dow_mult = dow_raw / dow_norm[:, None]

    k = 1.0 / np.clip(cv, 0.05, None) ** 2
    theta = cv ** 2
    noise = rng.gamma(np.broadcast_to(k[:, None], (n, d)), np.broadcast_to(theta[:, None], (n, d)))
    noise_df = pd.DataFrame(noise.T)   # rows=days, cols=accounts, for a cheap along-time smooth
    noise = noise_df.ewm(alpha=0.5, adjust=False).mean().to_numpy().T

    gallons = seasonal * dow_mult * noise
    gallons = np.clip(gallons, 0.0, None)

    # Vacant accounts need a real >50% zero_day_frac to be caught by
    # behavioral/segmentation.py's actual vacant_intermittent stratum rule
    # (zero_day_frac > 0.5) -- a low but continuous mean_gpd (the general
    # seasonal/noise path above) reads as a low-USE account, not a vacant
    # one. Found live: realized vacant share came out 0.8% vs the
    # calibrated 3.2% target because of this. A Bernoulli zero-gate on most
    # days (occasional real reads/leaks on the rest) is what actually
    # produces the zero_day_frac the real classifier looks for.
    vacant_mask = (accounts["stratum"] == "vacant").values
    if vacant_mask.any():
        zero_gate = rng.random((vacant_mask.sum(), d)) < 0.85
        sub = gallons[vacant_mask]
        sub[zero_gate] = 0.0
        gallons[vacant_mask] = sub

    return gallons


HETEROGENEITY_SPREAD_PP = 3.0   # +/- percentage points of shape-driven spread around the flat target


def apply_drought_response(gallons: np.ndarray, accounts: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Injects the calibrated June-1 step: a shared level shift for every
    account (both tiers show a step at the price-change date in the real
    ITS fit) PLUS an extra cut for tier2_3_exposed accounts specifically,
    sized so the two groups' AGGREGATE step percentages hit
    TIER1_STEP_PCT/TIER23_STEP_PCT exactly, with a modest
    summer_winter_ratio-shaped spread around that exact target for realism.

    Found live: an earlier version scaled RESPONSE_BINS' median_pct values
    (behavioral/response_profile.py's RAW, non-weather-normalized YoY %
    change metric) against TIER1_STEP_PCT (the weather-normalized ITS step
    %, a completely different metric/scale) -- subtracting one from the
    other doesn't make dimensional sense, and the resulting rescale flipped
    which tier ended up with the larger cut. Fixed by hitting the two
    groups' exact calibrated aggregate percentages via a flat per-tier
    target, then adding a small MEAN-ZERO (within tier23) shape component
    from RESPONSE_BINS' relative ordering only -- realism without
    corrupting the calibrated aggregate.
    """
    post = np.asarray([d >= PRICE_STEP_DATE for d in dates.date])
    tier23_mask = (accounts["tier"] == "tier2_3_exposed").values

    mult = np.ones_like(gallons)
    mult[:, post] = 1.0 + TIER1_STEP_PCT / 100   # shared step, both tiers

    swr = accounts["summer_winter_ratio"].values
    raw_shape = np.zeros(len(accounts))
    for lo, hi, med in RESPONSE_BINS:
        raw_shape[(swr >= lo) & (swr < hi)] = med
    weights = np.clip(accounts["mean_gpd"].values * tier23_mask, 1e-6, None)
    w_mean = np.average(raw_shape, weights=weights)
    w_std = np.sqrt(np.average((raw_shape - w_mean) ** 2, weights=weights))
    shape_centered = (raw_shape - w_mean) / w_std if w_std > 1e-9 else np.zeros(len(accounts))

    target_extra_pct = TIER23_STEP_PCT - TIER1_STEP_PCT
    extra_pct = target_extra_pct + shape_centered * HETEROGENEITY_SPREAD_PP

    # np.ix_ combines the row-boolean and column-boolean masks into a valid
    # fancy-index pair -- chained `arr[row_mask][:, col_mask] = ...` would
    # assign into a copy and silently do nothing.
    extra = np.ones_like(gallons)
    extra[np.ix_(tier23_mask, post)] = 1.0 + (extra_pct[tier23_mask][:, None] / 100)

    return gallons * mult * extra


def apply_data_artifacts(panel: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    panel = panel.copy()
    panel["estimated"] = rng.random(len(panel)) < ESTIMATED_FRAC
    neg = rng.random(len(panel)) < NEGATIVE_ARTIFACT_FRAC
    panel.loc[neg, "gallons"] = -panel.loc[neg, "gallons"].abs()
    drop = rng.random(len(panel)) < MISSING_FRAC
    return panel[~drop]


def build_customer_panel(accounts: pd.DataFrame, dates: pd.DatetimeIndex,
                         rng: np.random.Generator, weather: pd.DataFrame) -> pd.DataFrame:
    gallons = build_daily_series(accounts, dates, rng)
    gallons = apply_drought_response(gallons, accounts, dates)

    # Shared across every account identically (not per-account noise) --
    # see weather_shock()'s docstring for why this has to be a separate,
    # non-idiosyncratic step rather than folded into build_daily_series.
    shock = weather_shock(dates, weather)
    gallons = np.clip(gallons * (1.0 + shock[None, :]), 0.0, None)

    long = pd.DataFrame({
        "account_id": np.repeat(accounts["account_id"].values, len(dates)),
        "meter_id": np.repeat(accounts["meter_id"].values, len(dates)),
        "date": np.tile(dates.date, len(accounts)),
        "gallons": gallons.ravel(),
        "class_code": "RES1",
    })
    return apply_data_artifacts(long, rng)


def build_system_series(accounts: pd.DataFrame, panel: pd.DataFrame, sites: pd.DataFrame,
                        rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (production_df, entry_point_history_df)."""
    system_of = accounts.set_index("account_id")["system"]
    consumption = (panel.assign(system=panel["account_id"].map(system_of))
                   .groupby(["date", "system"])["gallons"].sum() / 1e6)
    consumption = consumption.unstack(fill_value=0.0)

    def _inject_input(consumption_mgd: pd.Series, nrw_pct: float, corr: float, rng) -> pd.Series:
        a = 1.0 / (1.0 - nrw_pct / 100)
        sd_c = consumption_mgd.std()
        sd_eps = a * sd_c * np.sqrt(max(1.0 / corr ** 2 - 1.0, 1e-6))
        eps = rng.normal(0, sd_eps, len(consumption_mgd))
        return (consumption_mgd * a + eps).clip(lower=0.01)

    input_a = _inject_input(consumption.get("system_a", pd.Series(0, index=consumption.index)),
                            NRW["system_a"]["mean_nrw_pct"], NRW["system_a"]["correlation"], rng)
    input_b = _inject_input(consumption.get("system_b", pd.Series(0, index=consumption.index)),
                            NRW["system_b"]["mean_nrw_pct"], NRW["system_b"]["correlation"], rng)

    production_df = pd.DataFrame({"date": input_b.index, "production_mgd": input_b.values})
    production_df["maintenance"] = 0
    maint_days = rng.choice(len(production_df), size=max(1, len(production_df) // 180), replace=False)
    production_df.loc[maint_days, ["production_mgd", "maintenance"]] = [0.0, 1]

    dw_sites = sites[sites["role"] == "dw_master"]
    weights = rng.dirichlet(np.ones(len(dw_sites)))
    rows = []
    for w, (_, site) in zip(weights, dw_sites.iterrows()):
        rows.append(pd.DataFrame({
            "date": input_a.index, "beacon_account": site["beacon_account"],
            "mgd": (input_a.values * w).clip(min=0.0),
        }))
    entry_history = pd.concat(rows, ignore_index=True)
    return production_df, entry_history


def write_outputs(panel: pd.DataFrame, accounts: pd.DataFrame,
                  production_df: pd.DataFrame, entry_history: pd.DataFrame) -> None:
    ch_dir = OUT_DIR / "state" / "customer_history"
    ch_dir.mkdir(parents=True, exist_ok=True)
    panel = panel.copy()
    # Real customer_history parquet stores `date` as plain python date
    # objects (object dtype, pyarrow date32), NOT a pandas Timestamp column
    # -- converting via pd.to_datetime() here produced a datetime64[ms]
    # column instead, which broke comparisons against plain `date` objects
    # elsewhere in the real pipeline (weather_norm.fit's `index <= train_end`).
    # Keep `date` untouched; use a separate temp key only for partitioning.
    month_key = pd.to_datetime(panel["date"].astype(str)).dt.to_period("M")
    for month, g in panel.groupby(month_key):
        g.to_parquet(ch_dir / f"{month}.parquet", index=False)

    (OUT_DIR / "state").mkdir(parents=True, exist_ok=True)
    entry_history.to_parquet(OUT_DIR / "state" / "entry_point_history.parquet", index=False)
    production_df.to_csv(OUT_DIR / "production_history.csv", index=False)

    zone_map = accounts[["account_id", "zone", "system"]].rename(
        columns={"account_id": "account_id_new"})
    zone_map["account_id_old"] = ""
    zone_map["meter_id"] = accounts["meter_id"].values
    zone_map["premise_ty"] = np.where(accounts["stratum"] == "top_consumers", "Commercial",
                              np.where(accounts["stratum"] == "vacant", "Vacant", "Residential 1 Unit"))
    zone_map.to_parquet(OUT_DIR / "state" / "meter_zone_map.parquet", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-accounts", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=20260723)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    zones, sites = _load_geography()
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    print("Loading weather (real, public GHCND data)...")
    weather = _load_weather(dates)

    print(f"Generating {args.n_accounts} synthetic accounts...")
    accounts = build_accounts(args.n_accounts, zones, rng)
    print(accounts["stratum"].value_counts())

    print(f"Building daily series over {len(dates)} days...")
    panel = build_customer_panel(accounts, dates, rng, weather)

    print("Building system aggregate series (production + entry-point history)...")
    production_df, entry_history = build_system_series(accounts, panel, sites, rng)

    print("Writing outputs...")
    write_outputs(panel, accounts, production_df, entry_history)
    accounts.to_parquet(OUT_DIR / "synthetic_accounts_debug.parquet", index=False)

    print(f"Wrote {len(panel):,} customer-history rows, "
          f"{len(entry_history):,} entry-point rows, "
          f"{len(production_df):,} production rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
