"""Tier-exposure difference-in-differences — isolates the price-specific
increment of a bundled drought-response package (restrictions + a tiered-
rate price step) from the concurrent restrictions.

Why this exists: a segment-level ITS answers "which behavioral cluster
responded" to the BUNDLED package (restrictions + price step, estimated
jointly). Neither that nor a system-level ITS separates price from
restrictions on its own. This does: tier-exposed customers (who faced
restrictions AND a real price increase on their marginal usage) vs.
tier-1-only customers (who faced the same restrictions but ~flat pricing)
— both hit by the same restrictions on the same date, so the DIFFERENCE in
their responses isolates the price-specific increment.

Standard DiD identifying assumption: absent the price step, both groups'
trends would have continued in parallel. check_parallel_trends() tests this
on pre-period data — required, not optional: a DiD estimate without it
passing is not trustworthy.

**Two classification sources.** classify_tier_exposure(source="proxy") uses
pre-period USAGE VOLUME as a stand-in for tier exposure — usable without any
per-account meter/share data. classify_tier_exposure(source="real") uses
each account's actual meter size/technology and, where available, a real
per-account share count against a tiered-rate schedule for an exact
breakpoint, falling back to a meter-size floor for accounts without a usable
share count. This path needs a local `tier_schedule` module (an account's
own real rate-schedule logic) that isn't included here — see this
function's docstring. Every DiDResult still carries an
is_proxy_classification flag so callers/reports can tell which mode
produced a given result — the machinery itself was validated on a
synthetic fixture (see _self_test()) independent of which classification
source feeds it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats

from behavioral import its as its_mod

log = logging.getLogger(__name__)

_HAC_MAXLAGS = 14
_MIN_PRE_PERIOD_DAYS = 60
_MIN_COVERAGE_FRACTION = 0.5   # fraction of pre-period days an account must report to be classified


def classify_tier_exposure(
    daily: pd.DataFrame,
    pre_period_start: date,
    pre_period_end: date,
    tier_breakpoints_gal_per_month: dict | None,
    *,
    source: str = "proxy",
    proj_root: Path | None = None,
) -> pd.Series:
    """account_id -> "tier1_only" | "tier2_3_exposed" | "unknown", from
    PRE-PERIOD usage volume against tier breakpoints (not post-period, so the
    price step itself can't contaminate the exposure classification — an
    account's post-step usage change is exactly what we're trying to
    measure, not something that should feed its own treatment assignment).

    daily: tidy account_id, date, gallons (customer_store schema).

    source="proxy": tier_breakpoints_gal_per_month={"tier1_max": float} is
    applied uniformly to every account, no meter-size/ERU adjustment. See
    config.yaml `pricing.tiers[*].proxy_breakpoints_gal_per_month`.

    source="real": per-account breakpoints from a local `tier_schedule`
    module (not included in this repo — see module docstring) — real share
    count where available, falling back to a meter-size/technology floor
    for the remainder — requires proj_root, tier_breakpoints_gal_per_month
    is ignored. Accounts with neither a real share count nor a resolvable
    meter type classify "unknown" regardless of usage, same as low-coverage
    accounts.
    """
    window = daily[(daily["date"] >= pre_period_start) & (daily["date"] <= pre_period_end)]
    n_calendar_days = (pre_period_end - pre_period_start).days + 1
    if window.empty:
        return pd.Series(dtype="object")

    total_gal = window.groupby("account_id")["gallons"].sum()
    n_days_reporting = window.groupby("account_id")["date"].nunique()
    coverage = n_days_reporting / n_calendar_days
    monthly_avg = total_gal / n_days_reporting * 30.44
    eligible = coverage >= _MIN_COVERAGE_FRACTION

    if source == "proxy":
        tier1_max = tier_breakpoints_gal_per_month["tier1_max"]
    elif source == "real":
        if proj_root is None:
            raise ValueError("classify_tier_exposure(source='real') requires proj_root")
        try:
            import tier_schedule
        except ImportError as e:
            raise ImportError(
                "classify_tier_exposure(source='real') needs a local `tier_schedule` "
                "module (per-account real-share-count rate-schedule logic) that isn't "
                "included in this repo -- use source='proxy' instead, or supply your "
                "own tier_schedule.py with a load_account_tier1_max(proj_root) function."
            ) from e
        account_tier1_max = tier_schedule.load_account_tier1_max(proj_root)
        tier1_max = total_gal.index.to_series().map(account_tier1_max)
        eligible = eligible & tier1_max.notna()
    else:
        raise NotImplementedError(f"classify_tier_exposure(source={source!r}) not implemented — "
                                  "only 'proxy' and 'real' exist")

    labels = pd.Series("unknown", index=total_gal.index)
    labels.loc[eligible & (monthly_avg <= tier1_max)] = "tier1_only"
    labels.loc[eligible & (monthly_avg > tier1_max)] = "tier2_3_exposed"

    counts = labels.value_counts().to_dict()
    log.info("classify_tier_exposure (%s): %s", source, counts)
    return labels


def check_parallel_trends(
    tier1_frame: pd.DataFrame, tier23_frame: pd.DataFrame, pre_period_end: date,
) -> tuple[bool, dict]:
    """Standard DiD pre-check: do the two groups' weather-normalized
    residuals trend together before ANY intervention?

    Compares pre-period TREND SLOPES, not raw correlation — the DiD
    identifying assumption is specifically about parallel trends (two
    seasonal series can correlate highly while their underlying trends
    diverge). Each frame is behavioral.weather_norm.WeatherNormResult.frame
    (index=date, residual_mgd) for one exposure group's aggregate series.

    Returns (passed, diagnostics) where passed = the two slopes are not
    statistically distinguishable (p>=0.05 on their difference) -- REQUIRED
    gate, not optional. A DiD estimate computed despite this failing is not
    trustworthy and estimate_did() records that in parallel_trends_ok rather
    than silently proceeding as if it passed.
    """
    def _slope(frame: pd.DataFrame) -> tuple[float, float, int]:
        df = frame.dropna(subset=["residual_mgd"])
        df = df[df.index <= pre_period_end]
        if len(df) < _MIN_PRE_PERIOD_DAYS:
            raise RuntimeError(f"check_parallel_trends: only {len(df)} pre-period days "
                               f"(<{_MIN_PRE_PERIOD_DAYS}) -- too few for a trend check")
        X = sm.add_constant(pd.DataFrame(
            {"t": [(d - df.index.min()).days for d in df.index]}, index=df.index), has_constant="add")
        ols = sm.OLS(df["residual_mgd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": _HAC_MAXLAGS})
        return float(ols.params["t"]), float(ols.bse["t"]), len(df)

    slope1, se1, n1 = _slope(tier1_frame)
    slope23, se23, n23 = _slope(tier23_frame)

    diff = slope23 - slope1
    se_diff = (se1 ** 2 + se23 ** 2) ** 0.5
    z = diff / se_diff if se_diff else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    passed = p_value >= 0.05

    diagnostics = {
        "tier1_slope_mgd_per_day": slope1, "tier23_slope_mgd_per_day": slope23,
        "slope_diff_mgd_per_day": diff, "p_value": p_value,
        "n_pre_days": {"tier1": n1, "tier23": n23},
    }
    log.info("check_parallel_trends: slope diff=%.5f MGD/day, p=%.3f -> %s",
             diff, p_value, "PASS" if passed else "FAIL")
    return passed, diagnostics


def _residual_correlation(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation of two residual series on their overlapping index,
    0.0 if too few overlapping points to estimate. Used as a proxy for the
    correlation between the two groups' ITS step-coefficient estimates --
    see estimate_did()'s docstring for why plain independence is the wrong
    default here."""
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    rho = aligned["a"].corr(aligned["b"])
    return float(rho) if pd.notna(rho) else 0.0


def _combined_se(se_a: float, se_23: float, rho: float) -> float:
    """sqrt(Var(A-B)) = sqrt(Var(A) + Var(B) - 2*Cov(A,B)), Cov approximated
    as rho * se_a * se_23. Falls back to the independence-assumption sum
    (rho=0) if sampling noise in `rho` would otherwise push the variance
    negative -- Var(A-B) is never actually negative, only its estimate can
    be, from a noisy correlation on a short/noisy pre-period."""
    var = se_a ** 2 + se_23 ** 2 - 2 * rho * se_a * se_23
    if var <= 0:
        var = se_a ** 2 + se_23 ** 2
    return var ** 0.5


@dataclass
class DiDResult:
    tier1_effect_mgd: float
    tier23_effect_mgd: float
    did_effect_mgd: float          # the isolated price-specific increment
    did_se_mgd: float
    did_p_value: float
    parallel_trends_ok: bool
    semi_elasticity: float | None  # None if price_differential not given
    placebo_clean: bool
    is_proxy_classification: bool  # True until real per-account tier data lands


def estimate_did(
    tier1_frame: pd.DataFrame,
    tier23_frame: pd.DataFrame,
    step_date: date,
    *,
    parallel_trends_ok: bool,
    price_differential: float | None = None,
    placebo_dates: list[date] = (),
    is_proxy_classification: bool = True,
) -> DiDResult:
    """The actual estimator: difference between the two groups' step
    coefficients (each from its.fit() on its own weather-normalized series),
    combined SE, z-test. Isolates the price-specific increment since both
    groups faced the same restrictions but only tier2_3_exposed faced the
    price step.

    The combined SE is NOT a plain independent-samples sum of variances:
    both groups' weather-normalized residuals come from OLS fit on the same
    system's shared weather features (weather_norm.fit), so a common demand
    shock the linear tmax term underfits (e.g. a heat wave) plausibly
    correlates the two residual series -- ignoring that overstates did_se
    (conservative direction: fewer false "significant" results, but
    unquantified until now). _combined_se() uses the two residual series'
    own correlation as a proxy for the step-coefficient estimates'
    correlation (both models share the same step-dummy design, differing
    only in outcome) -- a real, documented approximation, not an exact
    joint (SUR-style) estimate, but no longer a silent independence
    assumption either.

    parallel_trends_ok: pass the result of check_parallel_trends() —
    required argument, not computed internally, so the caller can't
    accidentally skip the gate and still get a result that looks validated.

    price_differential: known $/kgal difference between tiers (from
    config.yaml pricing.tiers), if available -- semi_elasticity =
    did_effect / price_differential. Labeled a semi-elasticity, not a
    continuous elasticity: assumes homogeneous within-tier response and
    coarse between-tier price variation, not the smooth marginal-price
    curve a formal Taylor-Nordin discrete-continuous model would estimate.

    placebo_dates: fake step dates tested in the pre-period, same pattern as
    its.placebo() -- placebo_clean=True only if the DID itself (not just
    each group individually) is statistically indistinguishable from zero
    at every fake date.
    """
    fit1 = its_mod.fit(tier1_frame, [("step", step_date)])
    fit23 = its_mod.fit(tier23_frame, [("step", step_date)])
    e1, e23 = fit1.steps[0], fit23.steps[0]

    rho = _residual_correlation(tier1_frame["residual_mgd"], tier23_frame["residual_mgd"])

    did_effect = e23.coef_mgd - e1.coef_mgd
    did_se = _combined_se(e1.se_mgd, e23.se_mgd, rho)
    z = did_effect / did_se if did_se else 0.0
    did_p = float(2 * (1 - stats.norm.cdf(abs(z))))

    semi_elasticity = (did_effect / price_differential) if price_differential else None

    placebo_clean = True
    for fake in placebo_dates:
        p1 = its_mod.placebo(tier1_frame, fake, step_date)
        p23 = its_mod.placebo(tier23_frame, fake, step_date)
        fake_did = p23.coef_mgd - p1.coef_mgd
        fake_se = _combined_se(p1.se_mgd, p23.se_mgd, rho)
        fake_p = 2 * (1 - stats.norm.cdf(abs(fake_did / fake_se))) if fake_se else 1.0
        if fake_p < 0.05:
            placebo_clean = False

    if not parallel_trends_ok:
        log.warning("estimate_did: parallel_trends_ok=False -- result computed anyway but is "
                    "NOT trustworthy, caller must surface this, not just the point estimate")

    return DiDResult(
        tier1_effect_mgd=e1.coef_mgd, tier23_effect_mgd=e23.coef_mgd,
        did_effect_mgd=did_effect, did_se_mgd=did_se, did_p_value=did_p,
        parallel_trends_ok=parallel_trends_ok, semi_elasticity=semi_elasticity,
        placebo_clean=placebo_clean, is_proxy_classification=is_proxy_classification,
    )


def convert_to_price_elasticity(
    semi_elasticity: float, price_baseline: float, quantity_baseline_mgd: float,
) -> float:
    """E = semi_elasticity * (P0/Q0) -- converts this module's dQ/dP
    semi-elasticity (MGD per $/kgal) into a dimensionless price elasticity
    comparable to published water-demand literature (e.g. Espey, Espey &
    Shaw 1997; Dalhuisen et al. 2003).

    semi_elasticity: this module's did_effect_mgd / price_differential output.
    price_baseline: P0, the baseline marginal price the treated group faced
    BEFORE the price step, $/kgal.
    quantity_baseline_mgd: Q0, the treated (tier2_3_exposed) group's own
    baseline daily volume BEFORE the step, MGD -- pull this from real data
    (e.g. the group's pre-period actual_mgd mean), never assumed.

    Units: semi_elasticity is MGD (= 1000 x kgal/day) per $/kgal, so
    semi_elasticity * (price_baseline / quantity_baseline_mgd) already reduces
    cleanly to a dimensionless ratio -- no extra unit conversion needed, both
    sides carry an implicit x1000 that cancels.

    Inherits the same homogeneous-within-tier-response simplification
    semi_elasticity itself already carries (see module docstring: not the
    smooth marginal-price curve a formal Taylor (1975)/Nordin (1976)
    discrete-continuous model would estimate) -- this is a gut-check
    conversion for plausibility comparison, not a re-estimation that resolves
    that limitation.
    """
    if quantity_baseline_mgd == 0:
        raise ValueError("convert_to_price_elasticity: quantity_baseline_mgd is zero -- cannot divide")
    return semi_elasticity * (price_baseline / quantity_baseline_mgd)


# ── Self-test: synthetic fixture with a known injected differential effect ──

def _self_test() -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", "2026-07-01", freq="D").date
    step_date = date(2026, 6, 1)

    # ── classify_tier_exposure ──
    n_accounts = 200
    acct_rows = []
    rng_usage = rng.normal(4000, 1500, n_accounts).clip(500, None)  # gal/month, pre-period baseline
    pre_start, pre_end = date(2026, 1, 1), date(2026, 5, 31)
    for i, base in enumerate(rng_usage):
        daily_gal = base / 30.44
        for d in dates:
            if pre_start <= d <= pre_end or True:  # full history so coverage is high everywhere
                acct_rows.append({"account_id": f"a{i}", "date": d,
                                  "gallons": max(0.0, daily_gal + rng.normal(0, daily_gal * 0.1))})
    daily_df = pd.DataFrame(acct_rows)

    labels = classify_tier_exposure(daily_df, pre_start, pre_end, {"tier1_max": 5000})
    assert set(labels.unique()) <= {"tier1_only", "tier2_3_exposed", "unknown"}
    assert (labels == "tier1_only").sum() > 0 and (labels == "tier2_3_exposed").sum() > 0, \
        "expected both tiers present with these synthetic usage levels"

    # ── classify_tier_exposure(source="real") ──
    # Needs a local `tier_schedule` module (not included in this repo, see
    # did.py's module docstring) to resolve real per-account breakpoints --
    # skip this block gracefully if it isn't present rather than fail the
    # whole self-test on an intentionally-omitted dependency.
    try:
        import tier_schedule  # noqa: F401
        has_tier_schedule = True
    except ImportError:
        has_tier_schedule = False

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        proj_root = Path(tmp)
        (proj_root / "state").mkdir()

        if has_tier_schedule:
            zm = pd.DataFrame({
                "meter_id": ["m1", "m2", "m3", "m4"],
                "account_id_new": ["r1", "r2", "r3", "r4"],
                "account_id_old": ["", "", "", ""],
                "meter_type": [
                    "5/8 Denver Displacement",   # tier1_max = 5,000 gal/mo
                    "5/8 Denver Displacement",   # tier1_max = 5,000 gal/mo
                    "8 MG Turbine",              # tier1_max = 630,000 gal/mo
                    "3 MG Compound",             # unclassified -- no breakpoint
                ],
            })
            zm.to_parquet(proj_root / "state" / "meter_zone_map.parquet")

            pre_dates = pd.date_range(pre_start, pre_end, freq="D").date
            # r3 is the case that actually distinguishes real-mode from proxy:
            # 100,000 gal/mo would be a massive tier2_3_exposed outlier under
            # ANY plausible residential-scale proxy threshold, but its real 8"
            # turbine breakpoint (630,000) puts it comfortably in tier1_only.
            usage = {"r1": 4_000.0, "r2": 8_000.0, "r3": 100_000.0, "r4": 4_000.0}
            real_rows = [
                {"account_id": acct, "date": d, "gallons": monthly_gal / 30.44}
                for acct, monthly_gal in usage.items() for d in pre_dates
            ]
            real_labels = classify_tier_exposure(
                pd.DataFrame(real_rows), pre_start, pre_end, None, source="real", proj_root=proj_root)

            assert real_labels["r1"] == "tier1_only"
            assert real_labels["r2"] == "tier2_3_exposed"
            assert real_labels["r3"] == "tier1_only", (
                f"8\" turbine at 100,000 gal/mo is well under its real 630,000 breakpoint -- "
                f"got {real_labels['r3']!r}, real mode must not fall back to a residential-scale proxy")
            assert real_labels["r4"] == "unknown", "Compound meters have no breakpoint and must classify unknown"
        else:
            real_rows = [{"account_id": "r1", "date": pre_start, "gallons": 100.0}]
            print("classify_tier_exposure(source='real'): tier_schedule module not present, "
                  "skipping the breakpoint-accuracy checks -- ValueError/NotImplementedError "
                  "checks below still run.")

        try:
            classify_tier_exposure(pd.DataFrame(real_rows), pre_start, pre_end, None,
                                   source="real", proj_root=None)
            raise AssertionError("source='real' without proj_root should raise ValueError")
        except ValueError:
            pass

        try:
            classify_tier_exposure(pd.DataFrame(real_rows), pre_start, pre_end, None, source="bogus")
            raise AssertionError("source='bogus' should raise NotImplementedError")
        except NotImplementedError:
            pass

    # ── check_parallel_trends + estimate_did: known injected differential ──
    doy = pd.DatetimeIndex(pd.to_datetime(list(dates))).dayofyear.values
    seasonal = 2.0 * np.sin((doy - 100) / 365 * 2 * 3.14159)
    common_noise = rng.normal(0, 0.15, len(dates))

    KNOWN_DID_EFFECT = -0.8  # MGD, injected only into tier23's post-step residual

    tier1_residual = seasonal * 0.5 + common_noise + rng.normal(0, 0.1, len(dates))
    tier23_residual = seasonal * 0.5 + common_noise + rng.normal(0, 0.1, len(dates))
    post_mask = pd.to_datetime(list(dates)) >= pd.Timestamp(step_date)
    tier23_residual = tier23_residual + np.where(post_mask, KNOWN_DID_EFFECT, 0.0)
    # both groups get the SAME restriction-only effect (not isolated by DiD) --
    # confirms the DiD subtracts it out rather than needing it to be absent.
    RESTRICTION_ONLY_EFFECT = -0.3
    restr_mask = pd.to_datetime(list(dates)) >= pd.Timestamp(date(2026, 4, 1))
    tier1_residual = tier1_residual + np.where(restr_mask, RESTRICTION_ONLY_EFFECT, 0.0)
    tier23_residual = tier23_residual + np.where(restr_mask, RESTRICTION_ONLY_EFFECT, 0.0)

    tier1_frame = pd.DataFrame({"residual_mgd": tier1_residual, "expected_mgd": 5.0}, index=dates)
    tier23_frame = pd.DataFrame({"residual_mgd": tier23_residual, "expected_mgd": 5.0}, index=dates)

    passed, diag = check_parallel_trends(tier1_frame, tier23_frame, date(2026, 3, 31))
    assert passed, f"expected parallel pre-period trends to pass, got {diag}"

    result = estimate_did(tier1_frame, tier23_frame, step_date, parallel_trends_ok=passed,
                          price_differential=2.45, is_proxy_classification=True)
    recovery_error = abs(result.did_effect_mgd - KNOWN_DID_EFFECT)
    assert recovery_error < 0.15, \
        f"DiD estimate {result.did_effect_mgd:.3f} too far from known injected {KNOWN_DID_EFFECT} MGD"
    assert result.did_p_value < 0.05, f"expected a significant DiD effect, p={result.did_p_value:.3f}"
    assert result.is_proxy_classification is True
    assert result.semi_elasticity is not None and abs(result.semi_elasticity - result.did_effect_mgd / 2.45) < 1e-9

    # ── check_parallel_trends: deliberately non-parallel synthetic trends should FAIL ──
    diverging_residual = tier1_residual + np.linspace(0, 3.0, len(dates))  # steadily diverging
    diverging_frame = pd.DataFrame({"residual_mgd": diverging_residual, "expected_mgd": 5.0}, index=dates)
    passed2, diag2 = check_parallel_trends(tier1_frame, diverging_frame, date(2026, 3, 31))
    assert not passed2, f"expected diverging trends to FAIL the parallel-trends check, got {diag2}"

    # ── convert_to_price_elasticity: hand-computed known-answer check ──
    # semi_elasticity=-0.5 MGD per $/kgal, P0=$6/kgal, Q0=3 MGD baseline
    # -> E = -0.5 * (6/3) = -1.0, computed by hand, not just re-run through
    # the same formula.
    e = convert_to_price_elasticity(semi_elasticity=-0.5, price_baseline=6.0, quantity_baseline_mgd=3.0)
    assert abs(e - (-1.0)) < 1e-9, f"convert_to_price_elasticity: expected -1.0, got {e:.6f}"
    try:
        convert_to_price_elasticity(semi_elasticity=-0.5, price_baseline=6.0, quantity_baseline_mgd=0.0)
        raise AssertionError("convert_to_price_elasticity should reject a zero quantity_baseline_mgd")
    except ValueError:
        pass

    print(f"SELF-TEST PASSED: DiD recovered {result.did_effect_mgd:+.3f} MGD "
          f"(known injected {KNOWN_DID_EFFECT:+.1f} MGD, error {recovery_error:.3f}), "
          f"p={result.did_p_value:.4f}, semi-elasticity={result.semi_elasticity:.4f} MGD per $/kgal; "
          f"parallel-trends pass/fail cases both correct; convert_to_price_elasticity known-answer check passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    _self_test()
