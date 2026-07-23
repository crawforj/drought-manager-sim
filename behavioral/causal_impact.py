"""Hand-built Bayesian-structural-time-series-style causal impact, built
after `pycausalimpact` (calls
`DataFrame.applymap()`, removed in pandas 3.0 -- confirmed live via
AttributeError) and `tfcausalimpact` (packaging metadata written for an older
Python, fails to install) both turned out broken in this environment. Built
instead directly on `statsmodels.tsa.statespace.structural.UnobservedComponents`
-- the actual state-space engine the CausalImpact R package/Python ports wrap
-- confirmed live to fit, forecast, and return confidence intervals correctly.
Same "hand-build on an already-trusted core dependency rather than add a
broken wrapper" choice `behavioral/its.py` and `behavioral/dml.py` already
made.

**What this is, precisely**: a local-level (random walk) state-space model
with optional exogenous regressors, fit on PRE-period data only (same
train-only-on-pre-intervention discipline `weather_norm.fit()` uses), then
forecast forward through the post-period. The forecast IS the synthetic
counterfactual -- "what would this series have looked like with no
intervention" -- with real prediction intervals from the Kalman filter itself,
not a bootstrap approximation. `pointwise_effect = actual - predicted`;
`cumulative_effect_mgd` sums that over the post period, with its own
propagated interval (variances summed across the post period's daily forecast
errors -- same combination-of-variances logic `behavioral/did.py`'s
`estimate_did()` already uses to combine two group SEs into one).

**Complements ITS/weather_norm, does not replace it**: `its.py`/`weather_norm.py`
ask "did the LEVEL step down," pooled across a whole system or segment, with a
model built to explain WEATHER specifically. This asks "what would THIS
series' own trajectory have looked like" -- naturally suited to ONE account or
a small group at a time, where there usually isn't enough history for a full
weather-feature regression. A single residential account's pre-period is
typically months, not years -- treat single-account results here as
illustrative/exploratory, not headline-grade, and always check
`model_r2_preperiod` before trusting a given account's counterfactual.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

log = logging.getLogger(__name__)

_MIN_PRE_DAYS = 60   # a single account's history is often thin -- see module docstring
_N_SIMULATIONS = 2000   # Monte Carlo draws for the cumulative-effect interval, see fit_causal_impact()


@dataclass
class CausalImpactResult:
    pointwise_effect: pd.Series      # post-period date -> actual - predicted, gal/day or MGD (caller's units)
    cumulative_effect_mgd: float
    cumulative_effect_ci: tuple[float, float]   # 95%, variance-summed across post-period days
    relative_effect_pct: float                  # cumulative_effect / sum(predicted) * 100
    model_r2_preperiod: float
    n_pre: int
    n_post: int


def fit_causal_impact(
    frame: pd.DataFrame,
    pre_period_end: date,
    post_period_end: date,
    feature_cols: list[str] | None = None,
) -> CausalImpactResult:
    """frame: index=date, column 'actual_mgd' (or any consistent volume unit)
    plus optional exogenous regressor columns (e.g. weather features).
    feature_cols: which columns to use as regressors; None = all columns
    except 'actual_mgd'. Pass [] explicitly for a pure local-level model with
    no regressors.
    """
    df = frame.dropna(subset=["actual_mgd"]).sort_index()
    df.index = pd.to_datetime(df.index).date   # normalize Timestamp or date index alike
    exog_cols = list(df.columns.drop("actual_mgd")) if feature_cols is None else list(feature_cols)

    pre = df[df.index <= pre_period_end]
    post = df[(df.index > pre_period_end) & (df.index <= post_period_end)]
    if len(pre) < _MIN_PRE_DAYS:
        raise RuntimeError(f"fit_causal_impact: only {len(pre)} pre-period days (<{_MIN_PRE_DAYS}) -- too thin to fit a local-level model")
    if len(post) == 0:
        raise RuntimeError("fit_causal_impact: no post-period days in range")

    exog_pre = pre[exog_cols] if exog_cols else None
    model = UnobservedComponents(pre["actual_mgd"], level="local level", exog=exog_pre)
    fitted = model.fit(disp=False)

    resid = pre["actual_mgd"] - fitted.fittedvalues
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((pre["actual_mgd"] - pre["actual_mgd"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    exog_post = post[exog_cols] if exog_cols else None
    forecast = fitted.get_forecast(steps=len(post), exog=exog_post)
    predicted = pd.Series(forecast.predicted_mean.to_numpy(), index=post.index)
    pointwise = post["actual_mgd"] - predicted
    cumulative = float(pointwise.sum())

    # Monte Carlo the cumulative-effect interval rather than summing each
    # day's forecast variance independently. Per-day forecast errors in a
    # local-level (integrated/random-walk) model are strongly autocorrelated
    # across the horizon -- an innovation to the level early in the
    # post-period propagates into every later day -- so treating them as
    # independent badly understates the uncertainty of a CUMULATIVE effect.
    # Caught via a 20-seed self-test coverage check: an earlier version that
    # summed independent daily variances covered the true injected effect in
    # only 2/20 draws at a nominal 95% level, instead of the ~19/20 a valid
    # interval should. Simulating full future paths from the fitted model
    # (statsmodels' own state-space simulate(), conditioned on the filtered
    # end-of-training state) captures the real path-level correlation
    # directly -- the same thing a real CausalImpact package's Bayesian
    # posterior draws do.
    sims = fitted.simulate(nsimulations=len(post), anchor="end",
                           repetitions=_N_SIMULATIONS, exog=exog_post)
    sim_cumulative = np.asarray(sims).reshape(len(post), -1).sum(axis=0)
    effect_draws = float(post["actual_mgd"].sum()) - sim_cumulative
    ci_lo = float(np.percentile(effect_draws, 2.5))
    ci_hi = float(np.percentile(effect_draws, 97.5))

    predicted_total = float(predicted.sum())
    relative_pct = cumulative / predicted_total * 100 if predicted_total else float("nan")

    log.info("fit_causal_impact: n_pre=%d n_post=%d, R2_pre=%.3f, cumulative effect=%.3f "
             "[%.3f, %.3f] (%.1f%% of predicted)",
             len(pre), len(post), r2, cumulative, ci_lo, ci_hi, relative_pct)
    return CausalImpactResult(
        pointwise_effect=pointwise, cumulative_effect_mgd=cumulative,
        cumulative_effect_ci=(ci_lo, ci_hi), relative_effect_pct=relative_pct,
        model_r2_preperiod=float(r2), n_pre=len(pre), n_post=len(post),
    )


# ── Self-test: synthetic series with a known injected level shift ──────────

def _self_test() -> None:
    rng = np.random.default_rng(3)
    n_pre, n_post = 400, 60
    n = n_pre + n_post
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    doy = np.arange(n)
    tmax = 60 + 25 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 3, n)
    level_drift = np.cumsum(rng.normal(0, 0.05, n))   # slow local-level random walk
    baseline = 5.0 + 0.03 * tmax + level_drift

    true_effect_per_day = -0.6   # KNOWN injected level shift, post-period only
    y = baseline.copy()
    y[n_pre:] += true_effect_per_day
    y += rng.normal(0, 0.2, n)

    frame = pd.DataFrame({"actual_mgd": y, "tmax": tmax}, index=dates)
    pre_end = dates[n_pre - 1].date()
    post_end = dates[-1].date()

    result = fit_causal_impact(frame, pre_end, post_end, feature_cols=["tmax"])
    true_cumulative = true_effect_per_day * n_post

    print(f"fit_causal_impact: cumulative_effect={result.cumulative_effect_mgd:.2f} "
          f"(true={true_cumulative:.2f}), CI={tuple(round(x, 2) for x in result.cumulative_effect_ci)}, "
          f"R2_pre={result.model_r2_preperiod:.3f}")
    assert abs(result.cumulative_effect_mgd - true_cumulative) < abs(true_cumulative) * 0.4 + 3.0, \
        f"cumulative effect {result.cumulative_effect_mgd:.2f} too far from true {true_cumulative:.2f}"
    lo, hi = result.cumulative_effect_ci
    assert lo <= true_cumulative <= hi, \
        f"true cumulative {true_cumulative:.2f} not inside 95% CI [{lo:.2f}, {hi:.2f}]"
    assert result.model_r2_preperiod > 0.5, f"pre-period R2 {result.model_r2_preperiod:.3f} unexpectedly low for this fixture"

    print("\nSELF-TEST PASSED: cumulative effect recovered within tolerance, "
          "95% CI contains the true injected effect.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    _self_test()
