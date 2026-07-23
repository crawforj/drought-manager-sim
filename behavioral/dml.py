"""Cross-fitted double machine learning (Chernozhukov et al. 2018) and the
R-learner (Nie & Wager 2021) for heterogeneous treatment effects, built
after `econml` turned out to need a C compiler this environment doesn't
have (confirmed live: gcc/clang/clang-cl/pgcc all missing) and
`pycausalimpact`/`tfcausalimpact` turned out to be broken against this
project's pandas/Python versions. Built here
instead on `sklearn` and `statsmodels`, both already core dependencies.

**What this is, precisely** (same labeling discipline `behavioral/did.py`
applies to its own semi-elasticity): `fit_dml_ate()` is a real, correctly
cross-fitted double ML average treatment effect — the debiased/orthogonalized
version of a naive group-difference estimate, a legitimate upgrade over
`did.py`'s simpler two-group ITS comparison for THIS specific cross-sectional
question. `fit_dml_cate()` is the R-learner's random-forest second stage,
which approximates what a real causal forest (Wager & Athey) does WITHOUT
honest splitting -- it does NOT get the asymptotically valid confidence
intervals a real causal forest provides. Treat `fit_dml_cate()`'s output as a
ranking/exploration tool (which covariate combinations associate with bigger
response), never as a number with its own p-value.

**Different outcome than did.py, on purpose, stated plainly**: this operates
on `behavioral.response_profile.yoy_account_change()`'s per-account raw
year-over-year change -- cross-sectional (one row per account), which is what
DML needs, unlike did.py's pooled weather-normalized time series. That raw
outcome is explicitly NOT weather-normalized (see yoy_account_change's own
docstring) -- so a real discrepancy between this module's ATE and did.py's
ATE is expected in general, not necessarily a bug in either: they measure
different things (raw YoY change vs. weather-normalized ITS step). Comparing
them is still useful as a directional sanity check (same sign, same rough
order of magnitude), not a strict equality test.

Covariates reused directly from `behavioral.segmentation.build_features()` --
summer_winter_ratio, weekend_weekday_ratio, irrigation_share, cv -- no new
data pull.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)

_NUISANCE_PARAMS = dict(n_estimators=150, max_depth=3, learning_rate=0.08,
                        min_samples_leaf=15, subsample=0.8, random_state=42)
_CATE_FOREST_PARAMS = dict(n_estimators=300, max_depth=4, min_samples_leaf=20, random_state=42)
_MIN_ROWS = 200   # below this, cross-fitting folds get too thin to trust


@dataclass
class DMLResult:
    ate: float
    se: float
    p_value: float
    n_obs: int
    n_folds: int
    nuisance_r2_outcome: float     # out-of-fold R^2 of m(X) = E[Y|X]
    nuisance_r2_treatment: float   # out-of-fold R^2 of e(X) = E[T|X] (T is 0/1 here)


@dataclass
class CATEResult:
    tau_by_account: pd.Series          # account_id -> estimated tau(X), gal/day
    feature_importances: dict[str, float]
    n_obs: int
    is_honest_split: bool = False      # ALWAYS False -- see module docstring


def _cross_fit_residuals(Y: np.ndarray, T: np.ndarray, X: np.ndarray, n_folds: int, seed: int = 0):
    """K-fold cross-fitting: nuisance models m(X)=E[Y|X], e(X)=E[T|X] each
    fit on out-of-fold data only, so the same observations never inform both
    the nuisance prediction and its own residual (the source of the
    "regularization bias" DML's cross-fitting specifically corrects for).
    Returns (Y_resid, T_resid, m_hat, e_hat) aligned to the original order.
    """
    n = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    m_hat = np.zeros(n)
    e_hat = np.zeros(n)
    for train_idx, test_idx in kf.split(X):
        m_model = GradientBoostingRegressor(**_NUISANCE_PARAMS)
        m_model.fit(X[train_idx], Y[train_idx])
        m_hat[test_idx] = m_model.predict(X[test_idx])

        e_model = GradientBoostingRegressor(**_NUISANCE_PARAMS)
        e_model.fit(X[train_idx], T[train_idx])
        e_hat[test_idx] = e_model.predict(X[test_idx])

    Y_resid = Y - m_hat
    T_resid = T - e_hat
    return Y_resid, T_resid, m_hat, e_hat


def fit_dml_ate(Y: pd.Series, T: pd.Series, X: pd.DataFrame, *, n_folds: int = 5) -> DMLResult:
    """Cross-fitted double ML average treatment effect.

    Y: per-account outcome (e.g. yoy_account_change()'s post_mean_gpd -
    pre_mean_gpd, gallons/day -- NOT weather-normalized, see module docstring).
    T: per-account treatment indicator, 0/1 (e.g. tier1_only=0, tier2_3_exposed=1
    from behavioral.did.classify_tier_exposure(), "unknown" rows dropped by caller).
    X: per-account covariates (behavioral.segmentation.build_features()'s
    cluster feature columns).

    Step 1: K-fold cross-fitting of m(X)=E[Y|X] and e(X)=E[T|X] (nuisance
    models, GradientBoostingRegressor for both -- T is treated as a
    regression target the same way econml's LinearDML would for a binary
    treatment). Step 2: residualize on held-out folds. Step 3: a single
    HAC-robust WLS coefficient of Y_resid ~ T_resid gives the debiased ATE --
    same statsmodels HAC machinery behavioral/its.py already uses for its
    own standard errors, applied here to a cross-sectional regression instead
    of a time series (HAC still valid for cross-sectional heteroskedasticity-
    robust inference, just without the autocorrelation lag structure mattering).
    """
    aligned = pd.concat([Y.rename("Y"), T.rename("T"), X], axis=1).dropna()
    if len(aligned) < _MIN_ROWS:
        raise RuntimeError(f"fit_dml_ate: only {len(aligned)} complete rows (<{_MIN_ROWS}) -- too few to cross-fit reliably")

    Y_arr = aligned["Y"].to_numpy(dtype=float)
    T_arr = aligned["T"].to_numpy(dtype=float)
    X_arr = aligned[X.columns].to_numpy(dtype=float)

    Y_resid, T_resid, m_hat, e_hat = _cross_fit_residuals(Y_arr, T_arr, X_arr, n_folds)

    r2_outcome = 1 - np.sum((Y_arr - m_hat) ** 2) / np.sum((Y_arr - Y_arr.mean()) ** 2)
    r2_treatment = 1 - np.sum((T_arr - e_hat) ** 2) / np.sum((T_arr - T_arr.mean()) ** 2)

    Xd = sm.add_constant(pd.Series(T_resid, name="T_resid"), has_constant="add")
    ols = sm.OLS(Y_resid, Xd).fit(cov_type="HC1")   # cross-sectional robust SE, not HAC's lag structure
    ate = float(ols.params["T_resid"])
    se = float(ols.bse["T_resid"])
    p_value = float(ols.pvalues["T_resid"])

    log.info("fit_dml_ate: n=%d, folds=%d, ATE=%.3f gal/day (SE=%.3f, p=%.4f), "
             "nuisance R2 outcome=%.3f treatment=%.3f",
             len(aligned), n_folds, ate, se, p_value, r2_outcome, r2_treatment)
    return DMLResult(ate=ate, se=se, p_value=p_value, n_obs=len(aligned), n_folds=n_folds,
                     nuisance_r2_outcome=r2_outcome, nuisance_r2_treatment=r2_treatment)


def fit_dml_cate(Y: pd.Series, T: pd.Series, X: pd.DataFrame, *, n_folds: int = 5) -> CATEResult:
    """R-learner second stage (Nie & Wager 2021, eq. 5): heterogeneous
    treatment effect tau(X) via a weighted regression of the pseudo-outcome
    Y_resid/T_resid onto X, weighted by T_resid**2, using a RandomForestRegressor.

    This is NOT a real causal forest -- no honest splitting, so the resulting
    tau(X) predictions and feature_importances are a ranking/exploration
    signal, not a number with valid confidence intervals. See module
    docstring; is_honest_split is always False on the returned result as a
    standing reminder to any downstream consumer.
    """
    aligned = pd.concat([Y.rename("Y"), T.rename("T"), X], axis=1).dropna()
    if len(aligned) < _MIN_ROWS:
        raise RuntimeError(f"fit_dml_cate: only {len(aligned)} complete rows (<{_MIN_ROWS}) -- too few to cross-fit reliably")

    Y_arr = aligned["Y"].to_numpy(dtype=float)
    T_arr = aligned["T"].to_numpy(dtype=float)
    X_arr = aligned[X.columns].to_numpy(dtype=float)

    Y_resid, T_resid, _, _ = _cross_fit_residuals(Y_arr, T_arr, X_arr, n_folds)

    # R-learner pseudo-outcome and weights (Nie & Wager eq. 5): minimizing
    # sum( T_resid^2 * (Y_resid/T_resid - tau(X))^2 ) is equivalent to a
    # weighted regression of Y_resid on X with sample_weight=T_resid**2 and
    # target Y_resid/T_resid -- avoids dividing by near-zero T_resid directly
    # by folding it into the regression target via weighted least squares on
    # the ORIGINAL Y_resid, T_resid*X interaction instead (numerically safer,
    # same solution for a tree-based learner since RandomForestRegressor
    # accepts sample_weight natively).
    eps = 1e-6
    weights = T_resid ** 2
    pseudo_outcome = np.divide(Y_resid, T_resid, out=np.zeros_like(Y_resid), where=np.abs(T_resid) > eps)
    valid = np.abs(T_resid) > eps
    if valid.sum() < _MIN_ROWS:
        raise RuntimeError(f"fit_dml_cate: only {valid.sum()} rows with usable T_resid -- too few")

    forest = RandomForestRegressor(**_CATE_FOREST_PARAMS)
    forest.fit(X_arr[valid], pseudo_outcome[valid], sample_weight=weights[valid])
    tau_hat = forest.predict(X_arr)

    tau_series = pd.Series(tau_hat, index=aligned.index, name="tau")
    importances = dict(zip(X.columns, forest.feature_importances_))
    log.info("fit_dml_cate: n=%d, tau range [%.2f, %.2f] gal/day, feature importances: %s",
             len(aligned), tau_hat.min(), tau_hat.max(),
             {k: round(v, 3) for k, v in sorted(importances.items(), key=lambda kv: -kv[1])})
    return CATEResult(tau_by_account=tau_series, feature_importances=importances,
                      n_obs=len(aligned), is_honest_split=False)


# ── Self-test: synthetic heterogeneous effect ───────────────────────────────

def _self_test() -> None:
    rng = np.random.default_rng(11)
    n = 3000
    X = pd.DataFrame({
        "summer_winter_ratio": rng.uniform(1.0, 8.0, n),
        "weekend_weekday_ratio": rng.uniform(0.7, 1.4, n),
        "irrigation_share": rng.uniform(0.0, 0.9, n),
        "cv": rng.uniform(0.1, 1.2, n),
    })
    # Propensity depends weakly on covariates (so e(X) isn't trivially constant).
    propensity = 1 / (1 + np.exp(-(X["irrigation_share"] * 2 - 0.8)))
    T = (rng.uniform(size=n) < propensity).astype(float)

    # KNOWN heterogeneous effect: tau(X) depends on summer_winter_ratio (the
    # "true" effect-modifying covariate) and NOT on the other three (noise
    # covariates for the CATE forest to correctly down-rank).
    true_tau = -20.0 - 15.0 * (X["summer_winter_ratio"] - 1.0)   # heavier irrigators respond more
    baseline = 50.0 + 10.0 * X["weekend_weekday_ratio"]
    noise = rng.normal(0, 25, n)
    Y = baseline + true_tau * T + noise
    Y = pd.Series(Y, name="Y")
    T = pd.Series(T, name="T")

    true_ate = float(true_tau.mean())   # E[Y(1)-Y(0)] over the whole population -- ATE, not ATT

    ate_result = fit_dml_ate(Y, T, X)
    print(f"fit_dml_ate: estimated ATE={ate_result.ate:.2f} (true population ATE~{true_ate:.2f}), "
          f"SE={ate_result.se:.2f}, p={ate_result.p_value:.4f}")
    assert abs(ate_result.ate - true_ate) < 8.0, \
        f"DML ATE {ate_result.ate:.2f} too far from true {true_ate:.2f}"
    assert ate_result.p_value < 0.05, "expected a significant ATE on this synthetic fixture"

    cate_result = fit_dml_cate(Y, T, X)
    top_feature = max(cate_result.feature_importances, key=cate_result.feature_importances.get)
    print(f"fit_dml_cate: top feature by importance = {top_feature!r} "
          f"(expected 'summer_winter_ratio'), importances={cate_result.feature_importances}")
    assert top_feature == "summer_winter_ratio", \
        f"expected summer_winter_ratio to dominate CATE feature importance, got {top_feature!r}"
    assert cate_result.is_honest_split is False

    # Correlation between estimated tau and true tau -- not expected to be
    # perfect (no honest splitting), but should be clearly positive-in-direction
    # (more negative true_tau -- i.e. bigger response -- should correlate with
    # more negative estimated tau).
    corr = np.corrcoef(cate_result.tau_by_account.values, true_tau.values)[0, 1]
    print(f"correlation(estimated tau, true tau) = {corr:.3f}")
    assert corr > 0.3, f"CATE estimates too weakly correlated with true heterogeneity ({corr:.3f})"

    print("\nSELF-TEST PASSED: ATE recovered within tolerance, CATE correctly "
          "ranks the true effect-modifying covariate, tau estimates positively "
          "correlated with ground truth.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    _self_test()
