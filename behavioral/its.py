"""Interrupted time series (segmented regression) on weather-residualized demand.

Per docs/module3-behavioral-plan.md and the user's explicit directive
(2026-07-15): the 2026 drought declaration and rate step are ONE combined
intervention — a package with a two-step phase-in (restrictions 2026-04-01,
surcharge rates 2026-06-01). Steps are estimated jointly but reported as one
combined package effect; step coefficients are internal decomposition, not
separate interventions.

Specification: residual_t = b0 + b1·t + Σ_k step_k·1[t ≥ d_k] + e_t, with
Newey-West (HAC) standard errors for autocorrelation. Steps whose date falls
beyond the data simply drop out (e.g. the June rate step is inactive until a
fresh SCADA import extends production past 2026-06-01).

Falsification: placebo() re-fits the same spec on PRE-intervention data only,
with a fake interruption date — a sound design shows ~zero placebo effects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import statsmodels.api as sm

log = logging.getLogger(__name__)

_HAC_MAXLAGS = 14  # two weeks of autocorrelation allowance on daily data


@dataclass
class StepEstimate:
    name: str
    step_date: date
    coef_mgd: float
    se_mgd: float
    p_value: float
    n_post: int


@dataclass
class ITSResult:
    steps: list[StepEstimate]
    combined_mgd: float                 # sum of active step coefficients
    combined_pct_of_expected: float     # vs mean weather-expected demand post-package
    n_obs: int
    inactive_steps: list[str] = field(default_factory=list)


def fit(frame: pd.DataFrame, steps: list[tuple[str, date]]) -> ITSResult:
    """frame: weather_norm output (index=date, residual_mgd, expected_mgd)."""
    df = frame.dropna(subset=["residual_mgd"]).sort_index()
    last_data = df.index.max()

    active = [(n, d) for n, d in steps if (df.index >= d).sum() >= 14]
    inactive = [n for n, d in steps if (n, d) not in active]
    if not active:
        raise RuntimeError("No intervention step has >=14 post days of data yet.")

    X = pd.DataFrame(index=df.index)
    X["t"] = [(d - df.index.min()).days for d in df.index]
    for name, d in active:
        X[f"step_{d.isoformat()}"] = (df.index >= d).astype(int)
    X = sm.add_constant(X, has_constant="add")

    ols = sm.OLS(df["residual_mgd"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": _HAC_MAXLAGS}
    )

    estimates = []
    for name, d in active:
        col = f"step_{d.isoformat()}"
        estimates.append(StepEstimate(
            name=name, step_date=d,
            coef_mgd=float(ols.params[col]),
            se_mgd=float(ols.bse[col]),
            p_value=float(ols.pvalues[col]),
            n_post=int((df.index >= d).sum()),
        ))

    first_step = min(d for _, d in active)
    post = df[df.index >= first_step]
    combined = sum(e.coef_mgd for e in estimates)
    combined_pct = combined / float(post["expected_mgd"].mean()) * 100 if len(post) else float("nan")

    log.info("ITS fit: %d obs through %s; combined package effect %.3f MGD (%.1f%% of expected)",
             len(df), last_data, combined, combined_pct)
    return ITSResult(
        steps=estimates, combined_mgd=combined,
        combined_pct_of_expected=combined_pct, n_obs=len(df),
        inactive_steps=inactive,
    )


def placebo(frame: pd.DataFrame, fake_date: date, true_start: date) -> StepEstimate:
    """Same spec, pre-intervention data only, fake interruption at fake_date."""
    df = frame.dropna(subset=["residual_mgd"]).sort_index()
    df = df[df.index < true_start]
    X = pd.DataFrame(index=df.index)
    X["t"] = [(d - df.index.min()).days for d in df.index]
    X["step"] = (df.index >= fake_date).astype(int)
    X = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(df["residual_mgd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": _HAC_MAXLAGS})
    return StepEstimate(
        name=f"placebo {fake_date.isoformat()}", step_date=fake_date,
        coef_mgd=float(ols.params["step"]), se_mgd=float(ols.bse["step"]),
        p_value=float(ols.pvalues["step"]), n_post=int((df.index >= fake_date).sum()),
    )


@dataclass
class EmpiricalNullResult:
    observed_mgd: float          # single-step effect at the real intervention date
    null_effects: list[float]    # same statistic at every feasible fake date
    n_draws: int
    window_days: int             # post-window length used for observed AND every draw
    exceedance_p: float          # (1 + #{|null| >= |obs|}) / (1 + n) — permutation-style p
    percentile: float            # share of |null| draws strictly below |observed|, in %


def _windowed_step(df: pd.DataFrame, step_date: date, window_days: int) -> float | None:
    """Single-step effect (const + trend + step) using only data through
    step_date + window_days — the same estimation a person running this
    analysis `window_days` after a (real or fake) intervention would do."""
    end = step_date + timedelta(days=window_days)
    sub = df[df.index <= end]
    n_post = int((sub.index >= step_date).sum())
    n_pre = len(sub) - n_post
    if n_post < window_days * 0.6 or n_pre < 365:
        return None   # not enough usable data on one side — skip this draw
    X = pd.DataFrame(index=sub.index)
    X["t"] = [(d - sub.index.min()).days for d in sub.index]
    X["step"] = (sub.index >= step_date).astype(int)
    X = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(sub["residual_mgd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": _HAC_MAXLAGS})
    return float(ols.params["step"])


def empirical_null(frame: pd.DataFrame, true_start: date) -> EmpiricalNullResult:
    """Randomization (placebo-in-time) inference: how big an 'effect' does this
    exact machinery find at dates where nothing happened?

    Rationale: a handful of hand-picked pass/fail placebos punishes honest
    weather-model imperfection
    without quantifying it — after the weather-station fix, 2/8 placebos stayed
    marginally significant (wet-2023 drift ~0.5 MGD), leaving 'how much should
    we trust -1.2 MGD?' unanswerable. This answers it: estimate the SAME
    windowed single-step statistic at every month-start in the pre-period, and
    report where the real effect falls in that distribution. The weather model
    doesn't need to be perfect; the real effect needs to beat the worst the
    imperfections produce.

    The statistic is a single step at the FIRST package date (the package
    phase-in is collinear over a short window; the combined effect is what the
    headline reports and what this validates), estimated with the same
    post-window length the real analysis currently has.
    """
    df = frame.dropna(subset=["residual_mgd"]).sort_index()
    window_days = int((df.index.max() - true_start).days)

    observed = _windowed_step(df, true_start, window_days)
    if observed is None:
        raise RuntimeError("empirical_null: not enough post-intervention data for the observed statistic")

    pre = df[df.index < true_start]
    # Every month-start where a full fake post-window still ends before the
    # real intervention (so no true-effect contamination in any draw).
    candidates = sorted({date(d.year, d.month, 1) for d in pre.index})
    nulls = []
    for fake in candidates:
        if fake + timedelta(days=window_days) >= true_start:
            continue
        eff = _windowed_step(pre, fake, window_days)
        if eff is not None:
            nulls.append(eff)

    if len(nulls) < 10:
        raise RuntimeError(f"empirical_null: only {len(nulls)} feasible null draws — pre-period too short")

    n_exceed = sum(1 for e in nulls if abs(e) >= abs(observed))
    exceedance_p = (1 + n_exceed) / (1 + len(nulls))
    percentile = 100.0 * sum(1 for e in nulls if abs(e) < abs(observed)) / len(nulls)
    log.info("empirical null: observed %.3f MGD vs %d draws, exceedance p=%.3f",
             observed, len(nulls), exceedance_p)
    return EmpiricalNullResult(
        observed_mgd=observed, null_effects=nulls, n_draws=len(nulls),
        window_days=window_days, exceedance_p=exceedance_p, percentile=percentile,
    )
