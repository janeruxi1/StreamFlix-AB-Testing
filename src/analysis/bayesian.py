"""Bayesian A/B test analysis for binary outcomes.

Uses the Beta-Binomial conjugate model — the textbook (and industry) choice
for conversion experiments. We sample from the posterior with numpy.random.beta
rather than running MCMC because conjugacy gives us closed-form posteriors.

Why conjugate Beta-Binomial?
  - Closed-form posterior → no MCMC, instant inference
  - Beta(1,1) = Uniform(0,1) gives the 'no prior knowledge' baseline
  - Beta(a,b) can encode prior business knowledge via (a-1)/(a+b-2) ≈ prior mean
  - Adopted by virtually every commercial experimentation platform
    (Statsig, Eppo, Optimizely, GrowthBook)

Why MC sampling rather than analytical Beta-Beta difference?
  - Difference of two Betas has no clean closed form
  - 100k Monte Carlo samples gives sub-0.001 precision in seconds
  - Same approach used in production at most large tech companies
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BayesianResult:
    """Result of a two-arm Bayesian comparison on a binary metric."""
    # Inputs
    n_control: int
    successes_control: int
    n_treatment: int
    successes_treatment: int
    prior_alpha: float
    prior_beta: float
    n_samples: int

    # Posterior summaries (absolute, in conversion-rate units)
    mean_control: float
    mean_treatment: float
    mean_lift: float
    credible_lower: float
    credible_upper: float

    # Decision metrics
    p_treatment_better: float
    expected_loss_ship: float       # E[(control - treatment)_+]  (units = conv rate)
    expected_loss_not_ship: float   # E[(treatment - control)_+]
    p_above_rope: float             # P(lift > ROPE_upper)
    p_below_rope: float             # P(lift < ROPE_lower)
    p_in_rope: float

    # Stored samples for visualization
    samples_control: np.ndarray
    samples_treatment: np.ndarray
    samples_lift: np.ndarray

    def __str__(self) -> str:
        return (
            f"Bayesian posterior summary\n"
            f"  Posterior mean control:    {self.mean_control:.4f}\n"
            f"  Posterior mean treatment:  {self.mean_treatment:.4f}\n"
            f"  Posterior mean lift:       {self.mean_lift:+.4f}\n"
            f"  95% credible interval:     [{self.credible_lower:+.4f}, {self.credible_upper:+.4f}]\n"
            f"  P(treatment > control):    {self.p_treatment_better:.4%}\n"
            f"  Expected loss of shipping: {self.expected_loss_ship:.6f}\n"
            f"  Expected gain of shipping: {self.expected_loss_not_ship:.6f}"
        )


def bayesian_ab_binary(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    rope: tuple[float, float] = (-0.005, 0.005),
    n_samples: int = 100_000,
    seed: int = 42,
) -> BayesianResult:
    """Posterior inference for two binary proportions.

    Parameters
    ----------
    successes_*, n_* : observed counts
    prior_alpha, prior_beta : Beta prior parameters (default 1,1 = uniform)
    rope : Region Of Practical Equivalence in absolute lift units (default ±0.5pp)
    n_samples : Monte Carlo posterior samples (default 100k = sub-0.001 precision)

    Returns
    -------
    BayesianResult with posterior summaries and decision metrics.
    """
    rng = np.random.default_rng(seed)

    # Posterior = Beta(prior_a + successes, prior_b + failures)
    post_c = rng.beta(
        prior_alpha + successes_control,
        prior_beta + (n_control - successes_control),
        size=n_samples,
    )
    post_t = rng.beta(
        prior_alpha + successes_treatment,
        prior_beta + (n_treatment - successes_treatment),
        size=n_samples,
    )
    lift = post_t - post_c

    # ROPE = "Region Of Practical Equivalence": a band near zero where
    # we'd treat the effect as 'no meaningful difference.'
    rope_lo, rope_hi = rope

    return BayesianResult(
        n_control=n_control, successes_control=successes_control,
        n_treatment=n_treatment, successes_treatment=successes_treatment,
        prior_alpha=prior_alpha, prior_beta=prior_beta,
        n_samples=n_samples,
        mean_control=float(post_c.mean()),
        mean_treatment=float(post_t.mean()),
        mean_lift=float(lift.mean()),
        credible_lower=float(np.quantile(lift, 0.025)),
        credible_upper=float(np.quantile(lift, 0.975)),
        p_treatment_better=float((lift > 0).mean()),
        # Expected loss of shipping = expected amount we'd regret if we
        # ship and control was actually better. Standard Bayesian decision
        # rule: ship when expected_loss_ship < small threshold.
        expected_loss_ship=float(np.maximum(0, -lift).mean()),
        expected_loss_not_ship=float(np.maximum(0, lift).mean()),
        p_above_rope=float((lift > rope_hi).mean()),
        p_below_rope=float((lift < rope_lo).mean()),
        p_in_rope=float(((lift >= rope_lo) & (lift <= rope_hi)).mean()),
        samples_control=post_c,
        samples_treatment=post_t,
        samples_lift=lift,
    )
