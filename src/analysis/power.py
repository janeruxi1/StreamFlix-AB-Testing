"""Power analysis & sample size calculators for two-proportion experiments.

Used both PRE-experiment (design phase: 'how big should we run this?') and
POST-experiment ('given our sample size, what's the smallest effect we could
have detected?'). Both questions matter in interviews.

References
----------
- Two-proportion z-test power: standard textbook formula
- Industry convention: alpha=0.05 (two-sided), power=0.80
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


# ---------------------------------------------------------------------
# Z-score helpers (use math.erf so we don't depend on scipy for the core)
# ---------------------------------------------------------------------
def _phi(x: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _z_alpha_two_sided(alpha: float) -> float:
    """Critical z for a two-sided test at level alpha."""
    # Inverse via bisection (no scipy dependency)
    target = 1 - alpha / 2
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _phi(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _z_beta(power: float) -> float:
    """Critical z for power = 1 - beta."""
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _phi(mid) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------
# Core calculators
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class SampleSizeResult:
    n_per_arm: int
    n_total: int
    baseline: float
    mde_absolute: float
    mde_relative: float
    alpha: float
    power: float

    def __str__(self) -> str:
        return (
            f"Required sample size\n"
            f"  Baseline (p0):     {self.baseline:.4f}\n"
            f"  MDE (absolute):    {self.mde_absolute:+.4f} ({self.mde_relative:+.2%} relative)\n"
            f"  alpha:             {self.alpha}\n"
            f"  power (1-beta):    {self.power}\n"
            f"  --> n per arm:     {self.n_per_arm:,}\n"
            f"  --> n total:       {self.n_total:,}"
        )


def sample_size_two_proportion(
    baseline: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> SampleSizeResult:
    """Per-arm sample size needed to detect an absolute lift `mde_absolute`
    over a baseline conversion rate `baseline`, at given alpha and power.

    Assumes a two-sided two-proportion z-test, equal arms.
    """
    p0 = baseline
    p1 = baseline + mde_absolute
    if not (0 < p0 < 1) or not (0 < p1 < 1):
        raise ValueError("baseline + mde must stay in (0, 1)")

    z_a = _z_alpha_two_sided(alpha)
    z_b = _z_beta(power)

    numerator = (z_a + z_b) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1))
    n_per_arm = math.ceil(numerator / (p1 - p0) ** 2)

    return SampleSizeResult(
        n_per_arm=n_per_arm,
        n_total=2 * n_per_arm,
        baseline=p0,
        mde_absolute=mde_absolute,
        mde_relative=mde_absolute / p0,
        alpha=alpha,
        power=power,
    )


def mde_for_sample_size(
    baseline: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Inverse of sample_size: given n per arm, what's the smallest
    ABSOLUTE lift we could detect at alpha & power?

    Useful for POST-HOC analysis: 'we had 50k per arm — what's our MDE?'
    """
    p0 = baseline
    z_a = _z_alpha_two_sided(alpha)
    z_b = _z_beta(power)

    # Solve numerically for mde: (z_a + z_b)^2 * (p0(1-p0) + p1(1-p1)) = n * mde^2
    lo, hi = 1e-6, 1 - p0 - 1e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p1 = p0 + mid
        rhs = (z_a + z_b) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / mid ** 2
        if rhs > n_per_arm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def power_for_sample_size(
    baseline: float,
    mde_absolute: float,
    n_per_arm: int,
    alpha: float = 0.05,
) -> float:
    """Given a sample size and a true effect, what's our power to detect it?

    Useful sanity check: 'if the true lift is X, can we even see it?'
    """
    p0 = baseline
    p1 = baseline + mde_absolute
    z_a = _z_alpha_two_sided(alpha)
    se = math.sqrt(p0 * (1 - p0) / n_per_arm + p1 * (1 - p1) / n_per_arm)
    z_diff = abs(p1 - p0) / se
    return _phi(z_diff - z_a)  # one-sided power approximation


def runtime_days(n_total: int, daily_traffic: int) -> float:
    """How many days do we need to run to reach n_total users?"""
    return n_total / daily_traffic



def sample_size_unequal_arms(
    baseline: float,
    mde_absolute: float,
    treatment_fraction: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Total sample size needed under an unequal allocation.
 
    With treatment fraction r (so n_treatment = r·n_total,
    n_control = (1-r)·n_total), the variance of the difference becomes:
 
        Var = p0(1-p0)/n_control + p1(1-p1)/n_treatment
            = [p0(1-p0)/(1-r) + p1(1-p1)/r] / n_total
 
    Solving for n_total at given alpha & power:
 
        n_total = (z_a + z_b)^2 · [p0(1-p0)/(1-r) + p1(1-p1)/r] / (p1-p0)^2
 
    Parameters
    ----------
    baseline : control conversion rate (0,1)
    mde_absolute : absolute lift to detect
    treatment_fraction : share of users assigned to treatment (e.g. 0.10)
    alpha, power : test parameters
 
    Returns
    -------
    dict with n_total, n_control, n_treatment, allocation, and efficiency
    (the ratio of unequal-split total n to the 50/50 baseline total n).
    """
    if not (0 < treatment_fraction < 1):
        raise ValueError("treatment_fraction must be in (0, 1)")
    p0 = baseline
    p1 = baseline + mde_absolute
    r = treatment_fraction
 
    z_a = _z_alpha_two_sided(alpha)
    z_b = _z_beta(power)
 
    num = (z_a + z_b) ** 2 * (p0 * (1 - p0) / (1 - r) + p1 * (1 - p1) / r)
    n_total = math.ceil(num / (p1 - p0) ** 2)
    n_treat = math.ceil(r * n_total)
    n_ctrl = n_total - n_treat
 
    equal = sample_size_two_proportion(baseline, mde_absolute, alpha, power)
    return {
        "n_total": n_total,
        "n_control": n_ctrl,
        "n_treatment": n_treat,
        "allocation": f"{int((1-r)*100)}/{int(r*100)}",
        "equal_split_total": equal.n_total,
        "efficiency_loss_pct": (n_total / equal.n_total - 1) * 100,
    }
