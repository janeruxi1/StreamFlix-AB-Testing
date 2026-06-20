"""Frequentist statistical tests for A/B experiment analysis.

Covers the two metric types you'll encounter in 95% of product experiments:
  - Binary outcomes (conversion, click, day7_active)  -> two-proportion z-test
  - Continuous outcomes (watch hours, latency)        -> Welch's two-sample t-test

Each test returns effect size, standard error, 95% CI, z/t statistic, and
a two-sided p-value. CIs are always reported alongside p-values — point
estimates without uncertainty bounds aren't decision-ready.

Why no scipy dependency in the core:
We use math.erf for the normal CDF so this module is portable and
testable without heavy scientific stack imports.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


# ---------------------------------------------------------------------
# Normal CDF / quantile helpers
# ---------------------------------------------------------------------
def _phi(x: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _two_sided_p(z: float) -> float:
    """Two-sided p-value for a z-statistic."""
    return 2 * (1 - _phi(abs(z)))


def _z_critical(alpha: float = 0.05) -> float:
    """Two-sided critical z for confidence level 1-alpha."""
    target = 1 - alpha / 2
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _phi(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class TestResult:
    """Generic result of a two-sample comparison."""
    name: str
    p_control: float
    p_treatment: float
    n_control: int
    n_treatment: int
    effect_absolute: float      # treatment - control (absolute units)
    effect_relative: float      # (treatment - control) / control
    se: float                   # standard error of the difference
    ci_lower: float             # 95% CI lower bound on absolute effect
    ci_upper: float             # 95% CI upper bound on absolute effect
    statistic: float            # z or t
    p_value: float              # two-sided
    test_type: str              # 'two_proportion_z' or 'welch_t'

    @property
    def significant(self) -> bool:
        """Is the effect significant at the 5% level (uncorrected)?"""
        return self.p_value < 0.05

    def __str__(self) -> str:
        sig = "✅ sig" if self.significant else "—  ns"
        return (
            f"{self.name:<24}  "
            f"Δ = {self.effect_absolute:+.4f}  "
            f"({self.effect_relative:+.2%})  "
            f"95% CI [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  "
            f"{self.test_type[0]}={self.statistic:+.3f}  "
            f"p={self.p_value:.4f}  {sig}"
        )


# ---------------------------------------------------------------------
# Two-proportion z-test (for binary outcomes)
# ---------------------------------------------------------------------
def two_proportion_test(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    name: str = "metric",
    alpha: float = 0.05,
) -> TestResult:
    """Two-sided two-proportion z-test with 95% CI on the absolute difference.

    Test statistic uses POOLED variance (more conservative for hypothesis test):
        z = (p_t - p_c) / sqrt( p_hat (1-p_hat) (1/n_t + 1/n_c) )

    CI uses UNPOOLED variance (correct for estimating the difference):
        SE = sqrt( p_t(1-p_t)/n_t + p_c(1-p_c)/n_c )

    This split is the standard textbook recommendation and matches what
    most experimentation platforms (e.g. Statsig, Eppo) report.
    """
    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment

    # Pooled p for the test statistic
    p_hat = (successes_control + successes_treatment) / (n_control + n_treatment)
    se_pooled = math.sqrt(p_hat * (1 - p_hat) * (1 / n_control + 1 / n_treatment))
    z = (p_t - p_c) / se_pooled if se_pooled > 0 else 0.0
    p_value = _two_sided_p(z)

    # Unpooled SE for the CI
    se_unpooled = math.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    z_crit = _z_critical(alpha)
    diff = p_t - p_c
    ci_lo = diff - z_crit * se_unpooled
    ci_hi = diff + z_crit * se_unpooled

    return TestResult(
        name=name,
        p_control=p_c, p_treatment=p_t,
        n_control=n_control, n_treatment=n_treatment,
        effect_absolute=diff,
        effect_relative=diff / p_c if p_c > 0 else float("nan"),
        se=se_unpooled,
        ci_lower=ci_lo, ci_upper=ci_hi,
        statistic=z, p_value=p_value,
        test_type="two_proportion_z",
    )


# ---------------------------------------------------------------------
# Welch's t-test (for continuous outcomes; unequal variances)
# ---------------------------------------------------------------------
def welch_t_test(
    control_values,
    treatment_values,
    name: str = "metric",
    alpha: float = 0.05,
) -> TestResult:
    """Welch's two-sample t-test on the mean difference (treatment - control).

    Welch's (vs. Student's) does NOT assume equal variances — almost always
    the right choice in product experiments because treatment often shifts
    variance, not just mean.

    For n > ~5000 (which is always true in our experiment), t-distribution
    is indistinguishable from normal, so we use the normal CDF for p-value.
    """
    c = [float(x) for x in control_values]
    t = [float(x) for x in treatment_values]
    n_c, n_t = len(c), len(t)
    m_c = sum(c) / n_c
    m_t = sum(t) / n_t

    # Sample variance (ddof=1)
    v_c = sum((x - m_c) ** 2 for x in c) / (n_c - 1)
    v_t = sum((x - m_t) ** 2 for x in t) / (n_t - 1)

    se = math.sqrt(v_c / n_c + v_t / n_t)
    diff = m_t - m_c
    t_stat = diff / se if se > 0 else 0.0
    p_value = _two_sided_p(t_stat)  # normal approx for large n

    z_crit = _z_critical(alpha)
    ci_lo = diff - z_crit * se
    ci_hi = diff + z_crit * se

    return TestResult(
        name=name,
        p_control=m_c, p_treatment=m_t,
        n_control=n_c, n_treatment=n_t,
        effect_absolute=diff,
        effect_relative=diff / m_c if m_c != 0 else float("nan"),
        se=se,
        ci_lower=ci_lo, ci_upper=ci_hi,
        statistic=t_stat, p_value=p_value,
        test_type="welch_t",
    )


# ---------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------
def holm_bonferroni(p_values: Iterable[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down procedure.

    Stronger than vanilla Bonferroni (less conservative), still controls
    family-wise error rate. Standard choice when reporting multiple metrics.

    Returns
    -------
    List of booleans aligned with input p_values: True = reject null after
    correction.
    """
    p_list = list(p_values)
    m = len(p_list)
    # sort with original indices
    order = sorted(range(m), key=lambda i: p_list[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if p_list[i] < threshold:
            reject[i] = True
        else:
            break  # step-down: once we fail to reject, all higher p's fail too
    return reject


def bonferroni(p_values: Iterable[float], alpha: float = 0.05) -> list[bool]:
    """Vanilla Bonferroni correction. More conservative than Holm."""
    p_list = list(p_values)
    m = len(p_list)
    return [p < alpha / m for p in p_list]
