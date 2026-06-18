"""Tests for src/analysis/frequentist.py."""
import math

import pytest

from src.analysis.frequentist import (
    two_proportion_test,
    welch_t_test,
    holm_bonferroni,
    bonferroni,
)


class TestTwoProportionTest:
    def test_positive_effect_when_treatment_higher(self):
        res = two_proportion_test(
            successes_control=100, n_control=1000,
            successes_treatment=150, n_treatment=1000,
        )
        assert res.effect_absolute > 0
        assert res.p_value < 0.05
        assert res.significant

    def test_negative_effect_when_treatment_lower(self):
        res = two_proportion_test(
            successes_control=200, n_control=1000,
            successes_treatment=100, n_treatment=1000,
        )
        assert res.effect_absolute < 0

    def test_ci_brackets_effect(self):
        res = two_proportion_test(
            successes_control=100, n_control=1000,
            successes_treatment=150, n_treatment=1000,
        )
        assert res.ci_lower < res.effect_absolute < res.ci_upper

    def test_null_effect_not_significant(self):
        """Same rate in both arms → p > 0.05."""
        res = two_proportion_test(
            successes_control=200, n_control=1000,
            successes_treatment=200, n_treatment=1000,
        )
        assert not res.significant

    def test_relative_effect_calculation(self):
        res = two_proportion_test(
            successes_control=100, n_control=1000,    # 10%
            successes_treatment=200, n_treatment=1000,  # 20%
        )
        assert math.isclose(res.effect_relative, 1.0, rel_tol=0.01)  # +100%


class TestWelchTTest:
    def test_detects_mean_shift(self):
        ctl = [10.0, 11.0, 9.5, 10.5, 10.2] * 200
        trt = [12.0, 13.0, 11.5, 12.5, 12.2] * 200
        res = welch_t_test(ctl, trt)
        assert res.effect_absolute > 0
        assert res.p_value < 0.001

    def test_unequal_variances_handled(self):
        """Welch's should work even when variances differ wildly."""
        ctl = [10.0] * 1000
        trt = [11.0 + i * 0.01 for i in range(1000)]  # higher mean + higher variance
        res = welch_t_test(ctl, trt)
        assert res.effect_absolute > 0
        # Should not crash; should produce a finite p-value
        assert 0 <= res.p_value <= 1

    def test_ci_brackets_effect(self):
        ctl = [10.0, 11.0, 9.0] * 500
        trt = [12.0, 13.0, 11.0] * 500
        res = welch_t_test(ctl, trt)
        assert res.ci_lower < res.effect_absolute < res.ci_upper


class TestMultipleTestingCorrection:
    def test_holm_all_significant(self):
        p_vals = [0.001, 0.002, 0.003]
        decisions = holm_bonferroni(p_vals, alpha=0.05)
        assert all(decisions)

    def test_holm_none_significant(self):
        p_vals = [0.4, 0.5, 0.6]
        decisions = holm_bonferroni(p_vals, alpha=0.05)
        assert not any(decisions)

    def test_holm_step_down(self):
        """If smallest p fails the strictest threshold, all larger p's also fail."""
        # alpha/m = 0.05/3 ≈ 0.0167. p=0.02 fails the strictest threshold.
        p_vals = [0.02, 0.03, 0.04]
        decisions = holm_bonferroni(p_vals, alpha=0.05)
        assert not any(decisions)

    def test_holm_less_conservative_than_bonferroni(self):
        """At identical p-values where Holm could reject, Bonferroni should reject equal or fewer."""
        # All p-values right at the Bonferroni boundary
        p_vals = [0.01, 0.02, 0.03]
        n_holm = sum(holm_bonferroni(p_vals, alpha=0.05))
        n_bonf = sum(bonferroni(p_vals, alpha=0.05))
        assert n_holm >= n_bonf

    def test_bonferroni_threshold(self):
        """Vanilla Bonferroni: p < alpha/m."""
        # alpha = 0.05, m = 5, threshold = 0.01
        p_vals = [0.005, 0.009, 0.011, 0.02, 0.5]
        decisions = bonferroni(p_vals, alpha=0.05)
        # First two should pass (< 0.01), rest should fail
        assert decisions[0] and decisions[1]
        assert not decisions[2] and not decisions[3] and not decisions[4]
