"""Tests for src/analysis/bayesian.py."""
import math

import numpy as np
import pytest

from src.analysis.bayesian import bayesian_ab_binary


class TestBayesianABBinary:
    def test_probabilities_in_unit_interval(self):
        res = bayesian_ab_binary(
            successes_control=100, n_control=1000,
            successes_treatment=150, n_treatment=1000,
            n_samples=20_000,
        )
        assert 0 <= res.p_treatment_better <= 1
        assert 0 <= res.p_above_rope <= 1
        assert 0 <= res.p_below_rope <= 1
        assert 0 <= res.p_in_rope <= 1

    def test_rope_probabilities_sum_to_one(self):
        res = bayesian_ab_binary(
            successes_control=100, n_control=1000,
            successes_treatment=150, n_treatment=1000,
            n_samples=20_000,
        )
        total = res.p_above_rope + res.p_in_rope + res.p_below_rope
        assert math.isclose(total, 1.0, abs_tol=0.01)

    def test_treatment_better_when_observed_higher(self):
        res = bayesian_ab_binary(
            successes_control=100, n_control=1000,
            successes_treatment=200, n_treatment=1000,
            n_samples=20_000,
        )
        assert res.p_treatment_better > 0.99

    def test_posterior_mean_close_to_observed(self):
        """With weak prior, posterior mean ≈ observed rate."""
        res = bayesian_ab_binary(
            successes_control=200, n_control=1000,
            successes_treatment=300, n_treatment=1000,
            prior_alpha=1.0, prior_beta=1.0,
            n_samples=50_000,
        )
        assert math.isclose(res.mean_control, 0.20, abs_tol=0.01)
        assert math.isclose(res.mean_treatment, 0.30, abs_tol=0.01)

    def test_credible_interval_brackets_mean(self):
        res = bayesian_ab_binary(
            successes_control=100, n_control=1000,
            successes_treatment=150, n_treatment=1000,
            n_samples=20_000,
        )
        assert res.credible_lower < res.mean_lift < res.credible_upper

    def test_larger_n_tighter_ci(self):
        small = bayesian_ab_binary(
            successes_control=50, n_control=500,
            successes_treatment=75, n_treatment=500,
            n_samples=20_000,
        )
        large = bayesian_ab_binary(
            successes_control=5000, n_control=50_000,
            successes_treatment=7500, n_treatment=50_000,
            n_samples=20_000,
        )
        small_width = small.credible_upper - small.credible_lower
        large_width = large.credible_upper - large.credible_lower
        assert large_width < small_width

    def test_expected_loss_small_when_treatment_clearly_wins(self):
        res = bayesian_ab_binary(
            successes_control=100, n_control=10_000,
            successes_treatment=500, n_treatment=10_000,
            n_samples=20_000,
        )
        assert res.expected_loss_ship < 0.001

    def test_reproducibility_with_seed(self):
        res1 = bayesian_ab_binary(
            successes_control=100, n_control=1000,
            successes_treatment=150, n_treatment=1000,
            n_samples=10_000, seed=42,
        )
        res2 = bayesian_ab_binary(
            successes_control=100, n_control=1000,
            successes_treatment=150, n_treatment=1000,
            n_samples=10_000, seed=42,
        )
        assert math.isclose(res1.mean_lift, res2.mean_lift, abs_tol=1e-12)

    def test_strong_prior_pulls_posterior(self):
        """With a strongly informative prior centered on 50%, posterior shifts toward 50%."""
        weak = bayesian_ab_binary(
            successes_control=20, n_control=100,    # observed 20%
            successes_treatment=25, n_treatment=100,  # observed 25%
            prior_alpha=1.0, prior_beta=1.0,
            n_samples=20_000,
        )
        strong = bayesian_ab_binary(
            successes_control=20, n_control=100,
            successes_treatment=25, n_treatment=100,
            prior_alpha=500.0, prior_beta=500.0,   # very strong, centered on 0.5
            n_samples=20_000,
        )
        # Strong prior pulls posterior mean closer to 0.5
        assert abs(strong.mean_control - 0.5) < abs(weak.mean_control - 0.5)
