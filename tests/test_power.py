"""Tests for src/analysis/power.py."""
import math

import pytest

from src.analysis.power import (
    sample_size_two_proportion,
    mde_for_sample_size,
    power_for_sample_size,
    sample_size_unequal_arms,
    runtime_days,
)


class TestSampleSize:
    def test_returns_positive_integer(self):
        res = sample_size_two_proportion(baseline=0.18, mde_absolute=0.01)
        assert res.n_per_arm > 0
        assert isinstance(res.n_per_arm, int)
        assert res.n_total == 2 * res.n_per_arm

    def test_known_textbook_value(self):
        """At baseline 18%, MDE 1pp, alpha 0.05, power 0.80, n ≈ 23,665/arm."""
        res = sample_size_two_proportion(0.18, 0.01, alpha=0.05, power=0.80)
        assert 23_000 <= res.n_per_arm <= 24_500

    def test_halving_mde_quadruples_n(self):
        """Sample size scales with 1/MDE^2."""
        a = sample_size_two_proportion(0.18, 0.01).n_per_arm
        b = sample_size_two_proportion(0.18, 0.005).n_per_arm
        ratio = b / a
        assert 3.5 < ratio < 4.5  # ~4x

    def test_higher_power_needs_more_sample(self):
        n_low = sample_size_two_proportion(0.18, 0.01, power=0.80).n_per_arm
        n_high = sample_size_two_proportion(0.18, 0.01, power=0.95).n_per_arm
        assert n_high > n_low

    def test_invalid_baseline_raises(self):
        with pytest.raises(ValueError):
            sample_size_two_proportion(baseline=1.5, mde_absolute=0.01)
        with pytest.raises(ValueError):
            sample_size_two_proportion(baseline=0.99, mde_absolute=0.02)  # p1 > 1


class TestMDEForSampleSize:
    def test_inverse_of_sample_size(self):
        """If we plug n back in, we recover the original MDE."""
        target_mde = 0.01
        res = sample_size_two_proportion(0.18, target_mde)
        recovered = mde_for_sample_size(0.18, res.n_per_arm)
        assert math.isclose(recovered, target_mde, rel_tol=0.05)

    def test_larger_n_smaller_mde(self):
        small = mde_for_sample_size(0.18, n_per_arm=5_000)
        large = mde_for_sample_size(0.18, n_per_arm=100_000)
        assert large < small


class TestPowerForSampleSize:
    def test_returns_probability(self):
        p = power_for_sample_size(0.18, 0.01, n_per_arm=25_000)
        assert 0 <= p <= 1

    def test_zero_effect_low_power(self):
        p = power_for_sample_size(0.18, mde_absolute=1e-6, n_per_arm=25_000)
        assert p < 0.10  # near alpha = 0.05

    def test_large_effect_near_one(self):
        p = power_for_sample_size(0.18, mde_absolute=0.10, n_per_arm=25_000)
        assert p > 0.99


class TestUnequalArms:
    def test_equal_split_matches_baseline(self):
        equal = sample_size_two_proportion(0.18, 0.01).n_total
        unequal_50 = sample_size_unequal_arms(0.18, 0.01, treatment_fraction=0.5)["n_total"]
        assert abs(equal - unequal_50) <= 2  # tolerate rounding

    def test_90_10_costs_more_than_50_50(self):
        equal = sample_size_two_proportion(0.18, 0.01).n_total
        unequal = sample_size_unequal_arms(0.18, 0.01, treatment_fraction=0.10)["n_total"]
        assert unequal > equal

    def test_efficiency_loss_positive_for_imbalanced(self):
        res = sample_size_unequal_arms(0.18, 0.01, treatment_fraction=0.10)
        assert res["efficiency_loss_pct"] > 100  # >2x sample needed

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            sample_size_unequal_arms(0.18, 0.01, treatment_fraction=0)
        with pytest.raises(ValueError):
            sample_size_unequal_arms(0.18, 0.01, treatment_fraction=1)


class TestRuntimeDays:
    def test_simple_division(self):
        assert runtime_days(70_000, 1_000) == 70.0
        assert runtime_days(47_330, 3_500) == pytest.approx(13.52, rel=0.01)
