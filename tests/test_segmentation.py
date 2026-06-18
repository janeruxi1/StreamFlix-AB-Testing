"""Tests for src/analysis/segmentation.py."""
import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.segmentation import (
    segment_lifts,
    cuped_adjust,
    cuped_t_test,
)


class TestSegmentLifts:
    def test_one_row_per_segment(self, small_experiment):
        out = segment_lifts(
            small_experiment, segment_col="segment", metric_col="converted",
        )
        assert set(out["segment"]) == set(small_experiment["segment"].unique())

    def test_skips_small_segments(self, small_experiment):
        df = small_experiment.copy()
        # Add a tiny segment that should be skipped
        df.loc[df.index[:5], "segment"] = "tiny"
        out = segment_lifts(df, segment_col="segment", metric_col="converted",
                            min_segment_n=100)
        assert "tiny" not in out["segment"].values

    def test_binary_metric_auto_detect(self, small_experiment):
        out = segment_lifts(
            small_experiment, segment_col="segment", metric_col="converted",
        )
        # Should not raise; should produce rows
        assert len(out) > 0
        assert "p_value" in out.columns

    def test_continuous_metric(self, small_experiment):
        out = segment_lifts(
            small_experiment, segment_col="segment",
            metric_col="metric_continuous", binary=False,
        )
        assert len(out) > 0
        for _, r in out.iterrows():
            assert r["ci_lower"] < r["mean_treatment"] - r["mean_control"] + 1e-6  # CI brackets effect


class TestCUPEDAdjust:
    def test_constant_covariate_yields_zero_theta(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([7.0, 7.0, 7.0, 7.0, 7.0])  # zero variance
        y_adj, theta, rho = cuped_adjust(y, x)
        assert theta == 0.0
        np.testing.assert_array_equal(y_adj, y)

    def test_perfect_correlation_centers_y(self):
        """If y = x + c, theta = 1 and y_adj should equal mean(y)."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000)
        y = x + 5.0
        y_adj, theta, rho = cuped_adjust(y, x)
        assert math.isclose(theta, 1.0, abs_tol=1e-9)
        assert math.isclose(rho, 1.0, abs_tol=1e-9)
        # After adjustment, y_adj should be constant (all equal to mean(y))
        assert y_adj.std() < 1e-9

    def test_no_correlation_preserves_y(self):
        """If x is independent of y, theta should be near 0 and y_adj ≈ y."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=10_000)
        y = rng.normal(loc=5.0, size=10_000)
        y_adj, theta, rho = cuped_adjust(y, x)
        assert abs(theta) < 0.05
        # Adjustment should be small
        assert np.abs(y_adj - y).mean() < 0.2


class TestCUPEDTTest:
    def test_naive_and_cuped_estimate_same_effect(self, small_experiment):
        """CUPED should not bias the effect estimate; it only reduces variance."""
        res = cuped_t_test(
            small_experiment,
            metric_col="metric_continuous",
            covariate_col="pre_covariate",
        )
        # Naive and CUPED estimates should agree to within ~10%
        assert math.isclose(
            res.naive_test.effect_absolute,
            res.cuped_test.effect_absolute,
            rel_tol=0.10,
        )

    def test_variance_reduction_positive_with_correlated_covariate(
        self, small_experiment
    ):
        res = cuped_t_test(
            small_experiment,
            metric_col="metric_continuous",
            covariate_col="pre_covariate",
        )
        # Our fixture has built-in correlation; expect positive reduction
        assert res.variance_reduction_pct > 0
        assert res.cuped_test.se < res.naive_test.se
