"""Tests for src/data/simulate.py."""
import pytest

from src.data.simulate import SimConfig, simulate_experiment


EXPECTED_COLS = {
    "user_id", "timestamp", "group", "landing_page",
    "country", "device", "source", "is_returning",
    "prior_watch_hours", "converted",
    "trial_watch_hours", "distinct_titles", "day7_active", "page_load_ms",
}


class TestSimulator:
    def test_default_config_runs(self):
        df = simulate_experiment(SimConfig(n_users=5_000, seed=42))
        assert len(df) == 5_000

    def test_all_expected_columns_present(self):
        df = simulate_experiment(SimConfig(n_users=2_000, seed=42))
        assert EXPECTED_COLS.issubset(set(df.columns))

    def test_groups_roughly_balanced(self):
        df = simulate_experiment(SimConfig(n_users=10_000, seed=42))
        counts = df["group"].value_counts(normalize=True)
        # Allow small drift due to engineered SRM bug + sampling
        for v in counts.values:
            assert 0.45 <= v <= 0.55

    def test_seed_makes_run_reproducible(self):
        df1 = simulate_experiment(SimConfig(n_users=2_000, seed=99))
        df2 = simulate_experiment(SimConfig(n_users=2_000, seed=99))
        assert (df1["converted"].values == df2["converted"].values).all()

    def test_conversion_rate_in_sane_range(self):
        df = simulate_experiment(SimConfig(n_users=10_000, seed=42))
        rate = df["converted"].mean()
        # Baseline ~18%, treatment shifts up; overall should land somewhere reasonable
        assert 0.10 <= rate <= 0.40

    def test_treatment_effect_positive(self):
        """Designed effect: treatment > control on conversion."""
        df = simulate_experiment(SimConfig(n_users=20_000, seed=42))
        ctl_rate = df.loc[df["group"] == "control", "converted"].mean()
        trt_rate = df.loc[df["group"] == "treatment", "converted"].mean()
        assert trt_rate > ctl_rate

    def test_landing_page_matches_group_mostly(self):
        """Engineered glitch should affect ≤ ~0.2% of rows."""
        df = simulate_experiment(SimConfig(n_users=20_000, seed=42))
        mismatched = df[
            ((df["group"] == "control") & (df["landing_page"] != "top_picks")) |
            ((df["group"] == "treatment") & (df["landing_page"] != "personalized"))
        ]
        assert len(mismatched) / len(df) < 0.005

    def test_no_negative_watch_hours(self):
        df = simulate_experiment(SimConfig(n_users=5_000, seed=42))
        assert (df["prior_watch_hours"] >= 0).all()
        assert (df["trial_watch_hours"] >= 0).all()

    def test_page_load_within_realistic_range(self):
        df = simulate_experiment(SimConfig(n_users=5_000, seed=42))
        assert (df["page_load_ms"] >= 50).all()
        assert (df["page_load_ms"] <= 2000).all()
