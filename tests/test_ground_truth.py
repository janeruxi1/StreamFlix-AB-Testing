"""The analysis pipeline should recover the effects the simulator planted."""
import numpy as np
import pytest

from src.analysis.frequentist import two_proportion_test
from src.analysis.segmentation import segment_lifts
from src.data.simulate import SimConfig, compute_ground_truth, simulate_experiment


@pytest.fixture(scope="module")
def sim():
    cfg = SimConfig(n_users=60_000, seed=7)
    df, truth = simulate_experiment(cfg, return_truth=True)
    return df, truth


def test_return_truth_does_not_change_data():
    cfg = SimConfig(n_users=3_000, seed=5)
    plain = simulate_experiment(cfg)
    with_truth, truth = simulate_experiment(cfg, return_truth=True)
    assert plain.equals(with_truth)
    assert len(truth) == len(plain) and {"p0", "p1"} == set(truth.columns)


def test_true_effects_have_expected_structure():
    truth = compute_ground_truth(SimConfig(n_users=30_000, seed=1))
    dev = truth["by_device"]
    assert truth["ate"] > 0
    assert dev["iOS"] > dev["Web"] > dev["TV"] >= 0
    assert truth["by_is_returning"][True] > truth["by_is_returning"][False]


def test_ate_ci_covers_true_ate(sim):
    df, truth = sim
    true_ate = float((truth["p1"] - truth["p0"]).mean())
    c, t = df[df["group"] == "control"], df[df["group"] == "treatment"]
    res = two_proportion_test(
        int(c["converted"].sum()), len(c), int(t["converted"].sum()), len(t)
    )
    assert res.ci_lower <= true_ate <= res.ci_upper


def test_segment_estimates_close_to_truth(sim):
    df, truth = sim
    true_by_dev = (truth["p1"] - truth["p0"]).groupby(df["device"]).mean()
    est = segment_lifts(df, "device", "converted").set_index("segment")
    for dev, true_val in true_by_dev.items():
        half_width = (est.loc[dev, "ci_upper"] - est.loc[dev, "ci_lower"]) / 2
        # 99.9%-ish band keeps this deterministic test from being flaky-looking
        assert abs(est.loc[dev, "effect_abs"] - true_val) < 1.7 * half_width
