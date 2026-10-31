"""Programme-level tools (src/analysis/programme.py) on synthetic tests with known truth."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import programme
from src.data import upworthy


@pytest.fixture(scope="module")
def synth():
    return upworthy.make_synthetic(900, seed=4, tau=0.15, broken_share=0.2)


def test_sample_ratio_check_finds_broken_tests(synth):
    srm = programme.srm_by_test(synth)
    broken = synth.groupby(programme.TEST)["broken"].first().reindex(srm[programme.TEST]).to_numpy()
    flagged = (srm["srm_p"] < 0.001).to_numpy()
    assert flagged[broken].mean() > 0.9
    assert flagged[~broken].mean() < 0.01


def test_analysis_sample_drops_flagged_and_window_tests(synth):
    s = synth.copy()
    s.loc[s[programme.TEST] == "t0000", "in_window"] = True
    srm = programme.srm_by_test(s)
    a = programme.analysis_sample(s, srm)
    assert "t0000" not in set(a[programme.TEST])
    assert not set(a[programme.TEST]) & set(srm.loc[srm["srm_p"] < 0.001, programme.TEST])
    assert (a.groupby(programme.TEST).size() >= 2).all() and (a["impressions"] >= 1000).all()


def test_fission_conserves_counts_and_halves_are_uncorrelated_noise():
    """Packages with the same true rate: clicks are a binomial draw, as in a real test. Splitting
    each package's sessions at random then gives two halves whose rates do not co-move. (Holding
    the click total fixed instead would make them negatively correlated: independence comes from
    the sessions, not from the totals.)"""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({programme.TEST: ["t"] * 2000, "impressions": 4000})
    df["clicks"] = rng.binomial(4000, 0.015, len(df))
    h = programme.fission(df, seed=1)
    assert ((h["n_a"] + h["n_b"]) == 4000).all() and ((h["c_a"] + h["c_b"]) == df["clicks"]).all()
    assert (h["c_a"] <= h["n_a"]).all()
    ra, rb = h["c_a"] / h["n_a"], h["c_b"] / h["n_b"]
    assert abs(np.corrcoef(ra, rb)[0, 1]) < 0.08


@pytest.mark.parametrize("tau", [0.1, 0.25])
def test_true_sd_recovers_the_planted_spread(tau):
    s = upworthy.make_synthetic(1500, seed=5, tau=tau, broken_share=0)
    a = programme.analysis_sample(s, programme.srm_by_test(s))
    est = np.mean([programme.true_sd(programme.fission(a, i)) for i in range(4)])
    assert est == pytest.approx(tau, rel=0.15)


def test_winners_curse_and_shrinkage(synth):
    a = programme.analysis_sample(synth, programme.srm_by_test(synth))
    w = programme.summarize_winners(programme.winners(programme.fission(a, 0), 0.15))
    assert w["lift_b_mean"] < w["lift_a_mean"]                         # the winner overstates itself
    assert w["shrunk_lift_mean"] == pytest.approx(w["lift_b_mean"], abs=0.04)
    assert w["same_winner_rate"] > w["chance_same_winner"]


def test_bootstrap_true_sd_brackets_the_estimate(synth):
    a = programme.analysis_sample(synth, programme.srm_by_test(synth))
    h = programme.fission(a, 0)
    b = programme.bootstrap_true_sd(h, n_boot=40, seed=1)
    assert np.percentile(b, 2.5) < programme.true_sd(h) < np.percentile(b, 97.5)
