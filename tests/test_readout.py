"""Readout functions for the e-mail test (src/analysis/readout.py, src/data/hillstrom.py)."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.analysis import readout
from src.data import hillstrom


@pytest.fixture(scope="module")
def synth():
    return hillstrom.prepare(hillstrom.make_synthetic(30_000, seed=3))


# ------------------------------------------------------------------ data
def test_synthetic_has_the_real_schema_and_exact_thirds(synth):
    raw = hillstrom.make_synthetic(9_000, seed=1)
    assert list(raw.columns) == hillstrom.COLUMNS
    counts = synth["arm"].value_counts()
    assert counts.max() - counts.min() <= 1
    assert set(synth["purchase_line"]) <= {"mens only", "womens only", "both", "neither"}


def test_prepare_rejects_unknown_arms():
    raw = hillstrom.make_synthetic(30, seed=1)
    raw.loc[0, "segment"] = "Kids E-Mail"
    with pytest.raises(ValueError):
        hillstrom.prepare(raw)


def test_read_raw_handles_csv_and_gzip(tmp_path):
    raw = hillstrom.make_synthetic(50, seed=2)
    raw.to_csv(tmp_path / "hillstrom.csv", index=False)
    raw.to_csv(tmp_path / "hillstorm_no_indices.csv.gz", index=False, compression="gzip")
    a = hillstrom.read_raw(tmp_path / "hillstrom.csv")
    b = hillstrom.read_raw(tmp_path / "hillstorm_no_indices.csv.gz")
    pd.testing.assert_frame_equal(a, b)
    assert hillstrom.find_raw(tmp_path).name == "hillstorm_no_indices.csv.gz"


def test_load_falls_back_to_synthetic(tmp_path):
    df, source = hillstrom.load(raw_dir=tmp_path, n_synthetic=300)
    assert source == "synthetic" and len(df) == 300


# ------------------------------------------------------------------ checks
def test_srm_flags_a_broken_split():
    assert readout.srm({"a": 1000, "b": 1000, "c": 1000})["passed"]
    assert not readout.srm({"a": 1000, "b": 1000, "c": 1200})["passed"]


def test_smd_is_zero_for_identical_arms_and_detects_a_shift():
    df = pd.DataFrame({"arm": ["a", "b"] * 500, "x": np.tile([1.0, 1.0], 500)})
    df.loc[df.index[::2][:250], "x"] = 3.0
    s = readout.smd(df, ["x"])
    assert s.loc["x", "max_abs_smd"] > 0.5


def test_spend_cap_uses_positive_spend_only():
    s = pd.Series([0.0] * 900 + list(range(1, 101)))
    assert readout.spend_cap(s, 0.99) == pytest.approx(np.quantile(np.arange(1, 101), 0.99))


# ------------------------------------------------------------------ effects
def test_diff_in_means_matches_welch_by_hand():
    df = pd.DataFrame({"arm": ["t"] * 4 + ["c"] * 4, "y": [1, 2, 3, 4, 1, 1, 2, 2.0]})
    e = readout.diff_in_means(df, "y", "t", "c")
    assert e.diff == pytest.approx(1.0)
    assert e.se == pytest.approx(np.sqrt(np.var([1, 2, 3, 4], ddof=1) / 4 + np.var([1, 1, 2, 2], ddof=1) / 4))


def test_lin_adjustment_matches_statsmodels_and_recovers_the_truth(synth):
    cov = ["history", "recency", "mens", "womens", "newbie", "zip_code", "channel"]
    e = readout.lin_adjusted(synth, "visit", "mens", "none", cov)
    sub = synth[synth["arm"].isin(["mens", "none"])]
    X = readout.covariate_matrix(sub, cov)
    X = X - X.mean()
    T = (sub["arm"] == "mens").astype(float)
    design = sm.add_constant(pd.concat([T.rename("T"), X, X.mul(T, axis=0).add_prefix("Tx")], axis=1))
    ref = sm.OLS(sub["visit"].astype(float), design).fit(cov_type="HC2")
    assert e.diff == pytest.approx(ref.params["T"]) and e.se == pytest.approx(ref.bse["T"])
    true_lift = hillstrom.TRUTH["visit"]["mens"]
    assert e.ci[0] < true_lift < e.ci[1]


def test_adjustment_narrows_the_interval_when_covariates_predict():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    arm = np.where(rng.random(n) < 0.5, "t", "c")
    y = 2 * x + 0.3 * (arm == "t") + rng.normal(size=n)
    df = pd.DataFrame({"arm": arm, "x": x, "y": y})
    raw, adj = readout.diff_in_means(df, "y", "t", "c"), readout.lin_adjusted(df, "y", "t", "c", ["x"])
    assert adj.se < 0.6 * raw.se
    assert adj.ci[0] < 0.3 < adj.ci[1]


def test_value_and_breakeven():
    assert readout.value_per_1000(1.0, 0.4, 0.1) == pytest.approx(300.0)
    assert readout.breakeven_margin(0.5, 0.1) == pytest.approx(0.2)
    assert readout.breakeven_margin(-0.1, 0.1) == float("inf")


# ------------------------------------------------------------------ targeting
def test_split_halves_is_stratified_and_disjoint(synth):
    a, b = readout.split_halves(synth, seed=7)
    assert len(a) + len(b) == len(synth) and not set(a.index) & set(b.index)
    assert (a["arm"].value_counts() - b["arm"].value_counts()).abs().max() <= 1


def test_segment_policy_picks_the_best_arm_or_control():
    df = pd.DataFrame({"seg": ["x"] * 6 + ["y"] * 6,
                       "arm": ["none", "a", "b"] * 4,
                       "y": [1, 3, 2, 1, 3, 2, 5, 1, 2, 5, 1, 2.0]})
    assert readout.segment_policy(df, "seg", "y", ["a", "b"], "none") == {"x": "a", "y": "none"}


def test_ipw_value_is_unbiased_for_a_single_arm_policy(synth):
    est = readout.ipw_value(synth, np.full(len(synth), "mens"), "visit", 1 / 3).mean()
    direct = synth.loc[synth["arm"] == "mens", "visit"].mean()
    assert est == pytest.approx(direct, rel=0.01)


def test_policy_gain_of_a_policy_against_itself_is_zero(synth):
    a = np.full(len(synth), "mens")
    g = readout.policy_gain(synth, a, a, "visit", 1 / 3, n_boot=200)
    assert g["gain"] == 0 and g["ci"] == [0.0, 0.0]


def test_readout_intervals_cover_the_planted_effect():
    """Over 60 simulated e-mail tests with a known visit lift, the unadjusted and the
    regression-adjusted 95% intervals each cover the truth close to 95% of the time."""
    cov = ["history", "recency", "mens", "womens", "newbie", "zip_code", "channel"]
    hits = {"unadjusted": 0, "adjusted": 0}
    reps = 60
    for s in range(reps):
        d = hillstrom.prepare(hillstrom.make_synthetic(20_000, seed=1000 + s))
        # the planted lift varies with purchase line; its average over these customers is the target
        share = d.loc[d["arm"].isin(["mens", "none"]), "mens"].mean()
        target = hillstrom.TRUTH["visit"]["mens"] + hillstrom.TRUTH["mens_line_bonus"] * (share - 0.55)
        for name, e in (("unadjusted", readout.diff_in_means(d, "visit", "mens", "none")),
                        ("adjusted", readout.lin_adjusted(d, "visit", "mens", "none", cov))):
            hits[name] += e.ci[0] <= target <= e.ci[1]
    # 95% coverage over 60 reps: fewer than 52 hits would happen by chance under 1% of the time
    assert all(h >= 52 for h in hits.values()), hits


def test_mde_and_sample_size_are_inverses():
    n = readout.n_per_arm(sd=11.6, effect=0.25)
    assert readout.mde(11.6, n) == pytest.approx(0.25, rel=1e-3)
    assert readout.mde(0.3, 21_000) == pytest.approx(2.8016 * 0.3 * (2 / 21_000) ** 0.5, rel=1e-3)
