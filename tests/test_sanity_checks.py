"""Tests for src/analysis/sanity_checks.py."""
import pandas as pd

from src.analysis.sanity_checks import check_srm, outcome_maturity


def test_srm_flags_obvious_mismatch():
    df = pd.DataFrame({"group": ["control"] * 6000 + ["treatment"] * 4000})
    res = check_srm(df, expected_ratio={"control": 0.5, "treatment": 0.5})
    assert not res.passed


def test_srm_passes_balanced_split():
    df = pd.DataFrame({"group": ["control"] * 5010 + ["treatment"] * 4990})
    assert check_srm(df, expected_ratio={"control": 0.5, "treatment": 0.5}).passed


def test_outcome_maturity_partial_and_full():
    ts = pd.date_range("2026-04-01", "2026-04-28", freq="D")
    df = pd.DataFrame({"timestamp": ts})
    early = outcome_maturity(df, "2026-05-08", window_days=14)
    assert 0 < early["matured_share"] < 1
    assert early["n_immature"] == (ts + pd.Timedelta(days=14) > pd.Timestamp("2026-05-08")).sum()
    full = outcome_maturity(df, early["all_matured_on"], window_days=14)
    assert full["matured_share"] == 1.0 and full["n_immature"] == 0
