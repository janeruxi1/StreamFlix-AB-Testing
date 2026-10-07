"""Sanity checks for A/B test data quality.

Run these BEFORE any statistical analysis. If any check fails,
the experiment is likely invalid and should not be analyzed for ship/no-ship.
"""
from dataclasses import dataclass
import pandas as pd
from scipy import stats


@dataclass
class SRMResult:
    """Result of a Sample Ratio Mismatch check."""
    observed: dict[str, int]
    expected: dict[str, float]
    chi2: float
    p_value: float
    passed: bool  # True if NO mismatch detected (p > threshold)

    def __str__(self) -> str:
        verdict = "✅ PASS" if self.passed else "🚨 FAIL — SRM detected"
        return (
            f"{verdict}\n"
            f"  Observed: {self.observed}\n"
            f"  Expected: {self.expected}\n"
            f"  Chi-square: {self.chi2:.4f}, p-value: {self.p_value:.6f}"
        )


def check_srm(
    df: pd.DataFrame,
    group_col: str = "group",
    expected_ratio: dict[str, float] | None = None,
    alpha: float = 0.001,
) -> SRMResult:
    """Test for Sample Ratio Mismatch using chi-square goodness-of-fit.

    A 'mismatch' means observed group sizes deviate from expected.
    Industry standard: alpha = 0.001 (very strict — false positives are costly).

    Parameters
    ----------
    df : DataFrame containing the assignment column.
    group_col : Name of the column holding the variant label.
    expected_ratio : Dict mapping variant -> expected proportion.
        Defaults to equal split across observed variants.
    alpha : Significance threshold. p < alpha → SRM flagged.

    Returns
    -------
    SRMResult
    """
    observed = df[group_col].value_counts().to_dict()
    variants = list(observed.keys())
    n_total = sum(observed.values())

    if expected_ratio is None:
        expected_ratio = {v: 1 / len(variants) for v in variants}

    observed_counts = [observed[v] for v in variants]
    expected_counts = [expected_ratio[v] * n_total for v in variants]

    chi2, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

    return SRMResult(
        observed=observed,
        expected={v: expected_ratio[v] * n_total for v in variants},
        chi2=float(chi2),
        p_value=float(p_value),
        passed=p_value > alpha,
    )


def outcome_maturity(
    df: pd.DataFrame,
    analysis_date: str | pd.Timestamp,
    window_days: int = 14,
    time_col: str = "timestamp",
) -> dict:
    """Share of users whose outcome window had fully closed at `analysis_date`.

    A 14-day conversion metric is only final for users who signed up at least
    14 days before the data pull. Analysing earlier censors the latest
    cohorts (they have had less time to convert), which biases conversion
    down for both arms and can distort weekly trends.

    Returns the matured share, the last fully-matured signup time, and the
    date on which every user's window would have closed.
    """
    analysis_date = pd.Timestamp(analysis_date)
    window = pd.Timedelta(days=window_days)
    ts = pd.to_datetime(df[time_col])
    matured = (ts + window) <= analysis_date
    return {
        "matured_share": float(matured.mean()),
        "last_matured_signup": analysis_date - window,
        "all_matured_on": ts.max() + window,
        "n_immature": int((~matured).sum()),
    }
