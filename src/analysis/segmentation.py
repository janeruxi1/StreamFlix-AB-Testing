"""Segmentation and variance-reduction analysis for A/B experiments.

Contains:
  - segment_lifts: compute treatment effect within each segment + 95% CI
  - cuped_adjust: CUPED variance reduction using a pre-experiment covariate
  - cuped_t_test: two-sample test on the CUPED-adjusted outcome

CUPED (Controlled Experiment Using Pre-Experiment Data) is the industry-standard
technique used at Microsoft, Netflix, Booking, and similar to shrink confidence
intervals on an experiment without collecting more data. It works by removing
user-level variability that has nothing to do with the treatment, using a
pre-period covariate (e.g. prior behavior) as a control variable.

Formula:
    Y_cuped = Y - theta * (X - mean(X))
    theta   = Cov(Y, X) / Var(X)
Variance reduction ~ 1 - corr(Y, X)^2.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from src.analysis.frequentist import two_proportion_test, welch_t_test, TestResult


# ---------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------
def segment_lifts(
    df: pd.DataFrame,
    segment_col: str,
    metric_col: str,
    group_col: str = "group",
    binary: bool | None = None,
    min_segment_n: int = 100,
) -> pd.DataFrame:
    """Compute treatment effect within each segment of `segment_col`.

    Parameters
    ----------
    df : DataFrame with group, segment, and metric columns.
    segment_col : column to split on (e.g. 'device', 'country').
    metric_col : outcome variable (binary or continuous).
    binary : True for proportion test, False for Welch's t-test. Auto-detect if None.
    min_segment_n : skip segments with fewer than this many users in either arm.

    Returns
    -------
    DataFrame with one row per segment containing effect, CI, n, and p-value.
    """
    if binary is None:
        binary = set(df[metric_col].dropna().unique()).issubset({0, 1, 0.0, 1.0})

    rows = []
    for seg_value, seg in df.groupby(segment_col):
        ctl = seg[seg[group_col] == "control"]
        trt = seg[seg[group_col] == "treatment"]
        if len(ctl) < min_segment_n or len(trt) < min_segment_n:
            continue

        if binary:
            res = two_proportion_test(
                successes_control=int(ctl[metric_col].sum()),
                n_control=len(ctl),
                successes_treatment=int(trt[metric_col].sum()),
                n_treatment=len(trt),
                name=str(seg_value),
            )
        else:
            res = welch_t_test(
                ctl[metric_col].values, trt[metric_col].values,
                name=str(seg_value),
            )
        rows.append({
            "segment": seg_value,
            "n_control": res.n_control,
            "n_treatment": res.n_treatment,
            "mean_control": res.p_control,
            "mean_treatment": res.p_treatment,
            "effect_abs": res.effect_absolute,
            "effect_rel": res.effect_relative,
            "ci_lower": res.ci_lower,
            "ci_upper": res.ci_upper,
            "p_value": res.p_value,
            "significant": res.significant,
        })

    return pd.DataFrame(rows).sort_values("effect_abs", ascending=False)


# ---------------------------------------------------------------------
# CUPED variance reduction
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CUPEDResult:
    theta: float
    correlation: float
    variance_reduction_pct: float
    naive_test: TestResult
    cuped_test: TestResult

    def __str__(self) -> str:
        return (
            f"CUPED summary\n"
            f"  theta:                  {self.theta:.4f}\n"
            f"  corr(Y, X_pre):         {self.correlation:.4f}\n"
            f"  Variance reduction:     {self.variance_reduction_pct:.1f}%\n"
            f"  Naive  CI half-width:   "
            f"{(self.naive_test.ci_upper - self.naive_test.ci_lower) / 2:.5f}\n"
            f"  CUPED  CI half-width:   "
            f"{(self.cuped_test.ci_upper - self.cuped_test.ci_lower) / 2:.5f}"
        )


def cuped_adjust(
    y: np.ndarray, x_pre: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Apply CUPED adjustment to outcome y using pre-experiment covariate x_pre.

    Returns (y_adjusted, theta, correlation).

    Theta is computed on the COMBINED sample (not just control); both are
    valid choices but the combined estimator is unbiased and the most common
    industry default.
    """
    x = np.asarray(x_pre, dtype=float)
    y = np.asarray(y, dtype=float)

    var_x = x.var(ddof=1)
    if var_x <= 0:
        return y.copy(), 0.0, 0.0

    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    theta = cov_xy / var_x
    rho = np.corrcoef(x, y)[0, 1]
    x_mean = x.mean()

    y_cuped = y - theta * (x - x_mean)
    return y_cuped, float(theta), float(rho)


def cuped_t_test(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
    group_col: str = "group",
    name: str = "metric (CUPED)",
) -> CUPEDResult:
    """Run a Welch t-test on the CUPED-adjusted outcome and compare to the
    naive (unadjusted) test. Returns both, plus the variance reduction.
    """
    sub = df[[group_col, metric_col, covariate_col]].dropna()
    y = sub[metric_col].values
    x = sub[covariate_col].values
    g = sub[group_col].values

    # Naive test
    naive = welch_t_test(
        y[g == "control"], y[g == "treatment"], name=name.replace(" (CUPED)", " (naive)"),
    )

    # CUPED-adjusted test
    y_cuped, theta, rho = cuped_adjust(y, x)
    cuped = welch_t_test(
        y_cuped[g == "control"], y_cuped[g == "treatment"], name=name,
    )

    var_reduction = 100.0 * (1 - (cuped.se ** 2) / (naive.se ** 2))

    return CUPEDResult(
        theta=theta,
        correlation=rho,
        variance_reduction_pct=var_reduction,
        naive_test=naive,
        cuped_test=cuped,
    )
