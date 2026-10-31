"""Judging a testing programme from its tests' package-level counts.

Each test randomizes sessions across packages and records impressions and clicks per package.
The tools here answer the questions in reports/upworthy_analysis_plan.md: is the split broken,
how different are packages really, how much a winner's lift overstates itself, what picking
winners bought, and how big tests should be.

The central trick is **splitting each package's sessions in two at random** ("data fission").
Given the click counts, assigning each session to half A or B with probability 1/2 makes the two
halves independent draws from the same package, so one half can choose a winner and the other
can measure it without the selection bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TEST = "clickability_test_id"


# ---------------------------------------------------------------- sample ratio
def srm_by_test(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square of each test's package impressions against an equal split."""
    out = []
    for tid, g in df.groupby(TEST, sort=False):
        n = g["impressions"].to_numpy(float)
        p = float(stats.chisquare(n).pvalue) if len(n) > 1 and n.sum() > 0 else np.nan
        out.append({TEST: tid, "packages": len(n), "impressions": int(n.sum()), "srm_p": p,
                    "test_start": g["test_start"].iloc[0], "in_window": bool(g["in_window"].iloc[0])})
    return pd.DataFrame(out)


def analysis_sample(df: pd.DataFrame, srm: pd.DataFrame, alpha: float = 0.001, min_impressions: int = 1000) -> pd.DataFrame:
    """Tests outside the window that pass the sample-ratio check, keeping packages with at least
    ``min_impressions`` impressions in tests that still have two or more of them."""
    ok = srm.loc[~srm["in_window"] & (srm["srm_p"] >= alpha), TEST]
    sub = df[df[TEST].isin(ok) & (df["impressions"] >= min_impressions)]
    keep = sub.groupby(TEST)[TEST].transform("size") >= 2
    return sub[keep].reset_index(drop=True)


# ---------------------------------------------------------------- data fission
def fission(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split each package's sessions into halves A and B at random (each session independently,
    probability 1/2); clicks follow without replacement."""
    rng = np.random.default_rng(seed)
    n = df["impressions"].to_numpy(np.int64)
    c = df["clicks"].to_numpy(np.int64)
    n_a = rng.binomial(n, 0.5)
    c_a = rng.hypergeometric(c, n - c, n_a)
    return df.assign(n_a=n_a, c_a=c_a, n_b=n - n_a, c_b=c - c_a)


def log_ctr(c: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.log((c + 0.5) / (n + 1.0))


def deviations(df: pd.DataFrame, c: str, n: str) -> np.ndarray:
    """Each package's log click-through minus its test's pooled log click-through."""
    lc = log_ctr(df[c].to_numpy(float), df[n].to_numpy(float))
    tot_c = df.groupby(TEST)[c].transform("sum").to_numpy(float)
    tot_n = df.groupby(TEST)[n].transform("sum").to_numpy(float)
    return lc - log_ctr(tot_c, tot_n)


def true_sd(halves: pd.DataFrame) -> float:
    """Square root of the covariance of the two halves' deviations across packages: the standard
    deviation of *true* differences between packages (noise is independent between halves, so
    it drops out of the covariance)."""
    ya, yb = deviations(halves, "c_a", "n_a"), deviations(halves, "c_b", "n_b")
    # A deviation from the test's own mean has variance tau^2 (k-1)/k for a test of k packages,
    # so each product is scaled by k/(k-1) to estimate tau^2 itself.
    g = halves[TEST].to_numpy()
    ya = ya - pd.Series(ya).groupby(g).transform("mean").to_numpy()   # centre within each test
    yb = yb - pd.Series(yb).groupby(g).transform("mean").to_numpy()
    k = halves.groupby(TEST)[TEST].transform("size").to_numpy(float)
    cov = float(np.mean(ya * yb * k / (k - 1)))
    return float(np.sqrt(max(cov, 0.0)))


def bootstrap_true_sd(halves: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> list[float]:
    """``true_sd`` on test-level bootstrap resamples (each drawn test gets a fresh id, so a test
    drawn twice counts as two tests)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(halves[TEST])
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    sorted_h = halves.iloc[order].reset_index(drop=True)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([np.arange(starts[t], starts[t] + sizes[t]) for t in pick])
        b = sorted_h.iloc[rows].copy()
        b[TEST] = np.repeat(np.arange(len(pick)), sizes[pick])
        out.append(true_sd(b))
    return out


def winners(halves: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Per test: half A's winner, its lift over the test's other packages on half A and on half B,
    whether it is also half B's winner, and the lift predicted by empirical-Bayes shrinkage of
    half A's deviations with prior sd ``tau``."""
    h = halves.copy()
    h["ctr_a"] = (h["c_a"] + 0.5) / (h["n_a"] + 1)
    h["ctr_b"] = (h["c_b"] + 0.5) / (h["n_b"] + 1)
    h["y_a"] = deviations(h, "c_a", "n_a")
    se2 = 1 / (h["c_a"] + 0.5) - 1 / (h["n_a"] + 1)
    h["shrunk_a"] = h["y_a"] * tau ** 2 / (tau ** 2 + se2)
    h["_tie"] = np.random.default_rng(0).random(len(h)) * 1e-12   # deterministic tie-break
    h = h.reset_index(drop=True)
    ia = (h["ctr_a"] + h["_tie"]).groupby(h[TEST], sort=False).idxmax()
    ib = (h["ctr_b"] + h["_tie"]).groupby(h[TEST], sort=False).idxmax()
    tot = h.groupby(TEST, sort=False)[["c_a", "n_a", "c_b", "n_b", "shrunk_a"]].sum()
    k = h.groupby(TEST, sort=False).size()
    h["log_ctr_b"] = np.log(h["ctr_b"])
    mean_log_b = h.groupby(TEST, sort=False)["log_ctr_b"].mean()
    w = h.loc[ia.to_numpy()].set_index(TEST)
    oth = tot.subtract(w[["c_a", "n_a", "c_b", "n_b", "shrunk_a"]])
    out = pd.DataFrame({
        "k": k,
        "lift_a": np.log(w["ctr_a"]) - np.log((oth["c_a"] + 0.5) / (oth["n_a"] + 1)),
        "lift_b": np.log(w["ctr_b"]) - np.log((oth["c_b"] + 0.5) / (oth["n_b"] + 1)),
        "shrunk_lift": w["shrunk_a"] - oth["shrunk_a"] / (k - 1),
        "same_winner": ia.to_numpy() == ib.reindex(ia.index).to_numpy(),
        # value of publishing A's winner instead of a random package, measured on B
        # (log click-through against the test's average log click-through, as in the simulation)
        "gain_vs_random_b": np.log(w["ctr_b"]) - mean_log_b,
    })
    return out.reset_index()


def summarize_winners(w: pd.DataFrame) -> dict:
    return {"tests": int(len(w)), "lift_a_mean": float(w["lift_a"].mean()), "lift_b_mean": float(w["lift_b"].mean()),
            "survival": float(w["lift_b"].mean() / w["lift_a"].mean()),
            "shrunk_lift_mean": float(w["shrunk_lift"].mean()),
            "same_winner_rate": float(w["same_winner"].mean()),
            "chance_same_winner": float((1 / w["k"]).mean()),
            "gain_vs_random_b_mean": float(w["gain_vs_random_b"].mean())}
