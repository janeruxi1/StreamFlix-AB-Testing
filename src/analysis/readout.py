"""Reading out a multi-arm test: trust checks, effects, adjustment, value, targeting.

Everything follows reports/hillstrom_analysis_plan.md. Functions take a customer table with an
``arm`` column and return plain numbers, so notebooks and tests share one implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .frequentist import holm_bonferroni

Z95 = float(stats.norm.ppf(0.975))


# ---------------------------------------------------------------- trust checks
def srm(counts: dict[str, int], expected: dict[str, float] | None = None) -> dict:
    """Chi-square goodness of fit of arm counts to the planned split (equal by default)."""
    arms = list(counts)
    n = sum(counts.values())
    exp_share = expected or {a: 1 / len(arms) for a in arms}
    obs = np.array([counts[a] for a in arms], float)
    exp = np.array([exp_share[a] * n for a in arms])
    chi2, p = stats.chisquare(obs, exp)
    return {"counts": counts, "chi2": float(chi2), "p_value": float(p), "passed": bool(p >= 0.001)}


def smd(df: pd.DataFrame, cols: list[str], arm_col: str = "arm") -> pd.DataFrame:
    """Largest absolute standardized mean difference of each column between any two arms.
    Categorical columns are expanded to one indicator per level."""
    X = pd.get_dummies(df[cols], dtype=float)
    arms = list(df[arm_col].cat.categories) if hasattr(df[arm_col], "cat") else sorted(df[arm_col].unique())
    rows = {}
    for c in X.columns:
        sd = X[c].std(ddof=1)
        means = [X.loc[df[arm_col] == a, c].mean() for a in arms]
        rows[c] = 0.0 if sd == 0 else (max(means) - min(means)) / sd
    return pd.Series(rows, name="max_abs_smd").sort_values(ascending=False).to_frame()


def spend_cap(spend: pd.Series, q: float = 0.99) -> float:
    """The plan's cap: the q-quantile of positive spend, all arms pooled."""
    pos = spend[spend > 0]
    return float(pos.quantile(q)) if len(pos) else 0.0


# ---------------------------------------------------------------- effects
@dataclass
class Effect:
    metric: str
    treatment: str
    control: str
    mean_t: float
    mean_c: float
    diff: float
    se: float
    p_value: float
    adjusted: bool

    @property
    def ci(self) -> tuple[float, float]:
        return self.diff - Z95 * self.se, self.diff + Z95 * self.se

    @property
    def rel(self) -> float:
        return self.diff / self.mean_c if self.mean_c else float("nan")

    def as_dict(self) -> dict:
        lo, hi = self.ci
        return {"metric": self.metric, "comparison": f"{self.treatment} vs {self.control}",
                "mean_treatment": self.mean_t, "mean_control": self.mean_c, "diff": self.diff,
                "ci_low": lo, "ci_high": hi, "rel": self.rel, "se": self.se, "p_value": self.p_value,
                "adjusted": self.adjusted}


def diff_in_means(df: pd.DataFrame, y: str, t: str, c: str, arm_col: str = "arm") -> Effect:
    """Unadjusted difference in means with a Welch (unequal-variance) standard error."""
    yt = df.loc[df[arm_col] == t, y].to_numpy(float)
    yc = df.loc[df[arm_col] == c, y].to_numpy(float)
    d = yt.mean() - yc.mean()
    se = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))
    p = float(2 * stats.norm.sf(abs(d / se))) if se > 0 else float("nan")
    return Effect(y, t, c, float(yt.mean()), float(yc.mean()), float(d), se, p, False)


def covariate_matrix(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Numeric covariates as they are, categorical ones as indicators (first level dropped)."""
    return pd.get_dummies(df[covariates], drop_first=True, dtype=float)


def lin_adjusted(df: pd.DataFrame, y: str, t: str, c: str, covariates: list[str],
                 arm_col: str = "arm") -> Effect:
    """Regression adjustment (Lin 2013): outcome on treatment, centred covariates and their
    interactions with treatment; HC2 standard errors. With one covariate and no interaction this
    is CUPED; the interaction makes it never less precise than the unadjusted estimate
    asymptotically, even if the treatment changes how covariates relate to the outcome."""
    sub = df[df[arm_col].isin([t, c])]
    X = covariate_matrix(sub, covariates)
    X = X - X.mean()
    T = (sub[arm_col] == t).astype(float).to_numpy()[:, None]
    design = np.hstack([np.ones((len(sub), 1)), T, X.to_numpy(), T * X.to_numpy()])
    res = sm.OLS(sub[y].to_numpy(float), design).fit(cov_type="HC2")
    d, se = float(res.params[1]), float(res.bse[1])
    yt, yc = sub.loc[sub[arm_col] == t, y], sub.loc[sub[arm_col] == c, y]
    return Effect(y, t, c, float(yt.mean()), float(yc.mean()), d, se,
                  float(2 * stats.norm.sf(abs(d / se))), True)


def holm(effects: list[Effect], alpha: float = 0.05) -> list[bool]:
    return holm_bonferroni([e.p_value for e in effects], alpha)


# ---------------------------------------------------------------- value
def value_per_1000(spend_lift: float, margin: float, cost_per_email: float) -> float:
    """Net value of e-mailing 1,000 customers: lift in spend per customer times margin, less cost."""
    return 1000 * (spend_lift * margin - cost_per_email)


def mde(sd: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest true difference in means a two-arm test with n per arm detects with the given
    power (two-sided normal approximation, equal arms, outcome sd `sd`; for a rate p, sd is
    sqrt(p(1 - p)))."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(2 / n_per_arm))


def n_per_arm(sd: float, effect: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Customers per arm needed to detect a true difference `effect` (inverse of `mde`)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sd / effect) ** 2))


def breakeven_margin(spend_lift: float, cost_per_email: float) -> float:
    """Gross margin at which the e-mail exactly pays for itself."""
    return cost_per_email / spend_lift if spend_lift > 0 else float("inf")


# ---------------------------------------------------------------- targeting
def split_halves(df: pd.DataFrame, seed: int, arm_col: str = "arm") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random 50/50 split, stratified by arm, so each half keeps the thirds."""
    rng = np.random.default_rng(seed)
    disc = np.zeros(len(df), bool)
    for _, idx in df.groupby(arm_col, observed=True).indices.items():
        pick = rng.permutation(idx)[: len(idx) // 2]
        disc[pick] = True
    return df[disc], df[~disc]


def segment_policy(df: pd.DataFrame, segment: str, y: str, arms: list[str], control: str,
                   arm_col: str = "arm") -> dict:
    """For each level of ``segment``: the arm with the largest mean lift over control, or control
    when no arm beats it."""
    policy = {}
    for level, g in df.groupby(segment, observed=True):
        base = g.loc[g[arm_col] == control, y].mean()
        lifts = {a: g.loc[g[arm_col] == a, y].mean() - base for a in arms}
        best = max(lifts, key=lifts.get)
        policy[level] = best if lifts[best] > 0 else control
    return policy


def ipw_value(df: pd.DataFrame, assign: np.ndarray, y: str, p_arm: float, arm_col: str = "arm") -> np.ndarray:
    """Per-customer inverse-probability-weighted value of a policy that would give customer i
    the arm ``assign[i]``. Its mean is an unbiased estimate of the policy's value when arms were
    assigned at random with probability ``p_arm``."""
    match = (df[arm_col].astype(str).to_numpy() == np.asarray(assign, dtype=str)).astype(float)
    return df[y].to_numpy(float) * match / p_arm


def policy_gain(df: pd.DataFrame, policy_assign: np.ndarray, rival_assign: np.ndarray, y: str,
                p_arm: float, n_boot: int = 2000, seed: int = 0, arm_col: str = "arm") -> dict:
    """Value of a policy minus a rival policy, per customer, with a bootstrap 95% interval
    (customers resampled; both policies scored on the same draws)."""
    d = ipw_value(df, policy_assign, y, p_arm, arm_col) - ipw_value(df, rival_assign, y, p_arm, arm_col)
    rng = np.random.default_rng(seed)
    boots = np.concatenate([d[rng.integers(0, len(d), (min(200, n_boot - s), len(d)))].mean(axis=1)
                            for s in range(0, n_boot, 200)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"gain": float(d.mean()), "ci": [float(lo), float(hi)], "n": int(len(d))}
