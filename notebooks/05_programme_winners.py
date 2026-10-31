# %% [markdown]
# # 05 · A testing programme: how much is a "winning" headline worth?
#
# Three questions from the plan ([`reports/upworthy_analysis_plan.md`](../reports/upworthy_analysis_plan.md)),
# on tests outside the maintainers' problem window that pass the sample-ratio check (notebook 04):
#
# - **Q2.** How different are packages (headline plus image), really, once noise is removed?
# - **Q3.** How much does the winning package's observed lift overstate its true lift?
# - **Q4.** What did picking winners buy, compared with publishing a package at random?
#
# The holdout sample, which the maintainers kept apart for confirmation, is checked at the end.
#
# **Method: split every package's sessions in two at random.** Assigning each session to half A
# or half B with probability 1/2 makes the halves independent samples of the same package. Half A
# picks the winner; half B, which played no part in choosing it, measures it. The covariance of
# a package's two half-sample results estimates how much packages truly differ, with no model of
# the noise. Every result is averaged over 20 random splits (seeds 0–19).

# %%
import os
import sys
from pathlib import Path


def _root() -> Path:
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "src" / "data" / "upworthy.py").exists():
            return p
    raise FileNotFoundError("run from inside the project folder")


ROOT = _root()
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import records
from src.analysis import programme
from src.data import upworthy

SAMPLE = os.environ.get("UPWORTHY_SAMPLE", "confirmatory")
HOLDOUT = os.environ.get("UPWORTHY_HOLDOUT", "holdout")
SEEDS = range(20)
df = upworthy.load(SAMPLE)
srm = programme.srm_by_test(df)
sample = programme.analysis_sample(df, srm)
print(f"{SAMPLE}: analysis sample {sample[programme.TEST].nunique():,} tests, {len(sample):,} packages "
      f"(of {srm.shape[0]:,} tests)")

# %% [markdown]
# ## Q2. How different are packages, really?
#
# The standard deviation of true differences in log click-through between packages of the same
# test (0.10 would mean a typical package is about 10% better or worse than its test's average).
# The interval resamples tests (200 draws on the first split).

# %%
halves = {s: programme.fission(sample, s) for s in SEEDS}
taus = np.array([programme.true_sd(h) for h in halves.values()])
TAU = float(taus.mean())
boot = programme.bootstrap_true_sd(halves[0], n_boot=200, seed=99)
tau_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
naive_sd = float(np.std(programme.deviations(sample, "clicks", "impressions")))
print(f"true sd {TAU:.3f} (split-to-split range {taus.min():.3f}–{taus.max():.3f}; test-bootstrap 95% "
      f"[{tau_ci[0]:.3f}, {tau_ci[1]:.3f}]); the raw spread, noise included, is {naive_sd:.3f}")

# %% [markdown]
# ## Q3. How much does the winner overstate itself?
#
# In each test, half A's winner: its lift over the test's other packages on half A (where it was
# chosen) and on half B (where it was not). The empirical-Bayes prediction shrinks each package's
# half-A result toward its test's average by the share of its variance that is signal,
# tau² / (tau² + noise²), using Q2's tau.

# %%
per_split = pd.DataFrame([programme.summarize_winners(programme.winners(h, TAU)) for h in halves.values()])
w = per_split.mean()
print(f"winner's lift on the half that chose it {w['lift_a_mean']:.3f}, on the other half {w['lift_b_mean']:.3f}: "
      f"{w['survival']:.0%} survives. Shrinkage predicted {w['shrunk_lift_mean']:.3f}. "
      f"Same winner on both halves in {w['same_winner_rate']:.0%} of tests (chance {w['chance_same_winner']:.0%}).")
per_split.describe().loc[["mean", "min", "max"]].round(4)

# %% [markdown]
# ## Q4. What did picking winners buy?
#
# On half B: the log click-through of half A's winner against the test's average log
# click-through, which is what publishing a package at random would have given. Expressed as a
# relative gain.

# %%
gain = float(np.expm1(w["gain_vs_random_b_mean"]))
print(f"publishing the winner instead of a random package: {gain:+.1%} click-through "
      f"(each half has half the test's traffic, so this is a lower bound for full-size tests)")

# %%
w0 = programme.winners(halves[0], TAU)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
bins = np.linspace(-1, 2, 61)
axes[0].hist(w0["lift_a"], bins=bins, alpha=0.6, color="#BFBFBF", label="on the half that chose it")
axes[0].hist(w0["lift_b"], bins=bins, alpha=0.7, color="#5B8FF9", label="on the other half")
for v, c in ((w["lift_a_mean"], "#7F7F7F"), (w["lift_b_mean"], "#1D4ED8")):
    axes[0].axvline(v, color=c, lw=1.2, ls="--")
axes[0].set(xlabel="winner's lift over the test's other packages (log click-through)", ylabel="tests",
            title="The winner's curse")
axes[0].legend(frameon=False, fontsize=8)
q = pd.qcut(w0["lift_a"], 10)
cal = w0.groupby(q, observed=True).agg(a=("lift_a", "mean"), b=("lift_b", "mean"), s=("shrunk_lift", "mean"))
axes[1].plot(cal["a"], cal["b"], "o-", color="#5B8FF9", label="measured on the other half")
axes[1].plot(cal["a"], cal["s"], "s--", color="#5AD8A6", label="empirical-Bayes prediction")
lim = [0, float(cal["a"].max()) * 1.05]
axes[1].plot(lim, lim, ":", color="#7F7F7F", lw=1, label="no overstatement")
axes[1].set(xlabel="lift on the half that chose the winner", ylabel="lift on the other half",
            title="Shrinkage predicts how much survives (deciles of tests)")
axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
records.save_figure(fig, f"05_winners_{SAMPLE}.png", "real")

# %% [markdown]
# ## Holdout: Q1–Q4 once more
#
# The same steps on the holdout sample, run once. The question is only whether it agrees in sign
# and rough size.

# %%
hold = upworthy.load(HOLDOUT)
hsrm = programme.srm_by_test(hold)
hsrm["flagged"] = hsrm["srm_p"] < 0.001
rate = hsrm.groupby("in_window")["flagged"].mean()
hsample = programme.analysis_sample(hold, hsrm)
hhalves = [programme.fission(hsample, s) for s in SEEDS]
htau = float(np.mean([programme.true_sd(h) for h in hhalves]))
hw = pd.DataFrame([programme.summarize_winners(programme.winners(h, htau)) for h in hhalves]).mean()
holdout = {"sample": HOLDOUT, "tests": int(len(hsrm)), "flag_rate_inside": float(rate.get(True, np.nan)),
           "flag_rate_outside": float(rate.get(False, np.nan)), "tau": htau,
           "winners": {k: float(v) for k, v in hw.items()},
           "gain_vs_random_relative": float(np.expm1(hw["gain_vs_random_b_mean"]))}
c04 = records.read(f"04_sample_ratio_{SAMPLE}")
compare = pd.DataFrame({
    SAMPLE: [c04["flag_rate_inside"], c04["flag_rate_outside"], TAU, w["survival"], w["same_winner_rate"], gain],
    HOLDOUT: [holdout["flag_rate_inside"], holdout["flag_rate_outside"], htau, hw["survival"],
              hw["same_winner_rate"], holdout["gain_vs_random_relative"]]},
    index=["flagged inside window", "flagged outside window", "true sd (tau)", "winner's lift that survives",
           "same winner on both halves", "gain over a random package"])
agree = bool(all(np.sign(compare[SAMPLE]) == np.sign(compare[HOLDOUT]))
             and all((compare[HOLDOUT] / compare[SAMPLE]).between(0.5, 2) | (compare.index == "flagged outside window")))
print(f"holdout {'agrees' if agree else 'does not agree'} with the {SAMPLE} sample in sign and rough size")
compare.round(4)

# %%
record = {"sample": SAMPLE, "seeds": list(SEEDS), "tests": int(sample[programme.TEST].nunique()),
          "packages": int(len(sample)), "tau": TAU, "tau_split_range": [float(taus.min()), float(taus.max())],
          "tau_ci_tests": tau_ci, "raw_sd": naive_sd,
          "winners": {k: float(v) for k, v in w.items()},
          "winners_split_range": {k: [float(per_split[k].min()), float(per_split[k].max())] for k in per_split},
          "gain_vs_random_relative": gain, "holdout": holdout, "holdout_agrees": agree}
print("wrote", records.write(f"05_winners_{SAMPLE}", record, "real").name)
lines = [
    f"Packages truly differ by about {TAU:.0%} in click-through (sd of log click-through, [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]).",
    f"The winner's observed lift overstates itself: {w['lift_a_mean']:.2f} where it was chosen, {w['lift_b_mean']:.2f} "
    f"on fresh traffic, so {w['survival']:.0%} survives; shrinkage predicts {w['shrunk_lift_mean']:.2f}.",
    f"The same package wins both halves in {w['same_winner_rate']:.0%} of tests, against {w['chance_same_winner']:.0%} by chance.",
    f"Picking the winner beats publishing a random package by {gain:+.1%} in click-through.",
    f"Holdout ({holdout['tests']:,} tests): {'agrees in sign and rough size' if agree else 'differs; see the table'} "
    f"(true sd {htau:.2f}, {hw['survival']:.0%} of the winner's lift survives, gain {holdout['gain_vs_random_relative']:+.1%}).",
]
print("\n".join(f"{i}. {t}" for i, t in enumerate(lines, 1)))
