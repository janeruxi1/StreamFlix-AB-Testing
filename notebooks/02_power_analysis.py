"""
Phase 2 — Power Analysis & Sample Size
=======================================

PRE-EXPERIMENT design (the PM brief says: 'we want at least 1pp lift').
We answer FOUR questions here:

    Q1. How many users per arm do we need?            -> sample_size
    Q2. How long would that take at our daily traffic? -> runtime
    Q3. What's our sample's actual MDE? (post-hoc)     -> mde_for_n
    Q4. If the true lift is X, can we detect it?       -> power_for_n

The most important interview moment isn't the formula — it's the framing:
"MDE is a business decision, not a statistical one."
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from src.analysis.power import (
    sample_size_two_proportion,
    mde_for_sample_size,
    power_for_sample_size,
    runtime_days,
)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Assumptions (from the PM brief & historical data)
# ---------------------------------------------------------------------
BASELINE_RATE = 0.18    # current trial-to-paid rate
MDE_ABSOLUTE  = 0.01    # PM wants to detect at least +1pp lift
ALPHA         = 0.05
POWER         = 0.80
DAILY_TRIAFFIC = 3_500  # avg new trialists per day (typo on purpose? no - corrected below)
DAILY_TRAFFIC = 3_500


# ---------------------------------------------------------------------
# Q1. Required sample size
# ---------------------------------------------------------------------
print("=" * 70)
print("Q1. Required sample size at our target MDE")
print("=" * 70)
res = sample_size_two_proportion(
    baseline=BASELINE_RATE,
    mde_absolute=MDE_ABSOLUTE,
    alpha=ALPHA,
    power=POWER,
)
print(res)

print("\n💡 Interview framing:")
print("   - alpha = 5% false-positive tolerance (we'd ship a useless feature")
print("     1-in-20 tests if we run a lot of them)")
print("   - power = 80% true-positive tolerance (we'd miss a real +1pp lift")
print("     1-in-5 times)")
print("   - MDE is the BUSINESS choice: +1pp is the smallest lift the PM")
print("     thinks justifies the engineering cost. It's not a stats decision.")


# ---------------------------------------------------------------------
# Q2. Runtime at observed traffic
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q2. How long do we need to run?")
print("=" * 70)
days = runtime_days(res.n_total, DAILY_TRAFFIC)
print(f"  At {DAILY_TRAFFIC:,} new trialists/day, total {res.n_total:,} users")
print(f"  --> runtime ≈ {days:.1f} days")
print("\n💡 Real-world adjustments senior DS would call out:")
print("   - Pad +1 full week-cycle to absorb weekday/weekend seasonality")
print("   - Account for novelty effects (decay over the first ~7 days)")
print("   - 4-week run was chosen for those reasons, not because of math")


# ---------------------------------------------------------------------
# Q3. Sensitivity — sample size as a function of MDE
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q3. How sensitive is sample size to MDE?")
print("=" * 70)
mdes = np.linspace(0.003, 0.03, 30)  # 0.3pp to 3pp
ns = [sample_size_two_proportion(BASELINE_RATE, m).n_per_arm for m in mdes]

print("  MDE (pp)  ->  n per arm")
for m, n in zip(mdes[::6], ns[::6]):
    print(f"   {m*100:5.2f}      {n:>10,}")
print("\n💡 Halving the MDE → 4x the sample size. This is why detecting tiny")
print("   lifts is expensive, and why interviewers love asking about it.")


# ---------------------------------------------------------------------
# Q4. Post-hoc MDE — given our actual 50k/arm, what could we detect?
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q4. Post-hoc: given our actual n ≈ 50,000 per arm, what's our MDE?")
print("=" * 70)
ACTUAL_N = 50_000
mde_actual = mde_for_sample_size(BASELINE_RATE, ACTUAL_N, alpha=ALPHA, power=POWER)
print(f"  MDE at n={ACTUAL_N:,}/arm, alpha={ALPHA}, power={POWER}: "
      f"{mde_actual*100:.3f}pp absolute "
      f"({mde_actual/BASELINE_RATE:.2%} relative)")
print("\n💡 If our headline lift is BELOW this MDE, the experiment is")
print("   underpowered for that effect and the null result is inconclusive.")


# ---------------------------------------------------------------------
# Visualization 1 — Sample size vs MDE curve
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mdes * 100, ns, color="#5B8FF9", linewidth=2.5)
ax.fill_between(mdes * 100, ns, alpha=0.15, color="#5B8FF9")

# Mark our target MDE
ax.axvline(MDE_ABSOLUTE * 100, color="#F6BD16", linestyle="--", linewidth=2,
           label=f"PM's MDE = {MDE_ABSOLUTE*100:.1f}pp")
ax.axhline(res.n_per_arm, color="#F6BD16", linestyle=":", linewidth=1.5)
ax.scatter([MDE_ABSOLUTE * 100], [res.n_per_arm],
           color="#F6BD16", s=100, zorder=5,
           label=f"n required = {res.n_per_arm:,}/arm")

ax.set_xlabel("Minimum Detectable Effect (percentage points)", fontsize=11)
ax.set_ylabel("Required n per arm", fontsize=11)
ax.set_title(
    f"Sample Size vs MDE  (baseline={BASELINE_RATE:.0%}, α={ALPHA}, power={POWER})",
    fontsize=13, fontweight="bold",
)
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_sample_size_vs_mde.png", dpi=140, bbox_inches="tight")
print(f"\n📊 Saved sample-size curve -> {FIG_DIR}/02_sample_size_vs_mde.png")


# ---------------------------------------------------------------------
# Visualization 2 — Power curve at our actual sample size
# ---------------------------------------------------------------------
true_effects = np.linspace(0.001, 0.03, 60)
powers = [
    power_for_sample_size(BASELINE_RATE, eff, ACTUAL_N, alpha=ALPHA)
    for eff in true_effects
]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(true_effects * 100, powers, color="#5B8FF9", linewidth=2.5)
ax.fill_between(true_effects * 100, powers, alpha=0.15, color="#5B8FF9")
ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="80% power threshold")
ax.axvline(mde_actual * 100, color="#F6BD16", linestyle="--", linewidth=2,
           label=f"Post-hoc MDE = {mde_actual*100:.2f}pp")
ax.axvline(MDE_ABSOLUTE * 100, color="crimson", linestyle=":", linewidth=2,
           label=f"Target MDE = {MDE_ABSOLUTE*100:.1f}pp")

ax.set_xlabel("True effect size (percentage points)", fontsize=11)
ax.set_ylabel("Power (probability of detecting effect)", fontsize=11)
ax.set_title(
    f"Power Curve  (n={ACTUAL_N:,}/arm, baseline={BASELINE_RATE:.0%}, α={ALPHA})",
    fontsize=13, fontweight="bold",
)
ax.set_ylim(0, 1.02)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_power_curve.png", dpi=140, bbox_inches="tight")
print(f"📊 Saved power curve -> {FIG_DIR}/02_power_curve.png")


# ---------------------------------------------------------------------
# Visualization 3 — MDE heatmap: MDE x baseline rate
# ---------------------------------------------------------------------
baselines = np.linspace(0.05, 0.40, 25)
mde_grid  = np.linspace(0.003, 0.03, 25)
B, M = np.meshgrid(baselines, mde_grid)
N = np.array([
    [sample_size_two_proportion(b, m).n_per_arm for b in baselines]
    for m in mde_grid
])

fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(np.log10(N), origin="lower", aspect="auto",
               extent=[baselines.min()*100, baselines.max()*100,
                       mde_grid.min()*100, mde_grid.max()*100],
               cmap="viridis")

# Mark our scenario
ax.scatter([BASELINE_RATE*100], [MDE_ABSOLUTE*100],
           color="white", edgecolor="crimson", s=140, linewidth=2, zorder=5,
           label="Our scenario")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("log10(n per arm)")
ax.set_xlabel("Baseline conversion rate (%)", fontsize=11)
ax.set_ylabel("MDE (percentage points)", fontsize=11)
ax.set_title(
    "Required sample size as a function of baseline rate and MDE",
    fontsize=13, fontweight="bold",
)
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_sample_size_heatmap.png", dpi=140, bbox_inches="tight")
print(f"📊 Saved heatmap -> {FIG_DIR}/02_sample_size_heatmap.png")


# ---------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("POWER ANALYSIS CONCLUSION")
print("=" * 70)
print(f"""
The experiment is well-powered for the headline metric.

  - Required sample (MDE = 1pp):  {res.n_per_arm:,}/arm
  - Actual sample collected:       ~{ACTUAL_N:,}/arm  (~2x over-powered)
  - Post-hoc MDE:                  {mde_actual*100:.2f}pp absolute
                                   ({mde_actual/BASELINE_RATE:.2%} relative)

Any observed lift above {mde_actual*100:.2f}pp will be statistically credible.
A null result would represent a genuine absence of effect, not insufficient
sample size. Proceeding to inference (Phase 3).
""")
