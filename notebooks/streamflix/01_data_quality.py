"""
Phase 1 — Data Quality & Sanity Checks
=======================================

Walkthrough script. Run as `python notebooks/streamflix/01_data_quality.py` from the
project root, or convert to a Jupyter notebook cell by cell.

Before ANY statistical analysis, we answer 4 questions:
    1. Did randomization work?           -> SRM (chi-square on group sizes)
    2. Are users on the right page?      -> group vs landing_page cross-tab
    3. Are users in only one group?      -> duplicate user_id check
    4. Are baseline attributes balanced? -> covariate balance check

If ANY check fails, the experiment cannot be analyzed for ship/no-ship
without remediation. Verifying data integrity first is the foundation of
trustworthy experimentation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.loader import load_experiment
from src.analysis.sanity_checks import check_srm


# ---------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------
df = load_experiment("data/experiment.csv").dropna(subset=["group"])
FIG_DIR = Path("reports/figures/streamflix")
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")
print(df.head())


# ---------------------------------------------------------------------
# 2. SRM check — Did randomization work?
# ---------------------------------------------------------------------
# We expect a 50/50 split. We use the very strict alpha = 0.001 because
# false positives are costly (we'd block a valid experiment).
srm = check_srm(df, group_col="group", expected_ratio={"control": 0.5, "treatment": 0.5})
print("\n=== SRM check ===")
print(srm)
if srm.p_value < 0.01:
    print("\n⚠️  Borderline: this passes the strict alpha = 0.001 alert threshold but "
          "would fail at 0.01.\n   Treat it as a data-quality incident to localize, "
          "not as a clean pass.")
print("\n💡 Interpretation: If SRM fails, suspect a bug. Look for:")
print("   - Asymmetric platform/device behavior (e.g., Android-only bug)")
print("   - Bot or crawler traffic in one arm")
print("   - Survivorship in tracking (one arm drops events)")


# ---------------------------------------------------------------------
# 3. Assignment integrity — Are users on the right page?
# ---------------------------------------------------------------------
# Treatment users should ONLY see 'personalized'. Control -> 'top_picks'.
print("\n=== Group vs landing_page cross-tab ===")
crosstab = df.groupby(["group", "landing_page"]).size().unstack(fill_value=0)
print(crosstab)

mismatched = df[
    ((df["group"] == "control") & (df["landing_page"] != "top_picks")) |
    ((df["group"] == "treatment") & (df["landing_page"] != "personalized"))
]
mismatch_rate = len(mismatched) / len(df)
print(f"\nMismatched rows: {len(mismatched):,} ({mismatch_rate:.3%})")
print("💡 Interpretation: <0.5% is usually tolerable (caching, race conditions).")
print("   Higher than that = systemic issue. Decision: drop these rows OR")
print("   analyze as ITT (intent-to-treat, using assignment not actual exposure).")


# ---------------------------------------------------------------------
# 4. Duplicate users — Are any users in BOTH groups?
# ---------------------------------------------------------------------
users_per_group = df.groupby("user_id")["group"].nunique()
multi_group = users_per_group[users_per_group > 1]
print(f"\n=== Duplicate users ===")
print(f"Users appearing in multiple groups: {len(multi_group):,}")
print("💡 In real experiments this happens via: deleted cookies, multi-device,")
print("   shared accounts. Common cleaning rules: drop them, or keep first.")


# ---------------------------------------------------------------------
# 5. Covariate balance — finer-grained randomization check
# ---------------------------------------------------------------------
# Even when overall SRM passes, individual attribute distributions
# should be balanced across arms. Imbalances reveal subtle assignment
# leaks that an overall chi-square can miss (e.g. a device-specific bug).
# Industry rule of thumb: flag any category with |gap| > 1pp.
from scipy import stats

FLAG_THRESHOLD_PP = 1.0
CATEGORICAL = ["country", "device", "source", "is_returning"]

print("\n=== Covariate balance (categorical) ===")
flagged_cols = []
gaps = {}
for col in CATEGORICAL:
    share = (
        df.groupby(["group", col]).size()
        / df.groupby("group").size()
    ).unstack(level="group") * 100
    share.columns = [f"{c}_%" for c in share.columns]
    share["gap_pp"] = (share["treatment_%"] - share["control_%"]).round(3)
    share["flag"] = share["gap_pp"].abs() > FLAG_THRESHOLD_PP
    flag_marker = " 🚨 IMBALANCE" if share["flag"].any() else " ✅"
    print(f"\n--- {col} ---{flag_marker}")
    print(share.round(2).to_string())
    if share["flag"].any():
        flagged_cols.append(col)
    gaps[col] = share["gap_pp"]

fig, axes = plt.subplots(1, len(CATEGORICAL), figsize=(15, 3.4), sharex=True)
for ax, col in zip(axes, CATEGORICAL):
    g = gaps[col]
    colors = ["#E8684A" if abs(v) > FLAG_THRESHOLD_PP else "#5B8FF9" for v in g.values]
    ax.barh([str(i) for i in g.index], g.values, color=colors)
    ax.axvline(0, color="gray", linewidth=1)
    for x in (-FLAG_THRESHOLD_PP, FLAG_THRESHOLD_PP):
        ax.axvline(x, color="#E8684A", linestyle="--", linewidth=1)
    ax.set_title(col)
    ax.set_xlabel("Treatment − control share (pp)")
fig.suptitle("Covariate balance: share gap by category (dashed = ±1pp flag)", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_categorical_balance.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/01_categorical_balance.png")

# Continuous covariate — Welch's t-test
print("\n=== Covariate balance (continuous: prior_watch_hours) ===")
c = df.loc[df["group"] == "control", "prior_watch_hours"]
t = df.loc[df["group"] == "treatment", "prior_watch_hours"]
t_stat, p_val = stats.ttest_ind(c, t, equal_var=False)
print(f"Control   mean: {c.mean():.4f}  std: {c.std():.4f}  n: {len(c):,}")
print(f"Treatment mean: {t.mean():.4f}  std: {t.std():.4f}  n: {len(t):,}")
print(f"Welch t-test:   t = {t_stat:+.3f}, p = {p_val:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, np.percentile(df["prior_watch_hours"], 99), 60)
ax.hist(c, bins=bins, density=True, alpha=0.5, color="#5B8FF9", label="Control")
ax.hist(t, bins=bins, density=True, alpha=0.5, color="#F6BD16", label="Treatment")
ax.set_yscale("log")
ax.set_xlabel("Prior watch hours (30 days before trial)")
ax.set_ylabel("Density (log scale)")
ax.set_title(f"Continuous covariate balance (Welch p = {p_val:.2f})", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "01_continuous_balance.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/01_continuous_balance.png")


# ---------------------------------------------------------------------
# 6. Verdict
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("DATA QUALITY VERDICT")
print("=" * 70)
if flagged_cols:
    print(f"⚠️  Imbalance detected in: {flagged_cols}")
    print("   Combined with the borderline SRM result, this is consistent "
          "with a\n   segment-specific assignment leak. Downstream analysis "
          "should report\n   ITT as primary and include a sensitivity "
          "analysis excluding the affected\n   segment to demonstrate "
          "robustness of the headline conclusion.")
else:
    print("✅ Randomization checks pass. Proceeding to inference.")
