# %% [markdown]
# # 01 · The e-mail test: what could it detect, and can it be trusted?
#
# Kevin Hillstrom's e-mail test (MineThatData, 2008): 64,000 customers who bought in the past 12
# months were randomized in equal thirds to a men's merchandise e-mail, a women's merchandise
# e-mail, or no e-mail, and their visits, purchases and spend were recorded for two weeks.
#
# This notebook covers the first two stages of the experiment: **design** (was the test big
# enough to answer the question?) and **trust** (did the randomization work?). The trust checks
# are the ones the analysis plan
# ([`reports/hillstrom_analysis_plan.md`](../reports/hillstrom_analysis_plan.md)) fixes in
# advance; a failure there stops the readout.
#
# 0. **Design:** the smallest effect this test could detect, and the size it needed.
# 1. **Sample ratio:** did each arm get a third of the customers?
# 2. **Balance:** do the arms look alike on everything known before the send?
# 3. **Data:** missing values and impossible combinations (spend without a purchase, a purchase
#    without a visit).

# %%
import sys
from pathlib import Path


def _root() -> Path:
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "src" / "data" / "hillstrom.py").exists():
            return p
    raise FileNotFoundError("run from inside the project folder")


ROOT = _root()
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src import records
from src.analysis import readout
from src.data import hillstrom

df, SOURCE = hillstrom.load()
raw = hillstrom.find_raw()
print(f"source: {SOURCE}; {len(df):,} customers"
      + (f"; file {raw.name}, sha256 {hillstrom.sha256(raw)[:16]}..." if SOURCE == "real" and raw else ""))
df.head()

# %% [markdown]
# ## 0. Design: what could this test detect?
#
# A test should be sized before it runs, for the smallest effect worth acting on. This one was
# run in 2008, so this is a design review after the fact: baselines and spreads come from the
# no-e-mail arm. (Two-sided 5% test, 80% power, equal arms.)
#
# The effect worth acting on comes from the decision rule in notebook 02: with a 40% margin and
# $0.10 to send an e-mail, an e-mail pays for itself if it lifts spend by at least
# $0.10 / 0.40 = **$0.25 per customer**.

# %%
MARGIN, COST = 0.40, 0.10                              # the plan's assumptions (used again in notebook 02)
ctrl = df[df["arm"] == "none"]
n_arm = int(df["arm"].value_counts().min())
worth = {"visit": 0.01, "conversion": 0.002, "spend": COST / MARGIN}   # +1 pt, +0.2 pt, break-even spend
design = []
for m, w in worth.items():
    sd = float(ctrl[m].std())
    design.append({"metric": m, "baseline": float(ctrl[m].mean()), "sd": sd,
                   "mde_at_this_size": readout.mde(sd, n_arm), "effect_worth_detecting": w,
                   "n_per_arm_needed": readout.n_per_arm(sd, w)})
design = pd.DataFrame(design).set_index("metric")
d = design.loc["spend"]
design["n_per_arm_needed"] = design["n_per_arm_needed"].astype(int)
d = design.loc["spend"]
print(f"{n_arm:,} customers per arm: the test reliably detects a spend lift of ${d['mde_at_this_size']:.2f} per "
      f"customer; the break-even lift of ${d['effect_worth_detecting']:.2f} needs {int(d['n_per_arm_needed']):,} per arm. "
      + ("An e-mail that only just pays for itself could be missed; a clearly profitable one would not."
         if d["mde_at_this_size"] > d["effect_worth_detecting"] else "The test is big enough for the break-even lift."))
design.round(4)

# %% [markdown]
# Spend is the hard metric: a few large orders make it noisy, so it needs far more customers than
# visits do for an effect worth the same money. Notebook 02 checks how much regression adjustment
# on pre-send variables narrows the interval (the CUPED idea).

# %% [markdown]
# ## 1. Sample ratio

# %%
counts = {a: int((df["arm"] == a).sum()) for a in hillstrom.ARM_ORDER}
srm = readout.srm(counts)
print(f"arm counts {counts}; chi-square {srm['chi2']:.3f}, p = {srm['p_value']:.3f} -> "
      + ("passes" if srm["passed"] else "FAILS (p < 0.001): stop and investigate"))

# %% [markdown]
# ## 2. Balance on what was known before the send
#
# The largest standardized mean difference between any two arms, for every pre-send variable
# (categories expanded to one indicator per level). The plan's threshold is 0.05; anything above
# it would be added to the regression adjustment (the planned adjustment already includes every
# variable below except `history_segment`, which is `history` cut into bands).

# %%
balance = readout.smd(df, ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel",
                           "history_segment"])
worst = balance.iloc[0]
print(f"largest |SMD|: {worst.name} {worst['max_abs_smd']:.4f}; "
      f"{int((balance['max_abs_smd'] > 0.05).sum())} variable(s) above 0.05")
balance.head(8).round(4)

# %% [markdown]
# ## 3. Data checks

# %%
checks = {
    "missing values": int(df[hillstrom.COLUMNS[:-4] + hillstrom.OUTCOMES].isna().sum().sum()),
    "negative spend": int((df["spend"] < 0).sum()),
    "spend without a purchase": int(((df["spend"] > 0) & (df["conversion"] == 0)).sum()),
    "purchase without spend": int(((df["spend"] == 0) & (df["conversion"] == 1)).sum()),
    "purchase without a visit": int(((df["conversion"] == 1) & (df["visit"] == 0)).sum()),
}
pd.Series(checks, name="rows")

# %% [markdown]
# ## 4. What the outcomes look like
#
# Shown for orientation only, by arm. Spend is mostly zeros with a long tail, which is why the
# plan caps it at the 99th percentile of positive spend, computed on all arms pooled.

# %%
cap = readout.spend_cap(df["spend"])
pos = df.loc[df["spend"] > 0, "spend"]
profile = {"buyers": int((df["spend"] > 0).sum()), "share_buying": float((df["spend"] > 0).mean()),
           "spend_per_buyer_median": float(pos.median()), "spend_per_buyer_max": float(pos.max()),
           "cap_p99_positive": cap, "buyers_above_cap": int((df["spend"] > cap).sum())}
print({k: round(v, 4) if isinstance(v, float) else v for k, v in profile.items()})

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].hist(pos.clip(upper=pos.quantile(0.995)), bins=60, color="#5B8FF9")
axes[0].axvline(cap, color="#2B2B2B", ls="--", lw=1)
axes[0].set(title="Spend among buyers", xlabel="dollars in the two weeks", ylabel="buyers")
axes[0].annotate(f"cap ${cap:,.0f}", (cap, axes[0].get_ylim()[1] * 0.9), xytext=(-6, 0),
                 textcoords="offset points", ha="right", fontsize=9)
top = balance.head(10)[::-1]
axes[1].barh(top.index, top["max_abs_smd"], color="#5AD8A6")
axes[1].axvline(0.05, color="#2B2B2B", ls="--", lw=1)
axes[1].set(title="Balance: largest |SMD| between arms", xlabel="standardized mean difference")
for ax in axes:
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
records.save_figure(fig, "01_email_test_checks.png", SOURCE)

# %%
record = {"n": len(df), "design": design.reset_index().to_dict(orient="records"), "srm": srm, "balance_max_abs_smd": balance["max_abs_smd"].round(6).to_dict(),
          "balance_above_0_05": int((balance["max_abs_smd"] > 0.05).sum()), "data_checks": checks,
          "spend_profile": profile}
print("wrote", records.write("01_email_test_checks", record, SOURCE).name)

failed = [k for k, v in checks.items() if v and k != "missing values"]
verdict = (f"Sample ratio p = {srm['p_value']:.3f} ({'passes' if srm['passed'] else 'FAILS'}); "
           f"largest imbalance |SMD| {worst['max_abs_smd']:.3f} ({worst.name}); "
           + ("no impossible rows." if not failed else f"problems: {', '.join(failed)}."))
print(verdict)
