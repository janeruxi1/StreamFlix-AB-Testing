# %% [markdown]
# # 04 · A testing programme: can a sample-ratio check find broken randomization?
#
# Part 2 judges Upworthy's whole headline-testing programme, 2013–2015, from the Upworthy Research
# Archive (CC BY 4.0). The plan ([`reports/upworthy_analysis_plan.md`](../reports/upworthy_analysis_plan.md))
# was written on the exploratory sample and committed before this notebook ran on the
# confirmatory sample.
#
# In June 2024 the archive's maintainers reported that tests run between 25 June 2013 and
# 10 January 2014 may have broken randomization. That gives this notebook something rare: an
# answer key. If a routine sample-ratio check works, it should flag tests inside that window far
# more often than outside it, without being told where the window is.
#
# **The check:** each test should split its sessions evenly across its packages. Chi-square of
# each test's package impressions against an equal split; flag at p < 0.001.
# **Confirms if** the flag rate inside the window is at least ten times the rate outside.

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

from src import records
from src.analysis import programme
from src.data import upworthy

SAMPLE = os.environ.get("UPWORTHY_SAMPLE", "confirmatory")
df = upworthy.load(SAMPLE)
srm = programme.srm_by_test(df)
srm["flagged"] = srm["srm_p"] < 0.001
print(f"{SAMPLE}: {srm.shape[0]:,} tests, {len(df):,} packages, "
      f"{srm['in_window'].mean():.1%} of tests inside the window")

# %% [markdown]
# ## 1. Flag rate inside and outside the window

# %%
rates = srm.groupby("in_window")["flagged"].agg(["mean", "sum", "size"]).rename(
    index={False: "outside the window", True: "inside the window"})
ratio = rates.loc["inside the window", "mean"] / rates.loc["outside the window", "mean"]
confirmed = bool(ratio >= 10)
print(f"flag rate inside {rates.loc['inside the window', 'mean']:.1%}, outside "
      f"{rates.loc['outside the window', 'mean']:.2%}; ratio {ratio:.0f}x -> "
      + ("confirms" if confirmed else "does not confirm"))
rates

# %% [markdown]
# ## 2. Month by month
#
# If the check is finding the maintainers' problem, the flag rate should jump at the start of
# the window and fall back at its end, not drift.

# %%
srm["month"] = srm["test_start"].dt.to_period("M").dt.to_timestamp()
monthly = srm.groupby("month").agg(tests=("flagged", "size"), flag_rate=("flagged", "mean"))
outside_flags = srm[~srm["in_window"] & srm["flagged"]]
print(f"flags outside the window: {len(outside_flags)} tests "
      f"({len(outside_flags) / max(1, srm['flagged'].sum()):.1%} of all flags); "
      f"at p < 0.001 about {0.001 * (~srm['in_window']).sum():.0f} would be expected by chance")
monthly.tail(30).round(3)

# %%
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.axvspan(upworthy.WINDOW_START, upworthy.WINDOW_END, color="#F6BD16", alpha=0.25, lw=0)
ax.annotate("maintainers' problem window\n(25 Jun 2013 – 10 Jan 2014)", (upworthy.WINDOW_START, 0.97),
            xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points", va="top", fontsize=8)
ax.bar(monthly.index, monthly["flag_rate"], width=25, color="#5B8FF9")
ax.set(ylim=(0, 1.08), ylabel="share of tests flagged", title="Tests failing the sample-ratio check (p < 0.001), by month started")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(alpha=0.25, axis="y")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
records.save_figure(fig, f"04_sample_ratio_{SAMPLE}.png", "real")

# %%
record = {"sample": SAMPLE, "tests": int(len(srm)), "packages": int(len(df)),
          "share_in_window": float(srm["in_window"].mean()),
          "flag_rate_inside": float(rates.loc["inside the window", "mean"]),
          "flag_rate_outside": float(rates.loc["outside the window", "mean"]),
          "flags_inside": int(rates.loc["inside the window", "sum"]),
          "flags_outside": int(rates.loc["outside the window", "sum"]),
          "ratio": float(ratio), "confirmed": confirmed,
          "monthly": {d.strftime("%Y-%m"): {"tests": int(r.tests), "flag_rate": float(r.flag_rate)}
                      for d, r in monthly.iterrows()}}
print("wrote", records.write(f"04_sample_ratio_{SAMPLE}", record, "real").name)
print(f"Verdict: the check flags {record['flag_rate_inside']:.0%} of tests inside the window and "
      f"{record['flag_rate_outside']:.1%} outside, {ratio:.0f} times as often; "
      + ("the plan's test is passed." if confirmed else "the plan's test is not passed."))
