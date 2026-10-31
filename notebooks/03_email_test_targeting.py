# %% [markdown]
# # 03 · The e-mail test: should the e-mail differ by customer?
#
# Splitting a test's results by segment and keeping the segment where the treatment looks best
# is the most common way to find an effect that is not there. The analysis plan
# ([`reports/hillstrom_analysis_plan.md`](../reports/hillstrom_analysis_plan.md)) guards against
# it by searching on one random half of the customers and confirming on the other.
#
# **On the discovery half:** for each pre-defined segment variable (purchase line, newbie,
# channel, zip code, history band), give every level the e-mail with the largest spend lift, or
# no e-mail when neither lifts spend. That is one candidate policy per variable. The candidate
# with the highest value on the discovery half is carried forward, against the best single
# choice for everyone on that half. (Choosing among the candidates on the discovery half is how
# this notebook implements "the targeting policy" of the plan; the confirmation half is not
# looked at until both are fixed.)
#
# **On the confirmation half:** the value of each policy is estimated by inverse-probability
# weighting (a customer counts when the arm they were randomized to is the one the policy would
# have chosen, weighted by 3). The targeting policy is recommended only if its gain over the
# single choice has a 95% bootstrap interval above zero.

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
import numpy as np
import pandas as pd

from src import records
from src.analysis import readout
from src.data import hillstrom

df, SOURCE = hillstrom.load()
CAP = readout.spend_cap(df["spend"])                 # same pooled cap as notebook 02
df["spend_capped"] = df["spend"].clip(upper=CAP)
SEED = 20261030                                      # fixed in the plan
SEGMENTS = ["purchase_line", "newbie", "channel", "zip_code", "history_segment"]
EMAILS, P_ARM, Y = ["mens", "womens"], 1 / 3, "spend_capped"
disc, conf = readout.split_halves(df, SEED)
print(f"source: {SOURCE}; discovery {len(disc):,}, confirmation {len(conf):,} customers")

# %% [markdown]
# ## 1. Discovery half: one candidate policy per segment variable

# %%
def assign(frame, policy, segment):
    return frame[segment].map(policy).astype(str).to_numpy()


candidates, rows = {}, []
for seg in SEGMENTS:
    pol = readout.segment_policy(disc, seg, Y, EMAILS, "none")
    candidates[seg] = {str(k): v for k, v in pol.items()}
    v = readout.ipw_value(disc, assign(disc, pol, seg), Y, P_ARM).mean()
    rows.append({"segment variable": seg, "policy": candidates[seg], "discovery value per customer": v})
single_values = {a: readout.ipw_value(disc, np.full(len(disc), a), Y, P_ARM).mean() for a in hillstrom.ARM_ORDER}
best_single = max(single_values, key=single_values.get)
cand = pd.DataFrame(rows).sort_values("discovery value per customer", ascending=False)
chosen = cand.iloc[0]["segment variable"]
print("best single choice on discovery:", best_single, {k: round(v, 3) for k, v in single_values.items()})
print("candidate carried forward:", chosen, candidates[chosen])
cand.round(3)

# %% [markdown]
# ## 2. Confirmation half: does targeting beat one e-mail for everyone?

# %%
pol_assign = conf[chosen].astype(str).map(candidates[chosen]).fillna(best_single).to_numpy()
single_assign = np.full(len(conf), best_single)
confirm = readout.policy_gain(conf, pol_assign, single_assign, Y, P_ARM, n_boot=2000, seed=1)
conf_value = {"targeted": readout.ipw_value(conf, pol_assign, Y, P_ARM).mean(),
              "single": readout.ipw_value(conf, single_assign, Y, P_ARM).mean()}
recommend = confirm["ci"][0] > 0
print(f"targeted {conf_value['targeted']:.3f} vs {best_single} for everyone {conf_value['single']:.3f} per customer; "
      f"gain {confirm['gain']:+.3f} [{confirm['ci'][0]:+.3f}, {confirm['ci'][1]:+.3f}] -> "
      + ("recommend targeting" if recommend else "do not target"))

# %% [markdown]
# ## 3. For reading only: every candidate on the confirmation half
#
# Not part of the decision (the confirmation half was used for one comparison only); shown so the
# reader can see how much each candidate's discovery value shrinks on new customers.

# %%
shrink = []
for seg in SEGMENTS:
    a = conf[seg].astype(str).map(candidates[seg]).fillna(best_single).to_numpy()
    shrink.append({"segment variable": seg,
                   "discovery gain over single": cand.set_index("segment variable").loc[seg, "discovery value per customer"]
                   - single_values[best_single],
                   "confirmation gain over single": readout.ipw_value(conf, a, Y, P_ARM).mean() - conf_value["single"]})
shrink = pd.DataFrame(shrink)
shrink.round(3)

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.4))
y = np.arange(len(shrink))
ax.barh(y - 0.18, shrink["discovery gain over single"], height=0.34, color="#BFBFBF", label="discovery half")
ax.barh(y + 0.18, shrink["confirmation gain over single"], height=0.34, color="#5B8FF9", label="confirmation half")
ax.axvline(0, color="#2B2B2B", lw=0.8)
ax.set_yticks(y, shrink["segment variable"])
ax.invert_yaxis()
ax.set(xlabel="spend per customer over the best single choice ($)",
       title="Targeting by segment: what it seemed to gain, and what held up")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25, axis="x")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
records.save_figure(fig, "03_email_test_targeting.png", SOURCE)

# %%
record = {"seed": SEED, "segments": SEGMENTS, "candidates": candidates,
          "discovery_values": cand.set_index("segment variable")["discovery value per customer"].to_dict(),
          "single_values_discovery": single_values, "best_single": best_single, "chosen": chosen,
          "confirmation": {**confirm, **conf_value}, "recommend_targeting": bool(recommend),
          "shrinkage": shrink.set_index("segment variable").to_dict(orient="index")}
print("wrote", records.write("03_email_test_targeting", record, SOURCE).name)

lines = [
    f"Best single choice on the discovery half: {best_single}.",
    f"Best targeting candidate on the discovery half: by {chosen}, {candidates[chosen]}.",
    f"On the confirmation half it gains {confirm['gain']:+.3f} per customer [{confirm['ci'][0]:+.3f}, "
    f"{confirm['ci'][1]:+.3f}] over {best_single} for everyone: "
    + ("targeting is recommended." if recommend else "not enough to recommend targeting."),
]
print("\n".join(f"{i}. {t}" for i, t in enumerate(lines, 1)))
