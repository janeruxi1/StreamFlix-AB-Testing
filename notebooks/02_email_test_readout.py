# %% [markdown]
# # 02 · The e-mail test: what did each e-mail do?
#
# Every choice below was fixed in the analysis plan
# ([`reports/hillstrom_analysis_plan.md`](../reports/hillstrom_analysis_plan.md)) before this
# notebook ran: the metrics, the spend cap, the covariates for regression adjustment, the three
# comparisons and their multiple-testing correction, and the decision rule.
#
# - **Primary:** spend per customer over the two weeks (intention to treat), capped at the 99th
#   percentile of positive spend.
# - **Secondary:** conversion rate and visit rate.
# - **Comparisons:** men's vs no e-mail, women's vs no e-mail, men's vs women's; Holm at 5% within
#   each metric.
# - **Adjustment:** Lin (2013) regression adjustment on the pre-send variables, HC2 errors; the
#   decision uses the adjusted estimate, the unadjusted one is shown beside it.

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
from scipy import stats

from src import records
from src.analysis import readout
from src.data import hillstrom

df, SOURCE = hillstrom.load()
COVARIATES = ["history", "recency", "mens", "womens", "newbie", "zip_code", "channel"]   # fixed in the plan
MARGIN, COST = 0.40, 0.10                                                                # assumptions in the plan
COMPARISONS = [("mens", "none"), ("womens", "none"), ("mens", "womens")]
CAP = readout.spend_cap(df["spend"])
df["spend_capped"] = df["spend"].clip(upper=CAP)
print(f"source: {SOURCE}; {len(df):,} customers; spend cap ${CAP:,.2f}")

# %% [markdown]
# ## 1. The effects
#
# Each e-mail against no e-mail, and the two e-mails against each other, on every metric. Spend is
# shown capped (the primary) and uncapped.

# %%
METRICS = {"spend_capped": "Spend per customer, capped ($)", "spend": "Spend per customer, uncapped ($)",
           "conversion": "Conversion rate", "visit": "Visit rate"}
rows, effects = [], {}
for m in METRICS:
    adj = [readout.lin_adjusted(df, m, t, c, COVARIATES) for t, c in COMPARISONS]
    raw = [readout.diff_in_means(df, m, t, c) for t, c in COMPARISONS]
    holm = readout.holm(adj)
    effects[m] = {"adjusted": adj, "unadjusted": raw, "holm": holm}
    for a, r, h in zip(adj, raw, holm):
        rows.append({"metric": METRICS[m], "comparison": f"{a.treatment} vs {a.control}",
                     "control mean": a.mean_c, "adjusted diff": a.diff, "95% CI low": a.ci[0],
                     "95% CI high": a.ci[1], "relative": a.rel, "p": a.p_value, "Holm 5%": h,
                     "unadjusted diff": r.diff, "CI narrower by": 1 - a.se / r.se})
table = pd.DataFrame(rows)
table.round(4)

# %% [markdown]
# ## 2. What regression adjustment bought
#
# The pre-send variables explain part of each customer's outcome, so removing that part narrows
# the interval without moving the estimate (the arms were randomized, so the adjustment should
# change the point estimate by little). The width of the interval is reported, and the
# equivalent extra sample: an interval that is narrower by a share `s` would otherwise need
# 1/(1-s)^2 times as many customers.

# %%
gain = table.groupby("metric", sort=False)["CI narrower by"].mean()
gain_sample = 1 / (1 - gain) ** 2 - 1
pd.DataFrame({"CI narrower by": gain, "equivalent extra sample": gain_sample}).round(3)

# %% [markdown]
# ## 3. The decision rule, applied
#
# From the plan:
#
# 1. An e-mail is worth sending to everyone only if its adjusted, capped spend lift over no
#    e-mail passes Holm and the lift times the margin (40%) exceeds the cost of sending ($0.10 an
#    e-mail).
# 2. If both pass, the higher one is chosen only if men's vs women's also passes Holm; otherwise
#    the memo says they cannot be told apart.
# 3. Targeting by customer is decided in notebook 03.

# %%
prim = {f"{e.treatment} vs {e.control}": (e, h) for e, h in zip(effects["spend_capped"]["adjusted"],
                                                                  effects["spend_capped"]["holm"])}
value = {}
for arm in ("mens", "womens"):
    e, h = prim[f"{arm} vs none"]
    value[arm] = {"lift": e.diff, "ci": list(e.ci), "holm_pass": bool(h),
                  "value_per_1000": readout.value_per_1000(e.diff, MARGIN, COST),
                  "value_per_1000_ci": [readout.value_per_1000(x, MARGIN, COST) for x in e.ci],
                  "breakeven_margin": readout.breakeven_margin(e.diff, COST),
                  "passes": bool(h and e.diff * MARGIN > COST)}
head, head_pass = prim["mens vs womens"]
passing = [a for a in ("mens", "womens") if value[a]["passes"]]
if not passing:
    decision = "none"
elif len(passing) == 1:
    decision = passing[0]
else:
    better = "mens" if head.diff > 0 else "womens"
    decision = better if head_pass else "either"
print({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in value.items()})
print(f"men's vs women's: {head.diff:+.3f} [{head.ci[0]:+.3f}, {head.ci[1]:+.3f}], Holm pass {bool(head_pass)}")
print("decision under the plan's rule:", decision)

# %% [markdown]
# ## 4. Figure

# %%
LABEL = {"mens vs none": "Men's vs no e-mail", "womens vs none": "Women's vs no e-mail",
         "mens vs womens": "Men's vs women's"}
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), sharey=True)
for ax, m, unit in zip(axes, ["spend_capped", "conversion", "visit"], ["$ per customer", "points", "points"]):
    scale = 1 if m.startswith("spend") else 100
    for i, (a, r) in enumerate(zip(effects[m]["adjusted"], effects[m]["unadjusted"])):
        key = f"{a.treatment} vs {a.control}"
        ax.plot([a.ci[0] * scale, a.ci[1] * scale], [i, i], color="#5B8FF9", lw=3)
        ax.plot(a.diff * scale, i, "o", color="#5B8FF9", ms=7, mec="white", mew=1.5)
        ax.plot([r.ci[0] * scale, r.ci[1] * scale], [i + 0.22, i + 0.22], color="#BFBFBF", lw=2)
    ax.axvline(0, color="#2B2B2B", lw=0.8)
    ax.set_yticks(range(3), [LABEL[f"{t} vs {c}"] for t, c in COMPARISONS])
    ax.invert_yaxis()
    ax.set(title=METRICS[m].split(",")[0], xlabel=f"difference ({unit})")
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("The e-mail test: effect of each e-mail over two weeks", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.text(0.99, 0.97, "blue: regression-adjusted 95% CI · grey: unadjusted", ha="right", va="top",
         fontsize=9, color="#555555")
records.save_figure(fig, "02_email_test_effects.png", SOURCE)

# %%
record = {
    "cap": CAP, "covariates": COVARIATES, "margin_assumed": MARGIN, "cost_per_email_assumed": COST,
    "effects": {m: [{**a.as_dict(), "holm_pass": bool(h), "unadjusted_diff": r.diff,
                     "unadjusted_ci": list(r.ci), "ci_narrower_by": 1 - a.se / r.se}
                    for a, r, h in zip(v["adjusted"], v["unadjusted"], v["holm"])] for m, v in effects.items()},
    "value": value, "head_to_head": {"diff": head.diff, "ci": list(head.ci), "holm_pass": bool(head_pass)},
    "decision": decision,
}
print("wrote", records.write("02_email_test_readout", record, SOURCE).name)

# %% [markdown]
# ## Verdict
#
# Rendered from the record written above.

# %%
DECISION_TEXT = {"none": "send nothing", "mens": "send the men's e-mail to everyone",
                 "womens": "send the women's e-mail to everyone",
                 "either": "send either e-mail; they cannot be told apart"}


def fmt(v):
    return f"{v['lift']:+.2f} per customer [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]"


lines = [
    f"Men's e-mail: spend {fmt(value['mens'])}; worth ${value['mens']['value_per_1000']:,.0f} per 1,000 "
    f"customers e-mailed at a 40% margin (break-even margin {value['mens']['breakeven_margin']:.0%}).",
    f"Women's e-mail: spend {fmt(value['womens'])}; ${value['womens']['value_per_1000']:,.0f} per 1,000 "
    f"(break-even margin {value['womens']['breakeven_margin']:.0%}).",
    f"Men's against women's: {head.diff:+.2f} [{head.ci[0]:+.2f}, {head.ci[1]:+.2f}]; "
    + ("the difference passes Holm." if head_pass else "the two cannot be told apart."),
    f"Regression adjustment narrowed the spend intervals by {gain[METRICS['spend_capped']]:.0%} on average, "
    f"worth {gain_sample[METRICS['spend_capped']]:.0%} more customers.",
    f"Decision under the plan's rule: {DECISION_TEXT[decision]}.",
]
print("\n".join(f"{i}. {t}" for i, t in enumerate(lines, 1)))

# %% [markdown]
# ## 5. Rollout and follow-up
#
# The decision is to e-mail everyone, but the e-mail's value has to keep being measured: once
# everyone gets it, there is no comparison group left. The usual answer is a **permanent
# holdback**: a small random share of customers who never get the campaign e-mail.
#
# How small can it be? With 64,000 customers per send, the table shows the smallest spend lift
# each holdback size detects in one send, and how many sends must be pooled before it can detect
# **half** the lift measured here (an early warning that the e-mail is wearing out).

# %%
LIST = len(df)
ctrl_sd = float(df.loc[df["arm"] == "none", "spend_capped"].std())
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
ARM = decision if decision in ("mens", "womens") else "mens"   # the e-mail being rolled out
target = value[ARM]["lift"] / 2
holdback = []
for share in (0.02, 0.05, 0.10, 0.20):
    n_c, n_t = LIST * share, LIST * (1 - share)
    per_send = z * ctrl_sd * np.sqrt(1 / n_c + 1 / n_t)
    holdback.append({"holdback_share": share, "customers_held_back": int(n_c), "mde_one_send": per_send,
                     "sends_to_detect_half_the_lift": int(np.ceil((per_send / target) ** 2)),
                     "margin_given_up_per_send": share * LIST * value[ARM]["lift"] * MARGIN})
holdback = pd.DataFrame(holdback)
pick = holdback[holdback["holdback_share"] == 0.05].iloc[0]
print(f"a 5% holdback ({int(pick['customers_held_back']):,} customers) gives up about "
      f"${pick['margin_given_up_per_send']:,.0f} of margin a send and detects half the measured lift "
      f"after {int(pick['sends_to_detect_half_the_lift'])} sends pooled")
holdback.round(3)

# %% [markdown]
# **Guardrails to watch during rollout.** This data has none, but a real rollout would track
# unsubscribes and spam complaints per send (an e-mail that sells this week but loses the
# address costs every later send), deliverability, and whether e-mailing men's merchandise to
# women's-line buyers shifts their spending rather than adding to it.
#
# **The next test.** Men's against women's was the closest call. The natural next test is the
# men's e-mail against a new version, for example one that features the customer's own line.
# Sized with notebook 01's calculation, to detect a difference worth the break-even amount:

# %%
mens_sd = float(df.loc[df["arm"] == "mens", "spend_capped"].std())
next_test = {"effect_to_detect": COST / MARGIN, "n_per_arm": readout.n_per_arm(mens_sd, COST / MARGIN),
             "n_per_arm_for_head_to_head_seen_here": readout.n_per_arm(mens_sd, abs(head.diff))}
next_test["sends_needed"] = 2 * next_test["n_per_arm"] / LIST
print(f"two arms of {next_test['n_per_arm']:,} customers ({next_test['sends_needed']:.1f} sends of this list) "
      f"to detect a ${next_test['effect_to_detect']:.2f} difference in spend. That is more than notebook 01's "
      f"{readout.n_per_arm(ctrl_sd, COST / MARGIN):,}, because spend varies more among e-mailed customers "
      f"(sd ${mens_sd:.2f} against ${ctrl_sd:.2f}).")
rollout = {"holdback": holdback.to_dict(orient="records"), "next_test": next_test}
record["rollout"] = rollout
print("updated", records.write("02_email_test_readout", record, SOURCE).name)
