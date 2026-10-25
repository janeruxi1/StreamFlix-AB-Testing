"""StreamFlix A/B test -- interactive companion app.

Three tabs:
  1. The StreamFlix result   -- the experiment's readout, recomputed from data/experiment.csv
  2. Plan a test             -- sample size, runtime and detectable effect
  3. Analyse your own results -- frequentist + Bayesian readout on any counts

Every number comes from the same `src/analysis/` functions the notebooks use
and the test suite checks.
"""
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Make src/ importable when Streamlit runs from the app/ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.bayesian import bayesian_ab_binary
from src.analysis.frequentist import two_proportion_test, welch_t_test
from src.analysis.power import (
    mde_for_sample_size,
    runtime_days,
    sample_size_two_proportion,
)
from src.analysis.sanity_checks import check_srm
from src.analysis.segmentation import segment_lifts
from src.data.loader import load_experiment

REPO_URL = "https://github.com/janeruxi1/StreamFlix-AB-Testing"
DATA_PATH = PROJECT_ROOT / "data" / "experiment.csv"

BLUE, ORANGE, GREY = "#2C5282", "#DD6B20", "#A0AEC0"

st.set_page_config(page_title="StreamFlix A/B Test", page_icon="🧪", layout="wide")


# ---------------------------------------------------------------------
# Data: the experiment readout, computed once
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Reading the experiment data...")
def experiment_readout():
    if not DATA_PATH.exists():  # fresh clone without the CSV
        from src.data.simulate import main as simulate_main
        simulate_main(DATA_PATH)
    df = load_experiment(DATA_PATH).dropna(subset=["group", "converted"])
    ctl, trt = df[df["group"] == "control"], df[df["group"] == "treatment"]

    def conv(c, t, name):
        return two_proportion_test(
            successes_control=int(c["converted"].sum()), n_control=len(c),
            successes_treatment=int(t["converted"].sum()), n_treatment=len(t),
            name=name,
        )

    primary = conv(ctl, trt, "conversion")
    no_android = df[df["device"] != "Android"]
    sens = conv(no_android[no_android["group"] == "control"],
                no_android[no_android["group"] == "treatment"],
                "conversion excl. Android")
    day7 = two_proportion_test(
        successes_control=int(ctl["day7_active"].sum()), n_control=len(ctl),
        successes_treatment=int(trt["day7_active"].sum()), n_treatment=len(trt),
        name="day-7 active",
    )
    load = welch_t_test(ctl["page_load_ms"].values, trt["page_load_ms"].values,
                        name="page load")
    srm = check_srm(df)
    bayes = bayesian_ab_binary(
        successes_control=int(ctl["converted"].sum()), n_control=len(ctl),
        successes_treatment=int(trt["converted"].sum()), n_treatment=len(trt),
        n_samples=40_000,
    )
    segments = {}
    for col, label in [("device", "Device"), ("country", "Country"),
                       ("source", "Acquisition source"),
                       ("is_returning", "New or returning")]:
        seg = segment_lifts(df, segment_col=col, metric_col="converted")
        seg["segment"] = seg["segment"].astype(str)
        if col == "is_returning":
            seg["segment"] = seg["segment"].map(
                {"True": "Returning", "False": "New"}).fillna(seg["segment"])
        segments[label] = seg
    counts = {"n_ctl": len(ctl), "k_ctl": int(ctl["converted"].sum()),
              "n_trt": len(trt), "k_trt": int(trt["converted"].sum())}
    return dict(primary=primary, sens=sens, day7=day7, load=load, srm=srm,
                p_better=bayes.p_treatment_better, segments=segments,
                counts=counts)


R = experiment_readout()
C = R["counts"]


def p_fmt(p: float) -> str:
    """p-value to 4 decimals, rounded half-up as in the reports."""
    return str(Decimal(f"{p:.6f}").quantize(Decimal("0.0001"), ROUND_HALF_UP))


def pct_better(p: float) -> str:
    return ">99.99%" if p >= 0.9999 else f"{p:.2%}"


def pp(x: float, signed: bool = True) -> str:
    return f"{x * 100:+.2f}pp" if signed else f"{x * 100:.2f}pp"


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🧪 StreamFlix Homepage A/B Test")
st.markdown(
    "Should StreamFlix ship a **personalised “Recommended For You” homepage** to "
    "every free-trial user? 100,000 trialists were split 50/50 for four weeks. "
    "This app shows the result and lets you plan or analyse a test of your own. "
    "*Synthetic data with a known true effect, so every estimate can be checked.*"
)

with st.expander("ℹ️ How to use this demo", expanded=False):
    st.markdown(
        """
1. **🏁 The StreamFlix result**: the decision, the evidence behind it and the
   checks a reviewer would ask about (sample ratio, guardrails, segments).
2. **📐 Plan a test**: choose a baseline rate and the smallest lift worth
   detecting; see how many users and days you need.
3. **🔬 Analyse your own results**: type in users and conversions for each arm
   (pre-filled with the StreamFlix numbers) to get a frequentist and a
   Bayesian readout side by side.

Hover over the **?** icons for definitions. *pp* = percentage points
(21% → 23% is +2pp).
"""
    )

tab_result, tab_plan, tab_analyse = st.tabs(
    ["🏁 The StreamFlix result", "📐 Plan a test", "🔬 Analyse your own results"]
)


# =====================================================================
# TAB 1 -- THE STREAMFLIX RESULT
# =====================================================================
with tab_result:
    p, s = R["primary"], R["sens"]
    st.success(
        f"**Decision: ship the personalised homepage**, with an engineering ticket "
        f"for its {R['load'].effect_absolute:+.0f} ms page-load regression. "
        f"Trial-to-paid conversion rose from {p.p_control:.2%} to "
        f"{p.p_treatment:.2%}."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Conversion lift", pp(p.effect_absolute),
              f"{p.effect_relative:+.1%} relative", border=True,
              help="Treatment minus control, trial-to-paid conversion within 14 days.")
    k2.metric("95% confidence interval",
              f"{p.ci_lower * 100:+.2f} to {p.ci_upper * 100:+.2f}pp", border=True,
              help="The whole interval is above the +1pp lift the PM said was worth shipping.")
    k3.metric("Chance personalised is better", pct_better(R["p_better"]), border=True,
              help="Bayesian posterior probability, flat Beta(1,1) prior.")
    k4.metric("Page load (guardrail)", f"{R['load'].effect_absolute:+.1f} ms",
              f"{R['load'].effect_relative:+.1%}", delta_color="inverse", border=True,
              help="A real slowdown, but inside the 50 ms guardrail and below the "
                   "~100 ms users tend to notice.")

    st.subheader("Checks before trusting the result")
    srm = R["srm"]
    checks = pd.DataFrame([
        {"Check": "Sample ratio (50/50 split)",
         "Result": f"{C['n_ctl']:,} vs {C['n_trt']:,} users, p = {p_fmt(srm.p_value)}",
         "Verdict": "⚠️ Passes the 0.001 alarm, not a clean pass",
         "What it means": "Traced to Android (likely an assignment bug)."},
        {"Check": "Lift without Android users",
         "Result": f"{pp(s.effect_absolute)} (95% CI {s.ci_lower * 100:+.2f} to "
                   f"{s.ci_upper * 100:+.2f}pp)",
         "Verdict": "✅ Same conclusion",
         "What it means": "The data-quality issue does not drive the result."},
        {"Check": "Day-7 active rate (guardrail)",
         "Result": f"{pp(R['day7'].effect_absolute)}",
         "Verdict": "✅ No harm" if R["day7"].ci_lower > -0.01 else "⚠️ Review",
         "What it means": "Guardrail: no drop of more than 1pp."},
        {"Check": "Page-load time (guardrail)",
         "Result": f"{R['load'].effect_absolute:+.1f} ms "
                   f"(95% CI {R['load'].ci_lower:+.1f} to {R['load'].ci_upper:+.1f})",
         "Verdict": "⚠️ Real, inside the 50 ms limit",
         "What it means": "Shipped with a ticket to fix it before wider rollout."},
    ])
    st.dataframe(checks, hide_index=True, width="stretch")

    st.subheader("Does every group benefit?")
    seg_by = st.radio("Split the lift by", list(R["segments"]), horizontal=True,
                      key="seg_by")
    seg = R["segments"][seg_by].copy()
    seg["lift_pp"] = seg["effect_abs"] * 100
    seg["lo"], seg["hi"] = seg["ci_lower"] * 100, seg["ci_upper"] * 100
    seg["users"] = seg["n_control"] + seg["n_treatment"]
    order = seg.sort_values("lift_pp", ascending=False)["segment"].tolist()
    base = alt.Chart(seg).encode(y=alt.Y("segment:N", sort=order, title=None))
    bars = base.mark_rule(strokeWidth=3, color=BLUE).encode(
        x=alt.X("lo:Q", title="Lift in conversion (pp), with 95% CI"), x2="hi:Q")
    dots = base.mark_circle(size=120, color=BLUE).encode(
        x="lift_pp:Q",
        tooltip=[alt.Tooltip("segment:N", title="Segment"),
                 alt.Tooltip("lift_pp:Q", title="Lift (pp)", format="+.2f"),
                 alt.Tooltip("lo:Q", title="CI low", format="+.2f"),
                 alt.Tooltip("hi:Q", title="CI high", format="+.2f"),
                 alt.Tooltip("users:Q", title="Users", format=",")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GREY).encode(x="x:Q")
    overall = alt.Chart(pd.DataFrame({"x": [p.effect_absolute * 100]})).mark_rule(
        color=ORANGE, strokeDash=[6, 4]).encode(x="x:Q")
    st.altair_chart((zero + overall + bars + dots).properties(
        height=max(140, 45 * len(seg))), width="stretch")
    all_up = bool((seg["lift_pp"] > 0).all())
    unsure = seg.loc[seg["lo"] <= 0, "segment"].tolist()
    note = ("Every group's estimated lift is positive." if all_up else
            "Not every group gains: check the groups left of zero before rollout.")
    if all_up and unsure:
        note += (f" {', '.join(unsure)}: the interval crosses zero because the "
                 "group is small, not because the effect reverses.")
    st.caption(
        "Each dot is the lift inside one group; the line is its 95% interval. "
        "The dashed orange line is the overall lift. " + note
    )
    st.markdown(
        f"Full write-up: [decision memo]({REPO_URL}/blob/main/reports/decision_memo.md)"
        f" · [project summary]({REPO_URL}/blob/main/reports/PROJECT_SUMMARY.md)"
    )


# =====================================================================
# TAB 2 -- PLAN A TEST
# =====================================================================
with tab_plan:
    st.markdown(
        "Before a test starts, decide **the smallest lift worth shipping** and work "
        "out how many users that needs. The defaults are the StreamFlix plan: an "
        "18% historical conversion rate, a 1pp minimum lift and about 3,500 new "
        "trialists a day."
    )
    c1, c2 = st.columns(2)
    with c1:
        baseline_pct = st.slider(
            "Current conversion rate (%)", 1.0, 50.0, 18.0, 0.5, key="baseline",
            help="The control group's expected rate. StreamFlix planned with its "
                 "historical 18% (the test itself measured 21.3%).")
        mde_pp = st.slider(
            "Smallest lift worth detecting (pp)", 0.1, 5.0, 1.0, 0.1, key="mde",
            help="Minimum detectable effect: a business choice agreed with the PM.")
    with c2:
        alpha = st.selectbox(
            "False-alarm rate (alpha)", [0.10, 0.05, 0.01, 0.001], index=1,
            key="alpha", help="Chance of declaring a winner when there is no real lift.")
        power = st.selectbox(
            "Power", [0.80, 0.85, 0.90, 0.95], index=0, key="power",
            help="Chance of detecting the lift if it is real.")
        daily_traffic = st.number_input(
            "New users per day (both arms)", 10, 500_000, 3_500, 100, key="traffic")

    baseline, mde_abs = baseline_pct / 100, mde_pp / 100
    res = sample_size_two_proportion(baseline, mde_abs, alpha=alpha, power=power)
    days = runtime_days(res.n_total, daily_traffic)
    weeks = int(np.ceil(days / 7))

    m1, m2, m3 = st.columns(3)
    m1.metric("Users per arm", f"{res.n_per_arm:,}", border=True)
    m2.metric("Users in total", f"{res.n_total:,}", border=True)
    m3.metric("Days to collect them", f"{days:.1f}", border=True,
              help="Total users divided by daily traffic.")

    if days > 28:
        st.warning(
            f"That is {days:.0f} days, over four weeks. Accept a larger minimum "
            f"lift, add traffic, or plan for a longer test.")
    else:
        st.info(
            f"Run for **{weeks} full week{'s' if weeks > 1 else ''}** "
            f"({weeks * 7} days) so every weekday is covered equally, and decide "
            f"the stopping date before the test starts.")

    mde_grid = np.round(np.arange(0.2, 5.01, 0.1), 2)
    curve = pd.DataFrame({
        "mde": mde_grid,
        "n": [sample_size_two_proportion(baseline, m / 100, alpha=alpha,
                                         power=power).n_per_arm for m in mde_grid],
    })
    curve["days"] = [runtime_days(2 * n, daily_traffic) for n in curve["n"]]
    line = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=3).encode(
        x=alt.X("mde:Q", title="Smallest lift worth detecting (pp)"),
        y=alt.Y("n:Q", title="Users needed per arm (log scale)",
                scale=alt.Scale(type="log")),
        tooltip=[alt.Tooltip("mde:Q", title="Lift (pp)", format=".1f"),
                 alt.Tooltip("n:Q", title="Users per arm", format=","),
                 alt.Tooltip("days:Q", title="Days", format=".1f")])
    point = alt.Chart(pd.DataFrame({"mde": [mde_pp], "n": [res.n_per_arm]})).mark_circle(
        size=160, color=ORANGE).encode(x="mde:Q", y="n:Q")
    st.altair_chart((line + point).properties(height=320), width="stretch")
    st.caption(
        "Halving the lift you want to detect roughly quadruples the users you "
        "need. The orange dot is your current choice."
    )

    with st.expander("Already have a fixed audience? See what it can detect"):
        n_fixed = st.number_input("Users per arm you can get", 100, 5_000_000,
                                  50_000, 1_000, key="n_fixed")
        mde_fixed = mde_for_sample_size(baseline, int(n_fixed), alpha=alpha, power=power)
        st.metric("Smallest lift you can reliably detect", pp(mde_fixed, signed=False),
                  border=True)


# =====================================================================
# TAB 3 -- ANALYSE YOUR OWN RESULTS
# =====================================================================
with tab_analyse:
    st.markdown(
        "Enter users and conversions for each arm. The boxes start with the "
        "StreamFlix experiment's counts; change them to analyse any two-arm test."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Control (current page)**")
        n_ctl = st.number_input("Users", 10, 50_000_000, C["n_ctl"], 100, key="nc")
        k_ctl = st.number_input("Conversions", 0, 50_000_000, C["k_ctl"], 10, key="kc")
    with c2:
        st.markdown("**Treatment (new page)**")
        n_trt = st.number_input("Users", 10, 50_000_000, C["n_trt"], 100, key="nt")
        k_trt = st.number_input("Conversions", 0, 50_000_000, C["k_trt"], 10, key="kt")

    if k_ctl > n_ctl or k_trt > n_trt:
        st.error("Conversions can't be more than users. Check the numbers.")
    else:
        freq = two_proportion_test(
            successes_control=int(k_ctl), n_control=int(n_ctl),
            successes_treatment=int(k_trt), n_treatment=int(n_trt))
        bayes = bayesian_ab_binary(
            successes_control=int(k_ctl), n_control=int(n_ctl),
            successes_treatment=int(k_trt), n_treatment=int(n_trt),
            n_samples=40_000)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Conversion: control → treatment",
                  f"{freq.p_control:.2%} → {freq.p_treatment:.2%}", border=True)
        m2.metric("Lift", pp(freq.effect_absolute),
                  f"{freq.effect_relative:+.1%} relative", border=True)
        m3.metric("95% confidence interval",
                  f"{freq.ci_lower * 100:+.2f} to {freq.ci_upper * 100:+.2f}pp",
                  border=True, help=f"Two-proportion z-test, p = {freq.p_value:.2g}.")
        m4.metric("Chance treatment is better", pct_better(bayes.p_treatment_better),
                  border=True, help="Bayesian posterior probability, flat prior.")

        if freq.significant and bayes.p_treatment_better > 0.95:
            st.success("Both methods agree: **treatment beats control.**")
        elif freq.significant and freq.effect_absolute < 0:
            st.error("Both methods agree: **treatment is worse than control.**")
        elif not freq.significant and 0.05 <= bayes.p_treatment_better <= 0.95:
            st.info("**No clear winner.** The interval includes zero; more data "
                    "or a larger effect is needed.")
        else:
            st.warning("The two methods lean the same way but only one is "
                       "conclusive; the test is close to the edge of its power.")

        draws = bayes.samples_lift * 100
        counts, edges = np.histogram(draws, bins=60)
        hist_df = pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "draws": counts})
        lo_x = min(0.0, float(edges[0])) - 0.1
        hi_x = max(0.0, float(edges[-1])) + 0.1
        xscale = alt.Scale(domain=[lo_x, hi_x], nice=False)
        hist = alt.Chart(hist_df).mark_bar(color=BLUE, opacity=0.85).encode(
            x=alt.X("lo:Q", scale=xscale, title="Plausible lift (pp), Bayesian posterior",
                    axis=alt.Axis(grid=False)),
            x2="hi:Q", y=alt.Y("draws:Q", title="Posterior draws"), y2=alt.datum(0))
        band = alt.Chart(pd.DataFrame({"lo": [bayes.credible_lower * 100],
                                       "hi": [bayes.credible_upper * 100]})).mark_rect(
            color=ORANGE, opacity=0.15).encode(x=alt.X("lo:Q", scale=xscale), x2="hi:Q")
        zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
            color="#4A5568", strokeWidth=2).encode(x=alt.X("x:Q", scale=xscale))
        st.altair_chart((band + hist + zero).properties(height=280), width="stretch")
        st.caption(
            f"The shaded band is the 95% credible interval "
            f"({bayes.credible_lower * 100:+.2f} to {bayes.credible_upper * 100:+.2f}pp). "
            "The grey line is zero lift: the further the bars sit from it, the "
            "surer the result."
        )

        d1, d2 = st.columns(2)
        loss_ship = bayes.expected_loss_ship * 100
        d1.metric("Expected cost of shipping by mistake",
                  "<0.001pp" if loss_ship < 0.001 else f"{loss_ship:.3f}pp",
                  border=True,
                  help="Average conversion lost if we ship and treatment is in fact "
                       "worse, weighted by how likely that is.")
        d2.metric("Expected cost of not shipping", f"{bayes.expected_loss_not_ship * 100:.3f}pp",
                  border=True,
                  help="Average conversion left on the table if we keep the old page "
                       "and treatment is in fact better.")

st.divider()
st.caption(
    f"Built on the tested `src/analysis/` code used by the notebooks · "
    f"[Source on GitHub]({REPO_URL})"
)
