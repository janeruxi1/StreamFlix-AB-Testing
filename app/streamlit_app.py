"""StreamFlix A/B Test Toolkit -- interactive Streamlit demo.

Two modes:
  1. Pre-experiment design  -- sample size, runtime, MDE explorer
  2. Post-experiment analysis -- frequentist + Bayesian on counts you paste in

Reuses the same `src/analysis/` modules tested in CI, so the math here is
the same math the analysis notebooks use.
"""
import sys
from pathlib import Path

# Make src/ importable when Streamlit runs from app/ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.analysis.power import (
    sample_size_two_proportion,
    mde_for_sample_size,
    runtime_days,
)
from src.analysis.frequentist import two_proportion_test
from src.analysis.bayesian import bayesian_ab_binary


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="A/B Test Toolkit",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 A/B Test Toolkit")
st.markdown(
    "Interactive companion to the StreamFlix A/B testing portfolio project. "
    "Both modes use the same code as the notebooks and are covered by 55 "
    "unit tests."
)

mode = st.sidebar.radio(
    "Choose a mode",
    ["Pre-experiment design", "Post-experiment analysis"],
)


# =====================================================================
# MODE 1 -- PRE-EXPERIMENT DESIGN
# =====================================================================
if mode == "Pre-experiment design":
    st.header("Pre-experiment design")
    st.caption(
        "Plug in business assumptions to see required sample size, runtime, "
        "and how sample size scales with the smallest effect you want to detect."
    )

    col1, col2 = st.columns(2)
    with col1:
        baseline = st.slider(
            "Baseline conversion rate (control)",
            min_value=0.01, max_value=0.50,
            value=0.18, step=0.005,
            help="Historical conversion rate for the metric you're testing.",
        )
        mde_pp = st.slider(
            "Minimum detectable effect (percentage points)",
            min_value=0.1, max_value=5.0,
            value=1.0, step=0.1,
            help="Smallest absolute lift worth shipping. Negotiated with the PM.",
        )
    with col2:
        alpha = st.selectbox(
            "Significance level (alpha)",
            options=[0.10, 0.05, 0.01, 0.001],
            index=1,
        )
        power = st.selectbox(
            "Statistical power (1 - beta)",
            options=[0.80, 0.85, 0.90, 0.95],
            index=0,
        )
        daily_traffic = st.number_input(
            "Daily traffic (users/day)",
            min_value=10, max_value=500_000,
            value=3500, step=100,
        )

    mde_abs = mde_pp / 100.0
    res = sample_size_two_proportion(baseline, mde_abs, alpha=alpha, power=power)
    days = runtime_days(res.n_total, daily_traffic)

    st.divider()
    st.subheader("Required sample size")
    m1, m2, m3 = st.columns(3)
    m1.metric("Users per arm", f"{res.n_per_arm:,}")
    m2.metric("Total users", f"{res.n_total:,}")
    m3.metric("Runtime", f"{days:.1f} days")

    if days > 28:
        st.warning(
            f"Runtime is {days:.1f} days. Consider negotiating a larger MDE "
            "with the PM, or expect a longer commitment."
        )
    else:
        st.success(
            f"Runtime is reachable in {days:.1f} days. Pad to a full "
            f"{int(np.ceil(days/7)+1)}-week cycle to absorb seasonality."
        )

    st.subheader("Sample size vs MDE")
    mde_grid = np.linspace(0.001, 0.05, 60)
    ns = [
        sample_size_two_proportion(baseline, m, alpha=alpha, power=power).n_per_arm
        for m in mde_grid
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mde_grid * 100, ns, color="#2C5282", linewidth=2.5)
    ax.fill_between(mde_grid * 100, ns, alpha=0.12, color="#2C5282")
    ax.axvline(mde_pp, color="#F6AD55", linestyle="--",
               linewidth=2, label=f"Selected MDE = {mde_pp:.1f}pp")
    ax.scatter([mde_pp], [res.n_per_arm], color="#F6AD55", s=110, zorder=5)
    ax.set_xlabel("MDE (percentage points)")
    ax.set_ylabel("Required n per arm")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    st.pyplot(fig)


# =====================================================================
# MODE 2 -- POST-EXPERIMENT ANALYSIS
# =====================================================================
else:
    st.header("Post-experiment analysis")
    st.caption(
        "Paste in your observed conversion counts to get a full frequentist "
        "and Bayesian readout in one place."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Control arm**")
        n_ctl = st.number_input("Control: total users",
                                min_value=10, value=50_000, step=100, key="nc")
        k_ctl = st.number_input("Control: conversions",
                                min_value=0, value=10_726, step=10, key="kc")
    with col2:
        st.markdown("**Treatment arm**")
        n_trt = st.number_input("Treatment: total users",
                                min_value=10, value=50_000, step=100, key="nt")
        k_trt = st.number_input("Treatment: conversions",
                                min_value=0, value=11_789, step=10, key="kt")

    if k_ctl > n_ctl or k_trt > n_trt:
        st.error("Conversions can't exceed total users. Check your inputs.")
        st.stop()

    # --- Frequentist ---
    freq = two_proportion_test(
        successes_control=int(k_ctl), n_control=int(n_ctl),
        successes_treatment=int(k_trt), n_treatment=int(n_trt),
    )

    # --- Bayesian ---
    bayes = bayesian_ab_binary(
        successes_control=int(k_ctl), n_control=int(n_ctl),
        successes_treatment=int(k_trt), n_treatment=int(n_trt),
        n_samples=40_000,
    )

    st.divider()
    st.subheader("Headline results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Absolute lift",
              f"{freq.effect_absolute*100:+.2f}pp",
              delta=f"{freq.effect_relative:+.1%} relative")
    m2.metric("95% confidence interval",
              f"[{freq.ci_lower*100:+.2f}, {freq.ci_upper*100:+.2f}]pp")
    m3.metric("Frequentist p-value", f"{freq.p_value:.2g}")
    m4.metric("P(treatment > control)",
              f"{bayes.p_treatment_better:.2%}")

    if freq.significant and bayes.p_treatment_better > 0.95:
        st.success(
            "Both frequentist and Bayesian agree: treatment beats control "
            "with high confidence."
        )
    elif not freq.significant and bayes.p_treatment_better < 0.95:
        st.info("No clear winner. The CI brackets zero and P(T>C) is below 95%.")
    else:
        st.warning(
            "Frequentist and Bayesian results disagree -- usually a power "
            "issue. Consider collecting more data."
        )

    st.divider()
    st.subheader("Bayesian posterior of the lift")

    samples_pp = bayes.samples_lift * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(samples_pp, bins=60, color="#2C5282", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="gray", linewidth=1)
    ax.axvline(bayes.mean_lift * 100, color="#F6AD55",
               linestyle="--", linewidth=2,
               label=f"Posterior mean = {bayes.mean_lift*100:+.2f}pp")
    ax.axvspan(bayes.credible_lower * 100, bayes.credible_upper * 100,
               alpha=0.15, color="#F6AD55",
               label=f"95% CrI [{bayes.credible_lower*100:+.2f}, "
                     f"{bayes.credible_upper*100:+.2f}]pp")
    ax.set_xlabel("Lift (percentage points)")
    ax.set_ylabel("Posterior samples")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig)

    st.subheader("Decision-theoretic summary")
    d1, d2 = st.columns(2)
    d1.metric("Expected loss if we SHIP",
              f"{bayes.expected_loss_ship:.6f}",
              help="How wrong we'd be on average if we ship and treatment is "
                   "actually worse.")
    d2.metric("Expected gain (cost of NOT shipping)",
              f"{bayes.expected_loss_not_ship*100:.2f}pp",
              help="Conversion we'd leave on the table by not shipping.")

    st.markdown(
        f"**Verdict:** P(treatment > control) = {bayes.p_treatment_better:.2%}. "
        f"95% credible interval [{bayes.credible_lower*100:+.2f}, "
        f"{bayes.credible_upper*100:+.2f}]pp. "
        f"Expected gain = {bayes.expected_loss_not_ship*100:.2f}pp; "
        f"expected loss of shipping = {bayes.expected_loss_ship:.4g}."
    )


st.sidebar.divider()
st.sidebar.markdown(
    "📂 [View source on GitHub]"
    "(https://github.com/janeruxi1/ab-testing-project)"
)
st.sidebar.markdown(
    "Powered by the same `src/analysis/` modules used in the analysis "
    "notebooks and covered by 55 pytest unit tests."
)
