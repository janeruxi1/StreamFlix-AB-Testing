"""
Phase 8 — Ground-truth validation, outcome maturity & formal heterogeneity
===========================================================================

Because the dataset is synthetic, we know the true effects. That lets us ask
the question real experiments can never answer: *does the pipeline recover
the truth?* This script also closes three gaps a skeptical reviewer would
raise about Phases 1-7:

  A. Ground-truth recovery   - do the CIs cover the true ATE / segment effects?
  B. Coverage simulation     - over many replications, do 95% CIs cover ~95%?
  C. Outcome maturity        - was the 14-day conversion window closed for
                               every user at analysis time?
  D. Formal HTE test         - treatment x segment interaction (likelihood
                               ratio test), instead of eyeballing per-segment p's
  E. Segment multiplicity    - Holm correction across ALL segment tests
  F. Skewed secondary metric - does the watch-hours conclusion survive
                               a log transform / bootstrap?

Run: python notebooks/08_validation_and_heterogeneity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.analysis.frequentist import holm_bonferroni, two_proportion_test, welch_t_test
from src.analysis.sanity_checks import outcome_maturity
from src.analysis.segmentation import segment_lifts
from src.data.loader import load_experiment
from src.data.simulate import SimConfig, simulate_experiment


def ate_test(df: pd.DataFrame):
    c, t = df[df["group"] == "control"], df[df["group"] == "treatment"]
    return two_proportion_test(
        int(c["converted"].sum()), len(c), int(t["converted"].sum()), len(t)
    )


# ---------------------------------------------------------------------
# A. Ground-truth recovery (same dataset as Phases 1-7)
# ---------------------------------------------------------------------
print("=== A. Ground-truth recovery (data/experiment.csv, seed 42) ===")
df_truth, truth = simulate_experiment(SimConfig(), return_truth=True)
lift = truth["p1"] - truth["p0"]
df = load_experiment("data/experiment.csv")
assert (df["converted"].values == df_truth["converted"].values).all(), \
    "data/experiment.csv is out of sync with simulate.py - regenerate it"

res = ate_test(df)
true_ate = lift.mean()
rows = [("OVERALL ATE", true_ate, res.effect_absolute, res.ci_lower, res.ci_upper)]
for col in ["device", "is_returning"]:
    true_seg = lift.groupby(df[col]).mean()
    est = segment_lifts(df, col, "converted").set_index("segment")
    for seg, tv in true_seg.items():
        rows.append((f"{col}={seg}", tv, est.loc[seg, "effect_abs"],
                     est.loc[seg, "ci_lower"], est.loc[seg, "ci_upper"]))
rec = pd.DataFrame(rows, columns=["effect", "true_pp", "est_pp", "ci_lo", "ci_hi"])
rec[["true_pp", "est_pp", "ci_lo", "ci_hi"]] *= 100
rec["covered"] = (rec["ci_lo"] <= rec["true_pp"]) & (rec["true_pp"] <= rec["ci_hi"])
print(rec.round(2).to_string(index=False))
print(f"\nTrue ATE {true_ate * 100:+.2f}pp vs estimated {res.effect_absolute * 100:+.2f}pp. "
      "Note the pre-experiment brief asked for >=1pp; the simulated truth is larger,\n"
      "so this is a clear win by construction. Real effects are rarely this big - a "
      "+12% relative lift in real life deserves extra bug-hunting.")


# ---------------------------------------------------------------------
# B. Coverage simulation: do nominal 95% CIs cover the truth ~95% of time?
# ---------------------------------------------------------------------
print("\n=== B. CI coverage over 300 replications (n=20,000 each) ===")
N_REPS, covered, errors = 300, 0, []
for seed in range(1000, 1000 + N_REPS):
    d, tr = simulate_experiment(SimConfig(n_users=20_000, seed=seed), return_truth=True)
    r = ate_test(d)
    t_ate = (tr["p1"] - tr["p0"]).mean()
    covered += r.ci_lower <= t_ate <= r.ci_upper
    errors.append(r.effect_absolute - t_ate)
print(f"Coverage of true ATE: {covered / N_REPS:.1%} (nominal 95%)")
print(f"Mean estimation error: {np.mean(errors) * 100:+.3f}pp "
      f"(SD {np.std(errors) * 100:.3f}pp)")
print("Reading: coverage near 95% means the CI machinery is calibrated; a clearly\n"
      "lower number would mean the SE is understated. The simulator's SRM leak is\n"
      "random within Android, so it does not bias the ATE.")


# ---------------------------------------------------------------------
# C. Outcome maturity
# ---------------------------------------------------------------------
print("\n=== C. Outcome maturity (14-day conversion window) ===")
for label, when in [("memo-date draft (2026-05-08)", "2026-05-08"),
                    ("2026-05-15 data pull", "2026-05-15")]:
    m = outcome_maturity(df, when, window_days=14)
    print(f"{label}: {m['matured_share']:.1%} of users matured "
          f"({m['n_immature']:,} immature); all matured on {m['all_matured_on']:%Y-%m-%d}")
print("Reading: signups run Apr 1-28, so the last 14-day window closes ~May 12. "
      "Analysing before that\ncensors late cohorts. The memo is dated after "
      "May 12 for this reason; in a real project, restrict to matured\ncohorts "
      "or wait. Weekly ATEs for week 4 are the ones most exposed to this.")


# ---------------------------------------------------------------------
# D. Formal heterogeneity: LR test of treatment x segment interaction
# ---------------------------------------------------------------------
print("\n=== D. Formal HTE test (logit, LR test of interaction) ===")
df["treat"] = (df["group"] == "treatment").astype(int)
df["returning"] = df["is_returning"].astype(int)
base = "converted ~ treat + C(device) + C(country) + C(source) + returning"
m0 = smf.logit(base, data=df).fit(disp=0)
for term, name in [("treat:C(device)", "device"), ("treat:returning", "is_returning"),
                   ("treat:C(country)", "country"), ("treat:C(source)", "source")]:
    m1 = smf.logit(f"{base} + {term}", data=df).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf)
    dof = int(m1.df_model - m0.df_model)
    print(f"  treat x {name:<12} LR chi2={lr:7.2f}, df={dof}, p={stats.chi2.sf(lr, dof):.4g}")
print("Reading: only the device interaction is significant (p~0.01). The planted\n"
      "returning-user bump is tiny on the log-odds scale (+0.04), so there is no\n"
      "detectable returning x treatment interaction: returning users show a larger\n"
      "lift in percentage points mostly because their baseline conversion is higher\n"
      "(their RELATIVE lift is actually lower). Country and source were given no\n"
      "interaction, and the tests agree. Per-segment differences in the forest plot\n"
      "outside device should therefore not be over-interpreted.")


# ---------------------------------------------------------------------
# E. Multiplicity across all segment tests
# ---------------------------------------------------------------------
print("\n=== E. Holm correction across all segment-level tests ===")
seg_rows = []
for col in ["device", "country", "source", "is_returning"]:
    s = segment_lifts(df, col, "converted").assign(dimension=col)
    seg_rows.append(s)
seg = pd.concat(seg_rows, ignore_index=True)
seg["survives_holm"] = holm_bonferroni(seg["p_value"].tolist())
print(seg[["dimension", "segment", "effect_abs", "p_value", "significant",
           "survives_holm"]].assign(effect_abs=lambda x: (x.effect_abs * 100).round(2))
      .to_string(index=False))
print(f"\n{len(seg)} segment tests; {int(seg['significant'].sum())} significant "
      f"uncorrected, {int(seg['survives_holm'].sum())} after Holm.")


# ---------------------------------------------------------------------
# F. Skewed secondary metric
# ---------------------------------------------------------------------
print("\n=== F. trial_watch_hours: skew-robust checks ===")
c = df.loc[df["group"] == "control", "trial_watch_hours"].values
t = df.loc[df["group"] == "treatment", "trial_watch_hours"].values
print(f"skewness (control): {stats.skew(c):.2f}")
w = welch_t_test(c, t, name="watch_hours")
print(f"Welch mean diff: {w.effect_absolute:+.3f}h  CI [{w.ci_lower:+.3f}, {w.ci_upper:+.3f}]")
rng = np.random.default_rng(0)
boot = [rng.choice(t, len(t)).mean() - rng.choice(c, len(c)).mean() for _ in range(2000)]
print(f"Bootstrap (2k) CI: [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")
lt = np.log1p(t).mean() - np.log1p(c).mean()
print(f"Log1p mean diff: {lt:+.3f} (~{(np.exp(lt) - 1) * 100:.1f}% geometric-mean lift)")
mw = stats.mannwhitneyu(t, c, alternative="two-sided")
print(f"Mann-Whitney p = {mw.pvalue:.3g}")
print("Reading: all four agree on direction and significance, so the Welch result\n"
      "in Phase 3 is not an artifact of the gamma-shaped distribution.")
