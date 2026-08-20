# Project Summary — StreamFlix Personalized Homepage A/B Test

**What was built + what was found.** Single-page catalog of the project's
components and headline results. Every number here is reproducible from
`notebooks/` and verified against the underlying test output. All figures
live in `reports/figures/`.

---

## 60-second overview

StreamFlix's Growth team hypothesized that replacing the generic
"Top Picks" homepage with a personalized "Recommended For You" homepage
would improve trial-to-paid conversion by at least 1pp. A 4-week
user-level 50/50 randomized experiment was run on ~100,000 new
trialists in April 2026. This project delivers the full data-science
workflow — experiment design validation, primary + secondary +
guardrail inference across two statistical frameworks, heterogeneous
treatment effects, variance reduction, sensitivity/robustness testing,
and a shipping recommendation for the PM.

**Headline outcome.** Recommend **ship** the personalized homepage,
conditional on an engineering ticket to mitigate a +29ms page-load
regression before broader rollout. Trial conversion improved by
**+2.54pp** (95% CI [+2.02, +3.06]), with **P(treatment > control) ≈
100%** under the Bayesian re-analysis. The decision is robust across
four independent sensitivity dimensions (novelty decay, sequential
looks, sample-size subsampling, framework choice).

---

## Headline numbers at a glance

| Metric | Control | Treatment | Δ | 95% CI |
|---|---:|---:|---:|---:|
| **Trial → paid conversion** (primary) | 21.26% | 23.80% | **+2.54pp (+11.96%)** | [+2.02, +3.06]pp |
| Trial watch hours | — | — | +0.87 hrs (+18.08%) | [+0.71, +1.02] |
| Distinct titles watched | — | — | +0.51 (+17.49%) | [+0.42, +0.60] |
| Day-7 active (guardrail) | — | — | +1.75pp (+4.05%) | [+1.13, +2.36] |
| Page load time (guardrail) | — | — | **+29.3ms (+6.87%)** ⚠️ | [+28.3, +30.3] |

**Design + inference summary:**

| Dimension | Value |
|---|---|
| Sample size | ~100,000 users (~50k control, ~50k treatment) |
| Experiment window | 4 weeks (April 2026) |
| Pre-committed MDE | 1pp absolute (~5.5% relative) |
| Post-hoc detectable effect | 0.69pp (2× over-powered) |
| Bayesian P(treatment > control) | ~100% (Beta-Binomial, 100k MC samples) |
| Multiple-testing correction | Holm-Bonferroni across 5 metrics — **all survive** |
| Variance reduction (CUPED) | 2.5% on watch-hours secondary |
| Segment coverage | **Positive lift in every segment** (device × country × source × tenure) |
| Sensitivity robustness | **All 6 checks support ship** (see Phase 7 below) |

---

## Phase 1 — Data quality & randomization checks

### What was built

- Sample-ratio mismatch (SRM) chi-square test at strict α = 0.001
- Covariate balance across categorical dimensions (country, device,
  source, is_returning) and continuous covariates (prior watch hours,
  page-load baseline)
- Diagnostic charts (`01_categorical_balance.png`,
  `01_continuous_balance.png`)

### Key findings

- SRM chi-square **p = 0.0036** — passes the strict 0.001 threshold but
  sits in the amber band (would fail at α = 0.005). Not sufficient
  grounds to discard the experiment.
- **Covariate balance revealed an Android-share imbalance of −1.08pp**
  in treatment. Consistent with a segment-specific assignment bug.
  Localizing the failure mode enabled a targeted sensitivity analysis
  (Phase 3) rather than a full experiment restart.

---

## Phase 2 — Power analysis

### What was built

- Required sample size at the 1pp MDE (agreed with PM at kickoff)
- Post-hoc MDE for the actual collected sample
- Sample-size curves across baseline conversion × MDE grid
  (`02_power_curve.png`, `02_sample_size_heatmap.png`,
  `02_sample_size_vs_mde.png`)

### Key findings

- **Required sample: 23,665 per arm** at 80% power / α = 0.05 for the
  1pp MDE
- **Actual sample: ~50,000 per arm** — 2× over-powered
- **Post-hoc MDE: 0.69pp** — the observed +2.54pp lift is comfortably
  above the detection floor
- Additional power confirmed adequate for segment-level analyses (TV
  cohort at ~5k per arm is the tightest constraint)

---

## Phase 3 — Frequentist analysis

### What was built

- Primary ITT two-proportion z-test on conversion
- Sensitivity analysis excluding the SRM-flagged Android segment
- Welch's t-tests on continuous secondaries (watch hours, distinct
  titles)
- Guardrail tests: two-proportion (day-7 active) + Welch's t (page
  load)
- Holm-Bonferroni correction across all 5 metrics
- Forest plot summary (`03_forest_plot.png`, embedded in the memo)

### Key findings

- **Primary lift: +2.54pp** (95% CI [+2.02, +3.06], z = 9.62, p < 10⁻²⁰)
- **One-sided test against the 1pp MDE bar: z = 5.83** — the lift is
  significantly above the ship threshold, not merely above zero
- **Android sensitivity: +2.32pp** (95% CI [+1.67, +2.96]) — direction
  and significance preserved; the SRM finding does not overturn the
  headline
- All 5 metrics survive Holm-Bonferroni correction at α = 0.05

---

## Phase 4 — Bayesian re-analysis

### What was built

- Beta-Binomial conjugate posterior with 100k Monte Carlo samples
- Prior sensitivity check across Beta(1,1) uninformative,
  Beta(18,82) mildly-informative, Beta(180,820)
  strongly-informative
- ROPE analysis at ±0.5pp practical-equivalence band
- Expected-loss decision metrics (loss of shipping vs loss of holding)
- Posterior density chart (`04_bayesian_posteriors.png`)

### Key findings

- **P(treatment > control) ≈ 100%**, posterior mean lift +2.54pp,
  95% credible interval [+2.02, +3.06]pp
- **Prior sensitivity: posterior mean varies by < 0.05pp** across all
  three priors tested — the result is not driven by prior choice
- **ROPE check: posterior mass sits entirely outside the ±0.5pp
  practical-equivalence band** — the effect is both statistically
  credible AND practically meaningful
- Expected loss of shipping ≈ 0 → decision-ready

---

## Phase 5 — Segmentation, CUPED, and Simpson's paradox

### What was built

- Within-segment two-proportion z-tests across 4 pre-registered
  dimensions: device (4 levels), country (5), source (4),
  new-vs-returning (2)
- Segment forest plot with 95% CIs (`05_segment_forest.png`)
- CUPED variance reduction on the trial_watch_hours secondary using
  prior_watch_hours as the pre-experiment covariate
- Simpson's-paradox toy demonstration + real-data verification

### Key findings

- **Positive lift in every segment examined.** Mobile (iOS +3.24pp,
  Android +2.96pp) leads; TV +0.42pp is directionally positive but
  underpowered at the segment level. Returning users +3.29pp beats new
  users +2.26pp — consistent with the mechanism (personalization has
  more signal to work with when users have prior behavior).
- **CUPED variance reduction: 2.5%** on watch-hours. Modest because
  prior_watch_hours barely correlates with trial outcomes (ρ ≈ 0.16 —
  most trialists are literally new users with no prior history). The
  framework is now in place for future experiments on returning-user
  cohorts where ρ ≈ 0.5-0.7 and CUPED typically delivers 20-50% CI
  reduction.
- **Simpson's paradox is not a live risk in this experiment.** Covariate
  balance (Phase 1) + segment forest confirm that treatment/control
  mix is balanced enough within-segments that the aggregate lift is not
  a composition artifact.

---

## Phase 6 — Decision memo + hero figure

### What was built

- Stakeholder-facing decision memo (`reports/decision_memo.md`) with
  TL;DR, methodology, results tables, guardrails discussion, risks,
  rollout plan, and statistical appendix
- Hero summary figure (`reports/figures/06_hero_summary.png`) embedded
  at the top of both the memo and README

### Key findings

- **Recommendation: SHIP**, conditional on engineering opening a
  ticket to mitigate the +29ms page-load regression before broader
  rollout
- **Page-load discussion:** +29ms is real (tight CI, not noise) but
  sits below the ~100ms perceptibility threshold. Extrapolating from
  published latency studies (Amazon, Google), the implied conversion
  drag is ~0.3% — roughly **40× smaller** than the +12% conversion
  gain. Ship, disclose, mitigate.
- **Harm-test summary:** of all 5 declared metrics, only page-load
  moves against treatment. Every other metric is positive and
  significant. Ideal pattern for a shipping recommendation — one
  focused concern, no scattered secondary-metric harm.

---

## Phase 7 — Sensitivity & robustness

### What was built

- Weekly ATE decomposition (novelty-decay check)
- Sequential-look Pocock correction (peeking analysis)
- Sample-size sensitivity via stratified subsampling (10% / 25% / 50%
  / 75% / 100%)
- Explicit Bayesian ↔ frequentist reconciliation
- A/A test — 500 random splits of control-only data validate the
  pipeline's empirical false-positive rate against nominal α = 5%
- Bootstrap CI — 10k Bernoulli bootstrap validates the analytical
  normal-approximation CI
- Diagnostic charts (`07_weekly_ate.png`, `07_sequential_looks.png`)

### Key findings — all six checks support ship

| Check | Result | Verdict |
|---|---|---|
| **A. Weekly ATE** | Lift +2.68 / +2.51 / +3.03 / +1.95pp across weeks 1-4; pooled +2.54pp | ✅ Stable, no novelty decay |
| **B. Sequential looks (Pocock K=4)** | \|z\| = 5.08 at end of week 1, well above Pocock threshold 2.361 | ✅ Could have stopped early at day 7 |
| **C. Sample-size sensitivity** | Ship call holds down to **10% subsample** (~10k users) | ✅ Full 100k was not required |
| **D. Framework reconciliation** | Frequentist p ≈ 0 with +2.54pp; Bayesian P(T>C) = 100% with matching +2.54pp | ✅ Frameworks agree |
| **E. A/A test** | 500 random splits of control → empirical FPR 3.6%, median p-value 0.496 | ✅ Pipeline validated on known-null data |
| **F. Bootstrap CI** | 10k Bernoulli bootstrap [+2.03, +3.06]pp vs analytical [+2.02, +3.06]pp — max bound difference 0.004pp | ✅ Normal approximation is safe |

**Reading this for stakeholders:** the +2.54pp headline is robust to
(i) time-of-experiment novelty effects, (ii) peeking / sequential-look
inflation, (iii) sample size (would have shipped at N/10),
(iv) framework choice, (v) pipeline validation on known-null data, and
(vi) CI-computation method. None of these are typical A/B test failure
modes — the recommendation is not fragile.

---

## Key design decisions + rationale

**Why intent-to-treat (ITT) as the primary estimator.**
ITT analyzes users by assigned arm regardless of what treatment they
actually received. It's the industry-default estimator because it
captures the real-world effect of *deciding* to ship. It biases toward
null when there's leakage (like the Android assignment bug), so a
positive ITT result is a conservative estimate of the true effect —
which strengthens rather than weakens the ship recommendation.

**Why Welch's t-test over Student's for the continuous secondaries.**
Student's assumes equal variances between arms. Product experiments
almost always change the variance of the outcome, not just the mean.
Welch's is the safer default and costs essentially nothing at
large-N when variances actually are equal.

**Why Holm-Bonferroni over vanilla Bonferroni for multiple-testing.**
Both control family-wise error at α. Bonferroni divides α by K
uniformly (most conservative). Holm is a step-down procedure — less
conservative while still guaranteeing family-wise α = 0.05. Better
power at the same Type-I control.

**Why run both frequentist AND Bayesian.**
Frequentist gates the ship decision (institutional default, what the
p-value review process checks). Bayesian translates the same evidence
into decision-relevant probability statements — P(treatment > control),
expected loss of shipping — that stakeholders can act on directly.
Running both surfaces framework-robustness (Phase 7 Section D confirms
both agree here) and produces both compliance-ready and
stakeholder-ready outputs.

**Why 4 pre-registered segments and not more.**
Pre-registering segments before unblinding is the key discipline that
distinguishes principled heterogeneous-effect analysis from
post-hoc segment fishing. The 4 chosen (device, country, source,
tenure) reflect legitimate business priors about which segments are
most likely to respond differently to personalization. Adding more
segments after unblinding would inflate family-wise error and open
the door to false-positive segment claims.

**Why the 24-day experiment ran for 4 weeks (not shorter).**
The aggregate metric needed ~24k users per arm at the 1pp MDE. But
the SEGMENT analyses need power *within* segments — TV at ~5k per arm
is already borderline. Powering only for the aggregate would have
zeroed out the segment story. Rule of thumb applied: power for the
smallest segment you'd want to make a decision about, not just the
aggregate.

**Why disclose the +29ms page-load regression rather than exclude the metric.**
Suppression of an unfavorable secondary result is the fastest way to
lose credibility in a post-hoc review. The regression is real (tight
CI, low probability of noise) and needs to be visible in the memo.
Quantifying the tradeoff (40× smaller than the conversion gain in
expected impact, below perceptibility threshold) turns "concerning
finding" into "documented, quantified, gated" — which is the right
shape for a mature ship recommendation.

---

## Limitations honestly named

- **Android SRM-adjacent imbalance.** Localized to one device segment.
  Root cause (assignment bug) hypothesized but not confirmed in the
  data — engineering ticket to investigate.
- **Page-load regression** (+29ms) is real. Below perceptibility and
  40× smaller than the conversion gain in expected impact, but
  requires engineering follow-up before broader rollout to expanded
  surfaces (TV homepage, in-app home).
- **CUPED variance reduction is modest** (2.5%) because the pre-period
  covariate correlates weakly with trial outcomes on a
  first-time-signup cohort. Framework is in place for future
  experiments on returning-user metrics where CUPED typically
  delivers 20-50% CI reduction.
- **TV segment is underpowered** at ~5k users per arm. Point estimate
  is positive (+0.42pp) but not significant. Not evidence of "no
  effect on TV" — evidence of "insufficient N to conclude on TV."
  Natural target for a TV-specific follow-up experiment.
- **Novelty monitoring window.** 4-week run captures 2 full trial
  cycles, but a longer post-ship monitoring window (+30d, +60d)
  is prudent for a personalization feature. Rollout plan explicitly
  includes this checkpoint.
- **Single test, no meta-analysis.** This is the first personalization
  test on this surface — no historical prior distribution of similar
  effects exists to pool from. Bayesian priors are appropriately
  uninformative as a result.

See the "Path to next tests" section in the decision memo for the
prioritized follow-up experiment queue.

---

## Where to find each artifact

| Artifact | Path |
|---|---|
| Decision memo (stakeholder-facing) | `reports/decision_memo.md` |
| Hero summary figure | `reports/figures/06_hero_summary.png` |
| PM brief (kickoff) | `reports/scenario_brief.md` |
| Interactive tool | `streamlit run app/streamlit_app.py` |
| Data quality diagnostics | `notebooks/01_data_quality.py` |
| Power analysis | `notebooks/02_power_analysis.py` |
| Frequentist inference + forest plot | `notebooks/03_frequentist.py` |
| Bayesian re-analysis | `notebooks/04_bayesian.py` |
| Segmentation, CUPED, Simpson's demo | `notebooks/05_segmentation.py` |
| Hero figure regeneration | `notebooks/06_hero_figure.py` |
| Sensitivity & robustness (Phase 7) | `notebooks/07_sensitivity_robustness.py` |
| All charts | `reports/figures/` |
| Reusable inference primitives | `src/analysis/` (frequentist, bayesian, power, sanity_checks, segmentation) |
| Data loader + simulator | `src/data/` |
| Tests | `tests/` — 55 pytest, GitHub Actions CI |

---

## How to read the notebooks

Sequential order — each notebook builds on the previous:

```
01_data_quality.py           SRM + covariate balance
   ↓
02_power_analysis.py         required sample size + post-hoc MDE
   ↓
03_frequentist.py            primary ITT + sensitivity + secondaries + guardrails
   ↓
04_bayesian.py               Beta-Binomial posterior + priors + ROPE
   ↓
05_segmentation.py           within-segment cuts + CUPED + Simpson's demo
   ↓
06_hero_figure.py            composite chart for memo + README
   ↓
07_sensitivity_robustness.py weekly ATE + Pocock + N-subsample + framework
                             reconciliation
```

Read top-to-bottom for the full story; skip directly to Phase 3 for
the primary result; skip to Phase 7 for the robustness view.
