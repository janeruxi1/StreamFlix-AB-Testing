# 🎤 Mock Interview Q&A — StreamFlix A/B Test Project

A rehearsal pack for Product Data Scientist interviews. Each question is one an interviewer is *likely* to ask after seeing this repo. Answers are anchored to the actual numbers and decisions in the project, so what you say matches what they can verify by clicking through the code.

**How to use this:**
- Read each question + answer pair aloud at least twice
- Practice condensing the answers to ~60–90 seconds each
- Memorize the *numbers* (they're the most credible part)
- For each question, prepare to handle a follow-up "why?" or "what if?"

---

## 🧭 Project framing (warm-up questions)

### Q1. "Walk me through this project in 90 seconds."

**What they're testing:** Can you communicate the arc of an end-to-end analysis without rambling.

**Strong answer:**
> StreamFlix is a fictional streaming service running a 14-day free trial. The Growth PM hypothesized that replacing the generic "Top Picks" homepage with a personalized "Recommended For You" homepage would increase trial-to-paid conversion by at least 1pp. I designed a 4-week, 50/50, user-level randomized experiment on 100,000 trial signups.
>
> Before analyzing results I checked data quality — SRM passed the strict α = 0.001 threshold but covariate balance revealed an Android-share imbalance, so I flagged it and set up a sensitivity analysis. Power analysis showed the experiment was 2× over-powered for the agreed MDE, with a post-hoc detectable effect of 0.69pp.
>
> The frequentist analysis showed +2.54pp absolute conversion lift (95% CI [+2.02, +3.06]), robust to excluding the affected Android segment. The Bayesian re-analysis gave P(treatment > control) ≈ 100%. Segmentation showed the lift was positive in every device, country, source, and tenure group, but mobile and returning users gained the most.
>
> One guardrail concern: page-load regressed by +29ms — real, but sub-perceptibility. The decision memo recommends **ship**, conditional on an engineering follow-up to mitigate the latency. I built the project with `src/` modules covered by 55 pytest unit tests and a GitHub Actions CI workflow, so the math primitives are protected as the code evolves.

---

### Q2. "Why did you build a synthetic dataset instead of using a public one?"

**What they're testing:** Whether you understand the *design* side of experimentation, not just analysis.

**Strong answer:**
> Most public A/B test datasets I considered — like the Kaggle e-commerce one — are heavily reused and lack the structure to demonstrate experimentation maturity. They have a single binary metric, no user segments, no pre-period covariate, and no engineered edge cases.
>
> Building a synthetic dataset let me embed everything a senior product DS should be able to handle: multiple metric tiers, heterogeneous treatment effects, a realistic SRM-adjacent assignment bug, and a pre-experiment covariate for CUPED. I know the ground truth, so I can verify my analysis recovers the right answers.
>
> Designing the dataset is also itself the signal — it forces you to think about how real product experiments fail, which is much harder than just running a z-test on someone else's clean data.

---

## 🧮 Design & power

### Q3. "How did you decide the MDE?"

**What they're testing:** Whether you treat MDE as a stats variable (junior) or a business decision (senior).

**Strong answer:**
> MDE is a business decision, not a statistical one. I'd ask the PM the smallest lift that would justify the engineering cost of shipping. That's the MDE.
>
> For this project, the PM said anything below +1pp absolute (~5.5% relative) wouldn't be worth shipping the personalization infrastructure, so 1pp became the MDE. We then ran power analysis backwards to figure out the required sample size: about 23,665 users per arm at α=0.05 and 80% power. Our actual collection of ~50k per arm gives us a post-hoc MDE of 0.69pp, so we can detect anything above ~0.7pp — more than enough headroom.
>
> If the PM had wanted 0.5pp instead, the required sample would have *quadrupled* to ~94k per arm. I would have pushed back: the math says we'd need ~7.6 weeks of runtime, which is a calendar cost that may not justify chasing a smaller lift.

---

### Q4. "Why is sample size scaling so dramatic when you halve the MDE?"

**What they're testing:** Whether you understand the underlying math, not just the formula.

**Strong answer:**
> Sample size for a two-proportion test scales as 1/MDE². So halving the MDE quadruples the required n. This is a direct consequence of the formula: required n is proportional to `(z_α + z_β)² × p(1-p) / (effect size)²`. Variance lives in the numerator and squared effect in the denominator.
>
> Practically, this means detecting tiny effects is *brutally* expensive. Going from MDE = 1pp to 0.5pp at our 18% baseline jumps required n from ~24k to ~94k per arm. I always show this curve to PMs so they understand the tradeoff between precision and calendar time before they negotiate MDE.

---

## 🔬 Data quality

### Q5. "Your SRM came back at p = 0.0036 — that's not significant at α = 0.001 but it's not clearly clean either. What did you do?"

**What they're testing:** Whether you treat data quality as binary (junior) or graded (senior).

**Strong answer:**
> I didn't treat it as pass/fail. A p-value of 0.0036 means about 1-in-280 chance of seeing this imbalance under true 50/50 randomization. That's small enough to take seriously but not small enough to throw out the experiment.
>
> So I dropped down to covariate-level balance checks. The device dimension came back with Android share at 34.5% in treatment vs 35.6% in control — a 1.08pp gap that crossed my 1pp flag threshold. That's consistent with an Android-specific assignment bug.
>
> My response: don't discard the data, but report ITT as primary and add a sensitivity analysis that excludes Android. Both gave the same direction and significance (+2.54pp ITT, +2.32pp ex-Android), so the conclusion is robust. I also filed it as an engineering followup so the bug gets fixed before broader rollout.

---

### Q6. "Why both SRM *and* covariate balance? Isn't one enough?"

**What they're testing:** Whether you understand that data quality has multiple levels.

**Strong answer:**
> SRM is the broad check — does randomization preserve overall arm sizes? Covariate balance is the fine-grained check — does randomization preserve attribute distributions within arms? They detect different failure modes.
>
> In this experiment, SRM was borderline non-significant overall. A junior candidate would shrug and move on. But the covariate balance check localized the imbalance to Android specifically, which led to a precise root-cause hypothesis (an Android-specific assignment bug) and a precise mitigation (a sensitivity analysis excluding Android).
>
> The rule I follow: SRM is the alarm, covariate balance is the diagnostic.

---

## 📊 Inference

### Q7. "Walk me through your primary analysis."

**What they're testing:** Whether you can explain a z-test correctly without getting lost in math.

**Strong answer:**
> Primary metric is trial-to-paid conversion — binary outcome, so a two-proportion z-test. Control had 10,726 of 50,461 convert (21.3%), treatment had 11,789 of 49,539 (23.8%). Difference is +2.54 percentage points, or +12% relative.
>
> For the test statistic I used pooled variance, which assumes equal proportions under the null. That gave z = 9.62, p < 10⁻²⁰. For the confidence interval I used *unpooled* variance because we're estimating the actual difference, not assuming it's zero. That gave 95% CI [+2.02pp, +3.06pp].
>
> The lift comfortably clears the +1pp MDE bar. One-sided test against H₀: lift ≤ 1pp gave z = 5.83, so the lift isn't just significantly above zero — it's significantly above the ship threshold.

---

### Q8. "Why did you use Welch's t-test for the continuous metrics instead of Student's?"

**What they're testing:** Whether you know which test to use and why.

**Strong answer:**
> Student's t-test assumes equal variances in the two groups. Welch's doesn't. In product experiments, treatment almost always changes the variance of the outcome — not just the mean. For example, a feature might help heavy users disproportionately, which widens the treatment distribution.
>
> Welch's is the safer default. The cost is essentially zero — for large samples it's indistinguishable from Student's when variances *are* equal. For our trial watch hours metric, treatment had slightly higher variance (because heavy watchers got even more engaged), so Welch's was the right choice.

---

### Q9. "Why correct for multiple testing? Your effects are all huge."

**What they're testing:** Whether you understand FWER, not just whether you can apply Bonferroni.

**Strong answer:**
> Even if the effects are large, the *family-wise* false-positive rate matters. If I test 5 metrics each at α=0.05 independently, the chance of *at least one* false positive is 1 − 0.95⁵ ≈ 23%, not 5%. A skeptical reviewer can correctly point out: "you fished across 5 metrics, of course one looks significant."
>
> Holm-Bonferroni preserves family-wise α at 5% but is less conservative than vanilla Bonferroni. It's a step-down procedure: order p-values from smallest to largest, compare each to α/(m − rank). All 5 of our metrics survived correction, so I can defensibly claim significance across all of them — not just on the one I happened to feature.

---

## 🎲 Bayesian

### Q10. "When do you prefer Bayesian over frequentist?"

**What they're testing:** Whether you can articulate the *role* of each, not which is "better."

**Strong answer:**
> Frequentist is the institutional default — most companies require p-value gates for shipping decisions. Bayesian is the communication layer — it translates the same evidence into the probability statements stakeholders actually act on.
>
> Concretely: a PM doesn't know what to do with "we reject the null at p < 0.001." They do know what to do with "there's a 99.99% probability treatment is better, expected lift +12%, expected loss if we ship and we're wrong is essentially zero."
>
> I run both. Frequentist satisfies the process, Bayesian drives the decision. When they agree (as they did here), the result is unambiguous. When they diverge — say, P(T>C) = 95% but frequentist p = 0.06 — I explain the gap and recommend which framing to anchor the decision on.

---

### Q11. "How would you explain a 95% credible interval to a PM?"

**What they're testing:** Whether you can speak the language of stakeholders.

**Strong answer:**
> A 95% credible interval is what people *think* a confidence interval is. You can say: "There's a 95% probability the true lift is between X and Y." That's the interpretation that matches how PMs actually think.
>
> For our experiment: "We're 95% confident the true conversion lift is between +2.02 and +3.06 percentage points." That's a clean probability statement they can put in a decision deck.
>
> A frequentist 95% CI technically means something more convoluted — under repeated sampling, 95% of intervals constructed this way would contain the true parameter. That's correct but useless for a PM. So I quote the Bayesian credible interval when I can.

---

## 🧩 Segmentation, CUPED, Simpson's

### Q12. "How do you detect heterogeneous treatment effects without p-hacking?"

**What they're testing:** Whether you understand pre-registration discipline.

**Strong answer:**
> Pre-register a small set of *important* segments before the experiment unblinds. I picked four for this project — device, country, traffic source, new-vs-returning — based on business priors about who personalization would resonate with most.
>
> Then I ran within-segment two-proportion z-tests and reported all of them, applying Holm-Bonferroni across the segment family. Mobile and returning users had the biggest lift (~3pp), TV the smallest (~0.4pp, not significant). That matched my prior — personalization needs prior behavior to draw on.
>
> The discipline matters: if I'd gone fishing across 20 segments after unblinding, I'd guarantee finding a "significant" segment by chance alone. Pre-registration plus correction is the only defense.

---

### Q13. "What's CUPED and when does it actually help?"

**What they're testing:** Whether you understand the variance-reduction tradeoff.

**Strong answer:**
> CUPED — Controlled Experiment Using Pre-Experiment Data — uses a pre-period covariate to remove user-level variance that has nothing to do with the treatment. The formula is `Y_adjusted = Y − θ(X − mean(X))`, where θ = Cov(Y, X)/Var(X). The variance reduction equals approximately 1 − ρ², where ρ is the correlation between Y and X.
>
> So it helps a lot when your covariate strongly predicts your outcome. For our trial watch hours metric I got only a 2.5% variance reduction, because prior watch hours barely correlate (ρ ≈ 0.16) — most of our trial users are brand new with no history. Honest result.
>
> Where CUPED really shines is on metrics with strong pre-period signal: returning-user revenue, repeat engagement, retention. Microsoft, Netflix, and Booking report 20–50% CI reductions on those metrics, which is equivalent to running the experiment with 30–50% more users for free. I built the framework so it's ready for the next experiment that has stronger covariate signal.

---

### Q14. "What is Simpson's paradox and how would you detect it?"

**What they're testing:** Whether you understand a confounding pattern that bites real teams.

**Strong answer:**
> Simpson's paradox is when the aggregate effect contradicts the segment-level effects. It happens when segments have *different baselines* AND *different exposure ratios* across arms.
>
> A toy example I built: treatment wins +2pp on Mobile and +2pp on Desktop, but loses by 22pp in aggregate, because treatment is over-sampled into the low-baseline Mobile segment. Each segment says "ship!", aggregate says "don't ship!" — and the correct answer is "ship," because the segments are what actually generalize.
>
> Detection: always check covariate balance and segment composition before trusting the aggregate. If segment composition is balanced — which it is in our real experiment — the aggregate is a trustworthy summary. If it isn't, you either reweight or report segment-level results as the primary.

---

## 🚦 Guardrails & decisions

### Q15. "Your treatment regressed page-load by 29 ms. Do you still ship?"

**What they're testing:** Whether you can hold a real tradeoff in your head and make a defensible call.

**Strong answer:**
> Yes — ship, conditional on engineering opening a follow-up to mitigate the regression.
>
> The 29ms is *statistically real* — CI is [+28.3, +30.3], tight enough that this is not noise. But it sits below the ~100ms threshold at which users typically perceive latency change. Extrapolating from Amazon and Google's published latency studies, 29ms implies roughly 0.3% conversion drag — about 40× smaller than the +12% conversion gain we observed.
>
> So the math is one-sided. But disclosure is non-negotiable. I'd present it explicitly to engineering, file a tracking ticket, and not advance to expanded surfaces (TV homepage, in-app home) until the load-time hit is understood and ideally mitigated. That's how senior product DS frame "ship with caveats" rather than blocking on a small guardrail.

---

### Q16. "How would you communicate a 'no ship' result to a disappointed PM?"

**What they're testing:** Whether you can deliver bad news constructively.

**Strong answer:**
> I'd frame it around what we learned, not what we lost. Three things to land:
>
> First, the data is decisive — we can act on it. A null result with a tight CI is informative; a null result with a wide CI means we wasted runtime. I'd show the post-hoc MDE to make clear our power was real.
>
> Second, here's what the data tells us about *why*. If aggregate is null but a segment lit up, that's a path forward. If engagement secondaries moved without conversion, the funnel is leaking somewhere downstream — go find the leak.
>
> Third, here's the next test I'd recommend. Never end on "no" — end on "and here's what I think we should try next." That keeps the PM forward-leaning and shows you're a partner, not a gatekeeper.

---

## 🛠️ Engineering rigor

### Q17. "How do you make sure your A/B testing pipeline doesn't break over time?"

**What they're testing:** Whether you treat analysis code like production code.

**Strong answer:**
> Every reusable function in `src/` is covered by property-based unit tests. The 55 tests in this repo cover things like: sample size scales as 1/MDE², CUPED with constant covariate returns θ = 0, ROPE probabilities sum to 1. These are *math identities* that must hold for any input, so they catch a wide class of bugs, not just the input the test author thought of.
>
> CI runs the full suite on every push across Python 3.10, 3.11, and 3.12, plus a smoke test that every analysis notebook executes end-to-end. If anyone — including me — pushes a math regression, the green badge in the README turns red within a minute.
>
> When I find a bug in production, I write a *failing* test first, then fix the code. The bug is then permanently locked out. That's how you keep a 100-function library from rotting over years.

---

### Q18. "What would you do differently if you ran this project again?"

**What they're testing:** Whether you can self-critique.

**Strong answer:**
> Three things.
>
> First, pre-register the segments and analysis plan formally before any unblinding. I treated it that way in spirit but I'd write the pre-registration document up explicitly — it's a stronger interview artifact and a stronger team norm.
>
> Second, design the dataset so the pre-period covariate has stronger correlation with the outcome — currently ρ ≈ 0.16, so CUPED is honest but not dramatic. A version where returning users dominate (e.g. a retention experiment) would let me show CUPED reducing CI by 30%+.
>
> Third, add a CUPAC variant — the version that uses ML-predicted outcome instead of a single covariate. That'd be a small but compelling extension that distinguishes me from candidates who only know plain CUPED.

---

### Q19. "What would your *next* experiment be on this platform?"

**What they're testing:** Whether you think in roadmaps, not just one-offs.

**Strong answer:**
> Two natural followups.
>
> First, a holdout test on the personalization model itself. Now that personalized homepage wins, the question becomes: how much of that win is the *concept* vs the *quality of recommendations?* I'd run a 90/10 holdout that keeps showing personalized layout to control, but with shuffled recommendations. If conversion drops in the shuffled arm, we know the recommendations themselves are doing the work — which justifies further investment in the recommender system.
>
> Second, test the same treatment on the TV homepage. Our segment analysis showed TV got +0.4pp (not significant) on the web. TV has very different interaction patterns — fewer titles browsed, longer dwell time. The personalization model probably needs retuning for TV, but it might unlock the segment with the most ARPU.

---

## 🎯 Curveball questions

### Q20. "What's the difference between intent-to-treat (ITT) and as-treated analysis?"

**What they're testing:** Whether you know a piece of terminology that distinguishes graduate-level training.

**Strong answer:**
> ITT analyzes users by *assigned* arm, regardless of what treatment they actually received. As-treated analyzes by what they actually received.
>
> ITT is the conservative, industry-default estimator. It captures the real-world effect of *deciding* to ship the treatment, including any leakage or non-compliance. It biases toward the null when there's leakage, so a positive ITT result is *stronger* evidence, not weaker.
>
> As-treated estimates the effect on users who *successfully* received treatment — useful for understanding upper-bound effect size but vulnerable to selection bias if leakage isn't random.
>
> For this experiment I used ITT as primary. The Android assignment bug meant some treatment-assigned users got control, which would dilute the lift toward null in ITT. The fact that ITT still showed +2.54pp is therefore a conservative estimate of the true effect.

---

### Q21. "Why are you using 100k users — couldn't you have run a smaller experiment?"

**What they're testing:** Whether you can defend your sample size against pushback.

**Strong answer:**
> Power analysis says we need about 23,665 per arm to detect the agreed 1pp MDE at 80% power. So strictly speaking, ~47k total would have sufficed for the headline metric.
>
> But I budgeted 100k because product DS rarely run "just the headline." Segment-level analysis needs power *within* segments — the TV slice of our data has only ~5k per arm, which gave a wide CI and a non-significant result even though the point estimate is positive. If I'd run at 50k total, every segment-level analysis would be too underpowered to interpret.
>
> The rule I follow: power for the smallest segment you'd want to make a decision about, not just the aggregate. That's the difference between an experiment that ships a feature and an experiment that informs strategy.

---

### Q22. "If I gave you only 1 hour to analyze this experiment, what would you skip?"

**What they're testing:** Whether you can triage under time pressure.

**Strong answer:**
> Three things stay; the rest can wait.
>
> Required: SRM + covariate balance (5 min). Two-proportion z-test on the primary metric with CI (5 min). Page-load guardrail check (5 min). That's the minimum required to make a defensible ship/no-ship call.
>
> Strong-to-have: Bayesian P(T>C) and ROPE check (10 min) — for the stakeholder language layer. Sensitivity analysis excluding the SRM-affected segment (10 min) — for robustness.
>
> Can skip in a 1-hour pass: full segmentation forest plot, CUPED variance reduction, Simpson's paradox toy demo, prior sensitivity. Those are *defense* against followup questions, not the core decision. I'd do them next morning before the leadership review.
>
> The principle: in a time crunch, prioritize anything that could *change the recommendation* and skip anything that only *strengthens* it.

---

## 🧠 Behavioral signals to weave in

When answering any technical question, also try to drop in these signals where they fit:

- **Pre-experiment alignment with the PM** — "I agreed the MDE with the PM at kickoff"
- **Skepticism of your own results** — "I checked SRM, then I checked again at segment level"
- **Concrete numbers, always** — "+2.54pp, 95% CI [+2.02, +3.06]" beats "a meaningful lift"
- **Mechanism stories** — "engagement reinforces the conversion lift, which suggests recommendations are doing the work"
- **Action orientation** — "ship, conditional on engineering followup" beats "the result is significant"
- **Limits and tradeoffs** — "29ms is real, but below perceptibility threshold and ~40× smaller than the conversion gain"

---

## 🎯 The single most important rehearsal

Practice **Question 1** (the 90-second walkthrough) until you can deliver it cold. Almost every interview opens with some version of it. If you can land that one cleanly, you set the tone for the entire round.

Time yourself: aim for 70–90 seconds, hitting every major phase (setup → data quality → power → frequentist → Bayesian → segmentation → guardrails → recommendation → engineering rigor).

When you can do that without notes, you're interview-ready on this project.
