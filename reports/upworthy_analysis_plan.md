# Analysis plan: the Upworthy headline-testing programme

Status: **written after exploring the exploratory sample only, and committed before the
confirmatory or holdout samples were analysed.** The archive's maintainers split the 32,487 tests
into these three random samples, stratified by week, for exactly this purpose. Any later change
is a new dated section at the end.

## The data

The Upworthy Research Archive (Matias, Munger, Le Quere and Ebersole, *Scientific Data* 2021;
CC BY 4.0): every headline test Upworthy ran from January 2013 to April 2015. In a test, each
browser session that reached the page was randomly shown one of several "packages" (a headline,
an image, sometimes other text), and the archive records each package's **impressions** and
**clicks**. One row per package; packages share a `clickability_test_id`.

| Sample | Tests | Packages | Use |
|---|---|---|---|
| Exploratory | 4,873 | 22,666 | building the analysis (done) |
| Confirmatory | 22,743 | 105,551 | every number in the report |
| Holdout | 4,871 | 22,600 | one rerun at the end, to confirm |

In June 2024 the maintainers reported that tests run between **25 June 2013 and 10 January
2014** may have broken randomization (they attribute it to a caching fault) and advise
confirmatory research to exclude them. The files used here (dated March 2020) do not carry their
flag, so the window is applied by each test's first `created_at`.

## What exploring the exploratory sample showed (and so motivates the plan)

- Tests have 2 to 14 packages (median 4) and about 3,100 impressions per package; click-through
  is about 1.6%.
- A sample-ratio check per test (chi-square of impressions against an equal split, p < 0.001)
  fails for 70% of tests inside the maintainers' window and 0.5% outside it.

## Questions, in order

**Q1. Does a standard sample-ratio check find the broken period without being told?**
Per test, chi-square of package impressions against an equal split; fail at p < 0.001. Report the
failure rate inside and outside the window, by month, and how many failures fall outside it.
*Confirms if:* the rate inside the window is at least ten times the rate outside.

**Analysis sample for Q2–Q5:** tests whose first package was created outside the window, that
pass the sample-ratio check, and that have at least two packages with 1,000 or more impressions.

**Q2. How different are headlines, really?** For each package, its log click-through relative
to its test's pooled click-through. Each package's sessions are split at random into two halves
(the clicks drawn without replacement, which is the same as randomly assigning each session to a
half). The covariance of a package's two half-sample deviations, across packages, estimates the
variance of *true* differences between headlines with no model of the noise. Report its square
root as the typical true difference, with a bootstrap interval over tests.

**Q3. How much does the winner's lift overstate itself?** In each test, pick the package with
the highest click-through on half A. Compare its lift (over the test's other packages) on half A
with its lift on half B, which did not choose it. Report the share of half-A lift that survives
on half B, and how often half A's winner is also half B's winner. Compare the survival share
with the empirical-Bayes shrinkage factor implied by Q2's variance, to check that shrinkage
predicts the overstatement.

**Q4. What did testing buy?** On half B, the log click-through of half A's winner against the
average log click-through of the test's packages (what publishing a package at random would have
given), reported as a relative gain. This is the programme's value per test, honestly measured.

**Q5. Were the tests the right size?** With the true-difference distribution from Q2, simulate
the gain from picking the winner as a function of impressions per package and number of
packages, for a fixed traffic budget. Report whether, at the median test's traffic, fewer
packages with more impressions each would have gained more. *Check first:* simulated at half a
test's traffic, the model must be compared with Q4's measured gain; if they differ, the
simulated gains are scaled by the measured/simulated ratio and read for their shape only.

All half-sample results are averaged over 20 random splits (seeds 0–19), with the spread
reported.

## Holdout

Q1–Q4 are rerun once on the holdout sample at the end. The report quotes the confirmatory
numbers and says whether the holdout agrees in sign and rough size.

## What would change the plan

Nothing in the confirmatory outcomes. If the confirmatory file's columns differ from the
exploratory one, that is fixed and noted here before any result is computed.

## 2026-10-31 · Q5 dropped, before the confirmatory sample was analysed

Recorded before any confirmatory or holdout result was computed (the confirmatory file had not
been opened by the analysis code at this point).

- **Q5 is dropped.** On the exploratory sample, its own pre-planned check failed: the simulation
  overstated the measured gain from picking winners by about 25%, so its answers would rest on
  a correction factor rather than on the data. Q1–Q4 answer the programme's questions directly
  from the experiments and are kept unchanged.
- **The holdout rerun of Q1–Q4 is unchanged** and is reported in the same notebook as Q2–Q4
  (notebook 05), so the analysis sample's definition no longer mentions Q5.
