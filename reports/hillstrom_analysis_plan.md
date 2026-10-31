# Analysis plan: the Hillstrom e-mail test

Status: **written and committed before any outcome in the data was analysed.** The git history
shows this file landing before the code that reads `visit`, `conversion` or `spend`. Any later
change is a new dated section at the end, not an edit.

One honest limit: this dataset has been public since 2008 and its headline result is widely
quoted (the women's e-mail did better than the men's). The plan cannot make the analysis blind
to that. What it does fix in advance is everything a result could be tuned through: metrics,
caps, covariates, comparisons, the segment search and the decision rule.

## The test

- **Who:** 64,000 customers who bought in the last 12 months.
- **Arms:** randomized in equal thirds: a men's merchandise e-mail, a women's merchandise
  e-mail, or no e-mail.
- **Outcomes, over the two weeks after the send:** `visit` (visited the site), `conversion`
  (bought), `spend` (dollars).
- **Known before the send:** `recency` (months since last purchase), `history` (dollars spent in
  the past year), `history_segment`, `mens` and `womens` (bought from that line in the past
  year), `newbie` (new in the past 12 months), `zip_code` (urban, suburban, rural), `channel`
  (phone, web, multichannel).

## The decision

For the next campaign to customers like these: **send the men's e-mail, the women's e-mail, or
nothing, and should the choice differ by customer?**

## Trust checks (run first; a failure stops the readout)

1. **Sample ratio:** chi-square test of the three arm counts against equal thirds. p < 0.001
   stops the analysis until explained.
2. **Balance:** standardized mean differences of every pre-send variable across arms. Any
   |SMD| > 0.05 is reported and the variable is added to the regression adjustment.
3. **Data:** missing values, negative spend, spend without conversion, conversion without visit.

## Metrics

| Role | Metric | Definition |
|---|---|---|
| Primary | Spend per customer | Two-week spend per customer *sent the treatment* (intention to treat), capped |
| Secondary | Conversion rate | Share of customers who bought |
| Secondary | Visit rate | Share of customers who visited |
| Diagnostic | Spend per buyer | Spend among converters (not randomized, descriptive only) |

**Cap.** Spend is capped at the 99th percentile of positive spend, computed on all arms pooled
(so the cap cannot favour an arm). Uncapped results are reported beside the capped ones.

## Estimation

- Difference in means per customer, each e-mail against no e-mail, and men's against women's.
  Welch standard errors; 95% intervals.
- **CUPED / regression adjustment** with the pre-send variables (Lin 2013: treatment interacted
  with centred covariates, HC2 standard errors). The covariates are fixed now: `history`,
  `recency`, `mens`, `womens`, `newbie`, and indicators for `zip_code` and `channel`. The
  adjusted estimate is the one the decision uses; the unadjusted one is reported beside it, with
  how much the interval narrowed.
- **Multiple comparisons:** the three primary comparisons form one family, Holm at 5%. Secondary
  metrics are reported with Holm within each metric, and do not decide.

## The decision rule

1. An e-mail is worth sending to everyone only if its adjusted, capped spend lift over no e-mail
   is positive after Holm, and the lift times the gross margin exceeds the cost of sending.
   The value is reported per 1,000 customers at an assumed **40% gross margin** and **$0.10 per
   e-mail sent**, with the break-even margin shown so the reader can substitute their own.
2. If both e-mails pass, the one with the higher lift is chosen only if the men's-versus-women's
   comparison passes Holm; otherwise the memo says they cannot be told apart and recommends the
   cheaper or broader option.
3. **Targeting by customer** replaces one e-mail for everyone only if it wins on data it was not
   chosen on (next section).

## Segment search, without cherry-picking

- Customers are split at random into a **discovery half** and a **confirmation half**
  (seed 20261030, stratified by arm).
- On the discovery half only: estimate the spend lift of each e-mail within each of these
  pre-defined segments, and nothing else: `mens`/`womens` purchase history (men's only, women's
  only, both), `newbie`, `channel`, `zip_code`, `history_segment`. The targeting policy assigns
  each segment the e-mail with the highest estimated lift (or no e-mail when neither is
  positive).
- On the confirmation half: estimate the policy's value per customer by inverse-probability
  weighting (each arm has probability 1/3), against the best single e-mail for everyone, with a
  bootstrap 95% interval.
- The targeting policy is recommended only if that interval excludes zero.

## What would change the plan

Nothing in the outcomes. A trust-check failure would be investigated and reported before any
readout.

## 2026-10-31 · Two notes added after the data arrived

1. **Correction to the opening note.** It said the widely quoted result was that the women's
   e-mail did better. That was wrong: the men's e-mail lifts visits, purchases and spend more.
   Nothing in the plan depended on it; it is corrected here rather than edited, as the plan's
   own rule requires.
2. **How "the targeting policy" is chosen.** The segment search builds one candidate policy per
   segment variable. The candidate with the highest value on the discovery half is the one
   confirmed, against the best single choice on that half. This was fixed in code before the
   data arrived (notebook 03 first ran on a synthetic stand-in) and is recorded here so the plan
   says it too.

## 2026-10-31 · Two sections added that do not change the readout

The repository now tells one experiment from plan to rollout, with this test as the spine.
Two sections are added; neither changes a metric, comparison or the decision rule above.

1. **Design (notebook 01).** What this test could detect: the minimum detectable effect for
   visit, conversion and spend at about 21,000 customers per arm (two-sided alpha 0.05, 80%
   power), and the sample size a planner would have needed for effects a retailer would care
   about. Baseline rates and spreads come from the no-e-mail arm, so this is a design review
   after the fact, not a pre-test calculation, and it is labelled that way.
2. **Rollout and follow-up (notebook 02).** How the decision would be rolled out: a small
   permanent no-e-mail holdback to keep measuring the e-mail's value, the guardrails to watch,
   and the next test, sized with the same calculation.
