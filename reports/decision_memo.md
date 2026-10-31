# Decision memo: two real experiments, read out

**Author:** Xi Ru · **Data:** Hillstrom e-mail test (2008); Upworthy Research Archive (2013–2015)
· Every number below is read from `reports/metrics/` and checked by `tests/test_reported_numbers.py`.

---

## Part 1 · The e-mail test: which e-mail, to whom?

**Recommendation: send the men's merchandise e-mail to every customer, keep a 5% no-e-mail
holdback, and do not target by segment.**

64,000 recent customers were randomized in equal thirds to a men's e-mail, a women's e-mail or no
e-mail. The plan ([`hillstrom_analysis_plan.md`](./hillstrom_analysis_plan.md)) fixed the metric
(spend per customer over two weeks), the comparisons and the decision rule before any outcome
was read.

### The test can be trusted

- **Sample ratio:** 21,306 / 21,307 / 21,387 customers, p = 0.90.
- **Balance:** no pre-send variable differs between arms by more than 0.02 standard deviations.
- **Data:** no missing values, no spend without a purchase, no purchase without a visit.

### What each e-mail did (spend per customer, two weeks, regression-adjusted, 95% CI)

| | Lift | Value per 1,000 e-mailed* | Break-even margin |
|---|---|---|---|
| **Men's e-mail** vs none | **+$0.77** [+0.48, +1.05] | **$207** [$93, $320] | 13% |
| Women's e-mail vs none | +$0.42 [+0.17, +0.68] | $70 [−$32, $172] | 24% |
| Men's vs women's | +$0.34 [+0.03, +0.65] | | |

\* at the plan's assumed 40% margin and $0.10 per e-mail.

All three comparisons pass Holm correction. The men's e-mail also wins on visits (+7.6 points)
and purchases (+0.68 points). The women's e-mail pays for itself on its point estimate, but its
interval includes a loss; the men's e-mail is profitable across its whole interval, at any margin
above 13%.

### Targeting does not survive a fair test

A segment search on half the customers found a policy (by past-spend band) that looked 10 cents
per customer better than sending the men's e-mail to everyone. On the other half, which played no
part in choosing it, the gain was **−$0.01 [−$0.28, +$0.26]**. The apparent gain was noise picked
by the search, so the simple policy stands.

### Two things worth knowing

- **Regression adjustment barely helped.** Adjusting for past spend, recency and the other
  pre-send variables narrowed the spend interval by under 1% and the visit interval by about
  1.5%. CUPED-style adjustment only pays when pre-period data predicts the outcome; past-year
  spend says little about a two-week window in which 99% of customers buy nothing.
- **The test was only just big enough.** At 21,300 per arm it reliably detects a spend lift of
  $0.31. The lift that just pays for an e-mail is $0.25, which would have needed 33,700 per arm.
  The men's e-mail is far above both, so the decision is safe; a marginal e-mail could have been
  missed.

### Rollout

- **Keep a 5% holdback** (3,200 customers a send) who never get the campaign e-mail. It costs
  about $980 of margin a send and, pooled over 3 sends, still detects an effect half the size
  measured here, so a fading e-mail shows up early.
- **Watch** unsubscribes, spam complaints and deliverability per send; this data has none of them.
- **Next test:** the men's e-mail against a new version. Detecting a $0.25 difference needs
  79,000 customers per arm, about 2.5 sends of this list (spend varies more among e-mailed
  customers than in the no-e-mail baseline).

---

## Part 2 · A testing programme: what do 22,743 headline tests say?

Upworthy tested headlines and images ("packages") on its stories from 2013 to 2015, showing each
visitor one package at random. The archive's maintainers split the tests into exploratory,
confirmatory and holdout samples. The plan ([`upworthy_analysis_plan.md`](./upworthy_analysis_plan.md))
was written on the exploratory sample and pushed before the confirmatory sample was analysed;
every number here is confirmatory, and the holdout agrees with each.

### 1. A routine sample-ratio check would have caught a six-month outage

The maintainers later found that tests between 25 June 2013 and 10 January 2014 may have broken
randomization. A per-test chi-square check, not told the dates, flags **70%** of tests inside
that window and **0.5%** outside it (136 times as often). It also flags a smaller excess in
early 2013 (about 2–5% of tests a month against 0.1% expected by chance).

**Recommendation:** run the check automatically on every test and block the readout on a failure.

### 2. Headlines matter, but winners overstate themselves by half

- **Packages truly differ** by about **26%** in click-through (standard deviation of true log
  click-through within a test, 95% CI 25%–26%), once noise is removed by splitting each
  package's traffic in two at random.
- **The winner's curse:** the package that wins on one half of the traffic shows a lift of 0.46
  (log click-through) over its rivals; on the other half it shows 0.22. **47% of a winner's
  observed lift survives.** The same package wins both halves in 43% of tests, against 22% by
  chance.
- **Shrinkage predicts most of this:** an empirical-Bayes estimate puts the surviving lift at
  0.25 against the 0.22 measured, so it is close but slightly optimistic.

**Recommendation:** when forecasting from a test, report the winner's shrunk lift, not its
observed lift; roughly halve the headline number.

### 3. Testing still paid

Publishing the winner instead of a random package raised click-through by **+20%**, measured on
traffic that played no part in picking the winner. Because each half had only half a test's
traffic, this is a lower bound for full-size tests.

---

## Limits

- **Hillstrom** is from 2008, its retailer is unnamed, and margin and e-mail cost are
  assumptions. It is public and widely analysed, so the plan cannot make the analysis blind to
  the headline result; it fixes everything a result could be tuned through.
- **Upworthy** measures clicks, not reading or sharing; a headline that wins clicks can still
  disappoint readers.
- The method checks against simulated data with a known answer are in the
  [StreamFlix appendix](../notebooks/streamflix/README.md) and `tests/`.
