# 📝 PM Brief — Personalized Homepage Trial Test

**From:** Sarah Chen, Sr. PM, Growth
**To:** Data Science Team
**Re:** Trial-to-paid uplift experiment

---

## Background

Our 14-day free trial converts at **~18%** to paid. The growth team believes our current "Top Picks" homepage (generic editorial picks) is missing an opportunity to hook trialists with personalized content.

## Proposal

Replace the generic homepage with a **"Recommended For You"** homepage powered by our recommendation model, shown immediately after trial signup.

## Hypothesis

> Personalized homepage will **increase trial-to-paid conversion by at least 1 percentage point** (relative lift ~5.5%).

## Decision sought

**Ship / Don't ship** the personalized homepage to 100% of trial users.

## Experiment design

- **Population:** All new trial signups during the test window
- **Randomization:** 50/50, user-level, persistent
- **Control:** Existing "Top Picks" homepage
- **Treatment:** "Recommended For You" homepage
- **Duration:** 4 weeks (covers 2x the 14-day trial cycle)
- **Sample size:** ~50,000 users per arm (per our power analysis — see Phase 2)

## Metrics

| Tier | Metric | Threshold for ship |
|---|---|---|
| Primary | Trial-to-paid conversion (14d) | +1pp lift, p < 0.05 |
| Secondary | Watch hours during trial | Directionally positive |
| Secondary | Distinct titles watched | Directionally positive |
| Guardrail | Day-7 active rate | No drop > 1pp |
| Guardrail | Page load time (ms) | No regression > 50ms |

## Stakeholders

- **Primary decision-maker:** Sarah Chen (PM)
- **Engineering:** Web platform team (concerned about page load)
- **Content:** Recommendations team (owns the model)

## Risks / Open questions

- Novelty effect: lift could decay after 1-2 weeks
- Segments: does the effect vary by country / device / new vs returning?
- Cannibalization: do users find fewer "new to them" titles?
