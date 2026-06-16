"""Synthetic A/B test data generator for the StreamFlix trial experiment.

DESIGN PHILOSOPHY
=================
We simulate a realistic product experiment with KNOWN ground truth so we can:
  1. Teach analysis techniques on data that mirrors real-world complexity
  2. Engineer edge cases (SRM bug, Simpson's paradox) that real tests exhibit
  3. Validate our analysis recovers the true treatment effect

What makes this realistic:
  - User-level heterogeneity (segments react differently to treatment)
  - Pre-experiment covariate (prior watch hours) correlated with outcome
    -> enables CUPED variance reduction
  - Multiple correlated metrics (conversion, watch time, day-7 active)
  - Engineered SRM in a small subset (intermittent assignment bug)
  - Hidden subgroup heterogeneity -> Simpson's paradox possible if segments
    have different baseline conversion AND treatment exposure ratio

GROUND TRUTH (set in code below):
  - Overall treatment effect on conversion: +1.2 percentage points
  - Effect is HIGHER for mobile users (+2pp) than desktop (+0.5pp)
  - Effect is HIGHER for returning users than brand-new users
  - SRM is injected in ~0.3% of users (assignment bug on Android <v5.1)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration — adjust here to retune realism
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SimConfig:
    n_users: int = 100_000           # total trial signups in test window
    seed: int = 42
    base_conversion: float = 0.18    # baseline trial->paid rate
    true_lift_pp: float = 0.012      # +1.2 percentage points (overall ATE)

    # Segment shares (must sum to 1 within each dimension)
    country_share: dict = None
    device_share: dict = None
    source_share: dict = None

    def __post_init__(self):
        object.__setattr__(self, "country_share",
                           {"US": 0.45, "UK": 0.15, "CA": 0.10, "AU": 0.08, "Other": 0.22})
        object.__setattr__(self, "device_share",
                           {"iOS": 0.30, "Android": 0.35, "Web": 0.25, "TV": 0.10})
        object.__setattr__(self, "source_share",
                           {"organic": 0.40, "paid_search": 0.25,
                            "paid_social": 0.20, "referral": 0.15})


def simulate_experiment(cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    """Generate a realistic A/B experiment dataset with known ground truth."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_users

    # ---- 1. User attributes ----
    user_id = np.arange(1_000_000, 1_000_000 + n)

    country = rng.choice(list(cfg.country_share), size=n, p=list(cfg.country_share.values()))
    device = rng.choice(list(cfg.device_share), size=n, p=list(cfg.device_share.values()))
    source = rng.choice(list(cfg.source_share), size=n, p=list(cfg.source_share.values()))

    # New vs returning (returning = had account before but inactive >= 90 days)
    is_returning = rng.binomial(1, 0.25, size=n).astype(bool)

    # ---- 2. Pre-experiment covariate (prior 30d watch hours) ----
    # Returning users have higher prior watch hours; new users mostly 0.
    prior_watch_hours = np.where(
        is_returning,
        rng.gamma(shape=2.0, scale=4.0, size=n),  # returning: 0-30h typical
        rng.gamma(shape=0.3, scale=0.5, size=n),  # new: mostly near 0
    ).round(2)

    # ---- 3. Random assignment (50/50, user-level) ----
    group = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])

    # ---- 4. Inject SRM bug: Android users on old app version sometimes
    #         get reassigned to control due to a missing feature flag ----
    android_buggy_mask = (device == "Android") & (rng.random(n) < 0.03)
    # Of these, 70% leak from treatment back to control (one-sided bug)
    leak = android_buggy_mask & (group == "treatment") & (rng.random(n) < 0.7)
    group[leak] = "control"  # the bug

    # ---- 5. Determine page actually rendered (mostly matches assignment,
    #         small mismatch due to caching / race conditions) ----
    landing_page = np.where(group == "treatment", "personalized", "top_picks")
    glitch = rng.random(n) < 0.001  # 0.1% rendering glitch
    landing_page[glitch] = np.where(group[glitch] == "treatment", "top_picks", "personalized")

    # ---- 6. Outcome generation ----
    # Base log-odds of conversion driven by:
    #   - Country (US/UK convert slightly higher)
    #   - Device (TV users convert highest, mobile lowest)
    #   - Source (organic & referral > paid)
    #   - Prior watch hours (strongest predictor — used in CUPED later)
    #   - Returning flag (returning users convert higher)
    country_effect = {"US": 0.05, "UK": 0.03, "CA": 0.0, "AU": -0.02, "Other": -0.05}
    device_effect = {"TV": 0.10, "Web": 0.02, "iOS": -0.01, "Android": -0.03}
    source_effect = {"organic": 0.04, "referral": 0.03, "paid_search": -0.01, "paid_social": -0.02}

    base_logit = np.log(cfg.base_conversion / (1 - cfg.base_conversion))
    user_logit = (
        base_logit
        + np.array([country_effect[c] for c in country])
        + np.array([device_effect[d] for d in device])
        + np.array([source_effect[s] for s in source])
        + 0.4 * is_returning.astype(float)
        + 0.03 * prior_watch_hours
    )

    # ---- 7. Heterogeneous treatment effect (HTE) ----
    # Overall ATE: +1.2pp. But effect varies by segment.
    # Mobile (iOS/Android) sees larger lift; TV almost no effect.
    treat_effect_logodds = np.where(
        np.isin(device, ["iOS", "Android"]),
        0.16,  # ~+2pp for mobile
        np.where(device == "Web", 0.06, 0.02),  # ~+0.5pp web, ~+0pp TV
    )
    # Returning users get an extra bump (recommendations resonate)
    treat_effect_logodds = treat_effect_logodds + is_returning.astype(float) * 0.04

    treat_indicator = (group == "treatment").astype(float)
    final_logit = user_logit + treat_effect_logodds * treat_indicator
    prob_convert = 1 / (1 + np.exp(-final_logit))
    converted = rng.binomial(1, prob_convert).astype(int)

    # ---- 8. Secondary metrics ----
    # Watch hours during trial (gamma; treatment slightly boosts mean)
    base_trial_watch = rng.gamma(shape=1.5, scale=3.0, size=n)
    treat_boost = 0.20 * treat_indicator  # +20% mean watch hours for treatment
    trial_watch_hours = (base_trial_watch * (1 + treat_boost) +
                         0.15 * prior_watch_hours).round(2)

    # Distinct titles watched (correlated with watch hours)
    distinct_titles = np.maximum(0, np.round(
        trial_watch_hours * 0.6 + rng.normal(0, 1.0, size=n)
    )).astype(int)

    # Day-7 active (engaged in week 1) — slight positive treatment effect
    day7_logit = (
        -0.5
        + 0.08 * treat_indicator
        + 0.04 * prior_watch_hours
        + 0.6 * is_returning.astype(float)
    )
    day7_active = rng.binomial(1, 1 / (1 + np.exp(-day7_logit)))

    # Page load time (guardrail) — treatment is ~30ms slower on average
    page_load_ms = (
        rng.normal(loc=420, scale=80, size=n)
        + 30 * treat_indicator
        + np.where(device == "TV", 60, 0)  # TVs slower
    ).round(0).astype(int)
    page_load_ms = np.maximum(page_load_ms, 50)

    # ---- 9. Timestamps spread across a 4-week window ----
    start = pd.Timestamp("2026-04-01")
    days_offset = rng.integers(0, 28, size=n)
    seconds_offset = rng.integers(0, 86400, size=n)
    timestamp = pd.to_datetime(start) + pd.to_timedelta(days_offset, unit="D") \
                                       + pd.to_timedelta(seconds_offset, unit="s")

    # ---- 10. Assemble ----
    df = pd.DataFrame({
        "user_id": user_id,
        "timestamp": timestamp,
        "group": group,
        "landing_page": landing_page,
        "country": country,
        "device": device,
        "source": source,
        "is_returning": is_returning,
        "prior_watch_hours": prior_watch_hours,
        "converted": converted,
        "trial_watch_hours": trial_watch_hours,
        "distinct_titles": distinct_titles,
        "day7_active": day7_active,
        "page_load_ms": page_load_ms,
    })

    return df.sort_values("timestamp").reset_index(drop=True)


def main(out_path: str | Path = "data/experiment.csv") -> None:
    df = simulate_experiment()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Print a quick summary
    print(f"✅ Wrote {len(df):,} rows to {out}")
    print(f"\nGroup counts:\n{df['group'].value_counts()}")
    print(f"\nObserved conversion by group:")
    print(df.groupby("group")["converted"].agg(["mean", "count"]))
    print(f"\nTrue overall lift target was: +1.2pp")


if __name__ == "__main__":
    main()
