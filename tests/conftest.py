"""Shared pytest fixtures for the A/B test analysis project."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable from anywhere in the test suite
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def small_experiment():
    """Tiny synthetic experiment for fast unit tests.

    2,000 users per arm, baseline 20% conversion, treatment 25% conversion,
    seeded so results are deterministic across runs.
    """
    rng = np.random.default_rng(123)
    n = 2_000
    control = pd.DataFrame({
        "group": "control",
        "converted": rng.binomial(1, 0.20, size=n),
        "metric_continuous": rng.normal(loc=10.0, scale=3.0, size=n),
        "pre_covariate": rng.normal(loc=5.0, scale=2.0, size=n),
        "segment": rng.choice(["A", "B"], size=n, p=[0.6, 0.4]),
    })
    treatment = pd.DataFrame({
        "group": "treatment",
        "converted": rng.binomial(1, 0.25, size=n),
        "metric_continuous": rng.normal(loc=10.8, scale=3.2, size=n),
        "pre_covariate": rng.normal(loc=5.0, scale=2.0, size=n),
        "segment": rng.choice(["A", "B"], size=n, p=[0.6, 0.4]),
    })
    # Add correlation between pre_covariate and metric_continuous so CUPED has signal
    df = pd.concat([control, treatment], ignore_index=True)
    df["metric_continuous"] = df["metric_continuous"] + 0.5 * df["pre_covariate"]
    return df
