"""The Upworthy Research Archive: 32,487 headline tests run by Upworthy, 2013-2015.

Matias, Munger, Le Quere and Ebersole (2021), "The Upworthy Research Archive, a time series of
32,487 experiments in U.S. media", Scientific Data 8:195. Licensed CC BY 4.0
(https://upworthy.natematias.com/). Each row is one "package" (headline, image, ...) in one test,
with the impressions it received and the clicks it got.

The raw CSVs (March 2020 release) are downloaded into data/raw/; ``build_compact`` writes a
compact copy of the columns this project uses to data/upworthy/, which is committed (the licence
allows it, with attribution) so CI runs on the real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
COMPACT_DIR = ROOT / "data" / "upworthy"
SAMPLES = ("exploratory", "confirmatory", "holdout")
RAW_NAME = "upworthy-archive-{sample}-packages-03.12.2020.csv"
KEEP = ["clickability_test_id", "created_at", "headline", "eyecatcher_id", "impressions", "clicks", "winner"]

# Maintainers' June 2024 note: randomization may be broken for tests run in this window.
WINDOW_START, WINDOW_END = pd.Timestamp("2013-06-25"), pd.Timestamp("2014-01-10")


def build_compact(sample: str, raw_dir: Path = RAW_DIR, out_dir: Path = COMPACT_DIR) -> Path:
    raw = pd.read_csv(raw_dir / RAW_NAME.format(sample=sample), usecols=KEEP)
    raw["created_at"] = pd.to_datetime(raw["created_at"], format="ISO8601")
    raw = raw.sort_values(["clickability_test_id", "created_at"], kind="stable").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sample}.parquet"
    raw.to_parquet(path, index=False, compression="zstd")
    return path


def load(sample: str, compact_dir: Path = COMPACT_DIR) -> pd.DataFrame:
    """Packages of one sample, with each test's start date and whether it falls in the window."""
    if sample not in SAMPLES:
        raise ValueError(f"unknown sample {sample!r}; expected one of {SAMPLES}")
    df = pd.read_parquet(compact_dir / f"{sample}.parquet")
    df["test_start"] = df.groupby("clickability_test_id")["created_at"].transform("min")
    df["in_window"] = (df["test_start"] >= WINDOW_START) & (df["test_start"] < WINDOW_END + pd.Timedelta(days=1))
    return df


def make_synthetic(n_tests: int = 400, seed: int = 0, tau: float = 0.1, broken_share: float = 0.2) -> pd.DataFrame:
    """Tests with known true headline differences (log click-through, sd ``tau``) and a share of
    tests whose impressions were split unevenly, for checking the estimators."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tests):
        k = int(rng.integers(2, 7))
        base = rng.uniform(0.008, 0.025)
        theta = rng.normal(0, tau, k)
        broken = rng.random() < broken_share
        shares = rng.dirichlet(np.full(k, 3.0)) if broken else np.full(k, 1 / k)
        n = rng.multinomial(int(rng.integers(8_000, 25_000)), shares)
        p = np.clip(base * np.exp(theta - theta.mean()), 0, 1)
        c = rng.binomial(n, p)
        start = pd.Timestamp("2013-03-01") + pd.Timedelta(days=int(rng.integers(0, 700)))
        for j in range(k):
            rows.append({"clickability_test_id": f"t{t:04d}", "created_at": start, "headline": f"h{t}-{j}",
                         "eyecatcher_id": "", "impressions": int(n[j]), "clicks": int(c[j]), "winner": False,
                         "true_theta": float(theta[j] - theta.mean()), "broken": broken})
    df = pd.DataFrame(rows)
    df["test_start"] = df["created_at"]
    df["in_window"] = False
    return df
