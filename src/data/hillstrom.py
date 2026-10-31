"""The Hillstrom e-mail test: loading, checking, and a stand-in for CI.

Kevin Hillstrom's MineThatData challenge (March 2008): 64,000 customers who bought in the past
12 months, randomized in equal thirds to a men's e-mail, a women's e-mail or no e-mail, with
visits, purchases and spend recorded for the two weeks after the send.

The file carries no stated licence, so it is not committed. Download it once into
``data/raw/`` (see data/README.md). Without it, ``load()`` returns a synthetic table with the
same columns and planted effects, which is what CI runs on; synthetic runs never overwrite the
recorded results.
"""
from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CANDIDATES = ("Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
              "hillstorm_no_indices.csv.gz", "hillstrom.csv")

ARMS = {"No E-Mail": "none", "Mens E-Mail": "mens", "Womens E-Mail": "womens"}
ARM_ORDER = ["none", "mens", "womens"]
PRE = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
OUTCOMES = ["visit", "conversion", "spend"]
COLUMNS = PRE + ["segment"] + OUTCOMES


def find_raw(raw_dir: Path = RAW_DIR) -> Path | None:
    for name in CANDIDATES:
        p = raw_dir / name
        if p.exists():
            return p
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    return df[COLUMNS]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Arm labels, typed columns, and the derived flags the analysis plan names."""
    out = df.copy()
    unknown = set(out["segment"].unique()) - set(ARMS)
    if unknown:
        raise ValueError(f"unexpected arm labels {sorted(unknown)}")
    out["arm"] = pd.Categorical(out["segment"].map(ARMS), categories=ARM_ORDER)
    for c in ("mens", "womens", "newbie", "visit", "conversion"):
        out[c] = out[c].astype(int)
    out["recency"] = out["recency"].astype(int)
    out["history"] = out["history"].astype(float)
    out["spend"] = out["spend"].astype(float)
    out["purchase_line"] = np.select([(out["mens"] == 1) & (out["womens"] == 1), out["mens"] == 1,
                                      out["womens"] == 1], ["both", "mens only", "womens only"], "neither")
    return out.drop(columns="segment")


def use_synthetic() -> bool:
    return os.environ.get("HILLSTROM_SYNTHETIC", "") in ("1", "true", "yes")


def load(raw_dir: Path = RAW_DIR, n_synthetic: int = 64_000, seed: int = 0) -> tuple[pd.DataFrame, str]:
    """``(customers, source)``: the real file when present, else a synthetic stand-in."""
    path = None if use_synthetic() else find_raw(raw_dir)
    if path is None:
        return prepare(make_synthetic(n_synthetic, seed)), "synthetic"
    return prepare(read_raw(path)), "real"


# Planted effects for the synthetic stand-in, so tests can check the estimators recover them.
TRUTH = {"visit": {"mens": 0.06, "womens": 0.045}, "conversion": {"mens": 0.006, "womens": 0.004},
         "spend_per_buyer": 110.0, "mens_line_bonus": 0.03}


def make_synthetic(n: int = 64_000, seed: int = 0) -> pd.DataFrame:
    """A table with Hillstrom's columns, randomized in thirds, with known effects.

    Nothing is calibrated to reproduce the real results; it only has to exercise the code and
    give estimators a truth to recover. The men's e-mail works better for customers who bought
    men's merchandise (``mens_line_bonus`` on the visit rate), so the targeting step has
    something to find.
    """
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 13, n)
    history = np.round(np.exp(rng.normal(5.0, 0.9, n)).clip(29.99, 3345.93), 2)
    mens = rng.binomial(1, 0.55, n)
    womens = np.where(mens == 0, 1, rng.binomial(1, 0.55, n))
    newbie = rng.binomial(1, 0.5, n)
    zip_code = rng.choice(["Urban", "Surburban", "Rural"], n, p=[0.4, 0.45, 0.15])
    channel = rng.choice(["Phone", "Web", "Multichannel"], n, p=[0.44, 0.44, 0.12])
    bins = [0, 100, 200, 350, 500, 750, 1000, np.inf]
    labels = ["1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500", "5) $500 - $750",
              "6) $750 - $1,000", "7) $1,000 +"]
    history_segment = pd.cut(history, bins, labels=labels, right=False).astype(str)
    arm = rng.permutation(np.repeat(np.arange(3), -(-n // 3))[:n])   # exact thirds, shuffled
    base_visit = 1 / (1 + np.exp(-(-1.9 + 0.25 * np.log(history / 100) - 0.05 * (recency - 6) + 0.1 * (channel == "Web"))))
    lift = np.select([arm == 1, arm == 2], [TRUTH["visit"]["mens"] + TRUTH["mens_line_bonus"] * (mens - 0.55),
                                            TRUTH["visit"]["womens"]], 0.0)
    visit = rng.binomial(1, np.clip(base_visit + lift, 0, 1))
    conv_given_visit = np.clip(0.06 + np.select([arm == 1, arm == 2], [0.02, 0.015], 0.0), 0, 1)
    conversion = visit * rng.binomial(1, conv_given_visit)
    spend = np.where(conversion == 1, np.round(rng.gamma(2.0, TRUTH["spend_per_buyer"] / 2.0, n), 2), 0.0)
    names = {0: "No E-Mail", 1: "Mens E-Mail", 2: "Womens E-Mail"}
    return pd.DataFrame({"recency": recency, "history_segment": history_segment, "history": history,
                         "mens": mens, "womens": womens, "zip_code": zip_code, "newbie": newbie,
                         "channel": channel, "segment": [names[a] for a in arm], "visit": visit,
                         "conversion": conversion, "spend": spend})[COLUMNS]
