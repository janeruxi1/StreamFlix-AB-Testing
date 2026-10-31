"""Recorded results. Real-data runs write to reports/metrics/; synthetic runs (CI, or no data
downloaded) write to a scratch folder so they never overwrite the record."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 8)
    return x


def out_dir(source: str) -> Path:
    d = METRICS if source == "real" else Path(tempfile.gettempdir()) / "abtest_synthetic_metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write(name: str, payload: dict, source: str) -> Path:
    path = out_dir(source) / f"{name}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(_clean({"source": source, **payload}), indent=2) + "\n")
    return path


def read(name: str, source: str = "real") -> dict:
    return json.loads((out_dir(source) / f"{name}.json").read_text(encoding="utf-8"))


def save_figure(fig, name: str, source: str) -> None:
    if source == "real":
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
