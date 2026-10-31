"""Check that re-running the notebooks reproduces the committed results.

CI runs every notebook, then this script compares each file in reports/metrics/ with the
version committed in git. The Upworthy notebooks (04-06) and the known-truth notebook (07)
rewrite their records in CI, so they are checked for real. The Hillstrom notebooks (01-03)
run on a synthetic stand-in in CI, because the Hillstrom file is not redistributed, and write
to a scratch folder instead; their committed records are checked by tests/test_reported_numbers.py.

Values are compared with a relative tolerance, since the last digits can differ across
machines.

    python scripts/check_metrics.py            # compare with HEAD
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
RTOL, ATOL = 1e-4, 1e-6


def committed(path: Path) -> str | None:
    rel = path.relative_to(ROOT).as_posix()
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def compare(a, b, where: str, problems: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            problems.append(f"{where}: keys differ")
        for k in a.keys() & b.keys():
            compare(a[k], b[k], f"{where}.{k}", problems)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{where}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{where}[{i}]", problems)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            problems.append(f"{where}: committed {a} vs new {b}")
    elif a != b:
        problems.append(f"{where}: committed {a!r} vs new {b!r}")


def main() -> int:
    problems: list[str] = []
    files = sorted(METRICS.glob("*.json"))
    for path in files:
        old = committed(path)
        if old is None:
            problems.append(f"{path.name}: not committed yet")
            continue
        compare(json.loads(old), json.loads(path.read_text(encoding="utf-8")), path.name, problems)
    if problems:
        print("Re-running the notebooks changed the recorded results:")
        print("\n".join(f"  {p}" for p in problems[:50]))
        return 1
    print(f"All {len(files)} metrics files match the committed results (rtol {RTOL}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
