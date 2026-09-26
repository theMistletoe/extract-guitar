#!/usr/bin/env python
"""Failure-mode analysis of an experiment on the validation set (PRD §23, §37).

For every clip: SDR, retention, the interfering class with the largest attributed leakage,
and a failure label (guitar_removed / <class>_leak / ok). Writes reports/failure_modes.md.

    python scripts/failure_analysis.py [exp_id]     # default: current Champion
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator import tracking  # noqa: E402


def classify(r: dict) -> tuple[str, str | None, float | None]:
    leaks = {k[5:-3]: float(v) for k, v in r.items() if k.startswith("leak_") and k.endswith("_db")
             and k != "leakage_db" and v not in ("", None)}
    worst = max(leaks, key=leaks.get) if leaks else None
    ret = float(r["target_retention"])
    sdr = float(r["sdr"])
    if ret < 0.4:
        label = "guitar_removed"
    elif worst is not None and leaks[worst] > -12:
        label = f"{worst.rstrip('0123456789')}_leak"
    elif sdr < 3:
        label = "artifacts/other"
    else:
        label = "ok"
    return label, worst, leaks.get(worst) if worst else None


def main(exp: str | None) -> int:
    exp = exp or tracking.current_champion()["experiment"]
    rows = list(csv.DictReader(open(ROOT / "experiments" / exp / "per_clip.csv")))
    out = [f"# Failure modes — {exp}", "",
           "Per-clip labels on the ground-truth validation set: `guitar_removed` (retention < 0.4), "
           "`<class>_leak` (largest attributed leakage > −12 dB re. target energy), "
           "`artifacts/other` (SDR < 3 dB without a dominant leak), else `ok`.", "",
           "| clip | SDR | retention | worst leak class | leak dB | label |", "|---|---|---|---|---|---|"]
    labels = Counter()
    by_label = defaultdict(list)
    for r in sorted(rows, key=lambda r: float(r["sdr"])):
        label, worst, lv = classify(r)
        labels[label] += 1
        by_label[label].append(float(r["sdr"]))
        out.append(f"| {r['clip']} | {float(r['sdr']):.2f} | {float(r['target_retention']):.2f} | "
                   f"{worst or '-'} | {'' if lv is None else f'{lv:.1f}'} | {label} |")
    out += ["", "## Summary", "", "| label | clips | mean SDR |", "|---|---|---|"]
    for lab, n in labels.most_common():
        out.append(f"| {lab} | {n} | {np.mean(by_label[lab]):.2f} |")
    path = ROOT / "reports" / "failure_modes.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out[-(len(labels) + 3):]))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
