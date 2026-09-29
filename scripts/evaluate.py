#!/usr/bin/env python
"""Evaluation utilities.

    # re-aggregate an experiment's per-clip metrics (e.g. after adding new aggregate keys)
    python scripts/evaluate.py reaggregate exp001_A_htdemucs6s

    # evaluate a folder of estimates <clip>.wav against the validation set
    python scripts/evaluate.py folder path/to/estimates/

    # compare experiments per clip family / category
    python scripts/evaluate.py compare exp001_A_htdemucs6s exp002_A_becruily
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from acoustic_separator import tracking  # noqa: E402
from acoustic_separator.audio import load_audio  # noqa: E402
from acoustic_separator.evaluation import evaluate_estimate  # noqa: E402

VAL = ROOT / "datasets" / "validation"


def read_per_clip(exp: str) -> list[dict]:
    rows = list(csv.DictReader(open(ROOT / "experiments" / exp / "per_clip.csv")))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
        out.append(d)
    return out


def cmd_reaggregate(a):
    from benchmark import summarize

    for exp in a.experiments:
        rows = read_per_clip(exp)
        mfile = ROOT / "experiments" / exp / "metrics.json"
        m = json.loads(mfile.read_text())
        rt = m.get("validation", {}).get("runtime_s")
        m["validation"] = summarize(rows)
        if rt is not None:
            m["validation"]["runtime_s"] = rt
        mfile.write_text(json.dumps(m, indent=2, ensure_ascii=False))
        # patch results.csv row
        res = tracking.read_results()
        for r in res:
            if r["experiment"] == exp:
                v = m["validation"]
                r["sdr_ms"] = f"{v.get('sdr_ms_mean', float('nan')):.4f}"
                r["sdr_syn"] = f"{v.get('sdr_syn_mean', float('nan')):.4f}"
        with open(tracking.RESULTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=tracking.CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(res)
        print(exp, {k: round(v, 3) for k, v in m["validation"].items() if k.startswith("sdr")})


def cmd_folder(a):
    rows = []
    for d in sorted(p for p in VAL.iterdir() if (p / "mixture.wav").exists()):
        est_path = Path(a.folder) / f"{d.name}.wav"
        if not est_path.exists():
            continue
        mix, _ = load_audio(d / "mixture.wav")
        ref, _ = load_audio(d / "acoustic_guitar.wav")
        est, _ = load_audio(est_path)
        interf = {p.stem: load_audio(p)[0] for p in sorted((d / "stems").glob("*.wav"))}
        m = evaluate_estimate(ref, est, mix, interf)
        m["clip"] = d.name
        rows.append(m)
        print(f"{d.name:40s} sdr={m['sdr']:.2f}")
    print("mean SDR", np.mean([r["sdr"] for r in rows]))


def cmd_compare(a):
    table = {e: {r["clip"]: r for r in read_per_clip(e)} for e in a.experiments}
    clips = sorted(set.intersection(*[set(t) for t in table.values()]))
    key = a.metric
    print(f"{'clip':40s} " + " ".join(f"{e[:18]:>18s}" for e in a.experiments))
    for c in clips:
        print(f"{c:40s} " + " ".join(f"{table[e][c][key]:18.2f}" for e in a.experiments))
    for fam in ("ms_", "syn_", ""):
        sel = [c for c in clips if c.startswith(fam)]
        print(f"{'mean ' + (fam or 'all'):40s} " + " ".join(
            f"{np.mean([table[e][c][key] for c in sel]):18.2f}" for e in a.experiments))
    # paired comparison of every experiment against the first one
    rng = np.random.default_rng(0)
    base = np.array([table[a.experiments[0]][c][key] for c in clips])
    for e in a.experiments[1:]:
        d = np.array([table[e][c][key] for c in clips]) - base
        boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{e} - {a.experiments[0]}: mean diff {d.mean():+.3f} dB, 95% bootstrap CI "
              f"[{lo:+.3f}, {hi:+.3f}], wins {int((d > 0).sum())}/{len(d)} clips")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reaggregate")
    p.add_argument("experiments", nargs="+")
    p = sub.add_parser("folder")
    p.add_argument("folder")
    p = sub.add_parser("compare")
    p.add_argument("experiments", nargs="+")
    p.add_argument("--metric", default="sdr")
    a = ap.parse_args()
    {"reaggregate": cmd_reaggregate, "folder": cmd_folder, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
