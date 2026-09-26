#!/usr/bin/env python
"""Recompute an experiment's separation compute time from the stem cache (each cache
entry stores the time measured when it was computed) and patch metrics.json/results.csv.

    python scripts/backfill_runtime.py exp003_A_htdemucs6s_ft
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator import tracking  # noqa: E402
from acoustic_separator.audio import load_audio  # noqa: E402
from acoustic_separator.pipeline import ModelPool, PipelineRunner  # noqa: E402


def main(exps: list[str]) -> int:
    runner = PipelineRunner(ModelPool())
    val = sorted(p for p in (ROOT / "datasets" / "validation").iterdir() if (p / "mixture.wav").exists())
    mixt, sr = load_audio(ROOT / "outputs" / "target" / "original.wav")
    rows = tracking.read_results()
    for exp in exps:
        d = ROOT / "experiments" / exp
        pipe = yaml.safe_load((d / "pipeline.yaml").read_text())
        vrt = sum(runner.run(pipe, load_audio(c / "mixture.wav")[0], 44100)["runtime"] for c in val)
        trt = runner.run(pipe, mixt, sr)["runtime"]
        m = json.loads((d / "metrics.json").read_text())
        m["validation"]["runtime_s"] = vrt
        m.setdefault("target", {})["runtime_s"] = trt
        (d / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
        for r in rows:
            if r["experiment"] == exp:
                r["runtime_val_s"], r["runtime_target_s"] = f"{vrt:.4f}", f"{trt:.4f}"
        print(exp, round(vrt), round(trt))
    with open(tracking.RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tracking.CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
