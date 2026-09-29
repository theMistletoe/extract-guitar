#!/usr/bin/env python
"""Render the target song for an existing experiment (whose validation ran with
--no-target), then update its metrics.json, results.csv row and target candidate.

    python scripts/target_only.py exp012_sweep_xyz
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator import tracking  # noqa: E402
from acoustic_separator.audio import load_audio, save_audio  # noqa: E402
from acoustic_separator.pipeline import ModelPool, PipelineRunner  # noqa: E402
from acoustic_separator.proxy import proxy_scores  # noqa: E402


def main(exp: str) -> int:
    d = ROOT / "experiments" / exp
    pipe = yaml.safe_load((d / "pipeline.yaml").read_text())
    mix, sr = load_audio(ROOT / "outputs" / "target" / "original.wav")
    t = time.perf_counter()
    res = PipelineRunner(ModelPool()).run(pipe, mix, sr)
    rt = time.perf_counter() - t
    cand = ROOT / "outputs" / "target" / "candidates"
    save_audio(cand / f"{exp}.wav", res["output"], sr)
    save_audio(cand / f"{exp}_residual.wav", res["residual"], sr)
    save_audio(d / "output.wav", res["output"], sr)
    m = json.loads((d / "metrics.json").read_text())
    m["target"] = {"runtime_s": rt, "model_runtime_s": res["runtime"], "step_times": res["step_times"],
                   "proxy": proxy_scores(res["output"], res["residual"], sr)}
    (d / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    rows = tracking.read_results()
    for r in rows:
        if r["experiment"] == exp:
            r["runtime_target_s"] = f"{rt:.4f}"
            r["target_guitar_prob"] = f"{m['target']['proxy']['guitar_prob']:.4f}"
            r["target_leak_prob"] = f"{m['target']['proxy']['leak_prob_max']:.4f}"
    with open(tracking.RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tracking.CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    champ = tracking.current_champion()
    if champ and champ["experiment"] == exp:
        best = ROOT / "outputs" / "target" / "best"
        best.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cand / f"{exp}.wav", best / "acoustic_guitar.wav")
        shutil.copy2(cand / f"{exp}_residual.wav", best / "non_acoustic_guitar.wav")
        tracking.copy_champion_audio(best)
    print(exp, "target rendered in", round(rt), "s", m["target"]["proxy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
