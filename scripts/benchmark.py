#!/usr/bin/env python
"""Run one experiment: a pipeline on the ground-truth validation set + the target song.

Writes experiments/<id>/{config.yaml,metrics.json,notes.md,output.wav,per_clip.csv},
appends experiments/results.csv, saves the target candidate to
outputs/target/candidates/<id>.wav and runs the Champion/Challenger check.

    python scripts/benchmark.py --pipeline configs/pipelines/direct_becruily.yaml \
        --slug becruily --strategy A --hypothesis "..." --change "..."
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator import tracking  # noqa: E402
from acoustic_separator.audio import load_audio, save_audio  # noqa: E402
from acoustic_separator.evaluation import aggregate, evaluate_estimate  # noqa: E402
from acoustic_separator.pipeline import ModelPool, PipelineRunner, describe, load_pipeline  # noqa: E402

VAL_DIR = ROOT / "datasets" / "validation"
TARGET = ROOT / "outputs" / "target" / "original.wav"


def load_clip(d: Path):
    mix, sr = load_audio(d / "mixture.wav")
    ref, _ = load_audio(d / "acoustic_guitar.wav")
    meta = json.loads((d / "meta.json").read_text())
    interferers = {}
    for p in sorted((d / "stems").glob("*.wav")):
        interferers[p.stem], _ = load_audio(p)
    return mix, ref, interferers, meta, sr


def run_validation(runner: PipelineRunner, pipeline: dict, val_dir: Path, limit: int | None,
                   with_bss: bool) -> tuple[list[dict], float]:
    rows, runtime = [], 0.0
    clips = sorted(p for p in val_dir.iterdir() if (p / "mixture.wav").exists())
    if limit:
        clips = clips[:limit]
    for i, d in enumerate(clips):
        mix, ref, interf, meta, sr = load_clip(d)
        t = time.perf_counter()
        res = runner.run(pipeline, mix, sr)
        runtime += time.perf_counter() - t
        m = evaluate_estimate(ref, res["output"], mix, interf, with_bss=with_bss)
        m.update({"clip": d.name, "category": meta.get("category", ""),
                  "hard": int(bool(meta.get("hard", False)))})
        rows.append(m)
        print(f"  [{i + 1}/{len(clips)}] {d.name:40s} sdr={m['sdr']:6.2f} si_sdr={m['si_sdr']:6.2f} "
              f"ret={m['target_retention']:.2f} leak={m['leakage_db']:6.1f}", flush=True)
    return rows, runtime


def summarize(rows: list[dict]) -> dict:
    agg = aggregate(rows)
    hard = [r for r in rows if r.get("hard")]
    if hard:
        agg["hard_sdr_mean"] = float(np.mean([r["sdr"] for r in hard]))
    for fam in ("ms", "syn"):
        fr = [r for r in rows if r.get("clip", "").startswith(fam + "_")]
        if fr:
            for k in ("sdr", "si_sdr", "target_retention", "leakage_db", "sir_bss", "sar_bss"):
                vals = [r[k] for r in fr if k in r and np.isfinite(r[k])]
                if vals:
                    agg[f"{k}_{fam}_mean"] = float(np.mean(vals))
    cats = sorted({r["category"] for r in rows if r.get("category")})
    agg["by_category"] = {c: {k: float(np.mean([r[k] for r in rows if r["category"] == c]))
                              for k in ("sdr", "si_sdr", "target_retention", "leakage_db")}
                          for c in cats}
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--strategy", default="", help="A direct | B two-stage | C cascade | D ensemble | ...")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--next", default="")
    ap.add_argument("--val-dir", default=str(VAL_DIR))
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--no-target", action="store_true")
    ap.add_argument("--no-bss", action="store_true", help="skip BSS-Eval SIR/SAR (faster)")
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--no-champion", action="store_true", help="do not run Champion/Challenger")
    ap.add_argument("--exp-id", default=None)
    args = ap.parse_args()

    pipeline = load_pipeline(args.pipeline)
    exp_id = args.exp_id or tracking.next_experiment_id(args.slug)
    exp_dir = tracking.EXPERIMENTS / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"== {exp_id}: pipeline {pipeline['name']}")
    runner = PipelineRunner(ModelPool(capacity=2))

    metrics: dict = {"experiment": exp_id}
    val_dir = Path(args.val_dir)
    if val_dir.exists():
        rows, rt = run_validation(runner, pipeline, val_dir, args.val_limit, not args.no_bss)
        metrics["validation"] = summarize(rows)
        metrics["validation"]["runtime_s"] = rt
        with open(exp_dir / "per_clip.csv", "w", newline="") as f:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    v = metrics.get("validation", {})

    tgt_rt = None
    if not args.no_target and TARGET.exists():
        mix, sr = load_audio(TARGET)
        t = time.perf_counter()
        res = runner.run(pipeline, mix, sr)
        tgt_rt = time.perf_counter() - t
        cand_dir = ROOT / "outputs" / "target" / "candidates"
        save_audio(cand_dir / f"{exp_id}.wav", res["output"], sr)
        save_audio(cand_dir / f"{exp_id}_residual.wav", res["residual"], sr)
        save_audio(exp_dir / "output.wav", res["output"], sr)
        metrics["target"] = {"runtime_s": tgt_rt, "model_runtime_s": res["runtime"],
                             "step_times": res["step_times"]}
        if not args.no_proxy:
            from acoustic_separator.proxy import proxy_scores

            metrics["target"]["proxy"] = proxy_scores(res["output"], res["residual"], sr)

    provenance = describe(pipeline)
    config = {"experiment": exp_id, "strategy": args.strategy, **provenance, "seed": 0}
    (exp_dir / "pipeline.yaml").write_text(yaml.safe_dump(pipeline, sort_keys=False))

    promoted = False
    champ_before = tracking.current_champion()
    if v and not args.no_champion:
        promoted = tracking.challenge(exp_id, pipeline, v)
        if promoted:
            best = ROOT / "outputs" / "target" / "best"
            best.mkdir(parents=True, exist_ok=True)
            src = ROOT / "outputs" / "target" / "candidates"
            if (src / f"{exp_id}.wav").exists():
                shutil.copy2(src / f"{exp_id}.wav", best / "acoustic_guitar.wav")
                shutil.copy2(src / f"{exp_id}_residual.wav", best / "non_acoustic_guitar.wav")
                tracking.copy_champion_audio(best)
    metrics["is_champion"] = promoted

    champ_line = "no champion yet"
    if champ_before:
        prev = champ_before["metrics"][tracking.PRIMARY_METRIC]
        champ_line = (f"champion before: {champ_before['experiment']} "
                      f"({tracking.PRIMARY_METRIC}={prev:.3f}); delta = "
                      f"{v.get(tracking.PRIMARY_METRIC, float('nan')) - prev:+.3f} dB")
    p = metrics.get("target", {}).get("proxy", {})
    result = (f"- validation SDR mean/median: {v.get('sdr_mean', float('nan')):.3f} / "
              f"{v.get('sdr_median', float('nan')):.3f} dB, SI-SDR mean {v.get('si_sdr_mean', float('nan')):.3f}, "
              f"SDRi {v.get('sdri_mean', float('nan')):.3f}\n"
              f"- SIR/SAR (BSS): {v.get('sir_bss_mean', float('nan')):.2f} / {v.get('sar_bss_mean', float('nan')):.2f} dB\n"
              f"- by family: ms_* (real multitracks) SDR {v.get('sdr_ms_mean', float('nan')):.3f}, "
              f"syn_* (synthetic) SDR {v.get('sdr_syn_mean', float('nan')):.3f}\n"
              f"- target retention {v.get('target_retention_mean', float('nan')):.3f}, "
              f"leakage {v.get('leakage_db_mean', float('nan')):.2f} dB, hard-case SDR "
              f"{v.get('hard_sdr_mean', float('nan')):.3f}\n"
              f"- target-song proxy: guitar prob {p.get('guitar_prob', float('nan')):.3f}, "
              f"max leak prob {p.get('leak_prob_max', float('nan')):.3f}, "
              f"residual guitar prob {p.get('residual_guitar_prob', float('nan')):.3f}\n"
              f"- runtime: validation {v.get('runtime_s', float('nan')):.0f}s, target "
              f"{tgt_rt if tgt_rt is not None else float('nan'):.0f}s")
    notes = (f"# {exp_id}\n\n## Hypothesis\n{args.hypothesis}\n\n## Change\n{args.change}\n\n"
             f"## Result\n{result}\n\n## Conclusion\n"
             f"{'PROMOTED to Champion. ' if promoted else 'Not promoted. '}{champ_line}\n\n"
             f"## Next experiment\n{args.next}\n")
    tracking.write_experiment(exp_id, config, metrics, notes)

    first = next((s for s in pipeline["steps"] if "model" in s), {})
    prm = first.get("params", {})
    tracking.append_results({
        "experiment": exp_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "strategy": args.strategy, "pipeline": pipeline["name"],
        "models": "+".join(dict.fromkeys(s["model"] for s in pipeline["steps"] if "model" in s)),
        "checkpoints": "+".join((m.get("sha256") or "")[:12] for m in provenance["models"].values()),
        "chunk": prm.get("chunk_size", "default"), "overlap": prm.get("num_overlap", 4),
        "tta": ",".join(prm.get("tta", [])) or "none",
        "sdr": v.get("sdr_mean"), "si_sdr": v.get("si_sdr_mean"), "sdri": v.get("sdri_mean"),
        "sir": v.get("sir_bss_mean"), "sar": v.get("sar_bss_mean"),
        "target_retention": v.get("target_retention_mean"), "leakage": v.get("leakage_db_mean"),
        "mrstft": v.get("mrstft_mean"), "hard_sdr": v.get("hard_sdr_mean"),
        "sdr_ms": v.get("sdr_ms_mean"), "sdr_syn": v.get("sdr_syn_mean"),
        "target_guitar_prob": p.get("guitar_prob"), "target_leak_prob": p.get("leak_prob_max"),
        "runtime_val_s": v.get("runtime_s"), "runtime_target_s": tgt_rt,
        "is_champion": int(promoted), "git_commit": tracking.git_commit(),
    })
    print(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
