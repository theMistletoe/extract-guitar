#!/usr/bin/env python
"""Build outputs/target/report.html comparing every target-song candidate.

Candidates are the files outputs/target/candidates/<experiment>.wav written by
scripts/benchmark.py; metrics and provenance are read from experiments/<experiment>/.
Optionally (re-)renders the Champion pipeline in fp32 into outputs/target/best/.

    python scripts/run_target.py                 # report only
    python scripts/run_target.py --render-best   # re-render champion (fp32) + report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator import tracking  # noqa: E402
from acoustic_separator.audio import load_audio, save_audio  # noqa: E402
from acoustic_separator.report import build_report  # noqa: E402

TARGET_DIR = ROOT / "outputs" / "target"


def exp_info(exp: str) -> tuple[dict, dict]:
    d = ROOT / "experiments" / exp
    metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    config = yaml.safe_load((d / "config.yaml").read_text()) if (d / "config.yaml").exists() else {}
    return metrics, config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-best", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="experiment ids to include")
    ap.add_argument("--max-candidates", type=int, default=12)
    args = ap.parse_args()

    champ = tracking.current_champion()
    if args.render_best and champ:
        from acoustic_separator.pipeline import ModelPool, PipelineRunner

        pipe = json.loads(json.dumps(champ["pipeline"]))
        for s in pipe["steps"]:  # final render at full precision
            if "model" in s:
                s.setdefault("params", {})["precision"] = "fp32"
        mix, sr = load_audio(TARGET_DIR / "original.wav")
        res = PipelineRunner(ModelPool(), progress=True).run(pipe, mix, sr)
        best = TARGET_DIR / "best"
        save_audio(best / "acoustic_guitar.wav", res["output"], sr)
        save_audio(best / "non_acoustic_guitar.wav", res["residual"], sr)
        (best / "pipeline.yaml").write_text(yaml.safe_dump(pipe, sort_keys=False))
        tracking.copy_champion_audio(best)
        print("rendered champion", champ["experiment"])

    cand_dir = TARGET_DIR / "candidates"
    exps = sorted(p.stem for p in cand_dir.glob("exp*.wav") if not p.stem.endswith("_residual"))
    if args.only:
        exps = [e for e in exps if e in args.only]
    rows, cands = [], []
    for e in exps:
        m, cfg = exp_info(e)
        v = m.get("validation", {})
        p = m.get("target", {}).get("proxy", {})
        rows.append({"experiment": e, "strategy": cfg.get("strategy", ""),
                     "val SDR": v.get("sdr_mean", float("nan")),
                     "val SI-SDR": v.get("si_sdr_mean", float("nan")),
                     "SIR": v.get("sir_bss_mean", float("nan")), "SAR": v.get("sar_bss_mean", float("nan")),
                     "retention": v.get("target_retention_mean", float("nan")),
                     "leak dB": v.get("leakage_db_mean", float("nan")),
                     "hard SDR": v.get("hard_sdr_mean", float("nan")),
                     "tgt guitar p": p.get("guitar_prob", float("nan")),
                     "tgt max leak p": p.get("leak_prob_max", float("nan")),
                     "runtime s": m.get("target", {}).get("runtime_s", float("nan")),
                     "_best": bool(champ and champ["experiment"] == e)})
    order = sorted(rows, key=lambda r: -(r["val SDR"] if r["val SDR"] == r["val SDR"] else -99))
    shown = [r["experiment"] for r in order[: args.max_candidates]]
    for e in shown:
        m, cfg = exp_info(e)
        models = cfg.get("models", {})
        cands.append({
            "id": e, "label": f"{e}" + ("  ★ Champion" if champ and champ["experiment"] == e else ""),
            "acoustic": cand_dir / f"{e}.wav", "residual": cand_dir / f"{e}_residual.wav",
            "metrics": {k: v for k, v in next(r for r in rows if r["experiment"] == e).items()
                        if k not in ("experiment", "_best", "strategy")},
            "info": {"experiment": e, "strategy": cfg.get("strategy", ""),
                     "models": {k: {"checkpoint": v.get("checkpoint"), "sha256": (v.get("sha256") or "")[:16]}
                                for k, v in models.items()},
                     "pipeline": cfg.get("pipeline", {}).get("steps", []),
                     "runtime_s": m.get("target", {}).get("runtime_s")},
        })
    if (TARGET_DIR / "best" / "acoustic_guitar.wav").exists():
        cands.insert(0, {"id": "best", "label": "best/ (Champion, final render)",
                         "acoustic": TARGET_DIR / "best" / "acoustic_guitar.wav",
                         "residual": TARGET_DIR / "best" / "non_acoustic_guitar.wav",
                         "info": {"champion": champ["experiment"] if champ else None}})
    for e in rows:
        e["experiment"] = e["experiment"]
    intro = ("<p>Target: ko-ko-ya “Frevo!” (user-supplied YouTube audio, AAC 44.1 kHz). "
             "No ground-truth stem exists for this song: the table shows ground-truth metrics "
             "from the synthetic validation set plus reference-free proxies (AST classifier) on "
             "the target. Click a spectrogram to seek; the players of one candidate play in sync "
             "(use the solo buttons to switch).</p>")
    out = build_report(TARGET_DIR / "report.html", "Acoustic guitar extraction — target song report",
                       TARGET_DIR / "original.wav", cands, order, intro)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
