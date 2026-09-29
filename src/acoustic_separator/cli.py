"""Command line entry point.

    python -m acoustic_separator --input song.wav --output outputs/song/ --quality max
    python -m acoustic_separator --input "https://www.youtube.com/watch?v=..." --quality max
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

QUALITY_PIPELINES = {
    "fast": "fast",
    "standard": "standard",
    "max": "max",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="acoustic_separator",
                                description="Extract the acoustic guitar stem from a mix.")
    p.add_argument("--input", "-i", required=True, help="audio file (wav/mp3/m4a/flac/...) or URL")
    p.add_argument("--output", "-o", default=None,
                   help="output directory (default: outputs/<track-id>/)")
    p.add_argument("--quality", choices=sorted(QUALITY_PIPELINES), default="max",
                   help="preset pipeline; 'max' favours quality over run time")
    p.add_argument("--pipeline", default=None,
                   help="pipeline name in configs/pipelines/ or path to a YAML (overrides --quality)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--save-candidates", action="store_true",
                   help="also write every intermediate step to <output>/candidates/")
    p.add_argument("--no-cache", action="store_true", help="do not reuse cached model outputs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .audio import load_audio, save_audio, track_id_for
    from .fetch import download_audio, is_url
    from .inference import pick_device
    from .pipeline import REPO_ROOT, ModelPool, PipelineRunner, describe, load_pipeline

    track_id = track_id_for(args.input)
    out_dir = Path(args.output) if args.output else REPO_ROOT / "outputs" / track_id
    out_dir.mkdir(parents=True, exist_ok=True)
    src = args.input
    if is_url(src):
        print(f"[fetch] downloading {src}")
        src = str(download_audio(src, out_dir / "source"))
    mix, sr = load_audio(src)
    save_audio(out_dir / "original.wav", mix, sr)
    print(f"[audio] {src}: {mix.shape[1] / sr:.1f}s @ {sr} Hz")

    pipeline = load_pipeline(args.pipeline or QUALITY_PIPELINES[args.quality])
    device = pick_device(args.device)
    print(f"[run] pipeline={pipeline['name']} device={device}")
    runner = PipelineRunner(ModelPool(device=device), use_cache=not args.no_cache, progress=True)
    t = time.perf_counter()
    res = runner.run(pipeline, mix, sr)
    wall = time.perf_counter() - t
    save_audio(out_dir / "acoustic_guitar.wav", res["output"], sr)
    save_audio(out_dir / "non_acoustic_guitar.wav", res["residual"], sr)
    if args.save_candidates:
        for k, v in res["values"].items():
            if k != "mix":
                save_audio(out_dir / "candidates" / f"{k}.wav", v, sr)
    meta = describe(pipeline)
    meta.update({"input": args.input, "samplerate": sr, "wall_seconds": wall,
                 "model_seconds": res["runtime"], "device": str(device)})
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[done] {out_dir / 'acoustic_guitar.wav'} ({wall:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
