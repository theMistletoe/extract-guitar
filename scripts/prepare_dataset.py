#!/usr/bin/env python
"""Build the ground-truth validation (and training) sets from datasets/manifest.csv.

Every clip is a synthetic mixture: isolated acoustic guitar + interferers drawn from
scenario definitions (see acoustic_separator.mixing.SCENARIOS). The acoustic guitar
stem is known exactly, so SDR / SIR / SAR / leakage can be measured.

    python scripts/prepare_dataset.py --split val --n-per-scenario 2 --seconds 11
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

from acoustic_separator.mixing import SCENARIOS, Source, make_mixture, write_clip  # noqa: E402

MANIFEST = ROOT / "datasets" / "manifest.csv"


def load_pool(split: str) -> dict[str, list[Source]]:
    pool: dict[str, list[Source]] = {}
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["split"] not in (split, "any"):
                continue
            p = ROOT / r["path"]
            if not p.exists():
                continue
            pool.setdefault(r["instrument"], []).append(
                Source(id=r["id"], path=str(p), cls=r["instrument"], license=r["license"],
                       dataset=r["source"], split=r["split"]))
    return pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-per-scenario", type=int, default=2)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "datasets" / ("validation" if args.split == "val" else "train_mixes")
    pool = load_pool(args.split)
    print({k: len(v) for k, v in sorted(pool.items())})
    if not pool.get("acoustic_guitar"):
        raise SystemExit("no acoustic_guitar sources for this split in the manifest")
    rng = np.random.default_rng(args.seed)
    scenarios = args.scenarios or list(SCENARIOS)
    index = []
    for sc in scenarios:
        needed = SCENARIOS[sc]["classes"]
        if not any(pool.get(c) for c in needed):
            print(f"skip {sc}: no sources for {needed}")
            continue
        missing = [c for c in needed if not pool.get(c)]
        for k in range(args.n_per_scenario):
            clip = make_mixture(pool, sc, rng, seconds=args.seconds)
            name = f"{sc}_{k:02d}"
            write_clip(out / name, clip, sc)
            index.append({"clip": name, "scenario": sc, "hard": clip["info"]["hard"],
                          "guitar_to_rest_db": clip["info"]["guitar_to_rest_db"],
                          "missing_classes": missing,
                          "sources": [s["id"] for s in clip["info"]["sources"]]})
            print(f"{name}: g/rest={clip['info']['guitar_to_rest_db']:+.1f} dB "
                  f"sources={[s['id'] for s in clip['info']['sources']]}")
    (out / "index.json").write_text(json.dumps({"seed": args.seed, "seconds": args.seconds,
                                                "clips": index}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
