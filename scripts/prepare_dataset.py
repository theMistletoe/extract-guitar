#!/usr/bin/env python
"""Build the ground-truth validation set from datasets/manifest.csv.

Two families of clips, both with an exactly known acoustic-guitar stem
(mixture == acoustic_guitar + sum(stems/*)):

* ``ms_*``  – coherent excerpts of real multitrack songs (RawStems / Mixing Secrets):
  every stem of the song is summed after light per-stem processing and a linked mastering
  stage; the ground truth is the (processed) sum of the song's acoustic-guitar stems.
* ``syn_*`` – synthetic scenario mixtures (mixing.SCENARIOS): an isolated acoustic guitar
  (GAPS nylon / GuitarSet steel) plus hard-negative stems (violin, winds, piano, clean and
  distorted electric guitar, mandolin/banjo/ukulele, vocals, drums, percussion, bass).

    python scripts/prepare_dataset.py                    # both families
    python scripts/prepare_dataset.py --only ms --clips-per-song 2
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator.mixing import (SCENARIOS, SR, Source, load_song_groups,  # noqa: E402
                                       make_mixture, make_multitrack_clip, write_clip)

MANIFEST = ROOT / "datasets" / "manifest.csv"

# Songs whose clips best resemble the target (strings / winds / Latin): two clips each.
PRIORITY_SONGS = ("Swing_Bazar", "Catfolkin", "Spektakulatius", "Rod_Alexander", "Perdidos",
                  "Bolz__Knecht", "Enda_Reilly")
# Synthetic scenarios and how many clips of each (target-like ones get more).
SYN_PLAN = {"chamber_frevo": 3, "strings": 2, "winds": 2, "piano": 2, "clean_electric": 2,
            "plucked": 2, "band_pop": 1, "electric_band": 1, "distorted_electric": 1,
            "female_vocal": 1, "cymbal_drums": 1, "buried": 2, "dense": 1, "piano_band": 1}


def read_manifest(split: str):
    with open(MANIFEST) as f:
        return [r for r in csv.DictReader(f) if r["split"] in (split, "any")]


def build_pool(rows) -> dict[str, list[Source]]:
    pool: dict[str, list[Source]] = {}
    for r in rows:
        if r["type"] == "multitrack_song":
            continue
        p = ROOT / r["path"]
        if p.exists():
            pool.setdefault(r["instrument"], []).append(
                Source(id=r["id"], path=str(p), cls=r["instrument"], license=r["license"],
                       dataset=r["source"], split=r["split"]))
    # balance nylon (GAPS) and steel-string (GuitarSet) guitars: the target is nylon-string
    ag = pool.get("acoustic_guitar", [])
    nylon = [x for x in ag if x.id.startswith("gaps_")]
    steel = [x for x in ag if not x.id.startswith("gaps_")]
    if nylon and steel:
        reps = max(1, round(len(steel) / len(nylon)))
        pool["acoustic_guitar"] = steel + nylon * reps
    return pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "validation"))
    ap.add_argument("--only", choices=["ms", "syn"], default=None)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--clips-per-song", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--clean", action="store_true", help="delete the output folder first")
    args = ap.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    rows = read_manifest(args.split)
    rng = np.random.default_rng(args.seed)
    index = []

    if args.only in (None, "ms"):
        for r in rows:
            if r["type"] != "multitrack_song":
                continue
            song_dir = ROOT / r["path"]
            groups = load_song_groups(song_dir)
            if "acoustic_guitar" not in groups:
                print(f"skip {song_dir.name}: no acoustic guitar after filtering")
                continue
            n = args.clips_per_song + (1 if song_dir.name.startswith(PRIORITY_SONGS) else 0)
            used = []
            for k in range(n):
                clip = make_multitrack_clip(song_dir, rng, args.seconds, groups=groups, exclude=used)
                used.append((clip["start"], clip["start"] + int(args.seconds * SR)))
                if clip["info"]["guitar_to_rest_db"] > 15:  # (almost) solo guitar: not a separation test
                    print(f"skip {song_dir.name}: no accompaniment in the chosen window")
                    break
                name = f"ms_{r['id'][3:31]}_{k}"
                clip["info"]["source_ids"] = [r["id"]]
                clip["info"]["license"] = r["license"]
                write_clip(out / name, clip, "multitrack")
                index.append({"clip": name, "family": "ms", **{k2: clip["info"][k2] for k2 in
                              ("song", "start_s", "guitar_to_rest_db", "hard")},
                              "classes": sorted(clip["info"]["classes"])})
                print(f"{name:40s} g/rest={clip['info']['guitar_to_rest_db']:+6.1f} dB "
                      f"classes={sorted(clip['info']['classes'])}")

    if args.only in (None, "syn"):
        pool = build_pool(rows)
        print({k: len(v) for k, v in sorted(pool.items())})
        for sc, n in SYN_PLAN.items():
            classes = SCENARIOS[sc]["classes"]
            if not any(pool.get(c) for c in classes):
                print(f"skip {sc}: no sources for {classes}")
                continue
            for k in range(n):
                clip = make_mixture(pool, sc, rng, seconds=args.seconds)
                name = f"syn_{sc}_{k}"
                write_clip(out / name, clip, sc)
                srcs = clip["info"]["sources"]
                index.append({"clip": name, "family": "syn", "scenario": sc,
                              "hard": clip["info"]["hard"],
                              "guitar_to_rest_db": clip["info"]["guitar_to_rest_db"],
                              "sources": [s["id"] for s in srcs]})
                print(f"{name:40s} g/rest={clip['info']['guitar_to_rest_db']:+6.1f} dB "
                      f"sources={[s['id'][:28] for s in srcs]}")

    (out / "index.json").write_text(json.dumps({"seed": args.seed, "seconds": args.seconds,
                                                "clips": index}, indent=1, default=float))
    print(f"{len(index)} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
