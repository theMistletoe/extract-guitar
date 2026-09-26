#!/usr/bin/env python
"""Scan downloaded raw datasets (data/raw/) and write datasets/manifest.csv.

Each row is one isolated source file with its dataset, licence, instrument class and split.
Splits are made by performer / recording so validation never shares a player or song
with training material.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "datasets" / "manifest.csv"
FIELDS = ["id", "source", "license", "instrument", "type", "path", "split", "notes"]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def guitarset():
    for p in sorted((RAW / "guitarset" / "mic").glob("*.wav")):
        player = p.name[:2]
        yield {"id": f"gs_{p.stem}", "source": "GuitarSet (Xi et al. 2018, zenodo 3371780)",
               "license": "CC-BY-4.0", "instrument": "acoustic_guitar", "type": "isolated_mic",
               "path": rel(p), "split": "val" if player == "05" else "train",
               "notes": "steel-string acoustic, mono mic"}


SCANNERS = [guitarset]


def register(fn):
    SCANNERS.append(fn)
    return fn


# --- additional sources are appended below as they are downloaded --------------------


def main() -> int:
    rows = []
    for scan in SCANNERS:
        try:
            rows.extend(scan())
        except FileNotFoundError:
            continue
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        k = (r["instrument"], r["split"])
        counts[k] = counts.get(k, 0) + 1
    for k in sorted(counts):
        print(f"{k[0]:20s} {k[1]:6s} {counts[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
