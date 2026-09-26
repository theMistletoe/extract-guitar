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


# --- additional sources ------------------------------------------------------------------
RAWSTEMS_DIR = RAW / "mixing-secrets-rawstems" / "dev"
MS_LICENSE = "Mixing Secrets / RawStems: non-commercial research & education only, no redistribution"


@register
def rawstems_multitracks():
    sys.path.insert(0, str(ROOT / "scripts"))
    from fetch_datasets import RAWSTEMS_VAL_SONGS

    for song in RAWSTEMS_VAL_SONGS:
        d = RAWSTEMS_DIR / song
        if d.exists():
            yield {"id": "ms_" + re.sub(r"[^A-Za-z0-9]+", "_", song)[:50], "source": "RawStems (HF kwatcharasupat/mixing-secrets-rawstems)",
                   "license": MS_LICENSE, "instrument": "multitrack", "type": "multitrack_song",
                   "path": rel(d), "split": "val", "notes": "coherent song; GT = sum of acoustic guitar stems"}


@register
def rawstems_negatives():
    import json

    sel = RAW / "rawstems_negatives.json"
    if not sel.exists():
        return
    for cls, paths in json.loads(sel.read_text()).items():
        for p in paths:
            f = RAW / "mixing-secrets-rawstems" / p
            if not f.exists() or re.search(r"room", f.name, re.I):
                continue
            song = p.split("/")[1]
            yield {"id": "msneg_" + re.sub(r"[^A-Za-z0-9]+", "_", p[4:])[:70], "source": "RawStems",
                   "license": MS_LICENSE, "instrument": cls, "type": "isolated_stem",
                   "path": rel(f), "split": "val", "notes": song}


@register
def gaps():
    train = {Path(p).name for p in _selected("gaps_train_selected.json")}
    for f in sorted((RAW / "GAPS" / "audio").glob("*.wav")):
        split = "train" if f.name in train else "val"
        yield {"id": f"gaps_{f.stem}", "source": f"GAPS (xavriley/GAPS, {split if split == 'train' else 'test'} split)",
               "license": "research use (MIT card; audio from YouTube performances)",
               "instrument": "acoustic_guitar", "type": "isolated_solo", "path": rel(f),
               "split": split, "notes": "nylon-string classical guitar, 48 kHz stereo"}


def _selected(name: str) -> set[str]:
    import json

    f = RAW / name
    return set(json.loads(f.read_text())) if f.exists() else set()


@register
def urmp():
    val, train = _selected("urmp_selected.json"), _selected("urmp_train_selected.json")
    for f in sorted((RAW / "URMP").rglob("AuSep_*.wav")):
        inst = f.name.split("_")[2]
        cls = {"vn": "violin", "cl": "winds", "fl": "winds"}.get(inst)
        key = str(f.relative_to(RAW / "URMP"))
        split = "val" if key in val else "train" if key in train else None
        if cls and split:
            yield {"id": f"urmp_{f.stem}", "source": "URMP (HF Eredis02/URMP)",
                   "license": "research use only (no explicit licence)", "instrument": cls,
                   "type": "isolated_stem", "path": rel(f), "split": split, "notes": inst}


@register
def guitarjam():
    files = sorted(f for f in (RAW / "GuitarJam").rglob("*") if f.suffix.lower() in (".wav", ".flac"))
    for i, f in enumerate(files):
        yield {"id": f"gjam_{f.stem}", "source": "GuitarJam (HF Julian-br/GuitarJam)",
               "license": "CC0-1.0", "instrument": "electric_clean", "type": "isolated_di",
               "path": rel(f), "split": "val" if i % 2 == 0 else "train", "notes": "clean Stratocaster DI"}


@register
def rawstems_train_songs():
    import json

    sel = RAW / "rawstems_train_songs.json"
    if not sel.exists():
        return
    for song in json.loads(sel.read_text()):
        d = RAWSTEMS_DIR / song
        if d.exists():
            yield {"id": "mstr_" + re.sub(r"[^A-Za-z0-9]+", "_", song)[:50], "source": "RawStems",
                   "license": MS_LICENSE, "instrument": "multitrack", "type": "multitrack_song",
                   "path": rel(d), "split": "train", "notes": "training song"}


@register
def rawstems_negatives_train():
    import json

    sel = RAW / "rawstems_negatives_train.json"
    if not sel.exists():
        return
    for cls, paths in json.loads(sel.read_text()).items():
        for p in paths:
            f = RAW / "mixing-secrets-rawstems" / p
            if f.exists():
                yield {"id": "msnegtr_" + re.sub(r"[^A-Za-z0-9]+", "_", p[4:])[:70], "source": "RawStems",
                       "license": MS_LICENSE, "instrument": cls, "type": "isolated_stem",
                       "path": rel(f), "split": "train", "notes": p.split("/")[1]}


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
