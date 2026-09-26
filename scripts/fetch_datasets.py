#!/usr/bin/env python
"""Download the raw sources used for validation/training into data/raw/ (git-ignored).

All sources are fetched from Hugging Face / Zenodo with deterministic file selection.
Licences are recorded in datasets/manifest.csv by scripts/build_manifest.py.

    python scripts/fetch_datasets.py --what rawstems_val gaps urmp guitarjam negatives
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "src"))

RAWSTEMS = "kwatcharasupat/mixing-secrets-rawstems"

# Coherent multitracks with acoustic guitar, used as the realistic validation set.
# Live-room sessions with heavy bleed (Pretty_Saro, Mipso, Barnstar, Chad_Hollister) are
# excluded because bleed would put other instruments inside the "ground truth".
RAWSTEMS_VAL_SONGS = [
    "Swing_Bazar - Fleche_DOr",               # AG + violin + accordion + EG + bass (gypsy jazz)
    "Catfolkin - Sant_Jordi_v2.0",            # AG + violin + flute + accordion + EG + drums + vox
    "Spektakulatius - What_Child_Is_This",    # AG + bass clarinet + piano + bass + drums + vox
    "Bolz__Knecht - Summertime",              # AG + saxophone (duo)
    "Rod_Alexander - Tears_In_The_Rain",      # AG + nylon guitar + EG + drums + tambourine
    "Perdidos_Na_Zona_Sul - Meu_Bem",         # Brazilian: AGs + piano + EG + drums + vox
    "Egda_Carolyn - Saudade_Do_Teu_Beijo",    # Brazilian: AG + organ + EG + shaker/cowbell + vox
    "Enda_Reilly - An_Nasc_Nua",              # AG + fiddles + bass + vox
    "Wolfs_Head__Vixen_Morris_Band - Lament",  # AG + mandolin + bouzouki + hurdy-gurdy + whistles
    "Adam_Buckley - Drag_me_Down",            # pop band: AG + EG + piano + drums + vox
    "Andres_Guazzelli - Attention",           # AG + EG + organ + piano + drums + synth
    "Leslie_Mendelson - The_Hardest_Part",    # AG + vocals
    "Nikola_Stajic_feat._Vlasis_Kostas - Nalim",  # 2 AG + drums
]

# Hard-negative single stems for the synthetic set, drawn from songs NOT in the val list.
NEGATIVE_PATTERNS = {
    "violin": r"/Orch/STR/[^/]*(Violin|Fiddle)[^/]*\.flac$",
    "winds": r"/Orch/WW/[^/]*(Clarinet|Flute|Sax|Oboe)[^/]*\.flac$",
    "percussion": r"/Rhy/PERC/[^/]*(Shaker|Conga|Cajon|Tamb|Bongo|Djembe|Perc)[^/]*\.flac$",
    "plucked_other": r"/Gtr/AG/[^/]*(Mandolin|Banjo|Uke|Bazouki|Bouzouki|Dobro|DOBRO)[^/]*\.flac$",
    "vocals": r"/Voc/LV/[^/]*\.flac$",
    # RawStems names electric guitars generically (ElecGtrN[DI|Mic]); take amp/mic tracks, not DI
    "electric_dist": r"/Gtr/EG/\d*_?ElecGtr\d*(Mic\d?)?\.flac$",
    "piano": r"/Kbs/PN/[^/]*Piano[^/]*\.flac$",
    "drums": r"/Rhy/DK/[^/]*(Overhead|OH)[^/]*\.flac$",
    "bass": r"/Bass/[^/]*Bass[^/]*\.flac$",
}
NEGATIVES_PER_CLASS = {"violin": 10, "winds": 10, "percussion": 8, "plucked_other": 8, "vocals": 8,
                       "electric_dist": 6, "piano": 8, "drums": 6, "bass": 6}
MAX_NEG_BYTES = 60_000_000


def hf_download(repo: str, path: str, repo_type: str = "dataset") -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=path, repo_type=repo_type,
                                local_dir=str(RAW / repo.split("/")[1])))


def list_tree(repo: str, repo_type: str = "dataset") -> list[dict]:
    cache = RAW / f"tree_{repo.replace('/', '__')}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from huggingface_hub import HfApi

    items = [{"path": f.path, "size": getattr(f, "size", 0)}
             for f in HfApi().list_repo_tree(repo, repo_type=repo_type, recursive=True)
             if hasattr(f, "size")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items))
    return items


def rawstems_val():
    tree = list_tree(RAWSTEMS)
    for song in RAWSTEMS_VAL_SONGS:
        files = [f["path"] for f in tree if f["path"].startswith(f"dev/{song}/")]
        print(f"{song}: {len(files)} files")
        for p in files:
            hf_download(RAWSTEMS, p)


def negatives(only: list[str] | None = None):
    tree = list_tree(RAWSTEMS)
    rng = random.Random(1234)
    val = {f"dev/{s}/" for s in RAWSTEMS_VAL_SONGS}
    sel = RAW / "rawstems_negatives.json"
    chosen = json.loads(sel.read_text()) if (only and sel.exists()) else {}
    for cls, pat in NEGATIVE_PATTERNS.items():
        if only and cls not in only:
            continue
        cands = [f for f in tree if re.search(pat, f["path"], re.I)
                 and not any(f["path"].startswith(v) for v in val)
                 and 3_000_000 < f["size"] < MAX_NEG_BYTES and "musdb18hq" not in f["path"]]
        # at most one file per song so the negatives are diverse
        by_song = {}
        for f in sorted(cands, key=lambda x: x["path"]):
            by_song.setdefault(f["path"].split("/")[1], []).append(f)
        songs = sorted(by_song)
        rng.shuffle(songs)
        picks = [rng.choice(by_song[s]) for s in songs[: NEGATIVES_PER_CLASS[cls]]]
        chosen[cls] = [p["path"] for p in picks]
        print(f"{cls}: {len(cands)} candidates, picked {len(picks)}")
        for p in picks:
            hf_download(RAWSTEMS, p["path"])
    (RAW / "rawstems_negatives.json").write_text(json.dumps(chosen, indent=1))


def gaps(n: int = 12):
    meta = hf_download("xavriley/GAPS", "gaps_metadata_with_splits.csv")
    rows = list(csv.DictReader(open(meta, newline="")))
    split_key = next(k for k in rows[0] if "split" in k.lower())
    id_key = next((k for k in rows[0] if k.lower() in ("id", "yt_id", "gaps_id", "track_id")), None)
    tree = list_tree("xavriley/GAPS")
    audio = {Path(f["path"]).stem: f["path"] for f in tree if f["path"].startswith("audio/")}
    test = [r for r in rows if r[split_key].strip().lower() == "test"]
    picked = []
    for r in test:
        key = next((v for v in r.values() if v in audio), None) if id_key is None else r[id_key]
        if key in audio:
            picked.append(audio[key])
    rng = random.Random(7)
    rng.shuffle(picked)
    picked = sorted(picked[:n])
    print(f"GAPS test pieces: {len(test)}, downloading {len(picked)}")
    for p in picked:
        hf_download("xavriley/GAPS", p)
    (RAW / "gaps_selected.json").write_text(json.dumps(picked, indent=1))


def urmp(per_instrument: int = 8):
    tree = list_tree("Eredis02/URMP")
    sep = [f["path"] for f in tree if re.search(r"/AuSep_\d+_(vn|cl|fl)_[^/]+\.wav$", f["path"])]
    rng = random.Random(11)
    picked = []
    for inst in ("vn", "cl", "fl"):
        c = sorted(p for p in sep if f"_{inst}_" in p)
        by_piece = {}
        for p in c:
            by_piece.setdefault(p.split("/")[0], []).append(p)
        pieces = sorted(by_piece)
        rng.shuffle(pieces)
        picked += [by_piece[k][0] for k in pieces[:per_instrument]]
    print(f"URMP: {len(picked)} files")
    for p in picked:
        hf_download("Eredis02/URMP", p)
    (RAW / "urmp_selected.json").write_text(json.dumps(picked, indent=1))


def guitarjam(n: int = 30):
    tree = list_tree("Julian-br/GuitarJam")
    wavs = sorted(f["path"] for f in tree if f["path"].lower().endswith((".wav", ".flac")))
    rng = random.Random(5)
    picked = sorted(rng.sample(wavs, min(n, len(wavs))))
    print(f"GuitarJam: {len(wavs)} files, downloading {len(picked)}")
    for p in picked:
        hf_download("Julian-br/GuitarJam", p)
    (RAW / "guitarjam_selected.json").write_text(json.dumps(picked, indent=1))


def guitarset():
    dst = RAW / "guitarset"
    if (dst / "mic").exists():
        print("GuitarSet already present")
        return
    import zipfile

    url = "https://zenodo.org/api/records/3371780/files/audio_mono-mic.zip/content"
    dst.mkdir(parents=True, exist_ok=True)
    zpath = dst / "audio_mono-mic.zip"
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dst / "mic")
    zpath.unlink()


# Training multitracks: other RawStems songs with acoustic guitar (never the val songs,
# never live-bleed sessions), chosen deterministically among the smaller sessions.
BLEED_SESSIONS = ("Pretty_Saro", "Mipso", "Barnstar", "Chad_Hollister", "Timo_Carlier")


def rawstems_train(n_songs: int = 14, max_song_mb: int = 260):
    tree = list_tree(RAWSTEMS)
    songs = {}
    for f in tree:
        parts = f["path"].split("/")
        if len(parts) > 3 and parts[0] == "dev":
            songs.setdefault(parts[1], []).append(f)
    val = set(RAWSTEMS_VAL_SONGS)
    cands = []
    for song, files in songs.items():
        if song in val or song.startswith(BLEED_SESSIONS):
            continue
        ag = [f for f in files if "/Gtr/AG/" in f["path"]
              and not re.search(r"mandolin|banjo|uke|bazouki|bouzouki|dobro|autoharp|sitar", f["path"], re.I)]
        size = sum(f["size"] for f in files)
        if ag and size < max_song_mb * 1e6:
            cands.append(song)
    rng = random.Random(2024)
    cands.sort()
    rng.shuffle(cands)
    picked = sorted(cands[:n_songs])
    print(f"training songs ({len(cands)} candidates): {picked}")
    for song in picked:
        for f in songs[song]:
            hf_download(RAWSTEMS, f["path"])
    (RAW / "rawstems_train_songs.json").write_text(json.dumps(picked, indent=1))


def negatives_train(per_class: int = 6):
    """Extra single-stem negatives from songs used neither in val nor as val negatives."""
    tree = list_tree(RAWSTEMS)
    used = set(RAWSTEMS_VAL_SONGS)
    sel = RAW / "rawstems_negatives.json"
    if sel.exists():
        for paths in json.loads(sel.read_text()).values():
            used |= {p.split("/")[1] for p in paths}
    rng = random.Random(99)
    chosen = {}
    for cls, pat in NEGATIVE_PATTERNS.items():
        cands = [f for f in tree if re.search(pat, f["path"], re.I) and "musdb18hq" not in f["path"]
                 and f["path"].split("/")[1] not in used and 3_000_000 < f["size"] < MAX_NEG_BYTES
                 and not re.search("room", f["path"], re.I)]
        by_song = {}
        for f in sorted(cands, key=lambda x: x["path"]):
            by_song.setdefault(f["path"].split("/")[1], []).append(f)
        songs_ = sorted(by_song)
        rng.shuffle(songs_)
        chosen[cls] = [rng.choice(by_song[s])["path"] for s in songs_[:per_class]]
        for p in chosen[cls]:
            hf_download(RAWSTEMS, p)
        print(f"{cls}: {len(chosen[cls])}")
    (RAW / "rawstems_negatives_train.json").write_text(json.dumps(chosen, indent=1))


def gaps_train(n: int = 16):
    meta = hf_download("xavriley/GAPS", "gaps_metadata_with_splits.csv")
    rows = [r for r in csv.DictReader(open(meta, newline="")) if r["split"].strip() == "train"]
    rng = random.Random(8)
    rng.shuffle(rows)
    picked = sorted(r["audio_path"] for r in rows[:n])
    for p in picked:
        hf_download("xavriley/GAPS", p)
    (RAW / "gaps_train_selected.json").write_text(json.dumps(picked, indent=1))


def urmp_train(per_instrument: int = 5):
    tree = list_tree("Eredis02/URMP")
    used = set(json.loads((RAW / "urmp_selected.json").read_text())) if (RAW / "urmp_selected.json").exists() else set()
    used_pieces = {p.split("/")[0] for p in used}
    sep = [f["path"] for f in tree if re.search(r"/AuSep_\d+_(vn|cl|fl)_[^/]+\.wav$", f["path"])
           and f["path"].split("/")[0] not in used_pieces]
    rng = random.Random(12)
    picked = []
    for inst in ("vn", "cl", "fl"):
        c = sorted(p for p in sep if f"_{inst}_" in p)
        rng.shuffle(c)
        picked += c[:per_instrument]
    for p in picked:
        hf_download("Eredis02/URMP", p)
    (RAW / "urmp_train_selected.json").write_text(json.dumps(picked, indent=1))


def negatives_electric():
    negatives(only=["electric_dist"])


STEPS = {"negatives_electric": negatives_electric, "guitarset": guitarset, "rawstems_val": rawstems_val, "negatives": negatives,
         "gaps": gaps, "urmp": urmp, "guitarjam": guitarjam, "rawstems_train": rawstems_train,
         "negatives_train": negatives_train, "gaps_train": gaps_train, "urmp_train": urmp_train}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="*", default=list(STEPS))
    args = ap.parse_args()
    for w in args.what:
        print(f"== {w}")
        STEPS[w]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
