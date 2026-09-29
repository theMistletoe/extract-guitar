"""Synthetic mixture generation with exact acoustic-guitar ground truth.

A *source pool* maps a class name (e.g. ``acoustic_guitar``, ``violin``, ``drums``) to a
list of :class:`Source` entries (file + license). ``make_mixture`` draws excerpts,
applies per-stem augmentation and a linked mastering stage, and returns stems whose sum
is exactly the mixture.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import augment as A
from .audio import load_audio, save_audio

SR = 44100


@dataclass
class Source:
    id: str
    path: str
    cls: str
    license: str
    dataset: str
    split: str = "val"
    offset_s: float = 0.0  # optional fixed start
    duration_s: float | None = None


# Scenario = which interferer classes to draw and at what relative level.
# Levels are dB of each interferer's active RMS relative to the acoustic guitar.
SCENARIOS: dict[str, dict] = {
    "band_pop": {"classes": ["vocals", "drums", "bass", "electric_clean"], "level": (-3, 6)},
    "piano": {"classes": ["piano"], "level": (-3, 6), "hard": True},
    "piano_band": {"classes": ["piano", "drums", "bass"], "level": (-3, 5), "hard": True},
    "clean_electric": {"classes": ["electric_clean"], "level": (-3, 6), "hard": True},
    "electric_band": {"classes": ["electric_clean", "drums", "bass", "vocals"], "level": (-3, 5), "hard": True},
    "distorted_electric": {"classes": ["electric_dist", "drums", "bass"], "level": (0, 8)},
    "strings": {"classes": ["violin"], "level": (-3, 6), "hard": True},
    "winds": {"classes": ["winds"], "level": (-3, 6), "hard": True},
    "chamber_frevo": {"classes": ["violin", "winds", "bass", "percussion"], "level": (-2, 6), "hard": True},
    "female_vocal": {"classes": ["vocals"], "level": (0, 8)},
    "cymbal_drums": {"classes": ["drums"], "level": (0, 8)},
    "buried": {"classes": ["vocals", "drums", "bass", "piano", "electric_clean"], "level": (4, 10), "hard": True},
    "plucked": {"classes": ["plucked_other"], "level": (-3, 6), "hard": True},
    "dense": {"classes": ["vocals", "drums", "bass", "piano", "violin", "winds", "electric_clean"],
              "level": (-2, 5), "hard": True},
}


def _excerpt(src: Source, seconds: float, rng: np.random.Generator, min_active_rms: float = 3e-3):
    audio, sr = load_audio(src.path, sr=SR)
    n = int(seconds * SR)
    if audio.shape[1] < n:  # loop short material
        reps = int(np.ceil(n / audio.shape[1]))
        audio = np.tile(audio, (1, reps))
    best, best_e = None, -1.0
    for _ in range(12):  # pick an excerpt with enough activity
        s = int(rng.integers(0, audio.shape[1] - n + 1))
        seg = audio[:, s:s + n]
        e = A.active_rms(seg, SR)
        if e > best_e:
            best, best_e, best_s = seg, e, s
        if e > min_active_rms and np.mean(np.abs(seg).max(0) > 1e-3) > 0.5:
            break
    return best.copy(), best_s / SR, best_e


def process_stem(x: np.ndarray, rng: np.random.Generator, cls: str, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    fx = {}
    if rng.random() < 0.8 * strength:
        x = A.random_eq(x, SR, rng, strength_db=6.0 if cls != "acoustic_guitar" else 4.0)
        fx["eq"] = True
    if rng.random() < 0.5 * strength:
        x = A.compress(x, SR, rng)
        fx["compress"] = True
    if cls in ("electric_dist",) or (rng.random() < 0.1 * strength and cls != "acoustic_guitar"):
        x = A.saturate(x, rng)
        fx["saturate"] = True
    if rng.random() < 0.15 * strength:
        x = A.bandlimit(x, SR, rng)
        fx["bandlimit"] = True
    if x.shape[0] == 2 and np.allclose(x[0], x[1]):
        width = 0.0
    else:
        width = float(rng.uniform(0.3, 1.0))
    posn = float(rng.uniform(-0.7, 0.7))
    x = A.pan(x, posn, width)
    fx["pan"] = round(posn, 2)
    if rng.random() < 0.25 * strength:
        x = A.haas_delay(x, SR, rng)
        fx["haas"] = True
    if rng.random() < 0.15 * strength and cls not in ("drums", "bass"):
        x = A.echo(x, SR, rng)
        fx["echo"] = True
    if rng.random() < 0.7 * strength:
        x = A.reverb(x, SR, rng)
        fx["reverb"] = True
    return x.astype(np.float32), fx


def make_mixture(pool: dict[str, list[Source]], scenario: str, rng: np.random.Generator,
                 seconds: float = 11.0, target_level_db: float | None = None,
                 strength: float = 1.0, master: bool = True) -> dict:
    sc = SCENARIOS[scenario]
    g_src = pool["acoustic_guitar"][int(rng.integers(len(pool["acoustic_guitar"])))]
    g, g_off, _ = _excerpt(g_src, seconds, rng)
    g, g_fx = process_stem(g, rng, "acoustic_guitar", strength)
    g_level = A.active_rms(g, SR)
    stems = {"acoustic_guitar": g}
    info = {"scenario": scenario, "hard": bool(sc.get("hard", False)),
            "sources": [{**asdict(g_src), "offset_used_s": round(g_off, 3), "fx": g_fx}]}
    lo, hi = sc["level"]
    for cls in sc["classes"]:
        cands = pool.get(cls, [])
        if not cands:
            continue
        src = cands[int(rng.integers(len(cands)))]
        x, off, _ = _excerpt(src, seconds, rng)
        x, fx = process_stem(x, rng, cls, strength)
        rel = float(rng.uniform(lo, hi)) if target_level_db is None else float(target_level_db)
        x = x * (g_level / (A.active_rms(x, SR) + 1e-9)) * A.db2lin(rel)
        name = cls
        k = 2
        while name in stems:
            name = f"{cls}{k}"
            k += 1
        stems[name] = x
        info["sources"].append({**asdict(src), "offset_used_s": round(off, 3), "fx": fx,
                                "level_rel_guitar_db": round(rel, 2)})
    if master:
        stems = A.mastering(stems, SR, rng)
    else:
        peak = np.abs(sum(stems.values())).max()
        stems = {k: v / (peak + 1e-9) * 0.9 for k, v in stems.items()}
    mix = sum(stems.values()).astype(np.float32)
    interf = sum(v for k, v in stems.items() if k != "acoustic_guitar")
    info["guitar_to_rest_db"] = round(10 * np.log10(np.mean(stems["acoustic_guitar"] ** 2) /
                                                     (np.mean(interf ** 2) + 1e-12)), 2)
    return {"mixture": mix, "stems": stems, "info": info}


def write_clip(out_dir: Path, clip: dict, category: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_audio(out_dir / "mixture.wav", clip["mixture"], SR)
    save_audio(out_dir / "acoustic_guitar.wav", clip["stems"]["acoustic_guitar"], SR)
    for k, v in clip["stems"].items():
        if k != "acoustic_guitar":
            save_audio(out_dir / "stems" / f"{k}.wav", v, SR)
    meta = dict(clip["info"])
    meta["category"] = category
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=float))


# --- coherent multitrack songs (RawStems / Mixing Secrets layout) ----------------------
NON_GUITAR_PLUCKED = re.compile(r"mandolin|mando|banjo|uke|ukelele|ukulele|guitalele|bazouki|"
                                r"bouzouki|dobro|autoharp|sitar|marimba", re.I)
EXCLUDE_STEM = re.compile(r"room|bleed|click|plusvox|ambien|crowd|talkback", re.I)
CATEGORY_CLASS = {
    "Voc": "vocals", "Rhy/DK": "drums", "Rhy/PERC": "percussion", "Bass": "bass",
    "Kbs/PN": "piano", "Kbs/OR": "keys", "Kbs/EP": "keys", "Kbs": "keys",
    "Gtr/EG": "electric_guitar", "Orch/STR": "strings", "Orch/WW": "winds",
    "Orch/BR": "brass", "Synth": "synth",
}
# nominal mix level of each class relative to the acoustic guitar (dB), jittered per clip
CLASS_LEVEL_DB = {"vocals": 3, "drums": 2, "percussion": -3, "bass": 0, "piano": 0, "keys": -2,
                  "electric_guitar": 0, "strings": 0, "winds": 0, "brass": -1, "synth": -2,
                  "plucked_other": -1}


def classify_stem(rel_path: str) -> str | None:
    """Map 'Cat/Sub/NN_Name.flac' (path inside a song folder) to a class; None = skip."""
    name = rel_path.rsplit("/", 1)[-1]
    if EXCLUDE_STEM.search(name) or rel_path.startswith("Misc"):
        return None
    if rel_path.startswith("Gtr/AG"):
        return "plucked_other" if NON_GUITAR_PLUCKED.search(name) else "acoustic_guitar"
    for prefix in sorted(CATEGORY_CLASS, key=len, reverse=True):
        if rel_path.startswith(prefix):
            return CATEGORY_CLASS[prefix]
    return None


def load_song_groups(song_dir: Path) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    files = {}
    for p in sorted(song_dir.rglob("*.flac")) + sorted(song_dir.rglob("*.wav")):
        cls = classify_stem(str(p.relative_to(song_dir)))
        if cls is None:
            continue
        files.setdefault(cls, []).append(p)
    length = None
    for cls, paths in files.items():
        acc = None
        for p in paths:
            x, _ = load_audio(p, sr=SR)
            if acc is None:
                acc = x
            else:
                n = max(acc.shape[1], x.shape[1])
                acc = np.pad(acc, ((0, 0), (0, n - acc.shape[1]))) + np.pad(x, ((0, 0), (0, n - x.shape[1])))
        groups[cls] = acc
        length = max(length or 0, acc.shape[1])
    return {k: np.pad(v, ((0, 0), (0, length - v.shape[1]))) for k, v in groups.items()}


def _frame_energy(x: np.ndarray, hop: int) -> np.ndarray:
    m = np.mean(x ** 2, axis=0)
    n = len(m) // hop
    return m[: n * hop].reshape(n, hop).mean(1)


def pick_window(groups: dict[str, np.ndarray], seconds: float, rng: np.random.Generator,
                exclude: list[tuple[int, int]] = ()) -> int:
    """Start sample of a window where the guitar is active and many other classes play."""
    hop = SR // 2
    n = int(seconds * SR)
    g = _frame_energy(groups["acoustic_guitar"], hop)
    g_act = g > (np.percentile(g, 95) * 10 ** (-30 / 10))
    others = [k for k in groups if k != "acoustic_guitar"]
    o_act = [(_frame_energy(groups[k], hop) > np.percentile(_frame_energy(groups[k], hop), 95) * 1e-3)
             for k in others]
    w = int(seconds * 2)
    best, best_score = 0, -1
    for s in range(0, len(g) - w):
        start = s * hop
        if any(a <= start < b or a < start + n <= b for a, b in exclude):
            continue
        score = g_act[s:s + w].mean() * 4 + sum(o[s:s + w].mean() for o in o_act) + rng.uniform(0, 0.3)
        if score > best_score:
            best, best_score = start, score
    return best


def make_multitrack_clip(song_dir: Path, rng: np.random.Generator, seconds: float = 12.0,
                         groups: dict | None = None, exclude=(), strength: float = 0.6,
                         level_jitter_db: float = 3.0) -> dict:
    groups = groups or load_song_groups(song_dir)
    if "acoustic_guitar" not in groups:
        raise ValueError(f"{song_dir} has no acoustic guitar stem")
    start = pick_window(groups, seconds, rng, exclude)
    n = int(seconds * SR)
    stems = {k: v[:, start:start + n].astype(np.float32) for k, v in groups.items()}
    stems = {k: v for k, v in stems.items() if A.rms(v) > 1e-5}
    ref = A.active_rms(stems["acoustic_guitar"], SR)
    info = {"song": song_dir.name, "start_s": round(start / SR, 2), "classes": {}}
    out = {}
    for k, v in stems.items():
        v, fx = process_stem(v, rng, k, strength) if strength > 0 else (v, {})
        lvl = 0.0 if k == "acoustic_guitar" else CLASS_LEVEL_DB.get(k, 0) + rng.uniform(-level_jitter_db, level_jitter_db)
        v = v * (ref / (A.active_rms(v, SR) + 1e-9)) * A.db2lin(lvl)
        out[k] = v.astype(np.float32)
        info["classes"][k] = {"level_rel_guitar_db": round(lvl, 2), "fx": fx}
    out = A.mastering(out, SR, rng)
    mix = sum(out.values()).astype(np.float32)
    interf = sum(v for k, v in out.items() if k != "acoustic_guitar")
    info["guitar_to_rest_db"] = round(10 * np.log10(np.mean(out["acoustic_guitar"] ** 2) /
                                                     (np.mean(interf ** 2) + 1e-12)), 2)
    info["hard"] = any(k in out for k in ("strings", "winds", "piano", "plucked_other", "electric_guitar"))
    info["scenario"] = "multitrack"
    return {"mixture": mix, "stems": out, "info": info, "start": start}
