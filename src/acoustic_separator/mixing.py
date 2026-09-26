"""Synthetic mixture generation with exact acoustic-guitar ground truth.

A *source pool* maps a class name (e.g. ``acoustic_guitar``, ``violin``, ``drums``) to a
list of :class:`Source` entries (file + license). ``make_mixture`` draws excerpts,
applies per-stem augmentation and a linked mastering stage, and returns stems whose sum
is exactly the mixture.
"""
from __future__ import annotations

import json
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
    "band_pop": {"classes": ["vocals", "drums", "bass", "other_band"], "level": (-3, 6)},
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
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
