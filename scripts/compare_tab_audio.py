#!/usr/bin/env python
"""Compare what the tab plays with what the guitar stem plays: note by note, chord by chord, in time.

    python scripts/compare_tab_audio.py --bench   # benchmark (ground truth) -> reports/tab_compare_calibration.json
    python scripts/compare_tab_audio.py           # target song -> outputs/target/tab/compare/
    python scripts/compare_tab_audio.py probe outputs/target/tab/compare --t 12.34 --pitch 55 [--image x.png]

The tab is rendered at the recording's timing (scripts/render_tab_audio.py) and the rendering and
the stem go through the same analyses:

1. Transcription round trip: the note model (fl + gaps_paper) on the rendering vs on the stem:
   does the rendering reproduce the tab, and do both signals transcribe alike (per bar)?
2. Every tab note: onset evidence at its pitch in the stem vs in the rendering (constant-Q onset
   flux and level), the three distinct checkpoints and Basic Pitch on the stem, the residual stem
   (violin, clarinet) at its pitch, and its onset time in the stem vs in the rendering.
3. Notes the stem has and the tab lacks: candidates from each checkpoint, Basic Pitch and the
   ensemble below its threshold (0.1); their overtone / octave relation to tab notes.  (Constant-Q
   onset peaks absent from the rendering were tried as a source: on the benchmark 10 of 3538 were
   real, so they are only a feature, ``cq_peak``.)
4. Every bar: reproduction F1, chroma cosine per 8th note, local lag between stem and rendering.

With ``--bench`` the same is run on the benchmark (the tab pipeline applied to a mix with known
notes), every candidate is labelled with the ground truth, and logistic models for P(tab note is
wrong) and P(missing note is real) are cross-validated by clip (several feature sets; the best is
kept) and saved; the target is scored with them.  ``probe`` prints the evidence at one time and
pitch (harmonic levels in stem / rendering / residual, posteriors, nearby notes) from the cached
state of a comparison folder.  Needs FluidSynth + a GM SoundFont (render_tab_audio.py) and optionally Basic Pitch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from acoustic_separator.audio import load_audio  # noqa: E402
from acoustic_separator.tab import amt  # noqa: E402

import render_tab_audio as rta  # noqa: E402

TAB = ROOT / "outputs" / "target" / "tab"
BENCH = ROOT / "data" / "bench_tab"
CALIB = ROOT / "reports" / "tab_compare_calibration.json"
ENSEMBLE = ("fl", "gaps_paper")
LO, HI = 40, 83  # pitch range the tab pipeline keeps
SSR, HOP, CQ0 = 22050, 256, 36  # constant-Q: 22.05 kHz, 11.6 ms hop, MIDI 36..107
OPEN = [40, 45, 50, 55, 59, 64]
TOL = 0.05  # note matching: onset within 50 ms, same pitch
# feature sets compared by cross-validation on the benchmark (the best one is kept)
NOTE_SETS = {"confidence": ["post_ens"],
             "models": ["post_ens", "votes", "bp"],
             "models+context": ["post_ens", "votes", "bp", "alt_octave", "overtone", "reattack", "dur", "pitch"],
             "all": ["post_ens", "votes", "bp", "alt_octave", "overtone", "reattack", "dur", "pitch", "flux_s", "flux_gap",
                     "lvl_s", "lvl_gap", "resid", "rt_post", "n_chord", "is_lowest", "is_top"]}
MISS_SETS = {"confidence": ["post_ens"],
             "models": ["post_ens", "post_max", "n_models", "bp"],
             "models+context": ["post_ens", "post_max", "n_models", "bp", "harmonic", "octave", "same_pc", "pitch"],
             "all": ["post_ens", "post_max", "n_models", "bp", "harmonic", "octave", "same_pc", "pitch", "flux_s", "flux_r",
                     "lvl_s", "resid"]}


@dataclass
class Case:
    name: str
    notes: list[dict]  # onset_s, offset_s, midi, string (0 = low E), fret, bar
    stem: np.ndarray  # mono, rta.SR
    resid: np.ndarray  # mono, rta.SR
    stem_path: Path
    cents: float
    post: dict  # distinct checkpoint -> stem posteriors (all heads, model frames)
    ens: dict  # fl + gaps_paper stem posteriors, all heads
    bar_edges: np.ndarray  # bar b spans bar_edges[b-1] .. bar_edges[b]
    slots: np.ndarray  # 8th-note windows (target) or 0.25 s windows (benchmark)
    out: Path
    truth: list | None = None  # (source, onset, offset, midi)
    segs: list | None = None
    grid: tuple | None = None  # (beat times, swing centres) of the tab

    @property
    def f(self) -> float:
        return 2.0 ** (-self.cents / 1200.0)

    def bar_of(self, t: float) -> int:
        return int(np.searchsorted(self.bar_edges, t, side="right"))


# ------------------------------------------------------------------------------------ cases
def target_case(out: Path) -> Case:
    notes, meta = rta.read_tab(TAB, "frevo_guitar_tab")
    beats, c = np.asarray(meta["beat_times_s"]), np.asarray(meta["swing_centres"])
    n_bars = max(n["q"] for n in notes) // 8 + 1
    ibi = np.median(np.diff(beats[-8:]))
    b = np.r_[beats, beats[-1] + ibi * np.arange(1, max(2 * n_bars - len(beats), 0) + 4)]
    for n in notes:
        n["bar"] = n["q"] // 8 + 1
    stem, _ = load_audio(ROOT / meta["stem"], sr=rta.SR)
    resid, _ = load_audio(ROOT / "outputs" / "target" / "best" / "non_acoustic_guitar.wav", sr=rta.SR)
    heads = ("onset", "offset", "frame", "velocity")
    load = {m: {k: v for k, v in np.load(TAB / f"posteriors_{m}.npz").items() if k in heads}
            for m in ("fl", "gaps_paper", "gaps")}
    return Case("target", notes, stem.mean(0), resid.mean(0), ROOT / meta["stem"], meta["tuning_cents"], load,
                {k: np.mean([load[m][k] for m in ENSEMBLE], 0) for k in heads},
                b[: 2 * n_bars + 1: 2], np.sort(np.r_[b[:-1], b[:-1] + c[2] * np.diff(b)]), out, grid=(beats, c))


def bench_case(out: Path) -> Case:
    import bench_tab as bt

    wav = BENCH / "sep" / "acoustic_guitar.wav"
    arr, strings, f = bt._transcribe(wav, BENCH / "post_separated_fl+gaps_paper.npz", list(ENSEMBLE))
    notes = [{"onset_s": float(on), "offset_s": float(off), "midi": int(p), "string": int(s), "fret": int(p) - OPEN[s],
              "bar": int(on // 2.0) + 1} for (on, off, p, _), s in zip(arr, strings)]
    stem, _ = load_audio(wav, sr=rta.SR)
    resid, _ = load_audio(BENCH / "sep" / "non_acoustic_guitar.wav", sr=rta.SR)
    ens = dict(np.load(BENCH / "post_separated_fl+gaps_paper.npz"))
    post = {m: dict(np.load(BENCH / f"post_separated_{m}.npz")) for m in ("fl", "gaps_paper", "kroma")}
    with open(BENCH / "truth.csv") as fh:
        truth = [(r["source"], float(r["onset"]), float(r["offset"]), int(r["midi"])) for r in csv.DictReader(fh)]
    dur = stem.shape[1] / rta.SR
    return Case("benchmark", notes, stem.mean(0), resid.mean(0), wav, -1200 * np.log2(f), post, ens,
                np.arange(0, dur + 2.0, 2.0), np.arange(0, dur + 0.25, 0.25), out, truth,
                json.loads((BENCH / "segments.json").read_text()))


# ------------------------------------------------------------------------------- analyses
def semitone_db(x: np.ndarray, sr: int, cents: float) -> np.ndarray:
    """Constant-Q level (dB), one row per semitone MIDI 36..107 (tuned), 11.6 ms frames."""
    import librosa

    y = librosa.resample(x, orig_sr=sr, target_sr=SSR)
    fmin = librosa.midi_to_hz(CQ0) * 2 ** (cents / 1200)
    mag = np.abs(librosa.cqt(y, sr=SSR, hop_length=HOP, fmin=fmin, n_bins=72, bins_per_octave=12, filter_scale=0.8))
    return 20 * np.log10(mag + 1e-6)


def onset_flux(S: np.ndarray) -> np.ndarray:
    """Per semitone: max level in the 80 ms from the frame minus mean level 150-35 ms before it."""
    from numpy.lib.stride_tricks import sliding_window_view as win

    pad = np.pad(S, ((0, 0), (13, 7)), mode="edge")
    ahead = win(pad[:, 12:], 7, axis=1)[:, : S.shape[1]].max(-1)
    before = win(pad, 10, axis=1)[:, : S.shape[1]].mean(-1)
    return ahead - before


def match(a_on, a_p, b_on, b_p, tol: float = TOL) -> dict[int, int]:
    """Greedy one-to-one matching of notes a -> b: same pitch, closest onset within ``tol``."""
    pairs = sorted((abs(a_on[i] - b_on[j]), i, j) for i in range(len(a_on))
                   for j in np.nonzero((b_p == a_p[i]) & (np.abs(b_on - a_on[i]) <= tol))[0])
    out, used = {}, set()
    for _, i, j in pairs:
        if i not in out and j not in used:
            out[i] = int(j)
            used.add(int(j))
    return out


def model_posteriors(case: Case, render: np.ndarray) -> dict:
    """Ensemble posteriors of the rendering (cached per rendering)."""
    key = hashlib.sha1(np.round(render[:: 97], 4).tobytes()).hexdigest()[:16]
    cache = case.out / "posteriors_render.npz"
    if cache.exists():
        d = dict(np.load(cache))
        if str(d.pop("key")) == key:
            return d
    y16, _ = amt.to_model_input(render, rta.SR, case.cents)
    posts = [amt.posteriors(amt.load_model(m), y16, shifts=(0.0, 2.5)) for m in ENSEMBLE]
    post = {k: np.mean([p[k] for p in posts], 0) for k in posts[0]}
    np.savez_compressed(cache, key=key, **post)
    return post


def basic_pitch(case: Case) -> np.ndarray | None:
    """Basic Pitch notes on the stem (onset, offset, pitch), cached; None if it is not installed."""
    cache = case.out / "basic_pitch_stem.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text()), float).reshape(-1, 3)
    try:
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model, predict
    except Exception:  # noqa: BLE001 - optional
        return None
    _, _, ev = predict(str(case.stem_path), Model(build_icassp_2022_model_path(FilenameSuffix.onnx)))
    bp = [[float(e[0]), float(e[1]), int(e[2])] for e in ev]
    cache.write_text(json.dumps(bp))
    return np.array(bp, float).reshape(-1, 3)


def decoded(post: dict, f: float) -> np.ndarray:
    n = amt.decode(post, onset_thr=0.3, frame_thr=0.3)
    n[:, :2] *= f
    return n[(n[:, 2] >= LO) & (n[:, 2] <= HI)]


def near_max(post: np.ndarray, t: float, p: int, f: float, w: int = 4) -> float:
    a = int(round(t / f * amt.FPS))
    return float(post[max(a - w, 0):a + w + 1, p - amt.BEGIN_NOTE].max()) if 0 <= a < len(post) + w else 0.0


def analyse(case: Case, sf2: Path) -> dict:
    notes = case.notes
    on = np.array([n["onset_s"] for n in notes])
    off = np.array([n["offset_s"] for n in notes])
    pitch = np.array([n["midi"] for n in notes])
    render, *_ = rta.performance(notes, case.stem, rta.SR, case.cents, sf2)
    render = render.mean(0)[: len(case.stem)]
    render = np.pad(render, (0, len(case.stem) - len(render)))
    act = np.abs(case.stem) > 0.01 * np.abs(case.stem).max()
    render = render * np.sqrt(np.mean(case.stem[act] ** 2) / (np.mean(render[act] ** 2) + 1e-12))
    rpost = model_posteriors(case, render)
    rnotes = decoded(rpost, case.f)
    snotes = decoded(case.ens, case.f)
    bp = basic_pitch(case)
    Ss, Sr, Sx = (semitone_db(x, rta.SR, case.cents) for x in (case.stem, render, case.resid))
    np.savez_compressed(case.out / "state.npz", Ss=Ss.astype(np.float16), Sr=Sr.astype(np.float16),
                        Sx=Sx.astype(np.float16), f=case.f, ens=case.ens["onset"].astype(np.float16),
                        render=rpost["onset"].astype(np.float16), models=np.array(list(case.post)),
                        **{f"post_{m}": v["onset"].astype(np.float16) for m, v in case.post.items()})
    Fs, Fr = onset_flux(Ss), onset_flux(Sr)
    T = Ss.shape[1]
    fr = np.clip(np.round(on * SSR / HOP).astype(int), 0, T - 1)
    win = lambda S, i, r: S[r, max(i - 1, 0):i + 6].max()  # noqa: E731 - level around an onset
    top = lambda S, i: S[LO - CQ0:HI - CQ0 + 13, max(i - 1, 0):i + 6].max()  # noqa: E731

    # --- 2. every tab note
    rt = match(on, pitch, rnotes[:, 0], rnotes[:, 2].astype(int))
    bpm = match(on, pitch, bp[:, 0], bp[:, 2].astype(int)) if bp is not None else {}
    chord = [np.nonzero(np.abs(on - on[i]) < 0.03)[0] for i in range(len(on))]
    sounding = lambda t: pitch[(on < t - 0.02) & (off >= t - 0.03)]  # noqa: E731 - tab notes ringing at t
    rows = []
    for i, n in enumerate(notes):
        r, k = pitch[i] - CQ0, fr[i]
        seg = slice(max(k - 5, 0), k + 6)
        rise = lambda S: (np.argmax(np.diff(S[r, seg])) + seg.start) * HOP / SSR  # noqa: E731
        row = {"i": i, "bar": n["bar"], "onset_s": round(on[i], 3), "offset_s": round(off[i], 3), "pitch": int(pitch[i]),
               "string": 6 - n["string"], "fret": n["fret"],
               "votes": sum(near_max(p["onset"], on[i], pitch[i], case.f) >= 0.3 for p in case.post.values()),
               "post_ens": near_max(case.ens["onset"], on[i], pitch[i], case.f),
               "bp": (1 if i in bpm else 0) if bp is not None else -1,
               "flux_s": float(Fs[r, k]), "flux_r": float(Fr[r, k]), "flux_gap": float(Fr[r, k] - Fs[r, k]),
               "lvl_s": float(win(Ss, k, r) - top(Ss, k)), "lvl_gap": float((win(Sr, k, r) - top(Sr, k)) - (win(Ss, k, r) - top(Ss, k))),
               "resid": float(win(Sx, k, r) - win(Ss, k, r)),
               "rt_heard": int(i in rt), "rt_post": near_max(rpost["onset"], on[i], pitch[i], case.f),
               "dt_render_ms": round((rnotes[rt[i], 0] - on[i]) * 1000, 1) if i in rt else None,
               "dt_bp_ms": round((bp[bpm[i], 0] - on[i]) * 1000, 1) if i in bpm else None,
               "dt_rise_ms": round((rise(Ss) - rise(Sr)) * 1000, 1),
               "alt_octave": max(near_max(case.ens["onset"], on[i], q, case.f) for q in (pitch[i] - 12, pitch[i] + 12)
                                 if LO - 12 <= q <= HI + 12) - near_max(case.ens["onset"], on[i], pitch[i], case.f),
               "overtone": int(any(pitch[i] - q in (12, 19, 24, 28, 31) for q in sounding(on[i]))),
               "reattack": int(bool(np.any((pitch == pitch[i]) & (on < on[i] - 0.05) & (on > on[i] - 1.0) & (off > on[i] - 0.05)))),
               "dur": float(off[i] - on[i]), "n_chord": len(chord[i]),
               "is_lowest": int(pitch[i] == pitch[chord[i]].min()), "is_top": int(pitch[i] == pitch[chord[i]].max())}
        if case.grid is not None:  # where the played onset sits relative to its written 16th (in 16ths)
            row["q"] = n["q"]
            row["grid_dev"] = round(float(rta.time_to_q(np.array([on[i]]), *case.grid)[0] - n["q"]), 2)
        rows.append(row)

    # --- 3. notes the stem has and the tab lacks
    cands: list[dict] = []

    def add(t, p, src):
        for c in cands:
            if c["pitch"] == p and abs(c["onset_s"] - t) < 0.06:
                c["src"].add(src)
                return
        cands.append({"onset_s": float(t), "pitch": int(p), "src": {src}})

    tab_has = lambda t, p: bool(np.any((pitch == p) & (np.abs(on - t) < 0.06)))  # noqa: E731
    for m, p in case.post.items():
        for t, _, q, _ in decoded(p, case.f):
            if not tab_has(t, int(q)):
                add(t, int(q), m)
    if bp is not None:
        for t, _, q in bp:
            if LO <= q <= HI and not tab_has(t, int(q)):
                add(t, int(q), "bp")
    low = amt.decode(case.ens, onset_thr=0.1, frame_thr=0.3)  # the ensemble below its threshold
    low[:, :2] *= case.f
    for t, _, q, _ in low[(low[:, 2] >= LO) & (low[:, 2] <= HI)]:
        if not tab_has(t, int(q)):
            add(t, int(q), "ens_low")
    miss = []
    for c in cands:
        t, p = c["onset_s"], c["pitch"]
        k, r = int(np.clip(round(t * SSR / HOP), 0, T - 1)), p - CQ0
        snd = pitch[(on <= t + 0.03) & (off >= t - 0.03)]
        near = np.abs(on - t) < 0.06
        miss.append({"bar": case.bar_of(t), "onset_s": round(t, 3), "pitch": p, "sources": "+".join(sorted(c["src"])),
                     "n_models": len(c["src"] & set(case.post)), "bp": int("bp" in c["src"]),
                     "post_ens": near_max(case.ens["onset"], t, p, case.f),
                     "post_max": max(near_max(v["onset"], t, p, case.f) for v in case.post.values()),
                     "flux_s": float(Fs[r, k]), "flux_r": float(Fr[r, k]), "lvl_s": float(win(Ss, k, r) - top(Ss, k)),
                     "resid": float(win(Sx, k, r) - win(Ss, k, r)),
                     "harmonic": int(any(p - q in (12, 19, 24, 28, 31) for q in snd)),
                     "cq_peak": int(Fs[r, k] >= 12 and Fr[r, max(k - 2, 0):k + 3].max() < Fs[r, k] - 8),
                     "octave": int(bool(np.any(near & (np.abs(pitch - p) == 12)))),
                     "same_pc": int(bool(np.any(near & (pitch % 12 == p % 12)))),
                     "same_pitch_tab_ms": (round(float(1000 * (on[pitch == p] - t)[np.argmin(np.abs(on[pitch == p] - t))]), 1)
                                           if np.any(pitch == p) else None)})

    # --- 1 + 4. per bar
    import librosa

    cs = librosa.feature.chroma_cqt(y=librosa.resample(case.stem, orig_sr=rta.SR, target_sr=SSR), sr=SSR, hop_length=512)
    cr = librosa.feature.chroma_cqt(y=librosa.resample(render, orig_sr=rta.SR, target_sr=SSR), sr=SSR, hop_length=512)
    es = librosa.onset.onset_strength(y=librosa.resample(case.stem, orig_sr=rta.SR, target_sr=SSR), sr=SSR, hop_length=128)
    er = librosa.onset.onset_strength(y=librosa.resample(render, orig_sr=rta.SR, target_sr=SSR), sr=SSR, hop_length=128)
    slot_cos = []
    for a, b in zip(case.slots[:-1], case.slots[1:]):
        i0, i1 = int(a * SSR / 512), max(int(b * SSR / 512), int(a * SSR / 512) + 1)
        u, v = cs[:, i0:i1].mean(1), cr[:, i0:i1].mean(1)
        slot_cos.append((a, float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)), float(u.sum())))
    bars = []
    for b in range(1, len(case.bar_edges)):
        t0, t1 = case.bar_edges[b - 1], case.bar_edges[b]
        tb = (on >= t0) & (on < t1)
        sb = (snotes[:, 0] >= t0) & (snotes[:, 0] < t1)
        rb = (rnotes[:, 0] >= t0) & (rnotes[:, 0] < t1)
        f1 = lambda a, b_: 2 * len(match(a[:, 0], a[:, 2].astype(int), b_[:, 0], b_[:, 2].astype(int))) / max(len(a) + len(b_), 1)  # noqa: E731
        tabarr = np.c_[on[tb], off[tb], pitch[tb]]
        cos = [c for a, c, e in slot_cos if t0 <= a < t1 and e > 0.5]
        i0, i1 = int(t0 * SSR / 128), int(t1 * SSR / 128)
        m = int(0.04 * SSR / 128)
        u, v = es[i0:i1] - es[i0:i1].mean(), er[i0:i1] - er[i0:i1].mean()
        xc = [np.dot(u[max(0, -d):len(u) - max(0, d)], v[max(0, d):len(v) - max(0, -d)]) for d in range(-m, m + 1)] if len(u) > 2 * m else [0]
        bars.append({"bar": b, "t0": round(float(t0), 3), "t1": round(float(t1), 3), "n_tab": int(tb.sum()),
                     "reproduction_f1_tab_vs_rendering": round(f1(tabarr, rnotes[rb]), 3) if tb.any() or rb.any() else None,
                     "transcription_f1_stem_vs_rendering": round(f1(snotes[sb], rnotes[rb]), 3) if sb.any() or rb.any() else None,
                     "chroma_median": round(float(np.median(cos)), 3) if cos else None,
                     "chroma_min": round(float(np.min(cos)), 3) if cos else None,
                     "lag_ms": round((int(np.argmax(xc)) - m) * 128 / SSR * 1000, 1) if len(xc) > 1 else None})
    return {"notes": rows, "missing": miss, "bars": bars, "rnotes": rnotes, "snotes": snotes, "bp": bp,
            "S": (Ss, Sr, Sx), "render": render}


# ----------------------------------------------------------------------------- calibration
def label(case: Case, res: dict) -> None:
    """Benchmark: tab note wrong = not matched to the truth; missing candidate real = a truth note
    (50 ms, same pitch) that no tab note matches.  Only notes inside the benchmark clips count."""
    inseg = lambda t: any(s["start"] <= t < s["end"] + 0.5 for s in case.segs)  # noqa: E731
    tr_on = np.array([t[1] for t in case.truth])
    tr_p = np.array([t[3] for t in case.truth])
    on = np.array([r["onset_s"] for r in res["notes"]])
    p = np.array([r["pitch"] for r in res["notes"]])
    m = match(on, p, tr_on, tr_p)
    hit = set(m.values())
    for r in res["notes"]:
        r["inseg"] = int(inseg(r["onset_s"]))
        r["wrong"] = int(r["i"] not in m)
    for c in res["missing"]:
        c["inseg"] = int(inseg(c["onset_s"]))
        c["real"] = int(bool(np.any((tr_p == c["pitch"]) & (np.abs(tr_on - c["onset_s"]) <= TOL)
                                    & ~np.isin(np.arange(len(tr_on)), list(hit)))))


def fit_best(rows: list[dict], sets: dict[str, list[str]], y: str, groups: list[int]) -> dict:
    """Fit every feature set with clip-grouped CV; keep the best CV AUC (the others are reported)."""
    fits = {name: fit(rows, feats, y, groups) for name, feats in sets.items()}
    best = max(fits, key=lambda k: fits[k]["cv_auc"])
    out = fits[best] | {"feature_set": best}
    out["compared"] = {k: {"cv_auc": round(v["cv_auc"], 3), "cv_average_precision": round(v["cv_average_precision"], 3)}
                       for k, v in fits.items()}
    return out


def fit(rows: list[dict], feats: list[str], y: str, groups: list[int]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    X = np.array([[r[f] for f in feats] for r in rows], float)
    Y = np.array([r[y] for r in rows])
    g = np.array(groups)
    oof = np.zeros(len(Y))
    for k in np.unique(g):
        tr, te = g != k, g == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(C=0.5, max_iter=2000).fit((X[tr] - mu) / sd, Y[tr])
        oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(C=0.5, max_iter=2000).fit((X - mu) / sd, Y)
    bins = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]
    return {"features": feats, "mean": mu.tolist(), "std": sd.tolist(), "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]), "n": int(len(Y)), "positives": int(Y.sum()),
            "cv_auc": float(roc_auc_score(Y, oof)), "cv_average_precision": float(average_precision_score(Y, oof)),
            "cv_calibration": [{"p_range": list(b), "n": int(((oof >= b[0]) & (oof < b[1])).sum()),
                                "positive_rate": float(Y[(oof >= b[0]) & (oof < b[1])].mean()) if ((oof >= b[0]) & (oof < b[1])).any() else None}
                               for b in bins], "oof": oof.tolist()}


def score(rows: list[dict], model: dict, key: str) -> None:
    X = np.array([[r[f] for f in model["features"]] for r in rows], float).reshape(len(rows), -1)
    z = ((X - model["mean"]) / np.array(model["std"])) @ np.array(model["coef"]) + model["intercept"]
    for r, v in zip(rows, 1 / (1 + np.exp(-z))):
        r[key] = round(float(v), 3)


# --------------------------------------------------------------------------------- outputs
def evidence(case: Case, res: dict, bars: list[int], top_n: int = 999) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = case.out / "evidence"
    d.mkdir(parents=True, exist_ok=True)
    Ss, Sr, Sx = res["S"]
    for b in bars[:top_n]:
        t0, t1 = case.bar_edges[b - 1] - 0.15, case.bar_edges[b] + 0.15
        k0, k1 = max(int(t0 * SSR / HOP), 0), min(int(t1 * SSR / HOP), Ss.shape[1])
        if k1 - k0 < 4:
            continue
        fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
        ref = Ss[:, k0:k1].max()
        for a, S, title in zip(ax, (Ss, Sr, Sx), ("stem (original guitar)", "tab rendering", "residual (violin, clarinet)")):
            a.imshow(S[LO - 2 - CQ0:HI + 3 - CQ0, k0:k1], origin="lower", aspect="auto", cmap="magma", vmin=ref - 55, vmax=ref,
                     extent=[t0, t1, LO - 2.5, HI + 2.5])
            a.set_title(title, fontsize=9)
            a.set_ylabel("MIDI")
            for r in res["notes"]:
                if r["bar"] == b:
                    bad = r.get("p_wrong", 0) >= 0.5
                    a.add_patch(plt.Rectangle((r["onset_s"], r["pitch"] - 0.45), max(r["offset_s"] - r["onset_s"], 0.03), 0.9,
                                              fill=False, ec="red" if bad else "white", lw=1.4 if bad else 0.7))
            for c in res["missing"]:
                if c["bar"] == b and c.get("p_real", 0) >= 0.3:
                    a.plot(c["onset_s"], c["pitch"], "o", mfc="none", mec="cyan", ms=9, mew=1.5)
        ax[-1].set_xlabel("time (s)   boxes: tab notes (red: P(wrong) >= 0.5)   cyan circles: notes the tab may lack (P >= 0.3)")
        fig.suptitle(f"{case.name} bar {b}", fontsize=10)
        fig.savefig(d / f"bar_{b:03d}.png", dpi=72)
        plt.close(fig)


def write(case: Case, res: dict, summary: dict) -> None:
    out = case.out
    for name, rows in (("notes", res["notes"]), ("missing", res["missing"]), ("bars", res["bars"])):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / f"{name}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


def summarise(case: Case, res: dict) -> dict:
    notes, bars = res["notes"], res["bars"]
    rtr = [r["dt_render_ms"] for r in notes if r["dt_render_ms"] is not None]
    dbp = [r["dt_bp_ms"] for r in notes if r["dt_bp_ms"] is not None]
    lags = [b["lag_ms"] for b in bars if b["lag_ms"] is not None and b["n_tab"] >= 3]
    s = {"case": case.name, "tab_notes": len(notes),
         "reproduction": {"tab_notes_found_in_rendering": round(float(np.mean([r["rt_heard"] for r in notes])), 3),
                          "rendering_notes": int(len(res["rnotes"])),
                          "f1_tab_vs_rendering": round(2 * len(match(np.array([r["onset_s"] for r in notes]), np.array([r["pitch"] for r in notes]),
                                                                       res["rnotes"][:, 0], res["rnotes"][:, 2].astype(int))) / (len(notes) + len(res["rnotes"])), 3),
                          "f1_stem_vs_rendering_same_transcriber": round(2 * len(match(res["snotes"][:, 0], res["snotes"][:, 2].astype(int),
                                                                                         res["rnotes"][:, 0], res["rnotes"][:, 2].astype(int)))
                                                                         / (len(res["snotes"]) + len(res["rnotes"])), 3)},
         "timing": {"rendering_vs_tab_onset_ms_median_abs": round(float(np.median(np.abs(rtr))), 1) if rtr else None,
                    "basic_pitch_vs_tab_onset_ms_median": round(float(np.median(dbp)), 1) if dbp else None,
                    "basic_pitch_vs_tab_onset_ms_p90_abs": round(float(np.percentile(np.abs(dbp), 90)), 1) if dbp else None,
                    "bar_lag_ms_median_abs": round(float(np.median(np.abs(lags))), 1) if lags else None,
                    "bars_with_lag_over_20ms": [b["bar"] for b in bars if b["lag_ms"] is not None and b["n_tab"] >= 3 and abs(b["lag_ms"]) > 20]},
         "chroma": {"bar_median": round(float(np.median([b["chroma_median"] for b in bars if b["chroma_median"] is not None])), 3)},
         "candidates": {"missing": len(res["missing"])}}
    if "p_wrong" in notes[0]:
        s["expected_wrong_notes"] = round(float(sum(r["p_wrong"] for r in notes)), 1)
        s["notes_p_wrong_ge_0.5"] = int(sum(r["p_wrong"] >= 0.5 for r in notes))
    if res["missing"] and "p_real" in res["missing"][0]:
        s["expected_missing_found"] = round(float(sum(c["p_real"] for c in res["missing"])), 1)
        s["missing_p_real_ge_0.5"] = int(sum(c["p_real"] >= 0.5 for c in res["missing"]))
    return s


PARTIALS = (0, 12, 19, 24, 28, 31)  # semitones above the fundamental of harmonics 1-6


def probe(folder: Path, t: float, pitch: int, image: Path | None = None) -> str:
    """Evidence at one (time, pitch) from a comparison folder's cached state: levels of the note's
    first six harmonics in stem / rendering / residual (at the onset and their rise over the 150 ms
    before), onset posteriors at the pitch and its octaves, and nearby tab / Basic Pitch notes."""
    st = np.load(folder / "state.npz")
    Ss, Sr, Sx = (st[k].astype(np.float32) for k in ("Ss", "Sr", "Sx"))
    f, T = float(st["f"]), Ss.shape[1]
    k = int(np.clip(round(t * SSR / HOP), 0, T - 1))
    now, pre = slice(max(k - 1, 0), k + 6), slice(max(k - 13, 0), max(k - 3, 1))
    lines = [f"probe t={t:.3f} s pitch={pitch} (MIDI)", "harmonic  midi   stem dB (rise)   rendering dB (rise)   residual dB (rise)"]
    for h, d in zip(range(1, 7), PARTIALS):
        r = pitch + d - CQ0
        if 0 <= r < Ss.shape[0]:
            cells = [f"{S[r, now].max():7.1f} ({S[r, now].max() - S[r, pre].mean():+5.1f})" for S in (Ss, Sr, Sx)]
            lines.append(f"  h{h}      {pitch + d:4d}  " + "      ".join(cells))
    lines.append("onset posterior (max within 40 ms)   p-12    p     p+12")
    for name in [f"post_{m}" for m in st["models"]] + ["ens", "render"]:
        v = st[name].astype(np.float32)
        a = int(round(t / f * amt.FPS))
        vals = [float(v[max(a - 4, 0):a + 5, q - amt.BEGIN_NOTE].max()) if 0 <= q - amt.BEGIN_NOTE < v.shape[1] else float("nan")
                for q in (pitch - 12, pitch, pitch + 12)]
        label = {"ens": "stem, fl+gaps_paper", "render": "rendering, fl+gaps_paper"}.get(name, "stem, " + name[5:])
        lines.append(f"  {label:34s}" + "  ".join(f"{x:5.2f}" for x in vals))
    with open(folder / "notes.csv") as fh:
        tab = [r for r in csv.DictReader(fh) if abs(float(r["onset_s"]) - t) <= 0.25]
    lines.append("tab notes within 250 ms: " + (", ".join(f"{float(r['onset_s']) - t:+.3f}s midi {r['pitch']} (string {r['string']} fret {r['fret']})"
                                                          for r in tab) or "none"))
    bpf = folder / "basic_pitch_stem.json"
    if bpf.exists():
        bp = [b for b in json.loads(bpf.read_text()) if abs(b[0] - t) <= 0.25]
        lines.append("Basic Pitch notes within 250 ms: " + (", ".join(f"{b[0] - t:+.3f}s midi {int(b[2])}" for b in bp) or "none"))
    if image is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k0, k1 = max(k - 35, 0), min(k + 35, T)
        lo, hi = max(pitch - 14 - CQ0, 0), min(pitch + 34 - CQ0, Ss.shape[0])
        fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True, constrained_layout=True)
        ref = Ss[lo:hi, k0:k1].max()
        for a, S, title in zip(ax, (Ss, Sr, Sx), ("stem", "rendering", "residual")):
            a.imshow(S[lo:hi, k0:k1], origin="lower", aspect="auto", cmap="magma", vmin=ref - 50, vmax=ref,
                     extent=[k0 * HOP / SSR, k1 * HOP / SSR, lo + CQ0 - 0.5, hi + CQ0 - 0.5])
            for r in tab:
                a.add_patch(plt.Rectangle((float(r["onset_s"]), int(r["pitch"]) - 0.45), max(float(r["offset_s"]) - float(r["onset_s"]), 0.03),
                                          0.9, fill=False, ec="white", lw=0.8))
            for d in PARTIALS:
                a.plot([t - 0.03, t + 0.03], [pitch + d] * 2, color="cyan", lw=1.2 if d else 2.0)
            a.set_title(title)
        ax[0].set_ylabel("MIDI (cyan: queried pitch and its harmonics)")
        fig.savefig(image, dpi=70)
        plt.close(fig)
        lines.append(f"image: {image}")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        pp = argparse.ArgumentParser(prog="compare_tab_audio.py probe")
        pp.add_argument("folder")
        pp.add_argument("--t", type=float, required=True, help="time in s")
        pp.add_argument("--pitch", type=int, required=True, help="MIDI pitch")
        pp.add_argument("--image", default=None)
        a = pp.parse_args(sys.argv[2:])
        print(probe(Path(a.folder), a.t, a.pitch, Path(a.image) if a.image else None))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true", help="run on the benchmark and (re)fit the calibration")
    ap.add_argument("--sf2", default=None)
    ap.add_argument("--evidence-bars", type=int, default=60, help="bars to draw, most suspicious first")
    args = ap.parse_args()
    sf2 = rta.find_sf2(args.sf2)
    out = (BENCH if args.bench else TAB) / "compare"
    out.mkdir(parents=True, exist_ok=True)
    case = bench_case(out) if args.bench else target_case(out)
    res = analyse(case, sf2)
    if args.bench:
        label(case, res)
        seg = lambda t: int(np.searchsorted([s["start"] for s in case.segs], t, side="right"))  # noqa: E731
        nrows = [r for r in res["notes"] if r["inseg"]]
        mrows = [c for c in res["missing"] if c["inseg"]]
        cal = {"notes": fit_best(nrows, NOTE_SETS, "wrong", [seg(r["onset_s"]) for r in nrows]),
               "missing": fit_best(mrows, MISS_SETS, "real", [seg(c["onset_s"]) for c in mrows])}
        for rows, key, m in ((nrows, "p_wrong", cal["notes"]), (mrows, "p_real", cal["missing"])):
            for r, v in zip(rows, m.pop("oof")):
                r[key + "_cv"] = round(v, 3)
        # simple baselines for comparison
        from sklearn.metrics import roc_auc_score

        cal["baselines"] = {
            "notes_by_votes_auc": float(roc_auc_score([r["wrong"] for r in nrows], [-r["votes"] for r in nrows])),
            "notes_by_post_ens_auc": float(roc_auc_score([r["wrong"] for r in nrows], [-r["post_ens"] for r in nrows])),
            "missing_by_n_models_auc": float(roc_auc_score([c["real"] for c in mrows], [c["n_models"] for c in mrows]))}
        CALIB.write_text(json.dumps(cal, indent=1))
    cal = json.loads(CALIB.read_text()) if CALIB.exists() else None
    if cal:
        score(res["notes"], cal["notes"], "p_wrong")
        score(res["missing"], cal["missing"], "p_real")
    summary = summarise(case, res)
    if args.bench:
        summary["calibration"] = {k: {kk: v[kk] for kk in ("feature_set", "n", "positives", "cv_auc", "cv_average_precision",
                                                          "cv_calibration", "compared")}
                                  for k, v in cal.items() if k != "baselines"} | {"baselines": cal["baselines"]}
    susp = {}
    for r in res["notes"]:
        susp[r["bar"]] = susp.get(r["bar"], 0) + r.get("p_wrong", 0)
    for c in res["missing"]:
        susp[c["bar"]] = susp.get(c["bar"], 0) + c.get("p_real", 0)
    order = sorted((b for b in susp if 1 <= b < len(case.bar_edges)), key=lambda b: -susp[b])
    summary["most_suspicious_bars"] = [{"bar": b, "score": round(susp[b], 2)} for b in order[:30]]
    evidence(case, res, order, args.evidence_bars)
    write(case, res, summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
