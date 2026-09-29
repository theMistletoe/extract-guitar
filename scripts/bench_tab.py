#!/usr/bin/env python
"""End-to-end accuracy of the tab pipeline on music with known notes.

Guitar recordings with note ground truth (GuitarSet bossa-nova comping: notes *and* strings;
GAPS classical-guitar test pieces incl. choro) are mixed with real violin and clarinet stems
(URMP) and a percussion stem (RawStems), at the target song's balance (guitar ~4.7 dB below the
rest).  The mix is separated with the Champion pipeline exactly as the target song was, then
transcribed; clean-guitar transcription is the reference condition.

    python scripts/bench_tab.py build --guitarset data/raw/guitarset     # -> data/bench_tab/
    python -m acoustic_separator --input data/bench_tab/mix.wav --quality max --output data/bench_tab/sep
    python scripts/bench_tab.py eval                                     # -> reports/tab_benchmark.*
    python scripts/bench_tab.py guitarset --guitarset data/raw/guitarset  # clean GuitarSet, 60 excerpts

GuitarSet: Zenodo 3371780 (audio_mono-mic + annotation); GAPS, URMP, RawStems: Hugging Face.
``eval`` needs mir_eval and mido (both in the ``tab`` extra).
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

from acoustic_separator.audio import load_audio, resample, save_audio  # noqa: E402

BENCH = ROOT / "data" / "bench_tab"
SR = 44100
CLIP_S = 20.0
GAP_S = 1.5
GUITARSET = ["00_BN2-166-Ab_comp", "02_BN3-154-E_comp", "04_BN1-147-Gb_comp", "05_BN3-119-G_comp"]
GAPS = [("126_XD1wc", 30.0), ("112_mf1wc", 60.0)]  # Tico Tico no Fuba (2/4 choro), Gavota Choro
URMP_VN = ["01_Jupiter_vn_vc/AuSep_1_vn_01_Jupiter.wav", "02_Sonata_vn_vn/AuSep_1_vn_02_Sonata.wav",
           "08_Spring_fl_vn/AuSep_2_vn_08_Spring.wav"]
URMP_CL = ["03_Dance_fl_cl/AuSep_2_cl_03_Dance.wav", "14_Waltz_fl_fl_cl/AuSep_3_cl_14_Waltz.wav",
           "19_Pavane_cl_vn_vc/AuSep_1_cl_19_Pavane.wav"]
PERC_PATTERN = r"/Rhy/PERC/[^/]*(Shaker|Tamb|Conga|Perc)[^/]*\.flac$"
# relative levels (dB re guitar RMS): together the accompaniment is ~4.7 dB above the guitar
LEVELS = {"violin": 1.5, "clarinet": 0.5, "percussion": -4.0}
PAN = {"guitar": 0.0, "violin": -0.3, "clarinet": 0.3, "percussion": 0.0}


def _hf(repo: str, path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=path, repo_type="dataset",
                                cache_dir=str(ROOT / "data" / "raw" / "hf")))


def _mono(path: Path) -> np.ndarray:
    x, sr = load_audio(path)
    return resample(x, sr, SR).mean(0)


def _active_window(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """A window of n samples from x where the stem actually plays (RMS above 20 % of max)."""
    hop = SR // 2
    rms = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, max(len(x) - n, 1), hop)])
    ok = np.nonzero(rms > 0.2 * rms.max())[0] if len(rms) else np.array([0])
    start = int(rng.choice(ok)) * hop if len(ok) else 0
    seg = x[start:start + n]
    return np.pad(seg, (0, n - len(seg)))


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def _pan(x: np.ndarray, p: float) -> np.ndarray:
    """Constant-power pan, p in [-1 (left), 1 (right)]."""
    th = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(th), x * np.sin(th)])


def _guitarset_truth(ann: Path, t0: float, t1: float) -> list[tuple]:
    d = json.loads(ann.read_text())
    out = []
    for a in d["annotations"]:
        if a["namespace"] != "note_midi":
            continue
        s = int(a["annotation_metadata"]["data_source"])
        for ob in a["data"]:
            if t0 <= ob["time"] < t1:
                out.append((ob["time"] - t0, ob["time"] + ob["duration"] - t0, int(round(ob["value"])), s))
    return out


def _midi_truth(mid: Path, t0: float, t1: float) -> list[tuple]:
    import mido

    out, on = [], {}
    t = 0.0
    for msg in mido.MidiFile(str(mid)):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[(msg.channel, msg.note)] = t
        elif msg.type in ("note_off", "note_on") and (msg.channel, msg.note) in on:
            s = on.pop((msg.channel, msg.note))
            if t0 <= s < t1:
                out.append((s - t0, t - t0, msg.note, -1))
    return out


def build(args) -> int:
    import re

    from huggingface_hub import HfApi

    rng = np.random.default_rng(11)
    gs = Path(args.guitarset)
    mic = gs / "mic" if (gs / "mic").exists() else gs / "audio_mic"
    ann = gs / "annotation"
    if not ann.exists():
        import urllib.request
        import zipfile

        z = gs / "annotation.zip"
        urllib.request.urlretrieve("https://zenodo.org/api/records/3371780/files/annotation.zip/content", z)
        zipfile.ZipFile(z).extractall(ann)
    vn = [_mono(_hf("Eredis02/URMP", p)) for p in URMP_VN]
    cl = [_mono(_hf("Eredis02/URMP", p)) for p in URMP_CL]
    perc = None
    try:
        files = HfApi().list_repo_files("kwatcharasupat/mixing-secrets-rawstems", repo_type="dataset")
        cand = sorted(f for f in files if re.search(PERC_PATTERN, "/" + f))
        if cand:
            perc = _mono(_hf("kwatcharasupat/mixing-secrets-rawstems", cand[0]))
    except Exception as e:  # noqa: BLE001 - percussion is optional
        print("percussion stem unavailable:", e)

    n = int(CLIP_S * SR)
    gap = np.zeros(int(GAP_S * SR))
    clips = []
    for name in GUITARSET:
        g = _mono(mic / f"{name}_mic.wav")[:n]
        clips.append(("guitarset", name, np.pad(g, (0, n - len(g))),
                      _guitarset_truth(ann / f"{name}.jams", 0.0, CLIP_S)))
    for gid, t0 in GAPS:
        g = _mono(_hf("xavriley/GAPS", f"audio/{gid}.wav"))[int(t0 * SR):int(t0 * SR) + n]
        clips.append(("gaps", gid, np.pad(g, (0, n - len(g))),
                      _midi_truth(_hf("xavriley/GAPS", f"midi/{gid}.mid"), t0, t0 + CLIP_S)))

    mix_parts, clean_parts, truth, segs = [], [], [], []
    t = 0.0
    for i, (src, name, g, notes) in enumerate(clips):
        g = g / _rms(g) * 10 ** (-24 / 20)
        stems = {"guitar": g, "violin": _active_window(vn[i % len(vn)], n, rng),
                 "clarinet": _active_window(cl[i % len(cl)], n, rng)}
        if perc is not None:
            stems["percussion"] = _active_window(perc, n, rng)
        mix = np.zeros((2, n))
        for k, x in stems.items():
            if k != "guitar":
                x = x / _rms(x) * _rms(g) * 10 ** (LEVELS[k] / 20)
            mix += _pan(x, PAN[k])
        mix_parts += [mix, np.zeros((2, len(gap)))]
        clean_parts += [_pan(g, 0.0), np.zeros((2, len(gap)))]
        truth += [(src, name, on + t, off + t, p, s) for on, off, p, s in notes]
        segs.append({"source": src, "name": name, "start": t, "end": t + CLIP_S,
                     "accomp_db": round(20 * np.log10(_rms(mix - _pan(g, 0.0)) / _rms(_pan(g, 0.0))), 2)})
        t += CLIP_S + GAP_S
    BENCH.mkdir(parents=True, exist_ok=True)
    mix = np.concatenate(mix_parts, 1)
    peak = np.abs(mix).max()
    save_audio(BENCH / "mix.wav", (mix / peak * 0.9).astype(np.float32), SR)
    save_audio(BENCH / "clean.wav", (np.concatenate(clean_parts, 1) / peak * 0.9).astype(np.float32), SR)
    with open(BENCH / "truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "clip", "onset", "offset", "midi", "string"])
        w.writerows(truth)
    (BENCH / "segments.json").write_text(json.dumps(segs, indent=1))
    print(f"wrote {BENCH} ({t:.0f} s, {len(truth)} notes, percussion={'yes' if perc is not None else 'no'})")
    return 0


def _transcribe(wav: Path, cache: Path, models: list[str]) -> tuple[np.ndarray, list]:
    """Notes + fingering exactly as scripts/transcribe_tab.py does (no bar grid needed)."""
    from acoustic_separator.tab import amt, fretboard

    audio, sr = load_audio(wav)
    cents = amt.estimate_tuning(audio.mean(0), sr)
    y16, f = amt.to_model_input(audio, sr, cents)
    if cache.exists():
        post = dict(np.load(cache))
    else:
        posts = [amt.posteriors(amt.load_model(m), y16, shifts=(0.0, 2.5)) for m in models]
        post = {k: np.mean([p[k] for p in posts], 0) for k in posts[0]}
        np.savez_compressed(cache, **post)
    notes = amt.decode(post, onset_thr=0.3, frame_thr=0.3)
    notes[:, :2] *= f
    notes = notes[(notes[:, 2] >= 40) & (notes[:, 2] <= 83)]
    groups = fretboard.group_notes(notes[:, 0], notes[:, 1], notes[:, 2].astype(int), conf=notes[:, 3])
    strings = [None] * len(notes)
    for g, (sf, _) in zip(groups, fretboard.assign(groups)):
        for i, x in zip(g.idx, sf):
            strings[i] = None if x is None else x[0]
    keep = np.array([s is not None for s in strings])
    return notes[keep], [s for s in strings if s is not None]


def _score(ref, est, est_strings):
    import mir_eval

    hz = lambda m: 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)  # noqa: E731
    ri = np.array([[r[2], max(r[3], r[2] + 0.01)] for r in ref])
    ei = np.c_[est[:, 0], np.maximum(est[:, 1], est[:, 0] + 0.01)] if len(est) else np.zeros((0, 2))
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, hz([x[4] for x in ref]), ei, hz(est[:, 2]) if len(est) else np.zeros(0), offset_ratio=None)
    match = mir_eval.transcription.match_notes(ri, hz([x[4] for x in ref]), ei,
                                               hz(est[:, 2]) if len(est) else np.zeros(0), offset_ratio=None)
    s_ok = [ref[i][5] == est_strings[j] for i, j in match if ref[i][5] >= 0]
    return {"precision": p, "recall": r, "f1": f, "n_ref": len(ref), "n_est": int(len(est)),
            "string_acc_matched": float(np.mean(s_ok)) if s_ok else None}


def evaluate(args) -> int:
    segs = json.loads((BENCH / "segments.json").read_text())
    with open(BENCH / "truth.csv") as f:
        truth = [(r["source"], r["clip"], float(r["onset"]), float(r["offset"]), int(r["midi"]), int(r["string"]))
                 for r in csv.DictReader(f)]
    sep = Path(args.sep) / "acoustic_guitar.wav"
    out = {}
    for mset in args.sets:
        models = mset.split("+")
        for cond, wav in (("clean guitar", BENCH / "clean.wav"), ("separated from mix", sep)):
            notes, strings = _transcribe(wav, BENCH / f"post_{cond.split()[0]}_{mset}.npz", models)
            rows = {}
            for grp in ("guitarset", "gaps", "all"):
                ref = [x for x in truth if grp in ("all", x[0])]
                sel = np.array([any(s["start"] <= t < s["end"] + 0.5 and grp in ("all", s["source"]) for s in segs)
                                for t in notes[:, 0]], bool)
                rows[grp] = _score(ref, notes[sel], [s for s, k in zip(strings, sel) if k])
            out[f"{mset} | {cond}"] = rows
            print(mset, cond, json.dumps(rows["all"]))
    rep = ROOT / "reports"
    (rep / "tab_benchmark.json").write_text(json.dumps({"segments": segs, "results": out}, indent=2))
    lines = ["# Tab pipeline: end-to-end accuracy on music with known notes", "",
             "Guitar with note ground truth (GuitarSet bossa-nova comping, 4 players; GAPS classical-guitar "
             "test pieces incl. a choro) mixed with violin + clarinet (URMP) + percussion (RawStems) at the "
             "target song's balance (guitar about 4.7 dB below the rest), separated with the Champion pipeline "
             "(`--quality max`), then transcribed exactly like the target song.  Note F1: onset within 50 ms and "
             "same pitch (mir_eval).  String accuracy: share of correctly detected GuitarSet notes placed on the "
             "performer's string.  Built by `scripts/bench_tab.py` (129 s, "
             f"{len(truth)} notes).", "",
             "Caveat: `gaps_paper` may have been trained with GuitarSet (the GAPS paper reports a supervised "
             "GuitarSet setting), so its rows on GuitarSet material may be optimistic; the GAPS test pieces "
             "(`gaps` rows) are unseen by all checkpoints' documented training data.", "",
             "| checkpoints | condition | material | precision | recall | F1 | string acc. |",
             "|---|---|---|---|---|---|---|"]
    for key, rows in out.items():
        mset, cond = key.split(" | ")
        for grp, r in rows.items():
            sa = f"{r['string_acc_matched']:.3f}" if r["string_acc_matched"] is not None else "-"
            lines.append(f"| {mset} | {cond} | {grp} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {sa} |")
    (rep / "tab_benchmark.md").write_text("\n".join(lines) + "\n")
    print("wrote reports/tab_benchmark.md")
    return 0


def guitarset(args) -> int:
    """Note-onset F1 on clean GuitarSet mic audio: every 6th excerpt (60, all players and styles)."""
    import mir_eval

    from acoustic_separator.tab import amt

    gs = Path(args.guitarset)
    mic = gs / "mic" if (gs / "mic").exists() else gs / "audio_mic"
    wavs = sorted(mic.glob("*_mic.wav"))[::6]
    hz = lambda m: 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)  # noqa: E731
    models: dict = {}
    lines = ["# Tab note model on clean GuitarSet (60 mic excerpts, all players and styles)", "",
             "Every 6th excerpt of GuitarSet's mono mic recordings (steel-string), onset within 50 ms and "
             f"same pitch (mir_eval), onset threshold {args.thr}, mean over excerpts.  "
             "`scripts/bench_tab.py guitarset`.", "",
             "Caveat: the FL checkpoint is documented as zero-shot on GuitarSet, but the GAPS paper reports "
             "both supervised (GuitarSet-trained) and zero-shot results and does not say which released "
             "checkpoint is which, so `gaps_paper` (and ensembles containing it) may be optimistic here.", "",
             "| checkpoints | precision | recall | F1 |", "|---|---|---|---|"]
    for mset in args.sets:
        res = []
        for wav in wavs:
            base = wav.name.replace("_mic.wav", "")
            posts = []
            for m in mset.split("+"):
                cache = BENCH / "guitarset" / m / f"{base}.npz"
                if not cache.exists():
                    x, sr = load_audio(wav)
                    y16 = resample(x, sr, amt.SR).mean(0)
                    models[m] = models.get(m) or amt.load_model(m)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache, **amt.posteriors(models[m], y16))
                posts.append(dict(np.load(cache)))
            post = {k: np.mean([q[k] for q in posts], 0) for k in ("onset", "offset", "frame", "velocity")}
            est = amt.decode(post, onset_thr=args.thr, frame_thr=0.3)
            ref = _guitarset_truth(gs / "annotation" / f"{base}.jams", 0.0, 1e9)
            ri = np.array([[r[0], max(r[1], r[0] + 0.01)] for r in ref])
            ei = np.c_[est[:, 0], np.maximum(est[:, 1], est[:, 0] + 0.01)]
            res.append(mir_eval.transcription.precision_recall_f1_overlap(
                ri, hz([r[2] for r in ref]), ei, hz(est[:, 2]), offset_ratio=None)[:3])
        p, r, f = np.mean(res, 0)
        print(f"{mset:24s} P={p:.3f} R={r:.3f} F1={f:.3f}")
        lines.append(f"| {mset} | {p:.3f} | {r:.3f} | {f:.3f} |")
    (ROOT / "reports" / "tab_benchmark_guitarset.md").write_text("\n".join(lines) + "\n")
    print("wrote reports/tab_benchmark_guitarset.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--guitarset", default=str(ROOT / "data" / "raw" / "guitarset"))
    e = sub.add_parser("eval")
    e.add_argument("--sep", default=str(BENCH / "sep"))
    e.add_argument("--sets", nargs="+", default=["kroma", "fl", "gaps_paper", "fl+gaps_paper"],
                   help="checkpoint sets to compare; '+' joins checkpoints whose posteriors are averaged")
    g = sub.add_parser("guitarset")
    g.add_argument("--guitarset", default=str(ROOT / "data" / "raw" / "guitarset"))
    g.add_argument("--sets", nargs="+", default=["kroma", "fl", "gaps_paper", "fl+gaps_paper"])
    g.add_argument("--thr", type=float, default=0.3)
    args = ap.parse_args()
    return {"build": build, "eval": evaluate, "guitarset": guitarset}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
