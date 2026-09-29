#!/usr/bin/env python
"""Guitar tablature from an extracted acoustic-guitar stem.

    python scripts/transcribe_tab.py                       # target song -> outputs/target/tab/
    python scripts/transcribe_tab.py --stem x.wav --out outputs/x/tab --title "..."

Steps: tuning estimate -> CRNN note transcription (tuning-compensated, 16 kHz) -> beat grid from
the guitar's own onsets (thumb bass weighted) refined by least squares -> bar phase -> swing-aware
16th quantisation -> string/fret Viterbi -> chord symbols -> ASCII / MusicXML / GP5 / MIDI / CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator.audio import file_sha256, load_audio  # noqa: E402
from acoustic_separator.tab import amt, chords, export, fretboard, rhythm  # noqa: E402

TARGET = ROOT / "outputs" / "target"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", default=str(TARGET / "best" / "acoustic_guitar.wav"))
    p.add_argument("--out", default=str(TARGET / "tab"))
    p.add_argument("--name", default="frevo_guitar_tab", help="output file stem")
    p.add_argument("--title", default="Frevo!")
    p.add_argument("--artist", default="ko-ko-ya")
    p.add_argument("--composer", default="Shigeharu Sasago")
    p.add_argument("--beats-per-bar", type=int, default=2)
    p.add_argument("--models", nargs="+", default=["fl", "gaps_paper"],
                   help="guitar checkpoints whose posteriors are averaged (see amt.CHECKPOINTS)")
    p.add_argument("--onset-thr", type=float, default=0.3)
    p.add_argument("--tta", type=float, nargs="*", default=[0.0, 2.5],
                   help="window shifts (s) averaged over for the CRNN posteriors")
    p.add_argument("--tuning-cents", type=float, default=None, help="override the tuning estimate")
    p.add_argument("--no-cache", action="store_true")
    return p


def model_posteriors(model: str, y16: np.ndarray, out: Path, key: dict, tta, no_cache: bool) -> dict:
    """Posteriors of one checkpoint, cached per model in <out>/posteriors_<model>.npz."""
    cache = out / f"posteriors_{model}.npz"
    if cache.exists() and not no_cache:
        d = np.load(cache, allow_pickle=False)
        if json.loads(str(d["key"])) == key:
            return {k: d[k] for k in ("onset", "offset", "frame", "velocity")}
    post = amt.posteriors(amt.load_model(model), y16, shifts=tuple(tta))
    np.savez_compressed(cache, key=json.dumps(key), **post)
    return post


def render_pdf(ly: Path) -> Path | None:
    """Engrave the LilyPond file if LilyPond is installed (binary or ``pip install lilypond``)."""
    import shutil
    import subprocess

    exe = shutil.which("lilypond")
    if exe is None:
        try:
            import lilypond

            exe = str(lilypond.executable())
        except ImportError:
            print("[pdf] LilyPond not installed; skipping PDF", flush=True)
            return None
    r = subprocess.run([exe, "-s", "-o", str(ly.with_suffix("")), str(ly)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[pdf] LilyPond failed:\n{r.stderr[-2000:]}", flush=True)
        return None
    return ly.with_suffix(".pdf")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    audio, sr = load_audio(args.stem)
    mono = audio.mean(0)

    # 1. tuning + note transcription -------------------------------------------------
    cents = args.tuning_cents if args.tuning_cents is not None else amt.estimate_tuning(mono, sr)
    y16, f = amt.to_model_input(audio, sr, cents)
    stem_sha = file_sha256(args.stem)
    posts = [model_posteriors(m, y16, out, dict(stem_sha=stem_sha, cents=round(cents, 2), tta=list(args.tta)),
                              args.tta, args.no_cache) for m in args.models]
    post = {k: np.mean([p[k] for p in posts], 0) for k in posts[0]}  # ensemble = mean posterior
    notes = amt.decode(post, onset_thr=args.onset_thr, frame_thr=0.3)
    notes[:, :2] *= f  # model time -> original time
    lo, hi = fretboard.STANDARD[0], fretboard.STANDARD[-1] + fretboard.Weights().max_fret
    notes = notes[(notes[:, 2] >= lo) & (notes[:, 2] <= hi)]
    print(f"[notes] tuning {cents:+.1f} cents, {len(notes)} notes", flush=True)

    # 2. beat grid, swing, bar phase -------------------------------------------------
    env = rhythm.onset_envelope(post["onset"], f)
    beats = rhythm.track_beats(env)
    on = np.sort(notes[:, 0])
    grp = on[np.r_[True, np.diff(on) > 0.04]]
    bass = np.array([((np.abs(notes[:, 0] - t) < 0.04) & (notes[:, 2] < 52)).any() for t in grp])
    wts = np.where(bass, 3.0, 1.0)
    beats = rhythm.extend_beats(beats, grp.min() - 1.0, notes[:, 1].max() + 1.0)
    centres = np.arange(4) / 4
    for _ in range(3):
        beats = rhythm.refine_beats(grp, beats, wts, centres)
        centres = rhythm.swing_centres(grp, beats)
    bounds = rhythm.slot_bounds(centres)
    phase = rhythm.bar_phase(notes, beats, args.beats_per_bar)
    k_first = int(rhythm.quantize(notes[:1, 0], beats, bounds)[0] // 4)
    k_first -= (k_first - phase) % args.beats_per_bar
    beats = beats[k_first:]
    bpm = 60.0 / np.median(np.diff(beats))
    print(f"[rhythm] {bpm:.1f} BPM, swing 16ths at {np.round(centres, 3)}, bar 1 at {beats[0]:.3f}s",
          flush=True)

    # 3. quantise, merge duplicates ---------------------------------------------------
    q = rhythm.quantize(notes[:, 0], beats, bounds)
    kf, ph = rhythm.phases(notes[:, 1], beats)
    q_end = 4 * (kf + ph)
    best: dict[tuple[int, int], int] = {}
    for i, (qq, p) in enumerate(zip(q, notes[:, 2].astype(int))):
        j = best.get((qq, p))
        if j is None or notes[i, 3] > notes[j, 3]:
            best[(qq, p)] = i
    keep = np.array(sorted(best.values()))
    notes, q, q_end = notes[keep], q[keep], q_end[keep]

    # 4. strings and frets ------------------------------------------------------------
    groups = fretboard.group_notes(notes[:, 0], notes[:, 1], notes[:, 2].astype(int), keys=q,
                                   conf=notes[:, 3])
    tab_notes = []
    for g, (sf, _) in zip(groups, fretboard.assign(groups)):
        for i, x in zip(g.idx, sf):
            if x is not None:
                tab_notes.append(export.TabNote(float(notes[i, 0]), float(notes[i, 1]), int(notes[i, 2]),
                                                x[0], x[1], float(notes[i, 3]), int(q[i]), float(q_end[i])))
    dropped = len(notes) - len(tab_notes)

    # 5. chords and key -------------------------------------------------------------
    pit = np.array([n.pitch for n in tab_notes])
    qn = np.array([n.q for n in tab_notes])
    n_units = int(qn.max()) // 2 + 2
    chroma, bass_pc = chords.evidence(qn, np.array([n.q_end for n in tab_notes]), pit,
                                      np.array([n.conf for n in tab_notes]), n_units)
    per_unit = chords.label(chroma, bass_pc, 2 * args.beats_per_bar)
    first_q: dict[int, int] = {}
    for qq in sorted(qn):
        first_q.setdefault((int(qq) + 1) // 2, int(qq))
    symbols = {first_q.get(u, 2 * u): name for u, name in chords.changes(per_unit).items()}
    fifths, mode, key_name = export.key_signature(pit, np.array([3.0 if n.pitch < 52 else 1.0 for n in tab_notes]))

    t_first, t_last = min(n.onset for n in tab_notes), max(n.offset for n in tab_notes)
    header = [
        f"{args.title} - {args.artist} (comp. {args.composer})",
        "Acoustic guitar - automatic transcription of outputs/target/best/acoustic_guitar.wav",
        "",
        f"Tuning: standard E A D G B E (recording at A4 = {440 * 2 ** (cents / 1200):.1f} Hz, {cents:+.0f} cents)",
        f"Time: {args.beats_per_bar}/4, quarter = {bpm:.0f} (Brazilian 16th feel: 2nd 16th late, 4th early)",
        f"Key: {key_name}.  One column = one 16th note; bar numbers with the time in the recording.",
        f"Guitar enters at {t_first:.2f}s (bar 1) and ends at {t_last:.2f}s; the intro before bar 1 is guitar tacet.",
        "Chord symbols are derived from the transcribed notes.  Lines: e B G D A E (string 1 at top).",
    ]
    tab = export.Tab(tab_notes, beats, args.beats_per_bar, symbols, bpm, args.title,
                     subtitle=f"{args.artist} - guitar (automatic transcription)", composer=args.composer,
                     key_fifths=fifths, key_mode=mode, header=header)

    # 6. write -------------------------------------------------------------------------
    stem = out / args.name
    (stem.with_suffix(".txt")).write_text(export.ascii_tab(tab))
    (stem.with_suffix(".musicxml")).write_text(export.musicxml(tab))
    export.guitar_pro(tab, stem.with_suffix(".gp5"), artist=args.artist)
    export.midi(tab_notes, stem.with_suffix(".mid"))
    export.notes_csv(tab, out / f"{args.name}_notes.csv")
    (stem.with_suffix(".ly")).write_text(export.lilypond(tab))
    pdf = render_pdf(stem.with_suffix(".ly"))
    summary = {
        "stem": str(Path(args.stem).resolve().relative_to(ROOT)) if Path(args.stem).resolve().is_relative_to(ROOT) else args.stem,
        "stem_sha256": stem_sha,
        "models": {"repo": amt.CHECKPOINT["repo"], "revision": amt.CHECKPOINT["revision"],
                   "checkpoints": {m: amt.CHECKPOINTS[m] for m in args.models}},
        "tuning_cents": round(cents, 2),
        "tta_shifts_s": args.tta,
        "onset_threshold": args.onset_thr,
        "notes": len(tab_notes),
        "notes_unplayable_dropped": dropped,
        "bars": tab.n_bars,
        "bpm": round(bpm, 2),
        "swing_centres": [round(float(c), 3) for c in centres],
        "bar1_time_s": round(float(beats[0]), 3),
        "beat_times_s": [round(float(b), 4) for b in beats],
        "key": key_name,
        "chord_changes": len(symbols),
        "chord_symbols": {str(k): v for k, v in sorted(symbols.items())},  # 16th position -> symbol
        "pdf": pdf is not None,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    (out / f"{args.name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {len(tab_notes)} notes in {tab.n_bars} bars -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
