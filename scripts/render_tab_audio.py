#!/usr/bin/env python
"""Render the tab as guitar audio: a sampled guitar plays exactly what the tab says.

    python scripts/render_tab_audio.py        # -> outputs/target/tab/audio/

Writes
  frevo_tab_guitar.mp3               the tab at the recording's timing and pitch (A4 ~ 442 Hz): it
                                     lines up with the original, and its dynamics follow the recording
  frevo_tab_guitar_practice_110.mp3  the tab on a steady grid at 110 BPM (the recording is ~156) with
                                     the recording's 16th-note feel, standard pitch (A4 = 440 Hz), two
                                     bars of count-in and a woodblock click on every beat

Sound: FluidSynth (command-line program) with a General MIDI SoundFont, preset 24 "Acoustic Guitar
(nylon)"; by default FluidR3_GM (MIT licence; Debian/Ubuntu package fluid-soundfont-gm).  Each
string gets its own MIDI channel, so plucking a string again stops the note it was sounding, as on
a guitar, and at the end of each note (as transcribed) its string is damped within 40 ms, so the
short, cut-off chords of the comping stay short.  Each note's loudness is measured in the stem:
constant-Q magnitude at its pitch just after its onset, with the average trend over the register
removed.  A gentle EQ (1/3-octave smoothed, within +-6 dB, 80 Hz - 6 kHz) moves the rendering's
long-term spectrum toward the stem's, and the performance version is shifted by the measured onset
lag so that it lines up with the stem.

Needs the fluidsynth program and a GM SoundFont (apt install fluidsynth fluid-soundfont-gm; macOS:
brew install fluid-synth plus any GM SoundFont via --sf2), and ffmpeg (imageio-ffmpeg) for MP3.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator.audio import ffmpeg_exe, load_audio, resample, save_audio  # noqa: E402

TAB = ROOT / "outputs" / "target" / "tab"
SF2_PATHS = ["/usr/share/sounds/sf2/FluidR3_GM.sf2", "/usr/share/soundfonts/FluidR3_GM.sf2",
             "/usr/local/share/soundfonts/FluidR3_GM.sf2", "/opt/homebrew/share/soundfonts/FluidR3_GM.sf2"]
SR = 44100
TPS = 1920  # MIDI ticks per second (120 BPM, 960 ticks per beat)
NYLON, STEEL = 24, 25
CLICK_ACCENT, CLICK = 76, 77  # GM drum kit: hi / low wood block
VOLUME = 110  # channel volume (CC7)
DAMP_S = 0.04  # string damping fade, s


def read_tab(tabdir: Path, name: str) -> tuple[list[dict], dict]:
    meta = json.loads((tabdir / f"{name}.json").read_text())
    with open(tabdir / f"{name}_notes.csv") as f:
        notes = list(csv.DictReader(f))
    spb = 4 * meta.get("beats_per_bar", 2)
    for n in notes:
        n["q"] = (int(n["bar"]) - 1) * spb + (int(n["beat"]) - 1) * 4 + int(n["sixteenth"]) - 1
        n["string"] = 6 - int(n["string"])  # CSV: 1 = high e -> 0 = low E
        for k in ("midi", "fret"):
            n[k] = int(n[k])
        for k in ("onset_s", "offset_s"):
            n[k] = float(n[k])
    return notes, meta


def time_to_q(t: np.ndarray, beats: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Recording time -> fractional 16th index on the tab's grid (beat times + swung 16ths)."""
    ibi = np.median(np.diff(beats[-8:]))
    b = np.r_[beats, beats[-1] + ibi * np.arange(1, 64)]
    k = np.clip(np.searchsorted(b, t, side="right") - 1, 0, len(b) - 2)
    frac = (np.asarray(t, float) - b[k]) / (b[k + 1] - b[k])
    c = np.asarray(centres, float)  # sub-beat positions of the four 16ths (as in the tab's grid)
    return 4 * k + np.interp(frac, np.r_[c, 1 + c[0]], np.arange(5.0))


def grid_time(qf: np.ndarray, bpm: float, centres: np.ndarray, start: float) -> np.ndarray:
    """Fractional 16th index -> time on a steady grid with the same 16th-note feel."""
    qf = np.asarray(qf, float)
    k = np.floor(qf / 4)
    c = np.asarray(centres, float)
    sub = np.interp(qf - 4 * k, np.arange(5.0), np.r_[c, 1 + c[0]])
    return start + (k + sub) * 60.0 / bpm


def velocities(stem: np.ndarray, sr: int, onset: np.ndarray, midi: np.ndarray, cents: float) -> np.ndarray:
    """MIDI velocity per note from the stem: constant-Q level at the note's pitch in the 60 ms after
    its onset, minus the register trend (a quadratic in pitch), mapped onto FluidSynth's velocity
    curve (40 dB per decade of velocity) with a 0.6 compression."""
    import librosa

    ssr, hop, lo = 22050, 256, 36
    y = librosa.resample(stem, orig_sr=sr, target_sr=ssr)
    fmin = librosa.midi_to_hz(lo) * 2 ** (cents / 1200)
    mag = np.abs(librosa.cqt(y, sr=ssr, hop_length=hop, fmin=fmin, n_bins=60, bins_per_octave=12))
    db = 20 * np.log10(mag + 1e-6 * mag.max())
    fr = np.round(onset * ssr / hop).astype(int)
    width = int(np.ceil(0.06 * ssr / hop)) + 1
    level = np.array([db[m - lo, max(f, 0):max(f, 0) + width].max() for f, m in zip(fr, midi)])
    trend = np.polyval(np.polyfit(midi, level, 2), midi)
    res = np.clip(level - trend, -12.0, 12.0)
    return np.clip(np.round(80 * 10 ** (0.6 * res / 40)), 35, 118).astype(int)


def note_events(notes: list[dict], on: np.ndarray, off: np.ndarray, vel: np.ndarray,
                damp_s: float = DAMP_S) -> list[tuple]:
    """MIDI events (time, order, kind, channel, data1, data2), one channel per string (0 = low E).

    At a note's end its string is damped: the channel volume fades to 0 over ``damp_s`` and the
    voice is cut (All Sound Off), so a SoundFont's long release does not ring on where the player
    stopped the string; a string that is plucked again is damped just before.  The volume is
    restored before each pluck.  ``order`` sorts events at the same instant (damping, volume, pluck).
    """
    by_string: dict[int, list[int]] = {}
    for i in np.argsort(on, kind="stable"):
        by_string.setdefault(notes[i]["string"], []).append(int(i))
    ev = []
    for s, idx in by_string.items():
        for j, i in enumerate(idx):
            nxt = on[idx[j + 1]] if j + 1 < len(idx) else np.inf
            key, end = notes[i]["midi"], max(off[i], on[i] + 0.06)
            f1 = min(end + damp_s, nxt - 0.001)
            f0 = min(f1, max(on[i] + 0.01, min(end, f1 - 0.005)))
            ev += [(on[i], 1, "cc", s, 7, VOLUME), (on[i], 2, "note_on", s, key, int(vel[i]))]
            steps = int(np.clip(round((f1 - f0) / 0.005), 1, 8))
            ev += [(f0 + (f1 - f0) * k / steps, 0, "cc", s, 7, int(round(VOLUME * (1 - k / steps))))
                   for k in range(1, steps + 1)]
            ev += [(f1, 0, "note_off", s, key, 0), (f1, 0, "cc", s, 120, 0)]
    return sorted(ev, key=lambda e: (e[0], e[1]))


def onset_lag(x: np.ndarray, ref: np.ndarray, sr: int, max_s: float = 0.06) -> float:
    """Seconds by which ``x`` lags ``ref``: peak of the cross-correlation of their onset envelopes."""
    import librosa

    hop = 32
    a, b = (librosa.onset.onset_strength(y=librosa.resample(v, orig_sr=sr, target_sr=22050), sr=22050, hop_length=hop)
            for v in (ref, x))
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    m = int(max_s * 22050 / hop)
    cc = [np.dot(a[max(0, -k):n - max(0, k)], b[max(0, k):n - max(0, -k)]) for k in range(-m, m + 1)]
    return (int(np.argmax(cc)) - m) * hop / 22050


def ltas_db(x: np.ndarray, sr: int, n_fft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Long-term average power spectrum (dB, total power normalised) over the frames with sound."""
    import librosa

    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=n_fft // 4)) ** 2
    e = S.sum(0)
    m = S[:, e > 1e-3 * e.max()].mean(1)
    return librosa.fft_frequencies(sr=sr, n_fft=n_fft), 10 * np.log10(m / m.sum() + 1e-20)


def match_eq(x: np.ndarray, ref: np.ndarray, sr: int, max_db: float = 6.0, lo: float = 80.0,
             hi: float = 6000.0) -> tuple[np.ndarray, np.ndarray]:
    """(band centres Hz, gain dB): 1/3-octave-smoothed difference ref - x of the long-term spectra,
    capped at +-max_db and 0 dB outside lo..hi."""
    f, dx = ltas_db(x, sr)
    _, dr = ltas_db(ref, sr)
    fc = lo * 2 ** (np.arange(int(np.log2(hi / lo) * 6) + 1) / 6)
    band = lambda d, c: 10 * np.log10(np.mean(10 ** (d[(f >= c * 2 ** -(1 / 6)) & (f < c * 2 ** (1 / 6))] / 10)))  # noqa: E731
    g = np.array([band(dr, c) - band(dx, c) for c in fc])
    return fc, np.clip(g - np.median(g), -max_db, max_db)


def apply_eq(x: np.ndarray, sr: int, fc: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Zero-phase EQ of (channels, samples) with gains g (dB) at fc, log-interpolated, tapering to
    0 dB within a third of an octave beyond the first and last band."""
    n = x.shape[-1]
    f = np.fft.rfftfreq(n, 1 / sr)
    edge = 2 ** (1 / 3)
    gain = np.interp(np.log2(np.maximum(f, 1.0)), np.log2(np.r_[fc[0] / edge, fc, fc[-1] * edge]), np.r_[0.0, g, 0.0])
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * 10 ** (gain / 20), n=n, axis=-1).astype(np.float32)


def write_midi(path: Path, ev: list[tuple], cents: float, clicks: list[tuple] = (), program: int = NYLON,
               reverb: int = 48, tail_s: float = 3.0) -> Path:
    import mido

    mid = mido.MidiFile(ticks_per_beat=960)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    bend = int(round(8192 * cents / 200))  # default bend range: +-2 semitones
    for ch in range(6):
        tr += [mido.Message("program_change", channel=ch, program=program, time=0),
               mido.Message("control_change", channel=ch, control=7, value=VOLUME, time=0),
               mido.Message("control_change", channel=ch, control=10, value=54 + 4 * ch, time=0),  # low E left
               mido.Message("control_change", channel=ch, control=91, value=reverb, time=0),
               mido.Message("control_change", channel=ch, control=93, value=0, time=0),
               mido.Message("pitchwheel", channel=ch, pitch=bend, time=0)]
    msgs = list(ev) + [(t, 2, "note_on", 9, k, v) for t, k, v in clicks] \
        + [(t + 0.05, 0, "note_off", 9, k, 0) for t, k, _ in clicks]
    last = 0
    for t, _, kind, ch, a, b in sorted(msgs, key=lambda m: (m[0], m[1])):
        tick = int(round(t * TPS))
        dt = max(tick - last, 0)
        tr.append(mido.Message("control_change", channel=ch, control=a, value=b, time=dt) if kind == "cc"
                  else mido.Message(kind, channel=ch, note=a, velocity=b, time=dt))
        last = max(tick, last)
    end = max((m[0] for m in msgs), default=0.0) + tail_s  # let the reverb ring out
    tr.append(mido.MetaMessage("end_of_track", time=int(round(end * TPS)) - last))
    mid.save(path)
    return path


def fluidsynth(midi_path: Path, wav_path: Path, sf2: Path, gain: float = 0.5) -> Path:
    exe = shutil.which("fluidsynth")
    if exe is None:
        raise SystemExit("fluidsynth not found: apt install fluidsynth (or brew install fluid-synth)")
    subprocess.run([exe, "-ni", "-q", "-g", str(gain), "-r", str(SR), "-O", "float", "-T", "wav",
                    "-o", "synth.reverb.room-size=0.45", "-o", "synth.reverb.damp=0.35",
                    "-o", "synth.reverb.width=0.8", "-o", "synth.reverb.level=0.7", "-o", "synth.chorus.active=0",
                    "-F", str(wav_path), str(sf2), str(midi_path)], check=True, capture_output=True)
    return wav_path


def mp3(path: Path, x: np.ndarray, kbps: int = 192) -> None:
    """(channels, samples) float -> MP3, peak-normalised to -1 dBFS."""
    x = x / (np.abs(x).max() + 1e-9) * 10 ** (-1 / 20)
    with tempfile.TemporaryDirectory() as d:
        wav = save_audio(Path(d) / "x.wav", x.astype(np.float32), SR)
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame",
                        "-b:a", f"{kbps}k", str(path)], check=True)


def render(ev: list[tuple], cents: float, sf2: Path, clicks: list[tuple] = (), program: int = NYLON) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        mid = write_midi(Path(d) / "tab.mid", ev, cents, clicks, program=program)
        x, sr = load_audio(fluidsynth(mid, Path(d) / "tab.wav", sf2))
    assert sr == SR
    return x


def performance(notes: list[dict], stem: np.ndarray, stem_sr: int, cents: float, sf2: Path,
                program: int = NYLON) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """The notes at their recorded times and at the recording's pitch, with the onset lag removed
    and the EQ toward the (mono) stem applied: (audio (2, n) at SR, velocities, lag s, EQ fc, EQ gain)."""
    on = np.array([n["onset_s"] for n in notes])
    off = np.array([n["offset_s"] for n in notes])
    vel = velocities(stem, stem_sr, on, np.array([n["midi"] for n in notes]), cents)
    x = render(note_events(notes, on, off, vel), cents, sf2, program=program)
    ref = stem if stem_sr == SR else resample(stem[None], stem_sr, SR)[0]
    lag = onset_lag(x.mean(0), ref, SR)
    k = int(round(lag * SR))
    x = x[:, k:] if k > 0 else np.pad(x, ((0, 0), (-k, 0)))
    fc, gain = match_eq(x.mean(0), ref, SR)
    return apply_eq(x, SR, fc, gain)[:, : int((len(ref) / SR + 3) * SR)], vel, lag, fc, gain


def find_sf2(arg: str | None) -> Path:
    for p in ([arg] if arg else SF2_PATHS):
        if p and Path(p).exists():
            return Path(p)
    raise SystemExit("no SoundFont found: apt install fluid-soundfont-gm, or pass --sf2 PATH (any GM "
                     "SoundFont, e.g. MuseScore_General.sf2, MIT licence)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tab", default=str(TAB))
    ap.add_argument("--name", default="frevo_guitar_tab")
    ap.add_argument("--out", default=None, help="default: <tab>/audio")
    ap.add_argument("--sf2", default=None)
    ap.add_argument("--program", type=int, default=NYLON, help=f"GM program ({NYLON} nylon, {STEEL} steel)")
    ap.add_argument("--practice-bpm", type=float, default=110.0)
    ap.add_argument("--count-in-bars", type=int, default=2)
    args = ap.parse_args()
    tabdir = Path(args.tab)
    out = Path(args.out) if args.out else tabdir / "audio"
    out.mkdir(parents=True, exist_ok=True)
    sf2 = find_sf2(args.sf2)
    notes, meta = read_tab(tabdir, args.name)
    beats, centres = np.asarray(meta["beat_times_s"]), np.asarray(meta["swing_centres"])
    bpb = meta.get("beats_per_bar", 2)
    off = np.array([n["offset_s"] for n in notes])
    stem, sr = load_audio(ROOT / meta["stem"])

    # 1. the recording's timing and pitch
    perf, vel, lag, fc, gain = performance(notes, stem.mean(0), sr, meta["tuning_cents"], sf2, args.program)
    mp3(out / "frevo_tab_guitar.mp3", perf)

    # 2. practice: steady tempo, standard pitch, count-in and click
    bpm = args.practice_bpm
    start = args.count_in_bars * bpb * 60.0 / bpm
    q = np.array([n["q"] for n in notes], float)
    q_off = np.maximum(time_to_q(off, beats, centres), q + 0.5)
    g_on, g_off = grid_time(q, bpm, centres, start), grid_time(q_off, bpm, centres, start)
    # a chord's strings are never struck at exactly the same instant: 3 ms per string, low to high
    low = {}
    for n in notes:
        low[n["q"]] = min(low.get(n["q"], 6), n["string"])
    g_on = g_on + 0.003 * np.array([n["string"] - low[n["q"]] for n in notes])
    n_beats = int(np.ceil((g_off.max() - start) * bpm / 60)) + 1
    clicks = [(i * 60.0 / bpm, CLICK_ACCENT if i % bpb == 0 else CLICK, 100 if i % bpb == 0 else 75)
              for i in range(args.count_in_bars * bpb + n_beats)]
    prac = render(note_events(notes, g_on, g_off, vel), 0.0, sf2, clicks, program=args.program)
    prac = apply_eq(prac, SR, fc, gain)
    name = f"frevo_tab_guitar_practice_{bpm:g}.mp3"
    mp3(out / name, prac)
    summary = {"soundfont": sf2.name, "program": args.program, "notes": len(notes),
               "velocity_median": int(np.median(vel)), "velocity_p5_p95": [int(np.percentile(vel, 5)), int(np.percentile(vel, 95))],
               "eq_db": {f"{c:.0f}Hz": round(float(v), 1) for c, v in zip(fc, gain)},
               "performance": {"file": "frevo_tab_guitar.mp3", "tuning_cents": meta["tuning_cents"],
                               "onset_lag_removed_ms": round(lag * 1000, 1), "seconds": round(perf.shape[1] / SR, 1)},
               "practice": {"file": name, "bpm": bpm, "recording_bpm": meta.get("bpm"), "tuning_cents": 0.0,
                            "count_in_bars": args.count_in_bars, "seconds": round(prac.shape[1] / SR, 1)}}
    (out / "render.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
