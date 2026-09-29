#!/usr/bin/env python
"""Create datasets/validation_aac: the validation set with every mixture passed through
an AAC encoder (like the YouTube-sourced target, which is AAC with a 16 kHz low-pass).
Ground-truth stems stay clean, so codec damage counts as error.

    python scripts/make_codec_valset.py --bitrate 128k
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator.audio import ffmpeg_exe, load_audio, save_audio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "validation"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "validation_aac"))
    ap.add_argument("--bitrate", default="128k")
    ap.add_argument("--cutoff", default="16000")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    for d in sorted(p for p in src.iterdir() if (p / "mixture.wav").exists()):
        o = out / d.name
        if o.exists():
            shutil.rmtree(o)
        o.mkdir(parents=True)
        for f in ("meta.json",):
            shutil.copy2(d / f, o / f)
        # ground truth and interferer stems are shared with the clean set (symlinks)
        (o / "acoustic_guitar.wav").symlink_to(Path("../../validation") / d.name / "acoustic_guitar.wav")
        (o / "stems").symlink_to(Path("../../validation") / d.name / "stems")
        with tempfile.TemporaryDirectory() as tmp:
            m4a = Path(tmp) / "mix.m4a"
            wav = Path(tmp) / "mix16.wav"
            mix, sr = load_audio(d / "mixture.wav")
            peak = max(1.0, float(abs(mix).max()))
            save_audio(wav, mix / peak, sr)
            subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(wav), "-c:a", "aac",
                            "-b:a", a.bitrate, "-cutoff", a.cutoff, str(m4a)], check=True)
            dec, _ = load_audio(m4a)
            # AAC adds encoder delay; align by cross-correlation on the first seconds
            import numpy as np

            n = min(sr * 3, mix.shape[1])
            ref = mix[0, :n] / peak
            corr = np.correlate(dec[0, : n + 4096], ref, mode="valid")
            lag = int(np.argmax(corr))
            dec = dec[:, lag:lag + mix.shape[1]] * peak
            if dec.shape[1] < mix.shape[1]:
                dec = np.pad(dec, ((0, 0), (0, mix.shape[1] - dec.shape[1])))
            save_audio(o / "mixture.wav", dec, sr)
        print(d.name, "lag", lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
