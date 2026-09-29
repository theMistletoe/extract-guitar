"""Audio I/O: decode any input to float32 stereo PCM once, write float32 WAV."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

NATIVE_EXTS = {".wav", ".flac", ".aiff", ".aif", ".ogg"}


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _decode_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    probe = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True
    ).stderr
    m = re.search(r"Audio:.*?(\d+) Hz", probe)
    if not m:
        raise RuntimeError(f"no audio stream found in {path}")
    sr = int(m.group(1))
    # Decode at the native sample rate straight to float32 so the only lossy step is the
    # original codec; no intermediate re-encoding.
    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    return audio, sr


def load_audio(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio as float32 array (channels=2, samples). Mono is duplicated to stereo.

    If ``sr`` is given the signal is resampled with soxr (VHQ).
    """
    path = Path(path)
    if path.suffix.lower() in NATIVE_EXTS:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = data.T
    else:
        audio, file_sr = _decode_ffmpeg(path)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if sr is not None and sr != file_sr:
        audio = resample(audio, file_sr, sr)
        file_sr = sr
    return np.ascontiguousarray(audio, dtype=np.float32), file_sr


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    import soxr

    return soxr.resample(audio.T, sr_in, sr_out, quality="VHQ").T.astype(np.float32)


def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    """Write (channels, samples) float32 WAV. Float32 keeps values > 1.0 intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32).T, sr, subtype="FLOAT")
    return path


def track_id_for(source: str) -> str:
    """Stable, filesystem-safe id for a file path or URL."""
    yt = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", source)
    if yt:
        return f"yt-{yt.group(1)}"
    p = Path(source)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem)[:48] or "track"
    digest = hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:8] if p.exists() else "x"
    return f"{stem}-{digest}"


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rms_db(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + eps))
