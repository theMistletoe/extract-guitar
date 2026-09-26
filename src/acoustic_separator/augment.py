"""Audio augmentations for synthetic mixtures (validation and training).

Every effect is applied per stem, so the ground truth stays exact: mixture == sum(stems).
Mix-bus "mastering" is done by computing one gain curve from the mixture and applying it
to every stem (a linked, side-chained compressor/limiter), which keeps that identity.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def db2lin(db: float) -> float:
    return float(10 ** (db / 20))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def active_rms(x: np.ndarray, sr: int, win_s: float = 0.4) -> float:
    """RMS over the louder half of windows (ignores silence when setting levels)."""
    n = max(1, int(win_s * sr))
    m = x.mean(0) if x.ndim == 2 else x
    frames = m[: len(m) // n * n].reshape(-1, n)
    if len(frames) == 0:
        return rms(x)
    e = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    top = np.sort(e)[len(e) // 2:]
    return float(np.sqrt(np.mean(top ** 2)))


# --- filters ---------------------------------------------------------------------------
def _biquad(kind: str, f0: float, sr: int, gain_db: float = 0.0, q: float = 0.707):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
    elif kind == "highshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(kind)
    return np.array(b) / a[0], np.array(a) / a[0]


def random_eq(x: np.ndarray, sr: int, rng: np.random.Generator, strength_db: float = 6.0) -> np.ndarray:
    y = x
    b, a = _biquad("lowshelf", rng.uniform(80, 300), sr, rng.uniform(-strength_db, strength_db))
    y = signal.lfilter(b, a, y, axis=-1)
    b, a = _biquad("peak", float(np.exp(rng.uniform(np.log(300), np.log(5000)))), sr,
                   rng.uniform(-strength_db, strength_db), rng.uniform(0.5, 2.0))
    y = signal.lfilter(b, a, y, axis=-1)
    b, a = _biquad("highshelf", rng.uniform(3000, 10000), sr, rng.uniform(-strength_db, strength_db))
    y = signal.lfilter(b, a, y, axis=-1)
    if rng.random() < 0.5:
        b, a = _biquad("highpass", rng.uniform(40, 120), sr)
        y = signal.lfilter(b, a, y, axis=-1)
    return y.astype(np.float32)


# --- dynamics --------------------------------------------------------------------------
def _envelope(x: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Peak envelope follower on the channel max (vectorised via decimation)."""
    level = np.abs(x).max(0) if x.ndim == 2 else np.abs(x)
    hop = 32
    frames = level[: len(level) // hop * hop].reshape(-1, hop).max(1)
    if len(level) % hop:
        frames = np.append(frames, level[len(level) // hop * hop:].max())
    fs = sr / hop
    ga = np.exp(-1 / (attack_ms / 1000 * fs))
    gr = np.exp(-1 / (release_ms / 1000 * fs))
    env = np.empty_like(frames)
    e = 0.0
    for i, v in enumerate(frames):
        g = ga if v > e else gr
        e = g * e + (1 - g) * v
        env[i] = e
    return np.repeat(env, hop)[: len(level)]


def compressor_gain(x: np.ndarray, sr: int, threshold_db: float, ratio: float,
                    attack_ms: float = 10.0, release_ms: float = 120.0,
                    makeup_db: float = 0.0) -> np.ndarray:
    env_db = 20 * np.log10(_envelope(x, sr, attack_ms, release_ms) + 1e-9)
    over = np.maximum(env_db - threshold_db, 0)
    gain_db = -over * (1 - 1 / ratio) + makeup_db
    return (10 ** (gain_db / 20)).astype(np.float32)


def compress(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    peak_db = 20 * np.log10(np.abs(x).max() + 1e-9)
    g = compressor_gain(x, sr, peak_db - rng.uniform(6, 18), rng.uniform(2, 6),
                        rng.uniform(3, 30), rng.uniform(60, 300))
    return (x * g).astype(np.float32)


def saturate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    drive = rng.uniform(1.5, 4.0)
    peak = np.abs(x).max() + 1e-9
    y = np.tanh(drive * x / peak) / np.tanh(drive) * peak
    return y.astype(np.float32)


# --- space -----------------------------------------------------------------------------
def synthetic_rir(sr: int, rt60: float, rng: np.random.Generator, predelay_ms: float = 10.0,
                  stereo_decorrelate: bool = True) -> np.ndarray:
    n = int(sr * min(rt60 * 1.2, 3.0))
    t = np.arange(n) / sr
    decay = np.exp(-6.9 * t / rt60)
    chans = []
    for _ in range(2):
        noise = rng.standard_normal(n) if stereo_decorrelate or not chans else chans[0] / decay
        chans.append(noise * decay)
    rir = np.stack(chans)
    # darker tail: lowpass the late part
    b, a = _biquad("lowpass", rng.uniform(3000, 9000), sr)
    rir = signal.lfilter(b, a, rir, axis=-1)
    pd = int(predelay_ms / 1000 * sr)
    rir = np.pad(rir, ((0, 0), (pd, 0)))
    rir /= np.sqrt(np.sum(rir ** 2, axis=1, keepdims=True)) + 1e-9
    return rir.astype(np.float32)


def reverb(x: np.ndarray, sr: int, rng: np.random.Generator, wet: float | None = None,
           rt60: float | None = None) -> np.ndarray:
    rt60 = rt60 or rng.uniform(0.3, 2.2)
    wet = wet if wet is not None else rng.uniform(0.05, 0.35)
    rir = synthetic_rir(sr, rt60, rng, predelay_ms=rng.uniform(0, 30))
    mono = x.mean(0)
    tail = np.stack([signal.fftconvolve(mono, rir[c])[: x.shape[-1]] for c in range(2)])
    tail *= rms(x) / (rms(tail) + 1e-9)
    return ((1 - wet) * x + wet * tail).astype(np.float32)


def pan(x: np.ndarray, position: float, width: float = 1.0) -> np.ndarray:
    """position in [-1, 1] (constant power), width in [0, 1] (0 = mono)."""
    mid = x.mean(0)
    side = (x[0] - x[1]) / 2 * width
    theta = (position + 1) * np.pi / 4
    left = mid * np.cos(theta) * np.sqrt(2) + side
    right = mid * np.sin(theta) * np.sqrt(2) - side
    return np.stack([left, right]).astype(np.float32)


def haas_delay(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    d = int(rng.uniform(0.002, 0.025) * sr)
    ch = int(rng.integers(0, 2))
    y = x.copy()
    y[ch] = np.concatenate([np.zeros(d, dtype=np.float32), x[ch, :-d]])
    return y


def echo(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    d = int(rng.uniform(0.12, 0.45) * sr)
    fb = rng.uniform(0.15, 0.4)
    y = x.copy()
    tap = x
    for _ in range(3):
        tap = np.pad(tap, ((0, 0), (d, 0)))[:, : x.shape[-1]] * fb
        y = y + tap
    return y.astype(np.float32)


def bandlimit(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a lower-sample-rate source (e.g. 22.05/32 kHz material)."""
    from .audio import resample

    low = int(rng.choice([16000, 22050, 32000]))
    return resample(resample(x, sr, low), low, sr)[:, : x.shape[-1]]


def mastering(stems: dict[str, np.ndarray], sr: int, rng: np.random.Generator,
              target_peak_db: float = -0.3) -> dict[str, np.ndarray]:
    """Linked bus compression + limiting; the same gain curve is applied to every stem."""
    mix = sum(stems.values())
    peak_db = 20 * np.log10(np.abs(mix).max() + 1e-9)
    g = compressor_gain(mix, sr, peak_db - rng.uniform(3, 10), rng.uniform(1.5, 4),
                        rng.uniform(5, 30), rng.uniform(80, 400))
    mix2 = mix * g
    # fast look-ahead-free limiter (short attack) toward the target peak
    peak2 = 20 * np.log10(np.abs(mix2).max() + 1e-9)
    g2 = compressor_gain(mix2, sr, target_peak_db - 1.0, 20.0, 0.5, 60.0)
    gain = g * g2
    mix3 = mix * gain
    norm = db2lin(target_peak_db) / (np.abs(mix3).max() + 1e-9)
    del peak2
    return {k: (v * gain * norm).astype(np.float32) for k, v in stems.items()}
