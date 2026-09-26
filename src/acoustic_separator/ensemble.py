"""Ensembling and spectral post-processing for stem estimates.

All functions take/return (C, T) float32 arrays at a common sample rate.
"""
from __future__ import annotations

import numpy as np
import torch

N_FFT = 4096
HOP = 1024


def _stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return torch.stft(t, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
                      return_complex=True, center=True)


def _istft(X: torch.Tensor, length: int, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    y = torch.istft(X, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
                    center=True, length=length)
    return y.numpy().astype(np.float32)


def _weights(n: int, weights) -> np.ndarray:
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    return w / w.sum()


def wave_mean(ests: list[np.ndarray], weights=None, **_) -> np.ndarray:
    w = _weights(len(ests), weights)
    return np.sum([wi * e for wi, e in zip(w, ests)], axis=0).astype(np.float32)


def wave_median(ests: list[np.ndarray], **_) -> np.ndarray:
    return np.median(np.stack(ests), axis=0).astype(np.float32)


def _combine_mag(ests, mix, reducer, weights=None, phase="mix", n_fft=N_FFT, hop=HOP):
    T = ests[0].shape[-1]
    specs = [_stft(e, n_fft, hop) for e in ests]
    mags = torch.stack([s.abs() for s in specs])
    if reducer == "mean":
        w = torch.tensor(_weights(len(ests), weights), dtype=torch.float32).view(-1, 1, 1, 1)
        mag = (mags * w).sum(0)
    elif reducer == "median":
        mag = mags.median(0).values
    elif reducer == "min":
        mag = mags.min(0).values
    elif reducer == "max":
        mag = mags.max(0).values
    else:
        raise ValueError(reducer)
    if phase == "mix" and mix is not None:
        ph = torch.angle(_stft(mix, n_fft, hop))
    else:  # phase of the weighted complex mean of the estimates
        w = _weights(len(ests), weights)
        ph = torch.angle(sum(float(wi) * s for wi, s in zip(w, specs)))
    return _istft(torch.polar(mag, ph), T, n_fft, hop)


def mag_mean(ests, mix=None, weights=None, phase="est", **_):
    return _combine_mag(ests, mix, "mean", weights, phase)


def mag_median(ests, mix=None, phase="est", **_):
    return _combine_mag(ests, mix, "median", None, phase)


def mag_min(ests, mix=None, phase="est", **_):
    """Keeps only energy every model agrees on (less leakage, more loss)."""
    return _combine_mag(ests, mix, "min", None, phase)


def mag_max(ests, mix=None, phase="est", **_):
    return _combine_mag(ests, mix, "max", None, phase)


def mask_mean(ests, mix, weights=None, power: float = 1.0, **_):
    """Average soft ratio masks |Y_i|/|X| (clipped to [0,1]) and apply to the mixture STFT."""
    T = mix.shape[-1]
    X = _stft(mix)
    ax = X.abs() + 1e-8
    w = _weights(len(ests), weights)
    m = sum(float(wi) * (_stft(e).abs() / ax).clamp(0, 1) ** power for wi, e in zip(w, ests))
    return _istft(X * m, T)


def band_weighted(ests, mix=None, band_weights=None, crossovers=(250.0, 2000.0, 8000.0),
                  sr: int = 44100, normalize: bool = True, **_):
    """Frequency-dependent weighting: band_weights[i][b] = weight of estimate i in band b.
    With normalize=False the weights are used as given (e.g. least-squares fitted)."""
    T = ests[0].shape[-1]
    specs = [_stft(e) for e in ests]
    freqs = torch.fft.rfftfreq(N_FFT, 1 / sr)
    edges = [0.0, *crossovers, sr / 2 + 1]
    bw = np.asarray(band_weights, dtype=np.float64)  # (n_est, n_bands)
    if normalize:
        bw = bw / bw.sum(0, keepdims=True)
    W = torch.zeros(len(ests), len(freqs))
    for b in range(len(edges) - 1):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        for i in range(len(ests)):
            W[i, sel] = float(bw[i, b])
    out = sum(W[i].view(1, -1, 1) * specs[i] for i in range(len(ests)))
    return _istft(out, T)


def wiener_refine(target: np.ndarray, mix: np.ndarray, iterations: int = 1, power: float = 2.0,
                  **_) -> np.ndarray:
    """Soft-mask refinement with target / residual PSDs from the current estimate.

    Enforces the mixture as the phase and energy source: target + residual == mix.
    """
    T = mix.shape[-1]
    X = _stft(mix)
    est = target
    for _ in range(max(1, iterations)):
        S = _stft(est).abs() ** power
        R = _stft(mix - est).abs() ** power
        mask = S / (S + R + 1e-10)
        est = _istft(X * mask, T)
    return est


def spectral_gate(target: np.ndarray, mix: np.ndarray, floor_db: float = -60.0,
                  ratio: float = 0.0, **_) -> np.ndarray:
    """Attenuate TF bins where the estimate is far below the mixture (likely residue)."""
    T = target.shape[-1]
    Y = _stft(target)
    X = _stft(mix)
    rel = 20 * torch.log10(Y.abs() / (X.abs() + 1e-8) + 1e-8)
    g = torch.where(rel < floor_db, torch.tensor(ratio, dtype=torch.float32), torch.tensor(1.0))
    return _istft(Y * g, T)


METHODS = {
    "wave_mean": wave_mean,
    "wave_median": wave_median,
    "mag_mean": mag_mean,
    "mag_median": mag_median,
    "mag_min": mag_min,
    "mag_max": mag_max,
    "mask_mean": mask_mean,
    "band_weighted": band_weighted,
}

POSTPROCESS = {
    "wiener": wiener_refine,
    "gate": spectral_gate,
}
