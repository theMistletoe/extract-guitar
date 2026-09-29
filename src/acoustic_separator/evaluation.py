"""Objective metrics for acoustic-guitar extraction.

Ground-truth metrics (validation set, where the true acoustic guitar stem is known):
    sdr          -- MDX/MUSDB "utterance SDR": 10 log10(|s|^2 / |s - s_hat|^2)
    si_sdr       -- scale-invariant SDR (per channel, averaged)
    sdri         -- sdr(est) - sdr(mixture used as the estimate)
    sdr/sir/sar_bss -- BSS-Eval v4 style (512-tap distortion filter, fast_bss_eval),
                    target vs. accompaniment
    target_retention -- share of the target's time-frequency energy present in the
                    estimate: sum min(|S_hat|,|S|)^2 / sum |S|^2  (1.0 = nothing lost)
    leakage_db   -- energy of the estimate that exceeds the target, relative to the
                    estimate: 10 log10(sum max(|S_hat|-|S|,0)^2 / sum |S_hat|^2)
    leak_<class>_db -- excess estimate energy (max(|S_hat|-|S|,0)^2) attributed to each
                    interfering class by oracle weights |R_j|^2 / sum_k |R_k|^2, relative to
                    target energy (floored at -80 dB)
    mrstft       -- multi-resolution STFT error (spectral convergence + log-mag L1)

Reference-free proxies (target song) live in ``proxy.py``.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-10


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    num = np.sum(np.square(ref, dtype=np.float64))
    den = np.sum(np.square(ref - est, dtype=np.float64))
    return float(10 * np.log10((num + EPS) / (den + EPS)))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    vals = []
    for r, e in zip(np.atleast_2d(ref), np.atleast_2d(est)):
        r = r.astype(np.float64) - r.mean()
        e = e.astype(np.float64) - e.mean()
        alpha = np.dot(e, r) / (np.dot(r, r) + EPS)
        t = alpha * r
        vals.append(10 * np.log10((np.sum(t * t) + EPS) / (np.sum((e - t) ** 2) + EPS)))
    return float(np.mean(vals))


def stft_mag(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """x: (C, T) -> |STFT| (C, F, N) using a Hann window."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(np.atleast_2d(x), dtype=np.float32))
    S = torch.stft(t, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
                   return_complex=True, center=True)
    return S.abs().numpy().astype(np.float64)


def mrstft(ref: np.ndarray, est: np.ndarray, sizes=(512, 1024, 2048)) -> float:
    tot = 0.0
    for n in sizes:
        R, E = stft_mag(ref, n, n // 4), stft_mag(est, n, n // 4)
        sc = np.linalg.norm(R - E) / (np.linalg.norm(R) + EPS)
        mag = np.mean(np.abs(np.log(R + 1e-5) - np.log(E + 1e-5)))
        tot += sc + mag
    return float(tot / len(sizes))


def bss_eval(ref: np.ndarray, est: np.ndarray, accomp: np.ndarray,
             est_accomp: np.ndarray, filter_length: int = 512) -> dict:
    """BSS-Eval (v4 style) for the target, with accompaniment as the other source."""
    import fast_bss_eval
    import torch

    # torch backend: the numpy backend of fast_bss_eval 0.1.4 breaks on numpy>=2
    out = {"sdr_bss": [], "sir_bss": [], "sar_bss": []}
    for c in range(ref.shape[0]):
        R = torch.from_numpy(np.stack([ref[c], accomp[c]]).astype(np.float64))
        E = torch.from_numpy(np.stack([est[c], est_accomp[c]]).astype(np.float64))
        if float(torch.sum(E[0] ** 2)) < 1e-12:
            continue
        s, i, a = fast_bss_eval.bss_eval_sources(R, E, filter_length=filter_length,
                                                  compute_permutation=False, clamp_db=100,
                                                  load_diag=1e-6)
        out["sdr_bss"].append(float(s[0])); out["sir_bss"].append(float(i[0]))
        out["sar_bss"].append(float(a[0]))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


def tf_metrics(ref: np.ndarray, est: np.ndarray, interferers: dict[str, np.ndarray] | None = None,
               n_fft: int = 2048, hop: int = 512) -> dict:
    S, E = stft_mag(ref, n_fft, hop), stft_mag(est, n_fft, hop)
    out = {
        "target_retention": float(np.sum(np.minimum(E, S) ** 2) / (np.sum(S ** 2) + EPS)),
        "leakage_db": float(10 * np.log10((np.sum(np.maximum(E - S, 0) ** 2) + EPS) /
                                          (np.sum(E ** 2) + EPS))),
    }
    if interferers:
        # Excess estimate energy (above the true target magnitude) is attributed to the
        # interfering classes in proportion to their oracle energy in each TF bin.
        mags = {k: stft_mag(v, n_fft, hop) ** 2 for k, v in interferers.items()}
        denom = sum(mags.values()) + EPS
        excess = np.maximum(E - S, 0) ** 2
        tgt_energy = np.sum(S ** 2) + EPS
        for k, m in mags.items():
            if np.sum(m) <= 1e-8:
                continue
            attributed = np.sum(excess * m / denom)
            out[f"leak_{k}_db"] = float(max(-80.0, 10 * np.log10(attributed / tgt_energy + EPS)))
    return out


def evaluate_estimate(ref: np.ndarray, est: np.ndarray, mixture: np.ndarray,
                      interferers: dict[str, np.ndarray] | None = None,
                      with_bss: bool = True) -> dict:
    """All ground-truth metrics for one clip. Arrays are (C, T) at the same rate."""
    T = min(ref.shape[-1], est.shape[-1], mixture.shape[-1])
    ref, est, mixture = ref[..., :T], est[..., :T], mixture[..., :T]
    m = {
        "sdr": sdr(ref, est),
        "si_sdr": si_sdr(ref, est),
        "sdr_mix": sdr(ref, mixture),
        "mrstft": mrstft(ref, est),
    }
    m["sdri"] = m["sdr"] - m["sdr_mix"]
    m.update(tf_metrics(ref, est, {k: v[..., :T] for k, v in (interferers or {}).items()}))
    if with_bss:
        m.update(bss_eval(ref, est, mixture - ref, mixture - est))
    return m


def aggregate(rows: list[dict]) -> dict:
    """Mean and median across clips for every numeric metric."""
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=np.float64)
        if len(vals):
            agg[f"{k}_mean"] = float(vals.mean())
            agg[f"{k}_median"] = float(np.median(vals))
    agg["n_clips"] = len(rows)
    return agg
