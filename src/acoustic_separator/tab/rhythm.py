"""Beat grid, bar phase and swing-aware quantisation of note onsets to 16th notes."""
from __future__ import annotations

import numpy as np

SUB = 4  # 16ths per beat


def onset_envelope(onset_post: np.ndarray, time_scale: float, fps: float = 100.0,
                   bass_below: int = 52, bass_weight: float = 3.0) -> np.ndarray:
    """Pitch-summed onset posterior, bass pitches weighted (the thumb marks the beat), resampled
    from model time to the original timeline at ``fps``."""
    w = np.ones(onset_post.shape[1])
    w[: bass_below - 21] = bass_weight
    env = onset_post @ w
    t_model = np.arange(len(env)) / fps
    t_orig = np.arange(int(len(env) * time_scale)) / fps
    return np.interp(t_orig, t_model * time_scale, env)


def track_beats(env: np.ndarray, fps: float = 100.0, bpm: float | None = None,
                tightness: float = 400.0) -> np.ndarray:
    import librosa

    hop = 512
    sr = int(round(fps * hop))
    if bpm is None:
        bpm = float(librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=hop,
                                          start_bpm=120, max_tempo=240)[0])
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, start_bpm=bpm,
                                       tightness=tightness, units="frames")
    return beats / fps


def extend_beats(beats: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    b = list(beats)
    ibi0, ibi1 = np.median(np.diff(beats[:8])), np.median(np.diff(beats[-8:]))
    while b[0] > t_min:
        b.insert(0, b[0] - ibi0)
    while b[-1] < t_max:
        b.append(b[-1] + ibi1)
    return np.array(b)


def phases(t: np.ndarray, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.clip(np.searchsorted(beats, t) - 1, 0, len(beats) - 2)
    return k, (t - beats[k]) / (beats[k + 1] - beats[k])


def swing_centres(onsets: np.ndarray, beats: np.ndarray, iters: int = 20) -> np.ndarray:
    """Circular k-means of onset phases around the 4 nominal 16th positions (Brazilian 16ths are
    uneven: the 2nd is late, the 4th early)."""
    _, ph = phases(onsets, beats)
    c = np.arange(SUB) / SUB
    for _ in range(iters):
        d = ((ph[:, None] - c[None, :] + 0.5) % 1.0) - 0.5
        a = np.abs(d).argmin(1)
        for j in range(SUB):
            if (a == j).sum() >= 5:
                c[j] = c[j] + np.median(d[a == j, j])
    return c


def slot_bounds(centres: np.ndarray) -> np.ndarray:
    """Decision boundaries between consecutive 16th positions (last one wraps to next beat)."""
    nxt = np.append(centres[1:], centres[0] + 1.0)
    return (centres + nxt) / 2


def quantize(t: np.ndarray, beats: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Onset times -> global 16th index (beat * 4 + slot)."""
    k, ph = phases(np.asarray(t, float), beats)
    lo = bounds[-1] - 1.0  # below this the onset belongs to the previous beat's last slot
    slot = np.searchsorted(bounds, ph)  # 0..4
    q = k * SUB + slot
    q = np.where(ph < lo, k * SUB - 1, q)
    return q.astype(int)


def refine_beats(onsets: np.ndarray, beats: np.ndarray, weights: np.ndarray | None = None,
                 centres: np.ndarray | None = None, smooth: float = 30.0, iters: int = 5,
                 max_err: float = 0.45) -> np.ndarray:
    """Least-squares beat times: an onset assigned to slot j of beat k satisfies
    t = (1 - c_j) b_k + c_j b_{k+1}; a second-difference penalty keeps the tempo smooth."""
    b = np.array(beats, float)
    w = np.ones(len(onsets)) if weights is None else np.asarray(weights, float)
    c = np.arange(SUB) / SUB if centres is None else np.asarray(centres)
    K = len(b)
    for _ in range(iters):
        q = quantize(onsets, b, slot_bounds(c))
        k, j = np.divmod(q, SUB)
        valid = (k >= 0) & (k < K - 1)
        pos = b[np.clip(k, 0, K - 2)] + c[j] * (b[np.clip(k + 1, 1, K - 1)] - b[np.clip(k, 0, K - 2)])
        ibi = np.diff(b)[np.clip(k, 0, K - 2)]
        keep = valid & (np.abs(onsets - pos) < max_err * ibi / SUB)
        n = np.nonzero(keep)[0]
        A = np.zeros((len(n), K))
        A[np.arange(len(n)), k[n]] = 1 - c[j[n]]
        A[np.arange(len(n)), k[n] + 1] += c[j[n]]
        sw = np.sqrt(w[n])
        D = np.zeros((K - 2, K))
        r = np.arange(K - 2)
        D[r, r], D[r, r + 1], D[r, r + 2] = 1.0, -2.0, 1.0
        M = np.vstack([A * sw[:, None], np.sqrt(smooth) * D, 0.05 * np.eye(K)])
        v = np.concatenate([onsets[n] * sw, np.zeros(K - 2), 0.05 * b])
        b = np.linalg.lstsq(M, v, rcond=None)[0]
    return b


def bar_phase(notes: np.ndarray, beats: np.ndarray, beats_per_bar: int = 2,
              bass_below: int = 52) -> int:
    """Which beat parity starts a bar: bass onsets on the beat and harmonic change (pitch-class
    novelty between consecutive beats) are both expected at bar starts."""
    k, ph = phases(notes[:, 0], beats)
    on_beat = (ph < 0.12) | (ph > 0.88)
    kb = np.where(ph > 0.5, k + 1, k)
    chroma = np.zeros((len(beats) + 1, 12))
    for kk, p, dur in zip(kb, notes[:, 2].astype(int), notes[:, 1] - notes[:, 0]):
        chroma[kk, p % 12] += min(dur, 0.5)
    chroma /= np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9
    nov = np.r_[0.0, 1.0 - (chroma[1:] * chroma[:-1]).sum(1)]
    score = np.zeros(beats_per_bar)
    bass = (notes[:, 2] < bass_below) & on_beat
    for r in range(beats_per_bar):
        score[r] = np.mean(nov[r::beats_per_bar]) + 0.5 * np.mean((kb[bass] % beats_per_bar) == r)
    return int(np.argmax(score))
