"""Reference-free proxy scores for songs without a ground-truth stem.

Uses the AudioSet-trained Audio Spectrogram Transformer (MIT/ast-finetuned-audioset,
BSD-3-Clause) to estimate, over 5 s windows of the *extracted* stem, the probability of
acoustic guitar and of each interfering class. These are proxies only: the classifier
works at 16 kHz and never replaces ground-truth validation metrics.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

GROUPS = {
    "guitar": ["Acoustic guitar", "Guitar", "Plucked string instrument", "Steel guitar, slide guitar"],
    "vocal": ["Singing", "Speech", "Male singing", "Female singing", "Choir", "Vocal music"],
    "drums": ["Drum kit", "Drum", "Snare drum", "Bass drum", "Cymbal", "Hi-hat", "Percussion",
              "Tambourine", "Rimshot", "Drum roll"],
    "bass": ["Bass guitar", "Double bass"],
    "piano": ["Piano", "Keyboard (musical)", "Electric piano"],
    "electric_guitar": ["Electric guitar", "Distortion"],
    "strings": ["Violin, fiddle", "Bowed string instrument", "Cello", "String section", "Pizzicato"],
    "winds": ["Clarinet", "Flute", "Saxophone", "Wind instrument, woodwind instrument",
              "Brass instrument", "Trumpet", "Harmonica", "Accordion"],
    "synth": ["Synthesizer", "Organ", "Electronic organ", "Hammond organ"],
}


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    m = ASTForAudioClassification.from_pretrained(AST_MODEL).eval()
    label2id = {v: int(k) for k, v in m.config.id2label.items()}
    torch.set_grad_enabled(False)
    return fe, m, label2id


def window_probs(audio: np.ndarray, sr: int, win_s: float = 5.0) -> np.ndarray:
    import torch

    from .audio import resample

    fe, m, _ = _load()
    x = resample(audio, sr, 16000).mean(0)
    n = int(win_s * 16000)
    rows = []
    for s in range(0, max(1, len(x) - n + 1), n):
        seg = x[s:s + n]
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:  # silence: skip (no evidence either way)
            continue
        feats = fe(seg, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            rows.append(torch.sigmoid(m(**feats).logits)[0].numpy())
    return np.array(rows) if rows else np.zeros((0, len(_load()[2])))


def group_scores(P: np.ndarray) -> dict[str, float]:
    _, _, label2id = _load()
    out = {}
    for g, labels in GROUPS.items():
        ids = [label2id[lab] for lab in labels if lab in label2id]
        out[g] = float(P[:, ids].max(1).mean()) if len(P) else float("nan")
    return out


def proxy_scores(acoustic: np.ndarray, residual: np.ndarray, sr: int) -> dict[str, float]:
    """Returns guitar probability in the stem, per-class leak probabilities in the stem,
    and guitar probability left in the residual."""
    s = group_scores(window_probs(acoustic, sr))
    r = group_scores(window_probs(residual, sr))
    out = {"guitar_prob": s["guitar"], "residual_guitar_prob": r["guitar"]}
    leaks = []
    for g in GROUPS:
        if g == "guitar":
            continue
        out[f"leak_{g}_prob"] = s[g]
        leaks.append(s[g])
    out["leak_prob_max"] = float(np.nanmax(leaks))
    out["leak_prob_sum"] = float(np.nansum(leaks))
    e_mix = np.mean((acoustic + residual) ** 2) + 1e-12
    out["stem_energy_db"] = float(10 * np.log10(np.mean(acoustic ** 2) / e_mix + 1e-12))
    return out
