"""Chunked overlap-add inference with test-time augmentation.

The chunking follows the MSST ``demix`` scheme (reflect padding, linear fades, overlap
averaging) but exposes every parameter so experiments can sweep them.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .audio import resample


def pick_device(pref: str | None = None) -> torch.device:
    pref = pref or os.environ.get("ACOUSTIC_SEPARATOR_DEVICE", "auto")
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class InferenceParams:
    chunk_size: int | None = None  # samples; None = model default
    num_overlap: int = 4  # step = chunk_size / num_overlap
    batch_size: int = 1
    window: str = "linear"  # linear fades (MSST) or "hann"
    fade_ratio: float = 0.1
    tta: list[str] = field(default_factory=list)  # "swap" (L/R), "invert" (polarity)
    shifts: int = 1  # random time-shift passes (Demucs-style "shifts")
    shift_max_s: float = 0.5
    seed: int = 0
    precision: str = "fp32"  # "fp32" or "bf16" (autocast; ~1.5x faster on AMX CPUs)

    def to_dict(self) -> dict:
        return asdict(self)


def _window(chunk: int, kind: str, fade_ratio: float) -> torch.Tensor:
    if kind == "hann":
        w = torch.hann_window(chunk, periodic=False)
        return w.clamp_min(1e-3)
    fade = max(1, int(chunk * fade_ratio))
    w = torch.ones(chunk)
    w[:fade] = torch.linspace(0, 1, fade)
    w[-fade:] = torch.linspace(1, 0, fade)
    return w


def _run_chunks(model, mix: torch.Tensor, n_out: int, chunk: int, params: InferenceParams,
                device: torch.device, demucs_mode: bool, progress: bool) -> torch.Tensor:
    """mix: (C, T) float32 -> (n_out, C, T)."""
    step = max(1, chunk // params.num_overlap)
    length = mix.shape[-1]
    border = chunk - step
    padded = False
    if not demucs_mode and length > 2 * border and border > 0:
        mix = F.pad(mix, (border, border), mode="reflect")
        padded = True
    total = mix.shape[-1]
    result = torch.zeros((n_out,) + tuple(mix.shape), dtype=torch.float32)
    counter = torch.zeros((1, 1, total), dtype=torch.float32)
    win_base = torch.ones(chunk) if demucs_mode else _window(chunk, params.window, params.fade_ratio)

    starts = list(range(0, total, step))
    # drop trailing starts whose chunk would be fully covered by the previous one
    while len(starts) > 1 and starts[-2] + chunk >= total:
        starts.pop()
    iterator = starts
    if progress:
        from tqdm import tqdm

        iterator = tqdm(starts, desc="chunks", leave=False)

    batch, locs = [], []
    for n, s in enumerate(iterator):
        part = mix[:, s:s + chunk]
        seg = part.shape[-1]
        if seg < chunk:
            mode = "reflect" if (not demucs_mode and seg > chunk // 2) else "constant"
            part = F.pad(part, (0, chunk - seg), mode=mode)
        batch.append(part)
        locs.append((s, seg, n == 0, n == len(starts) - 1))
        if len(batch) >= params.batch_size or n == len(starts) - 1:
            with torch.inference_mode(), torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=params.precision == "bf16"):
                out = model(torch.stack(batch).to(device))
            out = out.float().cpu()
            if out.dim() == 3:  # (B, C, T) single-stem models
                out = out.unsqueeze(1)
            for j, (s0, seg0, first, last) in enumerate(locs):
                w = win_base.clone()
                if not demucs_mode:
                    fade = max(1, int(chunk * params.fade_ratio))
                    if first:
                        w[:fade] = 1
                    if last:
                        w[-fade:] = 1
                result[..., s0:s0 + seg0] += out[j, ..., :seg0] * w[:seg0]
                counter[..., s0:s0 + seg0] += w[:seg0]
            batch, locs = [], []
    est = result / counter.clamp_min(1e-8)
    if padded:
        est = est[..., border:-border]
    return est


class Separator:
    """Wraps a loaded model + its config for full-track inference."""

    def __init__(self, spec, model, config, device: torch.device):
        self.spec = spec
        self.model = model
        self.config = config
        self.device = device
        self.demucs_mode = spec.arch == "htdemucs"
        self.demucs_pkg = spec.arch == "demucs_pkg"

    @property
    def samplerate(self) -> int:
        return int(self.spec.samplerate)

    @property
    def stems(self) -> list[str]:
        return list(self.spec.stems)

    def default_chunk(self) -> int:
        if self.demucs_pkg:
            return int(float(self.model.segment) * self.model.samplerate)
        cfg = self.config
        if self.demucs_mode:
            return int(cfg.training.samplerate * cfg.training.segment)
        inf = getattr(cfg, "inference", None)
        if inf is not None and "chunk_size" in inf:
            return int(inf.chunk_size)
        return int(cfg.audio.chunk_size)

    def _single(self, mix: torch.Tensor, params: InferenceParams, progress: bool) -> torch.Tensor:
        chunk = params.chunk_size or self.default_chunk()
        return _run_chunks(self.model, mix, len(self.stems), chunk, params, self.device,
                           self.demucs_mode, progress)

    def separate(self, audio: np.ndarray, sr: int, params: InferenceParams | None = None,
                 progress: bool = False) -> dict[str, np.ndarray]:
        """audio: (2, T) at ``sr``. Returns {stem: (2, T) at ``sr``}."""
        params = params or InferenceParams()
        x = resample(audio, sr, self.samplerate)
        norm = None
        if self.demucs_pkg:  # track-level normalisation as in demucs.separate
            ref = x.mean(0)
            norm = (float(ref.mean()), float(ref.std()) + 1e-8)
            x = (x - norm[0]) / norm[1]
        mix = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        rng = np.random.default_rng(params.seed)
        views = [("id", mix)]
        if "swap" in params.tta:
            views.append(("swap", mix.flip(0)))
        if "invert" in params.tta:
            views.append(("invert", -mix))
        acc = None
        n = 0
        for kind, v in views:
            for k in range(max(1, params.shifts)):
                off = 0
                vin = v
                if k > 0:
                    off = int(rng.integers(1, int(params.shift_max_s * self.samplerate)))
                    vin = F.pad(v, (off, 0))
                est = self._single(vin, params, progress)[..., off:off + v.shape[-1]]
                if kind == "swap":
                    est = est.flip(1)
                elif kind == "invert":
                    est = -est
                acc = est if acc is None else acc + est
                n += 1
        est = (acc / n).numpy()
        if norm is not None:
            est = est * norm[1] + norm[0] / len(self.stems)
        out = {}
        for i, name in enumerate(self.stems):
            y = resample(est[i], self.samplerate, sr)
            T = audio.shape[-1]
            if y.shape[-1] < T:
                y = np.pad(y, ((0, 0), (0, T - y.shape[-1])))
            out[name] = np.ascontiguousarray(y[:, :T], dtype=np.float32)
        return out


def timed(fn, *a, **kw):
    t = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t
