"""Guitar note transcription with a high-resolution onset/offset-regression CRNN.

Architecture: Kong et al. 2021, "High-resolution piano transcription with pedals by regressing
onset and offset times" (re-implemented here, state-dict compatible).  Weights: X. Riley's guitar
checkpoints (xavriley/midi-transcription-models, MIT licence), used as released (not fine-tuned
here), the tab averaging two of them; only safetensors are ever loaded as weights (see
``_convert_pth``).

Note-onset F1 (mir_eval, 50 ms; reports/tab_benchmark*.md), ``fl`` + ``gaps_paper`` averaged
vs ``kroma`` alone: clean GuitarSet (60 mic excerpts) 0.913 vs 0.798; guitar mixed with violin,
clarinet and percussion and extracted with the Champion separator 0.904 vs 0.853 (on its GAPS
choro/classical part, 270 notes unseen by every checkpoint: 0.922 vs 0.914).  ``gaps_paper`` may have been
trained with GuitarSet, so GuitarSet-based numbers containing it can be optimistic; ``fl`` alone,
documented as zero-shot on GuitarSet, scores 0.896 there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
FPS = 100
N_FFT = 2048
HOP = SR // FPS
MEL_BINS = 229
BEGIN_NOTE = 21  # MIDI pitch of output class 0 (A0)
CLASSES = 88

_REPO = "xavriley/midi-transcription-models"
_REVISION = "b7bec65a2b860aca72856b0feef58b5df407b777"
# Guitar checkpoints of the same architecture (X. Riley, MIT).  gaps*: trained on GAPS (classical
# nylon-string guitar); fl: trained on the Francois Leduc guitar recordings; kroma: released as
# safetensors.  The .pth files are converted once with PyTorch's weights_only loader.
CHECKPOINTS = {
    "kroma": {"file": "guitar_kroma.safetensors",
              "sha256": "26919a2fa15652f3a63255ea413a64ffbbeba99efa0a2a2dab425d13f57f2de0"},
    "gaps": {"file": "guitar-gaps.pth",
             "sha256": "65483e7c0e340a90415b15b520687587698c8c728f5fa470a205f13ee45c6513"},
    "gaps_paper": {"file": "guitar-gaps-paper-version-12200_iterations.pth",
                   "sha256": "94a7c936ec9fde83686d29007dc256274384e832739cadece39e92cee3b69a7e"},
    "fl": {"file": "guitar-fl.pth",
           "sha256": "50d93dba89bdd3401849bc735614478e83d9f46d21fa3f71d8aca5acc0a52028"},
}
CHECKPOINT = {"repo": _REPO, "revision": _REVISION, **CHECKPOINTS["kroma"]}


# --------------------------------------------------------------------------------- model
class _STFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_real = nn.Conv1d(1, N_FFT // 2 + 1, N_FFT, stride=HOP, bias=False)
        self.conv_imag = nn.Conv1d(1, N_FFT // 2 + 1, N_FFT, stride=HOP, bias=False)


class _Spectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = _STFT()

    def forward(self, x):  # (B, L) -> (B, 1, T, F) power spectrogram
        x = F.pad(x[:, None, :], (N_FFT // 2, N_FFT // 2), mode="reflect")
        re, im = self.stft.conv_real(x), self.stft.conv_imag(x)
        return (re ** 2 + im ** 2)[:, None].transpose(2, 3)


class _Logmel(nn.Module):
    def __init__(self):
        super().__init__()
        self.melW = nn.Parameter(torch.zeros(N_FFT // 2 + 1, MEL_BINS), requires_grad=False)

    def forward(self, x):
        return 10.0 * torch.log10(torch.clamp(x @ self.melW, min=1e-10))


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=(1, 2))


class _AcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = _ConvBlock(1, 48)
        self.conv_block2 = _ConvBlock(48, 64)
        self.conv_block3 = _ConvBlock(64, 96)
        self.conv_block4 = _ConvBlock(96, 128)
        self.fc5 = nn.Linear(128 * 14, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768)
        self.gru = nn.GRU(768, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, CLASSES)

    def forward(self, x):  # (B, 1, T, mel)
        for blk in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4):
            x = blk(x)
        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x, _ = self.gru(x)
        return torch.sigmoid(self.fc(x))


class RegressCRNN(nn.Module):
    """Onset/offset regression + frame + velocity heads (Kong et al. 2021)."""

    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = _Spectrogram()
        self.logmel_extractor = _Logmel()
        self.bn0 = nn.BatchNorm2d(MEL_BINS)
        self.frame_model = _AcousticModel()
        self.reg_onset_model = _AcousticModel()
        self.reg_offset_model = _AcousticModel()
        self.velocity_model = _AcousticModel()
        self.reg_onset_gru = nn.GRU(CLASSES * 2, 256, batch_first=True, bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, CLASSES)
        self.frame_gru = nn.GRU(CLASSES * 3, 256, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(512, CLASSES)

    def forward(self, audio):  # (B, L) float32 @ 16 kHz
        x = self.logmel_extractor(self.spectrogram_extractor(audio))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        frame = self.frame_model(x)
        onset = self.reg_onset_model(x)
        offset = self.reg_offset_model(x)
        vel = self.velocity_model(x)
        h, _ = self.reg_onset_gru(torch.cat((onset, onset ** 0.5 * vel), dim=2))
        onset = torch.sigmoid(self.reg_onset_fc(h))
        h, _ = self.frame_gru(torch.cat((frame, onset, offset), dim=2))
        frame = torch.sigmoid(self.frame_fc(h))
        return {"onset": onset, "offset": offset, "frame": frame, "velocity": vel}


def checkpoint_path(name: str = "kroma") -> Path:
    """Local safetensors file of a guitar checkpoint (downloaded at a pinned revision and
    checksummed).  ``.pth`` releases are converted once, see ``_convert_pth``."""
    from ..audio import file_sha256
    from ..models.loader import _hf_download, cache_dir

    spec = CHECKPOINTS[name]
    path = _hf_download(_REPO, spec["file"], _REVISION)
    digest = file_sha256(path)
    if digest != spec["sha256"]:
        raise RuntimeError(f"checksum mismatch for {path}: {digest}")
    if path.suffix == ".safetensors":
        return path
    out = cache_dir() / "tab" / f"{Path(spec['file']).stem}.safetensors"
    if not out.exists():
        _convert_pth(path, out)
    return out


def _convert_pth(src: Path, dst: Path) -> None:
    """The .pth files also pickle the training sampler's state (NumPy arrays).  They are read
    with torch.load(weights_only=True), allowing only NumPy's array/dtype constructors besides
    tensors (checked statically: nothing else is referenced), and re-saved as safetensors."""
    import codecs

    import numpy.dtypes as ndt
    from safetensors.torch import save_file

    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # NumPy < 2
        from numpy.core.multiarray import _reconstruct
    # the pickles name the NumPy 1.x path; register the function under that name too
    allow = [_reconstruct, (_reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, np.dtype,
             codecs.encode]
    allow += [getattr(ndt, n) for n in dir(ndt) if n.endswith("DType")]
    with torch.serialization.safe_globals(allow):
        ck = torch.load(src, map_location="cpu", weights_only=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in ck["model"].items()}, str(dst))


def load_model(name: str | Path = "kroma") -> RegressCRNN:
    """Checkpoint name from ``CHECKPOINTS`` or a path to a safetensors file."""
    from safetensors.torch import load_file

    path = Path(name) if str(name).endswith(".safetensors") else checkpoint_path(str(name))
    sd = {k: v.float() for k, v in load_file(str(path)).items()}
    model = RegressCRNN()
    model.load_state_dict(sd, strict=True)
    return model.eval()


# ------------------------------------------------------------------------------ inference
@torch.inference_mode()
def posteriors(model: RegressCRNN, audio: np.ndarray, seg_s: float = 10.0, batch: int = 4,
               shifts: tuple[float, ...] = (0.0,)) -> dict[str, np.ndarray]:
    """(T, 88) posteriors at 100 fps.  10 s windows, 50 % overlap, only the central half of each
    window is kept (>= 2.5 s of context on both sides).  ``shifts`` (in s) averages several
    window phases (test-time augmentation)."""
    seg = int(seg_s * SR)
    n_frames = int(np.ceil(len(audio) / HOP)) + 1
    acc: dict[str, np.ndarray] = {}
    for shift in shifts:
        lead = seg // 4 + int(shift * SR)
        x = np.concatenate([np.zeros(lead, np.float32), audio.astype(np.float32),
                            np.zeros(seg, np.float32)])
        starts = range(0, len(x) - seg + 1, seg // 2)
        segs = np.stack([x[s:s + seg] for s in starts])
        outs: dict[str, list] = {}
        for i in range(0, len(segs), batch):
            for k, v in model(torch.from_numpy(segs[i:i + batch])).items():
                outs.setdefault(k, []).append(v.numpy())
        q = seg // HOP // 4
        skip = int(round(shift * SR / HOP))
        for k, v in outs.items():
            v = np.concatenate(v, 0)
            y = np.concatenate([s[q:3 * q] for s in v], 0)[skip:skip + n_frames]
            acc[k] = acc.get(k, 0) + y / len(shifts)
    return acc


def _regression_peaks(x: np.ndarray, thr: float, nb: int):
    """Frames/pitches where a regression curve peaks (monotonic over ``nb`` neighbours on each
    side) above ``thr``; returns (frame, pitch, sub-frame shift) as in Kong et al."""
    T = x.shape[0]
    ok = x > thr
    ok[:nb] = False
    ok[T - nb:] = False
    for i in range(nb):
        left = np.zeros_like(ok)
        left[nb:] = x[nb - i:T - i] >= x[nb - i - 1:T - i - 1]
        right = np.zeros_like(ok)
        right[:T - nb] = x[i:T - nb + i] >= x[i + 1:T - nb + i + 1]
        ok &= left & right
    n, k = np.nonzero(ok)
    a, b, c = x[n - 1, k], x[n, k], x[n + 1, k]
    den = np.where(a > c, b - c, b - a)
    shift = np.where(den > 1e-9, (c - a) / np.maximum(den, 1e-9) / 2, 0.0)
    return n, k, shift


def decode(post: dict[str, np.ndarray], onset_thr: float = 0.3, offset_thr: float = 0.3,
           frame_thr: float = 0.1, max_len_s: float = 6.0) -> np.ndarray:
    """Posteriors -> notes (N, 4): onset s, offset s, MIDI pitch, onset confidence."""
    onset, offset, frame = post["onset"], post["offset"], post["frame"]
    T = onset.shape[0]
    on_n, on_k, on_s = _regression_peaks(onset, onset_thr, 2)
    off_n, off_k, off_s = _regression_peaks(offset, offset_thr, 4)
    off_at = np.zeros(onset.shape, bool)
    off_sh = np.zeros(onset.shape)
    off_at[off_n, off_k] = True
    off_sh[off_n, off_k] = off_s
    max_len = int(max_len_s * FPS)
    notes = []
    for k in np.unique(on_k):
        sel = np.nonzero(on_k == k)[0]
        frames = on_n[sel]
        order = np.argsort(frames)
        frames, shifts = frames[order], on_s[sel][order]
        for j, (n0, s0) in enumerate(zip(frames, shifts)):
            stop = min(frames[j + 1] if j + 1 < len(frames) else T, n0 + max_len, T)
            fin, fin_sh = stop - 1, 0.0
            gone = np.nonzero(frame[n0 + 1:stop, k] <= frame_thr)[0]
            offs = np.nonzero(off_at[n0 + 1:stop, k])[0]
            if len(gone):
                d = n0 + 1 + gone[0]
                o = n0 + 1 + offs[0] if len(offs) and offs[0] < gone[0] else None
                if o is not None and o - n0 > d - o:
                    fin, fin_sh = o, off_sh[o, k]
                else:
                    fin = d
            elif len(offs):
                fin = n0 + 1 + offs[0]
                fin_sh = off_sh[fin, k]
            t_on = (n0 + s0) / FPS
            notes.append((t_on, max((fin + fin_sh) / FPS, t_on + 0.02), k + BEGIN_NOTE,
                          float(onset[n0, k])))
    notes.sort()
    return np.array(notes, dtype=float).reshape(-1, 4)


# ------------------------------------------------------------------------------- tuning
def estimate_tuning(audio: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 700.0) -> float:
    """Global tuning offset in cents relative to A4 = 440 Hz: magnitude-weighted circular mean
    of interpolated spectral-peak frequencies (mod 100 cents)."""
    from scipy.signal import find_peaks, stft

    n = 16384
    _, _, Z = stft(audio, fs=sr, nperseg=n, noverlap=n - 2048, window="hann")
    M = np.abs(Z)
    L = np.log(M + 1e-12)
    k0, k1 = int(fmin * n / sr), int(fmax * n / sr)
    dev, wts = [], []
    for j in range(M.shape[1]):
        col = M[k0:k1, j]
        if col.max() <= 0:
            continue
        for p in find_peaks(col, height=col.max() * 0.1)[0] + k0:
            a, b, c = L[p - 1, j], L[p, j], L[p + 1, j]
            d = 0.5 * (a - c) / (a - 2 * b + c)
            cents = 1200 * np.log2((p + d) * sr / n / 440.0)
            dev.append(((cents + 50) % 100) - 50)
            wts.append(M[p, j])
    ang = np.asarray(dev) / 100 * 2 * np.pi
    return float(np.angle(np.sum(np.asarray(wts) * np.exp(1j * ang))) / (2 * np.pi) * 100)


def to_model_input(audio: np.ndarray, sr: int, tuning_cents: float = 0.0) -> tuple[np.ndarray, float]:
    """Mono 16 kHz input with the tuning offset removed by resampling.  Returns the signal and
    the time scale ``f`` that maps model time back to the original: t_orig = f * t_model."""
    import soxr

    mono = audio.mean(0) if audio.ndim == 2 else audio
    f = 2.0 ** (-tuning_cents / 1200.0)
    y = soxr.resample(mono.astype(np.float32), sr * f, SR, quality="VHQ")
    return y.astype(np.float32), f
