"""Learned mask refiner ("stacking ensemble") on top of pretrained separators.

Inputs per clip (all at 44.1 kHz stereo):
    mix                 -- the mixture
    positives[k]        -- acoustic/all-guitar estimates from K pretrained models
    negatives[j]        -- estimates of interferers (e.g. violin, woodwind) used as evidence
The network predicts a soft mask on the mixture STFT. It is residual on the logit of the
mean positive ratio mask, and its last layer is zero-initialised, so an untrained refiner
is exactly the ``mask_mean`` ensemble; training can only move away from it if that lowers
the validation loss (checked with the ground-truth benchmark, not the training loss).

The network is small (≈100–300 k parameters) so it trains on a CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FFT = 2048
HOP = 512
EPS = 1e-6


def stft(x: torch.Tensor) -> torch.Tensor:
    """(..., T) -> complex (..., F, N)."""
    shape = x.shape
    X = torch.stft(x.reshape(-1, shape[-1]), N_FFT, HOP, window=torch.hann_window(N_FFT, device=x.device),
                   return_complex=True, center=True)
    return X.reshape(*shape[:-1], *X.shape[-2:])


def istft(X: torch.Tensor, length: int) -> torch.Tensor:
    shape = X.shape
    y = torch.istft(X.reshape(-1, *shape[-2:]), N_FFT, HOP, window=torch.hann_window(N_FFT, device=X.device),
                    center=True, length=length)
    return y.reshape(*shape[:-2], length)


def features(mix: torch.Tensor, pos: list[torch.Tensor], neg: list[torch.Tensor]):
    """mix/pos/neg: (B, 2, T). Returns (feat (B*2, C, F, N), X (B, 2, F, N), base_mask)."""
    X = stft(mix)
    ax = X.abs() + EPS
    pm = [(stft(p).abs() / ax).clamp(0, 1) for p in pos]
    nm = [(stft(n).abs() / ax).clamp(0, 1) for n in neg]
    base = torch.stack(pm).mean(0)
    lx = torch.log(ax)
    lx = (lx - lx.mean(dim=(-2, -1), keepdim=True)) / (lx.std(dim=(-2, -1), keepdim=True) + EPS)
    chans = [lx, base] + pm + nm
    if len(pm) > 1:
        chans.append(torch.stack(pm).std(0))  # model disagreement
    feat = torch.stack(chans, dim=2)  # (B, 2, C, F, N)
    B, S, C, Fq, N = feat.shape
    return feat.reshape(B * S, C, Fq, N), X, base


class ConvBlock(nn.Module):
    def __init__(self, c: int, dil: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=(1, dil), dilation=(1, dil))
        self.norm = nn.GroupNorm(4, c)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))


class MaskRefiner(nn.Module):
    def __init__(self, in_ch: int, width: int = 32, depth: int = 6, freq_emb: int = 8,
                 n_freq: int = N_FFT // 2 + 1):
        super().__init__()
        self.freq_emb = nn.Parameter(torch.zeros(freq_emb, n_freq, 1))
        self.inp = nn.Conv2d(in_ch + freq_emb, width, 3, padding=1)
        self.blocks = nn.ModuleList([ConvBlock(width, 2 ** (i % 4)) for i in range(depth)])
        # frequency mixing: a GRU across frequency would be heavier; use a strided conv pyramid
        self.down = nn.Conv2d(width, width, (4, 1), stride=(4, 1))
        self.mid = nn.Sequential(ConvBlock(width, 1), ConvBlock(width, 2))
        self.up = nn.ConvTranspose2d(width, width, (4, 1), stride=(4, 1))
        self.out = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        BS, C, Fq, N = feat.shape
        fe = self.freq_emb[:, :Fq].unsqueeze(0).expand(BS, -1, -1, N)
        h = F.gelu(self.inp(torch.cat([feat, fe], 1)))
        for b in self.blocks:
            h = b(h)
        f4 = (Fq // 4) * 4
        d = self.mid(F.gelu(self.down(h[:, :, :f4])))
        u = self.up(d)
        h = h + F.pad(u, (0, 0, 0, Fq - f4))
        delta = self.out(h).squeeze(1)  # (BS, F, N)
        b = base.reshape(BS, Fq, N).clamp(1e-4, 1 - 1e-4)
        return torch.sigmoid(torch.logit(b) + delta)


def apply_refiner(model: MaskRefiner, mix: np.ndarray, pos: list[np.ndarray], neg: list[np.ndarray],
                  seg_s: float = 12.0, sr: int = 44100) -> np.ndarray:
    """Full-track inference in overlapping segments (numpy in/out, (2, T))."""
    model.eval()
    T = mix.shape[-1]
    seg = int(seg_s * sr)
    hop = seg // 2
    out = np.zeros_like(mix)
    wsum = np.zeros(T, dtype=np.float32)
    win = np.hanning(seg).astype(np.float32).clip(1e-3)
    starts = list(range(0, max(1, T - seg + hop), hop)) or [0]
    for s in starts:
        e = min(T, s + seg)
        sl = slice(s, e)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[:, sl], dtype=np.float32))[None]  # noqa: E731
        with torch.no_grad():
            feat, X, base = features(t(mix), [t(p) for p in pos], [t(n) for n in neg])
            m = model(feat, base).reshape(1, 2, *X.shape[-2:])
            y = istft(X * m, e - s)[0].numpy()
        w = win[: e - s] if (e - s) == seg else np.hanning(e - s).astype(np.float32).clip(1e-3)
        if s == 0:
            w[: (e - s) // 2] = 1.0
        if e == T:
            w[(e - s) // 2:] = 1.0
        out[:, sl] += y * w
        wsum[sl] += w
    return (out / wsum.clip(1e-6)).astype(np.float32)


def sdr_loss(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    num = (ref ** 2).sum(-1) + 1e-8
    den = ((ref - est) ** 2).sum(-1) + 1e-8
    return -(10 * torch.log10(num / den)).mean()


def refiner_loss(est: torch.Tensor, ref: torch.Tensor, w_sdr: float = 1.0, w_spec: float = 1.0) -> torch.Tensor:
    E, R = stft(est), stft(ref)
    spec = (E.abs() - R.abs()).abs().mean() / (R.abs().mean() + 1e-6)
    return w_sdr * sdr_loss(est, ref) / 10 + w_spec * spec
