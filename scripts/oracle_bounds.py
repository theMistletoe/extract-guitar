#!/usr/bin/env python
"""Oracle upper bounds on the validation set (how much headroom separation has).

* mixture   -- the unprocessed mixture used as the estimate (lower bound, = sdr_mix)
* irm       -- ideal ratio mask |S|/(|S|+|N|) on the mixture STFT (mask-based upper bound)
* iwm2      -- ideal Wiener mask |S|^2/(|S|^2+|N|^2)
Writes reports/oracle_bounds.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator.audio import load_audio  # noqa: E402
from acoustic_separator.evaluation import sdr, si_sdr  # noqa: E402

N_FFT, HOP = 4096, 1024


def stft(x):
    return torch.stft(torch.from_numpy(x), N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True)


def istft(X, n):
    return torch.istft(X, N_FFT, HOP, window=torch.hann_window(N_FFT), length=n).numpy()


def main() -> int:
    rows = {}
    for d in sorted(p for p in (ROOT / "datasets" / "validation").iterdir() if (p / "mixture.wav").exists()):
        mix, _ = load_audio(d / "mixture.wav")
        s, _ = load_audio(d / "acoustic_guitar.wav")
        X, S, N = stft(mix), stft(s), stft(mix - s)
        irm = S.abs() / (S.abs() + N.abs() + 1e-8)
        iwm = S.abs() ** 2 / (S.abs() ** 2 + N.abs() ** 2 + 1e-8)
        est_irm, est_iwm = istft(X * irm, mix.shape[1]), istft(X * iwm, mix.shape[1])
        rows[d.name] = {"mixture": sdr(s, mix), "irm": sdr(s, est_irm), "iwm2": sdr(s, est_iwm),
                        "irm_si": si_sdr(s, est_irm)}
    agg = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))}
    for fam in ("ms_", "syn_"):
        for k in ("mixture", "irm", "iwm2"):
            agg[f"{k}_{fam[:-1]}"] = float(np.mean([r[k] for n, r in rows.items() if n.startswith(fam)]))
    out = ROOT / "reports" / "oracle_bounds.json"
    out.write_text(json.dumps({"mean": agg, "clips": rows}, indent=1))
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
