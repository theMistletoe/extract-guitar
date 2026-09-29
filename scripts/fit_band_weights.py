#!/usr/bin/env python
"""Fit frequency-dependent ensemble weights on TRAINING clips (never the validation set).

For every band, solve the least-squares problem min sum |S - sum_i w_i Y_i|^2 over the
complex STFTs of all training clips (S = true acoustic guitar, Y_i = member estimates),
then write a Strategy-D pipeline that applies those weights with band_weighted.

    python scripts/fit_band_weights.py --members sw6:guitar htdemucs6s_gtrft:guitar \
        --pipelines A_xlance A_htdemucs6s_ft --tag sw_htft
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from acoustic_separator.audio import load_audio  # noqa: E402
from acoustic_separator.ensemble import HOP, N_FFT  # noqa: E402
from acoustic_separator.pipeline import PIPELINE_DIR, ModelPool, PipelineRunner  # noqa: E402
from make_ensembles import member_steps  # noqa: E402
from train import CLIPS, _run  # noqa: E402

EDGES = [0, 150, 300, 600, 1200, 2400, 4800, 9600, 22051]


def stft(x):
    return torch.stft(torch.from_numpy(x), N_FFT, HOP, window=torch.hann_window(N_FFT),
                      return_complex=True).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="model[:stem] as in train.py")
    ap.add_argument("--pipelines", nargs="+", required=True, help="matching member pipelines")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    runner = PipelineRunner(ModelPool(capacity=1))
    K, nb = len(a.members), len(EDGES) - 1
    freqs = np.fft.rfftfreq(N_FFT, 1 / 44100)
    band = np.digitize(freqs, EDGES) - 1
    A = np.zeros((nb, K, K))
    b = np.zeros((nb, K))
    clips = sorted(p for p in CLIPS.iterdir() if (p / "mixture.wav").exists())
    for c in clips:
        mix, _ = load_audio(c / "mixture.wav")
        S = stft(load_audio(c / "acoustic_guitar.wav")[0])
        Y = [stft(_run(runner, m, mix)) for m in a.members]
        for j in range(nb):
            sel = band == j
            s = S[:, sel].ravel()
            ys = [y[:, sel].ravel() for y in Y]
            for i in range(K):
                b[j, i] += np.real(np.vdot(ys[i], s))
                for k in range(K):
                    A[j, i, k] += np.real(np.vdot(ys[i], ys[k]))
    W = np.stack([np.linalg.solve(A[j] + 1e-6 * np.eye(K) * np.trace(A[j]), b[j]) for j in range(nb)], 1)
    W = np.clip(W, 0, None)
    print("band edges (Hz):", EDGES)
    for i, m in enumerate(a.members):
        print(f"{m:28s}", " ".join(f"{w:5.2f}" for w in W[i]))
    steps, outs = [], []
    for i, p in enumerate(a.pipelines):
        st, out = member_steps(p, f"m{i}")
        steps += st
        outs.append(out)
    steps.append({"id": "ens", "ensemble": outs, "method": "band_weighted",
                  "band_weights": [[round(float(w), 4) for w in row] for row in W],
                  "crossovers": [float(e) for e in EDGES[1:-1]], "normalize": False})
    name = f"D_bandw_{a.tag}"
    pipe = {"name": name, "description": f"Strategy D: frequency-dependent least-squares weights "
            f"(fitted on {len(clips)} training clips) over {', '.join(a.pipelines)}",
            "steps": steps, "output": "ens"}
    (PIPELINE_DIR / f"{name}.yaml").write_text(yaml.safe_dump(pipe, sort_keys=False))
    (ROOT / "artifacts" / f"band_weights_{a.tag}.json").write_text(json.dumps(
        {"members": a.members, "edges": EDGES, "weights": W.tolist(), "n_clips": len(clips)}, indent=1))
    print("wrote", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
