#!/usr/bin/env python
"""Leakage vs preservation plot (PRD §17): one point per experiment.

x = mean excess-energy leakage (dB, lower = cleaner), y = target retention (higher = less
guitar destroyed), colour = validation SDR. Writes reports/tradeoff.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator import tracking  # noqa: E402


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in tracking.read_results() if r.get("leakage") and r.get("target_retention")]
    if not rows:
        print("no results")
        return 0
    x = [float(r["leakage"]) for r in rows]
    y = [float(r["target_retention"]) for r in rows]
    c = [float(r["sdr"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=60, edgecolor="k", linewidth=0.5)
    champ = tracking.current_champion()
    for r, xi, yi in zip(rows, x, y):
        lab = r["experiment"][:6] + " " + r["pipeline"].replace("_wave_mean", "")[:22]
        bold = champ and champ["experiment"] == r["experiment"]
        ax.annotate(lab + (" ★" if bold else ""), (xi, yi), fontsize=6.5, xytext=(3, 3),
                    textcoords="offset points", weight="bold" if bold else "normal")
    ax.set_xlabel("leakage: excess energy re. estimate (dB)  ← cleaner")
    ax.set_ylabel("target retention (fraction of guitar TF energy kept)  ↑ less loss")
    ax.set_title("Leakage vs preservation on the ground-truth validation set")
    fig.colorbar(sc, label="validation SDR (dB)")
    ax.grid(alpha=0.3)
    out = ROOT / "reports" / "tradeoff.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
