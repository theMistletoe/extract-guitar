"""HTML report with synchronized original / target / residual players and spectrograms."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np

SPEC_NFFT = 4096
SPEC_HOP = 1024


def _spec_db(x: np.ndarray, sr: int) -> np.ndarray:
    import torch

    mono = torch.from_numpy(np.ascontiguousarray(x.mean(0), dtype=np.float32))
    S = torch.stft(mono, SPEC_NFFT, SPEC_HOP, window=torch.hann_window(SPEC_NFFT),
                   return_complex=True).abs().numpy()
    return 20 * np.log10(S + 1e-6)


def save_spectrogram(x: np.ndarray, sr: int, path: Path, title: str, vmax: float | None = None,
                     fmax: float = 16000.0) -> float:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = _spec_db(x, sr)
    vmax = vmax if vmax is not None else float(np.percentile(D, 99.9))
    freqs = np.fft.rfftfreq(SPEC_NFFT, 1 / sr)
    keep = freqs <= fmax
    dur = x.shape[-1] / sr
    fig, ax = plt.subplots(figsize=(14, 3.2), dpi=90)
    ax.imshow(D[keep], origin="lower", aspect="auto", cmap="magma", vmin=vmax - 80, vmax=vmax,
              extent=[0, dur, 0, freqs[keep][-1] / 1000], interpolation="nearest")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("kHz")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlim(0, dur)
    ax.set_xlabel("s")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return vmax


def save_waveform(signals: dict[str, np.ndarray], sr: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 2.2), dpi=90)
    colors = {"original": "#9aa0a6", "acoustic_guitar": "#d9822b", "residual": "#2b83d9"}
    for name, x in signals.items():
        mono = x.mean(0)
        hop = max(1, len(mono) // 3000)
        env = np.abs(mono[: len(mono) // hop * hop]).reshape(-1, hop).max(1)
        t = np.arange(len(env)) * hop / sr
        ax.fill_between(t, -env, env, alpha=0.55, lw=0, color=colors.get(name), label=name)
    ax.set_xlim(0, signals["original"].shape[-1] / sr)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def segment_diagnostics(mix: np.ndarray, target: np.ndarray, sr: int, win_s: float = 2.0) -> list[dict]:
    """Per-window energy of target and residual relative to the mix (find problem regions)."""
    res = mix - target
    n = int(win_s * sr)
    rows = []
    for s in range(0, mix.shape[-1] - n + 1, n):
        e_m = np.mean(mix[:, s:s + n] ** 2) + 1e-12
        rows.append({
            "t": s / sr,
            "target_db": float(10 * np.log10(np.mean(target[:, s:s + n] ** 2) / e_m + 1e-12)),
            "residual_db": float(10 * np.log10(np.mean(res[:, s:s + n] ** 2) / e_m + 1e-12)),
        })
    return rows


def _rel(p: Path, base: Path) -> str:
    return os.path.relpath(p, base).replace(os.sep, "/")


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;max-width:1400px;color:#222}
h1{font-size:22px} h2{font-size:18px;margin-top:36px;border-bottom:1px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;font-size:13px;margin:8px 0} td,th{border:1px solid #ddd;padding:4px 8px;text-align:right}
th{background:#f4f4f4} td:first-child,th:first-child{text-align:left}
.best{background:#fff4e0;font-weight:600}
.cand{margin:18px 0 30px} .players{display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:13px}
.spec{position:relative} .spec img{width:100%;display:block;cursor:crosshair}
.cursor{position:absolute;top:0;bottom:0;width:2px;background:#0f0;pointer-events:none}
code{background:#f4f4f4;padding:1px 4px} .muted{color:#777;font-size:12px}
"""

JS = """
function syncGroup(g){
  const auds=[...document.querySelectorAll('audio[data-g="'+g+'"]')];
  let active=null;
  auds.forEach(a=>{
    a.addEventListener('play',()=>{active=a;auds.forEach(b=>{if(b!==a){b.muted=true;b.currentTime=a.currentTime;b.play();}});a.muted=false;});
    a.addEventListener('pause',()=>{if(active===a)auds.forEach(b=>{if(b!==a)b.pause();});});
    a.addEventListener('timeupdate',()=>{document.querySelectorAll('.cursor[data-g="'+g+'"]').forEach(c=>{const img=c.parentElement.querySelector('img');c.style.left=(a.currentTime/a.duration*img.clientWidth*0.93+img.clientWidth*0.045)+'px';});});
  });
}
function solo(g,idx){const auds=[...document.querySelectorAll('audio[data-g="'+g+'"]')];auds.forEach((a,i)=>a.muted=(i!==idx));}
function seek(ev,g){const img=ev.target;const r=img.getBoundingClientRect();const frac=Math.min(1,Math.max(0,((ev.clientX-r.left)/r.width-0.045)/0.93));
  document.querySelectorAll('audio[data-g="'+g+'"]').forEach(a=>{a.currentTime=frac*a.duration;});}
document.addEventListener('DOMContentLoaded',()=>{new Set([...document.querySelectorAll('audio[data-g]')].map(a=>a.dataset.g)).forEach(syncGroup);});
"""


def build_report(out_html: Path, title: str, original: Path, candidates: list[dict],
                 summary_rows: list[dict] | None = None, intro_md: str = "") -> Path:
    """candidates: [{id, label, acoustic (Path), residual (Path), metrics (dict),
    info (dict: model/checkpoint/params/runtime/experiment)}]."""
    from .audio import load_audio

    base = out_html.parent
    img_dir = base / "report_assets"
    img_dir.mkdir(parents=True, exist_ok=True)
    mix, sr = load_audio(original)
    vmax = save_spectrogram(mix, sr, img_dir / "original.png", "original")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
             f"<style>{CSS}</style><script>{JS}</script></head><body>",
             f"<h1>{html.escape(title)}</h1>"]
    if intro_md:
        parts.append(f"<div>{intro_md}</div>")
    if summary_rows:
        keys = list(summary_rows[0].keys())
        parts.append("<h2>Candidate comparison</h2><table><tr>" +
                     "".join(f"<th>{html.escape(k)}</th>" for k in keys) + "</tr>")
        for r in summary_rows:
            cls = " class='best'" if r.get("_best") else ""
            parts.append(f"<tr{cls}>" + "".join(
                f"<td>{html.escape(f'{v:.3f}' if isinstance(v, float) else str(v))}</td>"
                for k, v in r.items()) + "</tr>")
        parts.append("</table>")
    parts.append("<h2>Original</h2><div class='spec'>"
                 f"<img src='{_rel(img_dir / 'original.png', base)}'></div>"
                 f"<audio controls preload='none' src='{_rel(original, base)}'></audio>")
    for c in candidates:
        g = c["id"]
        acc, _ = load_audio(c["acoustic"])
        res = mix[:, : acc.shape[-1]] - acc
        save_spectrogram(acc, sr, img_dir / f"{g}_acoustic.png", f"{g}: extracted acoustic guitar", vmax)
        save_spectrogram(res, sr, img_dir / f"{g}_residual.png", f"{g}: residual (mix - acoustic)", vmax)
        save_waveform({"original": mix, "acoustic_guitar": acc, "residual": res}, sr,
                      img_dir / f"{g}_wave.png")
        diag = segment_diagnostics(mix[:, : acc.shape[-1]], acc, sr)
        parts.append(f"<div class='cand'><h2>{html.escape(c.get('label', g))}</h2>")
        info = c.get("info", {})
        if info:
            parts.append("<table>" + "".join(
                f"<tr><th>{html.escape(str(k))}</th><td style='text-align:left'><code>"
                f"{html.escape(json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v)}"
                f"</code></td></tr>" for k, v in info.items()) + "</table>")
        m = c.get("metrics", {})
        if m:
            parts.append("<table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in m) +
                         "</tr><tr>" + "".join(
                             f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
                             for v in m.values()) + "</tr></table>")
        parts.append("<div class='players'>"
                     f"<span>original <audio controls preload='none' data-g='{g}' src='{_rel(original, base)}'></audio></span>"
                     f"<span>acoustic <audio controls preload='none' data-g='{g}' src='{_rel(Path(c['acoustic']), base)}'></audio></span>")
        if c.get("residual"):
            parts.append(f"<span>residual <audio controls preload='none' data-g='{g}' src='{_rel(Path(c['residual']), base)}'></audio></span>")
        parts.append(f"<span>solo: <button onclick=\"solo('{g}',0)\">orig</button>"
                     f"<button onclick=\"solo('{g}',1)\">acoustic</button>"
                     f"<button onclick=\"solo('{g}',2)\">residual</button></span></div>")
        for kind in ("acoustic", "residual"):
            parts.append(f"<div class='spec'><img onclick=\"seek(event,'{g}')\" "
                         f"src='{_rel(img_dir / f'{g}_{kind}.png', base)}'>"
                         f"<div class='cursor' data-g='{g}'></div></div>")
        parts.append(f"<img style='width:100%' src='{_rel(img_dir / f'{g}_wave.png', base)}'>")
        worst = sorted(diag, key=lambda r: -r["target_db"])[:3]
        parts.append("<p class='muted'>Loudest extracted segments (check for leakage): " +
                     ", ".join(f"{r['t']:.0f}s ({r['target_db']:.1f} dB rel. mix)" for r in worst) +
                     "</p></div>")
    parts.append("</body></html>")
    out_html.write_text("\n".join(parts))
    return out_html
