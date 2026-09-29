#!/usr/bin/env python
"""Listening aids for checking the tab against the recording by ear.

    python scripts/make_tab_check.py        # -> outputs/target/tab/check/

Writes
  tab_synth.mp3  the tab rendered exactly as written (grid timing incl. the swing feel, the
                 recording's tuning, a string stops when it is plucked again)
  stem.mp3       the extracted guitar (outputs/target/best/acoustic_guitar.wav)
  ab.mp3         left ear = extracted guitar, right ear = tab rendering (use headphones)
  index.html     player that highlights the current 16th in the tab; A/B modes, slow-down
                 without pitch change, click a bar to jump, loop a bar or a range; bars flagged by
                 scripts/verify_tab.py (review.json) are marked with their reasons, and notes only
                 one checkpoint hears (verification.json) are marked in the tab
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator.audio import ffmpeg_exe, load_audio  # noqa: E402

TAB = ROOT / "outputs" / "target" / "tab"
SR = 44100
ROWS = ["e", "B", "G", "D", "A", "E"]


def pluck(f0: float, dur: float, amp: float, rng: np.random.Generator) -> np.ndarray:
    """Additive plucked nylon string: pluck-position harmonic weights, frequency-dependent decay,
    slight inharmonicity, a short noise transient and a damped release at ``dur``."""
    release = 0.06
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    beta = 0.16 + 0.02 * rng.random()
    t60 = float(np.clip(3.5 * (110.0 / f0) ** 0.5, 0.8, 5.0))
    y = np.zeros(n)
    for k in range(1, 25):
        fk = k * f0 * np.sqrt(1 + 1.5e-5 * k * k)
        if fk > 9000:
            break
        a = abs(np.sin(np.pi * k * beta)) / k ** 1.15
        tk = t60 / (1 + 0.09 * (k - 1) ** 1.5)
        y += a * np.exp(-6.91 * t / tk) * np.sin(2 * np.pi * fk * t + rng.random() * 6.283)
    att = int(0.003 * SR)
    y[:att] *= np.linspace(0, 1, att)
    noise = rng.standard_normal(int(0.006 * SR)) * np.exp(-np.arange(int(0.006 * SR)) / (0.0015 * SR))
    y[:len(noise)] += 0.15 * noise
    end = int(dur * SR)
    y[end:] *= np.exp(-np.arange(n - end) / (0.012 * SR))
    return amp * y


def render(notes: list[dict], times: np.ndarray, cents: float, length: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.zeros(length + SR * 6)
    by_string: dict[int, list[tuple[float, dict]]] = {}
    for n in notes:
        by_string.setdefault(n["string"], []).append((times[n["q"]], n))
    for s, lst in by_string.items():
        lst.sort(key=lambda x: x[0])
        for i, (t, n) in enumerate(lst):
            dur = max(n["offset_s"] - n["onset_s"], 0.08)
            if i + 1 < len(lst):
                dur = min(dur, lst[i + 1][0] - t)  # the string is plucked again
            f0 = 440.0 * 2 ** ((n["midi"] - 69) / 12 + cents / 1200)
            y = pluck(f0, max(dur, 0.03), 0.55 + 0.45 * n["confidence"], rng)
            i0 = int(t * SR)
            out[i0:i0 + len(y)] += y
    from scipy.signal import butter, sosfilt

    out = sosfilt(butter(1, 6000, fs=SR, output="sos"), out)  # nylon warmth
    return out[:length]


def mp3(path: Path, x: np.ndarray, kbps: int) -> None:
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = np.stack([x, x])
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-c:a", "libmp3lame", "-b:a", f"{kbps}k", str(path)],
                   input=np.ascontiguousarray(x.T).tobytes(), check=True)


def grid_times(meta: dict, n16: int) -> np.ndarray:
    beats = np.asarray(meta["beat_times_s"])
    c = np.asarray(meta["swing_centres"])
    ibi = np.median(np.diff(beats[-8:]))
    while len(beats) < n16 // 4 + 2:
        beats = np.append(beats, beats[-1] + ibi)
    q = np.arange(n16 + 1)
    k, j = q // 4, q % 4
    return beats[k] + c[j] * (beats[k + 1] - beats[k])


def page(meta: dict, notes: list[dict], chords: dict[int, str], times: np.ndarray, n_bars: int,
         review: dict[int, list[str]] | None = None, bars_per_line: int = 4,
         uncertain: set[tuple[int, int]] | None = None, uncertain_correct: float | None = None) -> str:
    """``uncertain``: (16th index, tab row with 0 = high e) of notes to mark; ``uncertain_correct``:
    the share of such notes that were right on the benchmark, for the legend."""
    review, uncertain = review or {}, uncertain or set()
    cell: dict[int, dict[int, int]] = {}
    for n in notes:
        cell.setdefault(n["q"], {})[5 - n["string"]] = n["fret"]
    systems = []
    for b0 in range(0, n_bars, bars_per_line):
        bars = range(b0, min(n_bars, b0 + bars_per_line))
        head, count = ["  "], ["  "]
        rows = [[f"{r}|"] for r in ROWS]
        chord_line = [" "] * (2 + len(bars) * 25 + 16)
        cursor = 0
        for bi, b in enumerate(bars):
            t = times[b * 8]
            label = f"{b + 1} ({int(t // 60)}:{t % 60:04.1f})"
            why = review.get(b + 1)
            attr = f' title="{html.escape(chr(10).join(why))}"' if why else ""
            mark = ("!!" if len(why) >= 2 else "!") if why else ""
            cls = (" rv" if len(why) >= 2 else " rv1") if why else ""
            head.append(f'<span class="bl{cls}" data-bar="{b}"{attr}>{html.escape(label)}{mark}</span>'
                        + " " * (25 - len(label) - len(mark)))
            for j in range(8):
                sym = chords.get(b * 8 + j)
                if sym:  # system-wide placement: long names may run into the next bar
                    pos = max(2 + bi * 25 + j * 3, cursor)
                    chord_line[pos:pos + len(sym)] = list(sym)
                    cursor = pos + len(sym) + 1
            for j in range(8):
                q = b * 8 + j
                cnt = (str(j // 4 + 1) if j % 4 == 0 else "e+a"[j % 4 - 1]).ljust(3)
                count.append(f'<span class="c q{q}">{cnt}</span>')
                for r in range(6):
                    fr = cell.get(q, {}).get(r)
                    txt = (str(fr) if fr is not None else "").ljust(3, "-")
                    if fr is not None and (q, r) in uncertain:
                        txt = f'<b class="u">{fr}</b>' + txt[len(str(fr)):]
                    rows[r].append(f'<span class="c q{q}">{txt}</span>')
            count.append(" ")
            for r in range(6):
                rows[r].append("|")
        chord_line = ['<span class="ch">' + html.escape("".join(chord_line).rstrip()) + "</span>"]
        lines = ["".join(head), "".join(chord_line)] + ["".join(r) for r in rows] + ["".join(count)]
        systems.append(f'<div class="sys" data-first="{b0}">' + "\n".join(lines) + "</div>")
    data = {"times": [round(float(x), 4) for x in times], "nBars": n_bars,
            "review": sorted((b - 1 for b in review), key=lambda b: (-len(review[b + 1]), b)),
            "priority": sorted(b - 1 for b in review if len(review[b]) >= 2)}
    legend = ""
    if uncertain:
        legend = ("<br>赤い波線のフレット番号：3 つの採譜モデルのうち 1 つしか検出しなかった音（"
                  f"{len(uncertain)} 音）。")
        if uncertain_correct is not None:
            legend += f"再現実験では、この種の音で正しかったのは約 {round(uncertain_correct * 100)} % でした。"
        legend += "<br>"
    return TEMPLATE.replace("__SYSTEMS__", "\n".join(systems)).replace("__DATA__", json.dumps(data)) \
        .replace("__TITLE__", html.escape(f"{meta.get('title', 'Frevo!')} タブ譜チェッカー")) \
        .replace("__LEGEND__", legend)


TEMPLATE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a64;--line:#dcd8cf;--hl:#ffd54a;--bar:#fff3c4;--accent:#1f6feb;--chord:#b35900;--unc:#c62828}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--fg:#e9e6df;--muted:#9c998f;--line:#35332f;--hl:#7a5f00;--bar:#2b2616;--accent:#5aa2ff;--chord:#ffae57;--unc:#ff7b72}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:17px;margin:0 0 6px}
.row{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
button,select{font:inherit;padding:5px 10px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.hint{color:var(--muted);font-size:13px;margin-top:6px}
main{padding:6px 16px 120px}
.sys{font:13px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre;overflow-x:auto;margin:14px 0;padding:6px 0;border-bottom:1px dashed var(--line)}
.bl{color:var(--accent);cursor:pointer;text-decoration:underline dotted}
.ch{color:var(--chord);font-weight:600}
.c.cur{background:var(--hl)}
.sys.cur{background:var(--bar)}
.bl.loop{background:var(--accent);color:#fff}
.bl.rv{color:#c62828;font-weight:700}
.bl.rv1{color:#d9730d;font-weight:600}
.u{color:var(--unc);text-decoration:underline wavy;text-underline-offset:2px}
#rvlist a{color:#c62828;cursor:pointer;margin-right:6px}
#time{font-variant-numeric:tabular-nums;min-width:120px;display:inline-block}
</style></head><body>
<header>
<h1>__TITLE__</h1>
<div class="row">
  <button id="play">▶ 再生</button>
  <span id="time">0:00.0 / 小節 -</span>
  <span>音源:</span>
  <button class="mode on" data-mode="stem">元のギター</button>
  <button class="mode" data-mode="tab">タブの合成音</button>
  <button class="mode" data-mode="ab">左右比較（左=元 右=タブ）</button>
  <span>速さ:</span>
  <select id="rate"><option value="1">100%</option><option value="0.85">85%</option><option value="0.7">70%</option><option value="0.5">50%</option></select>
  <button id="loop">小節ループ</button>
  <button id="nextrv">次の要確認小節 ▶</button>
</div>
</header>
<main>
<div class="hint" id="rvlist"></div>
<div class="hint">自動チェックで要確認になった小節：赤（!!）＝理由が 2 つ以上で優先、オレンジ（!）＝理由 1 つ。マウスを乗せると理由が出ます。__LEGEND__
小節番号をクリックでその小節へ移動。ループ中は Shift+クリックで範囲を広げられます。
キー: スペース=再生/停止、←/→=前後の小節、L=ループ。左右比較はヘッドホン推奨。</div>
__SYSTEMS__
</main>
<script>
const DATA = __DATA__;
const T = DATA.times, FILES = {stem: "stem.mp3", tab: "tab_synth.mp3", ab: "ab.mp3"};
const audio = new Audio(FILES.stem); audio.preload = "auto";
let mode = "stem", rate = 1, cur = -1, loop = null, curBar = 0;
const cells = {}; document.querySelectorAll(".c").forEach(e => { const q = +[...e.classList].find(c => /^q\\d+$/.test(c)).slice(1); (cells[q] = cells[q] || []).push(e); });
const systems = [...document.querySelectorAll(".sys")];
function qAt(t){ let lo = 0, hi = T.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (T[m] <= t) lo = m; else hi = m - 1; } return lo; }
function fmt(t){ return Math.floor(t / 60) + ":" + (t % 60).toFixed(1).padStart(4, "0"); }
function barStart(b){ return T[Math.max(0, Math.min(b * 8, T.length - 1))]; }
function seek(t){ audio.currentTime = Math.max(0, t); }
function setMode(m){ if (m === mode) return; const t = audio.currentTime, playing = !audio.paused; mode = m;
  document.querySelectorAll(".mode").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
  audio.src = FILES[m]; audio.addEventListener("loadedmetadata", () => { audio.currentTime = t; audio.playbackRate = rate; audio.preservesPitch = true; if (playing) audio.play(); }, {once: true}); }
function markLoop(){ document.querySelectorAll(".bl").forEach(e => { const b = +e.dataset.bar; e.classList.toggle("loop", !!loop && b >= loop[0] && b <= loop[1]); });
  document.getElementById("loop").classList.toggle("on", !!loop); }
function tick(){ const t = audio.currentTime;
  if (loop && t >= barStart(loop[1] + 1)) { seek(barStart(loop[0]) - 0.05); }
  const q = t < T[0] ? -1 : qAt(t);
  if (q !== cur) { (cells[cur] || []).forEach(e => e.classList.remove("cur")); (cells[q] || []).forEach(e => e.classList.add("cur")); cur = q;
    const c0 = (cells[q] || [])[0];
    if (c0) { const s = c0.closest(".sys"), cr = c0.getBoundingClientRect(), sr = s.getBoundingClientRect();  // narrow screens: follow sideways
      if (cr.left < sr.left + 16 || cr.right > sr.right - 16) s.scrollLeft += cr.left - sr.left - sr.width / 4; }
    const b = q < 0 ? 0 : Math.floor(q / 8); if (b !== curBar || q < 0) { curBar = b;
      systems.forEach(s => s.classList.remove("cur")); const s = systems[Math.floor(b / 4)];
      if (s && q >= 0) { s.classList.add("cur"); const r = s.getBoundingClientRect(); if (r.top < 120 || r.bottom > innerHeight - 20) s.scrollIntoView({block: "center", behavior: "smooth"}); } } }
  document.getElementById("time").textContent = fmt(t) + " / 小節 " + (q < 0 ? "-" : curBar + 1);
  requestAnimationFrame(tick); }
requestAnimationFrame(tick);
const rv = DATA.review;
const pri = new Set(DATA.priority);
document.getElementById("rvlist").innerHTML = rv.length ? "要確認の小節（優先 " + pri.size + " / 全 " + rv.length + "、優先度順）: " + rv.map(b => '<a data-bar="' + b + '"' + (pri.has(b) ? "" : ' style="color:#d9730d"') + ">" + (b + 1) + "</a>").join(" ") : "";
document.querySelectorAll("#rvlist a").forEach(a => a.onclick = () => { const b = +a.dataset.bar; if (loop) { loop = [b, b]; markLoop(); } seek(barStart(b) - 0.3); if (audio.paused) toggle(); });
const rvPos = [...rv].sort((a, b) => a - b);
document.getElementById("nextrv").onclick = () => { const b = rvPos.find(x => x > curBar) ?? rvPos[0]; if (b === undefined) return;
  if (loop) { loop = [b, b]; markLoop(); } seek(barStart(b) - 0.3); if (audio.paused) toggle(); };
const playBtn = document.getElementById("play");
function toggle(){ if (audio.paused) { audio.playbackRate = rate; audio.preservesPitch = true; audio.play(); } else audio.pause(); }
audio.addEventListener("play", () => playBtn.textContent = "❚❚ 停止"); audio.addEventListener("pause", () => playBtn.textContent = "▶ 再生");
playBtn.onclick = toggle;
document.querySelectorAll(".mode").forEach(b => b.onclick = () => setMode(b.dataset.mode));
document.getElementById("rate").onchange = e => { rate = +e.target.value; audio.playbackRate = rate; audio.preservesPitch = true; };
document.getElementById("loop").onclick = () => { loop = loop ? null : [curBar, curBar]; markLoop(); };
document.querySelectorAll(".bl").forEach(e => e.onclick = ev => { const b = +e.dataset.bar;
  if (loop && ev.shiftKey) { loop = [Math.min(loop[0], b), Math.max(loop[1], b)]; } else if (loop) { loop = [b, b]; }
  markLoop(); seek(barStart(b) - 0.3); if (audio.paused) toggle(); });
addEventListener("keydown", e => { if (e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); toggle(); }
  else if (e.code === "ArrowRight") { seek(barStart(curBar + 1) - 0.1); if (loop) { loop = [curBar + 1, curBar + 1]; markLoop(); } }
  else if (e.code === "ArrowLeft") { seek(barStart(Math.max(curBar - 1, 0)) - 0.1); if (loop) { loop = [Math.max(curBar - 1, 0), Math.max(curBar - 1, 0)]; markLoop(); } }
  else if (e.key === "l" || e.key === "L") document.getElementById("loop").click(); });
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", default=str(TAB))
    ap.add_argument("--name", default="frevo_guitar_tab")
    args = ap.parse_args()
    tabdir = Path(args.tab)
    meta = json.loads((tabdir / f"{args.name}.json").read_text())
    with open(tabdir / f"{args.name}_notes.csv") as f:
        notes = list(csv.DictReader(f))
    for n in notes:
        n["q"] = (int(n["bar"]) - 1) * 8 + (int(n["beat"]) - 1) * 4 + int(n["sixteenth"]) - 1
        n["string"] = 6 - int(n["string"])  # CSV: 1 = high e -> 0 = low E
        for k in ("midi", "fret"):
            n[k] = int(n[k])
        for k in ("onset_s", "offset_s", "confidence"):
            n[k] = float(n[k])
    n_bars = max(n["q"] for n in notes) // 8 + 1
    times = grid_times(meta, n_bars * 8)
    stem, sr = load_audio(ROOT / meta["stem"])
    mono = stem.mean(0)
    syn = render(notes, times, meta["tuning_cents"], len(mono))
    act = np.abs(mono) > 0.01 * np.abs(mono).max()
    syn *= np.sqrt(np.mean(mono[act] ** 2) / (np.mean(syn[act] ** 2) + 1e-12))
    out = tabdir / "check"
    out.mkdir(exist_ok=True)
    mp3(out / "tab_synth.mp3", syn, 128)
    mp3(out / "stem.mp3", stem, 160)
    mp3(out / "ab.mp3", np.stack([mono, syn]), 160)
    chords = {int(k): v for k, v in meta.get("chord_symbols", {}).items()}
    rv_path = tabdir / "review.json"
    review = {int(k): v for k, v in json.loads(rv_path.read_text()).items()} if rv_path.exists() else {}
    ver_path, bench_path = tabdir / "verification.json", ROOT / "reports" / "tab_benchmark.json"
    uncertain = {((n["bar"] - 1) * 8 + (n["beat"] - 1) * 4 + n["sixteenth"] - 1, n["string"] - 1)  # CSV string 1 = e
                 for n in json.loads(ver_path.read_text()).get("uncertain_notes", [])} if ver_path.exists() else set()
    share = None
    if bench_path.exists():
        one = json.loads(bench_path.read_text()).get("agreement", {}).get("separated from mix", {}) \
            .get("by_checkpoints", {}).get("1")
        share = one["correct"] / one["notes"] if one and one["notes"] else None
    (out / "index.html").write_text(page({"title": "Frevo!"}, notes, chords, times, n_bars, review,
                                         uncertain=uncertain, uncertain_correct=share))
    print(f"wrote {out}: tab_synth.mp3, stem.mp3, ab.mp3, index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
