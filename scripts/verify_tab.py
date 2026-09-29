#!/usr/bin/env python
"""Check a transcribed tab against the guitar stem (no ground truth exists for the target).

    python scripts/verify_tab.py            # outputs/target/tab/frevo_guitar_tab_notes.csv

1. Posterior support: every tab note's onset/frame evidence in the (ensemble) CRNN output, and
   how many of the individual guitar checkpoints detect it.
2. Missed notes: strong frame activations (>= 0.5 for >= 80 ms) not covered by any tab note, and
   notes that >= 2 individual checkpoints detect but the tab lacks.
3. Resynthesis: the tab (performance timing) is rendered with a plucked-string model; the
   chroma of that rendering is compared with the stem's chroma (cosine), per bar.  Chroma is
   octave- and largely timbre-independent, so low bars point at wrong or missing pitches.
4. Independent model: note-onset agreement with Basic Pitch (Spotify) on the same stem, if
   installed.
5. Playability: fret span per chord and hand shifts.
6. Repeats: a note played in one statement of a repeated passage but missing at the repeat,
   although >= 2 checkpoints hear it there.
Writes <tab dir>/verification.json, verification.md and review.json (bars to check by ear, with
reasons; shown by scripts/make_tab_check.py, which also marks the notes only one checkpoint hears).
If reports/tab_benchmark.json holds the benchmark's precision per agreement level
(scripts/bench_tab.py eval), the number of wrong notes in the tab is estimated from it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator.audio import load_audio  # noqa: E402
from acoustic_separator.tab import amt  # noqa: E402

TAB = ROOT / "outputs" / "target" / "tab"
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def read_notes(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("bar", "beat", "sixteenth", "midi", "string", "fret"):
            r[k] = int(r[k])
        for k in ("onset_s", "offset_s", "confidence"):
            r[k] = float(r[k])
    return rows


def karplus(pitch: int, dur: float, sr: int, rng) -> np.ndarray:
    f0 = 440.0 * 2 ** ((pitch - 69) / 12)
    n = int(sr * min(dur + 0.3, 3.0))
    period = sr / f0
    p = int(period)
    buf = rng.uniform(-1, 1, p)
    out = np.empty(n)
    decay = 0.996 if pitch < 52 else 0.994
    for i in range(n):  # simple KS; short notes so a python loop is fine
        out[i] = buf[i % p]
        buf[i % p] = decay * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out * np.exp(-np.arange(n) / sr / max(dur, 0.15) * 1.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", default=str(TAB))
    ap.add_argument("--name", default="frevo_guitar_tab")
    ap.add_argument("--stem", default=str(ROOT / "outputs" / "target" / "best" / "acoustic_guitar.wav"))
    args = ap.parse_args()
    tabdir = Path(args.tab)
    notes = read_notes(tabdir / f"{args.name}_notes.csv")
    meta = json.loads((tabdir / f"{args.name}.json").read_text())
    audio, sr = load_audio(args.stem)
    mono = audio.mean(0)
    rep: dict = {"n_notes": len(notes)}

    # 1-2. posterior support and missed energy ------------------------------------------
    members = list(meta.get("models", {}).get("checkpoints", {"kroma": None}))
    posts = {pth.stem.removeprefix("posteriors_"): np.load(pth) for pth in sorted(tabdir.glob("posteriors_*.npz"))}
    frame = np.mean([posts[m]["frame"] for m in members], 0)
    onset = np.mean([posts[m]["onset"] for m in members], 0)
    f = 2 ** (-meta["tuning_cents"] / 1200)
    fr = lambda t: int(round(t / f * amt.FPS))  # noqa: E731 original time -> model frame
    sup_on, sup_fr = [], []
    for n in notes:
        k = n["midi"] - amt.BEGIN_NOTE
        a = fr(n["onset_s"])
        sup_on.append(onset[max(a - 3, 0):a + 4, k].max())
        sup_fr.append(frame[a:a + 8, k].mean())
    sup_on, sup_fr = np.array(sup_on), np.array(sup_fr)
    rep["support"] = {"ensemble": members, "onset_post_median": float(np.median(sup_on)),
                      "weak_notes(onset<0.4)": int((sup_on < 0.4).sum()),
                      "frame_post_median": float(np.median(sup_fr))}
    # checkpoints with (near-)identical output count once (guitar_kroma is guitar-gaps re-saved)
    names, dup = [], {}
    for m in sorted(posts, key=lambda m: (m not in members, m)):
        same = [d for d in names if np.corrcoef(posts[m]["onset"].ravel()[::5], posts[d]["onset"].ravel()[::5])[0, 1] > 0.99]
        if same:
            dup[m] = same[0]
        else:
            names.append(m)
    votes = np.array([[posts[m]["onset"][max(fr(n["onset_s"]) - 4, 0):fr(n["onset_s"]) + 5,
                                         n["midi"] - amt.BEGIN_NOTE].max() >= 0.3 for m in names] for n in notes])
    agree = votes.sum(1)
    rep["model_agreement"] = {"checkpoints": names, "duplicates": dup,
                              "notes_by_n_checkpoints": {int(k): int((agree == k).sum()) for k in range(len(names) + 1)}}
    rep["uncertain_notes"] = [{k: n[k] for k in ("bar", "beat", "sixteenth", "string", "fret", "midi")} | {"checkpoints": int(a)}
                              for n, a in zip(notes, agree) if a <= 1]
    bench = ROOT / "reports" / "tab_benchmark.json"
    cal = json.loads(bench.read_text()).get("agreement", {}).get("separated from mix") if bench.exists() else None
    if cal and len(names) == len(cal["by_checkpoints"]):
        est, var = 0.0, 0.0
        for k, v in cal["by_checkpoints"].items():
            n_k, q = int((agree == int(k)).sum()), 1 - v["correct"] / max(v["notes"], 1)
            est += n_k * q
            var += n_k ** 2 * q * (1 - q) / max(v["notes"], 1)
        rep["expected_wrong_notes"] = {"estimate": round(est), "range_95": [round(est - 2 * var ** 0.5), round(est + 2 * var ** 0.5)],
                                       "basis": "benchmark precision per agreement level (separated from mix), "
                                                "applied to this tab's agreement counts"}
    tab_on = np.array([n["onset_s"] for n in notes])
    tab_p = np.array([n["midi"] for n in notes])
    extra: dict[tuple, set] = {}
    for m in names:
        est = amt.decode({k: posts[m][k] for k in ("onset", "offset", "frame", "velocity")}, onset_thr=0.3, frame_thr=0.3)
        est[:, :2] *= f
        for on, _, p, _ in est[(est[:, 2] >= 40) & (est[:, 2] <= 83)]:
            if not np.any((np.abs(tab_on - on) < 0.06) & (tab_p == p)):
                extra.setdefault((round(on / 0.05) * 0.05, int(p)), set()).add(m)
    extra2 = sorted((t, p, sorted(ms)) for (t, p), ms in extra.items() if len(ms) >= 2)
    rep["notes_heard_by_2plus_checkpoints_not_in_tab"] = {"count": len(extra2), "examples": extra2[:30]}
    covered = np.zeros_like(frame, bool)
    for n in notes:
        k = n["midi"] - amt.BEGIN_NOTE
        covered[max(fr(n["onset_s"]) - 5, 0):fr(n["offset_s"]) + 5, k] = True
    act = (frame >= 0.5) & ~covered
    missed = []
    for k in range(act.shape[1]):
        x = np.r_[0, act[:, k].astype(int), 0]
        st, en = np.nonzero(np.diff(x) == 1)[0], np.nonzero(np.diff(x) == -1)[0]
        for s, e in zip(st, en):
            if e - s >= 8 and 40 <= k + amt.BEGIN_NOTE <= 83:
                missed.append((round(s / amt.FPS * f, 2), k + amt.BEGIN_NOTE, round((e - s) / amt.FPS, 2)))
    rep["uncovered_activations"] = {"count": len(missed), "examples": missed[:30]}

    # 3. resynthesis correlation per bar ------------------------------------------------
    import librosa

    ssr = 22050
    stem = librosa.resample(mono, orig_sr=sr, target_sr=ssr)
    syn = np.zeros(len(stem) + ssr * 3)
    rng = np.random.default_rng(0)
    for n in notes:
        s = karplus(n["midi"], n["offset_s"] - n["onset_s"], ssr, rng)
        i = int(n["onset_s"] * ssr)
        syn[i:i + len(s)] += s * (0.4 + 0.6 * n["confidence"])
    syn = syn[:len(stem)]
    ca = librosa.feature.chroma_cqt(y=stem, sr=ssr, hop_length=512)
    cb = librosa.feature.chroma_cqt(y=syn, sr=ssr, hop_length=512)
    cos = (ca * cb).sum(0) / (np.linalg.norm(ca, axis=0) * np.linalg.norm(cb, axis=0) + 1e-9)
    rms = librosa.feature.rms(y=stem, frame_length=2048, hop_length=512)[0][: ca.shape[1]]
    active = rms > 0.02 * rms.max()  # frames where the guitar sounds
    hop_t = 512 / ssr
    bar_t: dict[int, list[float]] = {}
    for n in notes:
        bar_t.setdefault(n["bar"], []).append(n["onset_s"])
    corr = {}
    for b in sorted(bar_t):
        t0 = min(bar_t[b]) - 0.05
        t1 = min(bar_t[b + 1]) - 0.05 if b + 1 in bar_t else max(bar_t[b]) + 0.4
        a, c = int(t0 / hop_t), int(t1 / hop_t)
        m = active[a:c]
        if m.sum() > 3:
            corr[b] = float(cos[a:c][m].mean())
    vals = np.array(list(corr.values()))
    rep["chroma_cosine"] = {"median_frame": float(np.median(cos[active])),
                            "median_bar": float(np.median(vals)), "p10_bar": float(np.percentile(vals, 10)),
                            "bars_below_0.7": sorted(b for b, v in corr.items() if v < 0.7),
                            "lowest_bars": sorted(corr, key=corr.get)[:12]}

    # 4. Basic Pitch agreement -------------------------------------------------------
    try:
        import mir_eval
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model, predict

        _, _, ev = predict(args.stem, Model(build_icassp_2022_model_path(FilenameSuffix.onnx)))
        bp = np.array([[e[0], e[1], e[2]] for e in ev], float)
        bp = bp[(bp[:, 2] >= 40) & (bp[:, 2] <= 83)]
        ref = np.array([[n["onset_s"], n["offset_s"], n["midi"]] for n in notes])
        hz = lambda m: 440 * 2 ** ((m - 69) / 12)  # noqa: E731
        p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref[:, :2], hz(ref[:, 2]), bp[:, :2], hz(bp[:, 2]), offset_ratio=None)
        # pitch-class-only agreement (octave errors forgiven)
        _, _, f1c, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref[:, :2], hz(60 + ref[:, 2] % 12), bp[:, :2], hz(60 + bp[:, 2] % 12), offset_ratio=None)
        rep["basic_pitch_agreement"] = {"n_bp": len(bp), "onset_f1": float(f1), "precision_vs_tab": float(r),
                                        "recall_of_tab": float(p), "onset_f1_pitch_class": float(f1c)}
    except Exception as e:  # noqa: BLE001 - optional
        rep["basic_pitch_agreement"] = f"skipped: {e}"

    # 5. playability ---------------------------------------------------------------------
    by_q: dict = {}
    for n in notes:
        by_q.setdefault((n["bar"], n["beat"], n["sixteenth"]), []).append(n)
    spans, pos = [], []
    for k in sorted(by_q):
        fr_ = [n["fret"] for n in by_q[k] if n["fret"] > 0]
        spans.append(max(fr_) - min(fr_) if fr_ else 0)
        if fr_:
            pos.append(min(fr_))
    spans, pos = np.array(spans), np.array(pos)
    rep["playability"] = {"max_span": int(spans.max()), "chords_span_ge4": int((spans >= 4).sum()),
                          "max_fret": int(max(n["fret"] for n in notes)),
                          "median_position": float(np.median(pos)),
                          "shifts_gt_5_frets": int((np.abs(np.diff(pos)) > 5).sum())}

    c = rep["chroma_cosine"]

    # 6. repeated passages -------------------------------------------------------------
    beats = np.asarray(meta["beat_times_s"])
    cen = np.asarray(meta["swing_centres"])
    bars: dict[int, set] = {}
    for n in notes:
        bars.setdefault(n["bar"], set()).add(((n["beat"] - 1) * 4 + n["sixteenth"] - 1, n["midi"]))
    n_bars = max(bars)
    sim = lambda a, b: len(bars.get(a, set()) & bars.get(b, set())) / max(1, len(bars.get(a, set()) | bars.get(b, set())))  # noqa: E731
    pairs = set()
    for lag in range(8, n_bars):  # 8-bar windows that recur (pitch + position)
        for a in range(1, n_bars - lag - 6):
            if np.mean([sim(a + i, a + lag + i) for i in range(8)]) > 0.4:
                pairs.update((a + i, a + lag + i) for i in range(8))
    rep_miss = []
    for a, b in sorted(pairs):
        for x, y in ((a, b), (b, a)):
            if sim(x, y) < 0.4:
                continue
            for slot, p in sorted(bars.get(x, set()) - bars.get(y, set())):
                if any((s, p) in bars.get(y, set()) for s in (slot - 1, slot + 1)):
                    continue
                q = (y - 1) * 8 + slot
                k, j = divmod(q, 4)
                if k + 1 >= len(beats):
                    continue
                t = beats[k] + cen[j] * (beats[k + 1] - beats[k])
                heard = [m for m in names
                         if posts[m]["onset"][max(fr(t) - 4, 0):fr(t) + 5, p - amt.BEGIN_NOTE].max() >= 0.2]
                if len(heard) >= 2:
                    rep_miss.append((y, slot, p, x, heard))
    rep["repeats"] = {"repeated_bar_pairs": len(pairs), "missing_at_repeat_but_heard": len(set(r[:3] for r in rep_miss)),
                      "examples": sorted(set((y, s, p, x) for y, s, p, x, _ in rep_miss))[:30]}

    # review list: bars to check by ear first (strong reasons only) ------------------------
    names_ja = lambda p: NOTE_NAMES[p % 12] + str(p // 12 - 1)  # noqa: E731
    review: dict[int, list[str]] = {}
    for n, a in zip(notes, agree):
        if a <= 1:
            review.setdefault(n["bar"], []).append(
                f"{names_ja(n['midi'])}（{n['beat']}拍目）を検出したのは {len(names)} モデル中 {a} つだけ")
    for t, p, ms in extra2:
        if len(ms) >= 2:
            k = int(np.clip(np.searchsorted(beats, t) - 1, 0, len(beats) - 1))
            review.setdefault(k // 2 + 1, []).append(f"{names_ja(p)} が {len(ms)} モデルで聞こえるがタブに無い")
    for y, slot, p, x, heard in rep_miss:
        review.setdefault(y, []).append(
            f"繰り返しの {x} 小節目にある {names_ja(p)} が無い（{len(heard)} モデルがここでも検出）")
    for b in c["bars_below_0.7"]:
        review.setdefault(b, []).append("和音の響きの一致度が低い")
    review = {b: sorted(set(v)) for b, v in review.items()}
    ranked = sorted(review, key=lambda b: (-len(review[b]), b))
    rep["review_bars"] = {"count": len(review), "top": ranked[:25]}
    (tabdir / "review.json").write_text(json.dumps({str(b): review[b] for b in sorted(review)}, ensure_ascii=False, indent=1))

    (tabdir / "verification.json").write_text(json.dumps(rep, indent=2))
    (tabdir / "verification.md").write_text(markdown(rep))
    print(json.dumps(rep, indent=2))
    return 0


def markdown(rep: dict) -> str:
    s, c, p = rep["support"], rep["chroma_cosine"], rep["playability"]
    lines = ["# Tab verification (no ground truth exists for this song)", "",
             "| check | result |", "|---|---|",
             f"| notes in the tab | {rep['n_notes']} |",
             f"| CRNN onset evidence per note (median) | {s['onset_post_median']:.2f} |",
             f"| notes with weak onset evidence (< 0.4) | {s['weak_notes(onset<0.4)']} |",
             f"| strong activations not in the tab (>= 80 ms) | {rep['uncovered_activations']['count']} |",
             f"| chroma cosine stem vs. resynthesised tab (median frame / bar) | {c['median_frame']:.2f} / {c['median_bar']:.2f} |",
             f"| bars with chroma cosine < 0.7 | {', '.join(map(str, c['bars_below_0.7'])) or 'none'} |"]
    bp = rep["basic_pitch_agreement"]
    if isinstance(bp, dict):
        lines.append(f"| agreement with Basic Pitch (onset F1, 50 ms) | {bp['onset_f1']:.2f} "
                     f"(pitch class {bp['onset_f1_pitch_class']:.2f}) |")
    ma = rep["model_agreement"]
    lines.append(f"| tab notes detected by k of {len(ma['checkpoints'])} guitar checkpoints (k: count) | "
                 + ", ".join(f"{k}: {v}" for k, v in ma["notes_by_n_checkpoints"].items()) + " |")
    ew = rep.get("expected_wrong_notes")
    if ew:
        lines.append(f"| expected wrong notes (benchmark precision per agreement level) | about {ew['estimate']} "
                     f"({ew['range_95'][0]}-{ew['range_95'][1]}) of {rep['n_notes']} |")
    lines.append(f"| notes heard by >= 2 checkpoints but not in the tab | {rep['notes_heard_by_2plus_checkpoints_not_in_tab']['count']} |")
    lines.append(f"| repeated passages: note missing at the repeat although heard | {rep['repeats']['missing_at_repeat_but_heard']} |")
    lines += [f"| max fret span in a chord / max fret | {p['max_span']} / {p['max_fret']} |",
              f"| hand shifts > 5 frets | {p['shifts_gt_5_frets']} |", "",
              f"Bars to double-check by ear first (most review reasons, see review.json): "
              f"{', '.join(map(str, rep['review_bars']['top']))}.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
