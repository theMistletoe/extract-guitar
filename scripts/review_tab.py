#!/usr/bin/env python
"""Agent review of the tab: build review batches from the comparison, and turn verdicts into edits.

    python scripts/review_tab.py pack      # compare/ flags -> review items in batches (prints folder, work dir, batch ids)
    python scripts/review_tab.py collect   # verdicts -> <name>_edits.json (applied by transcribe_tab.py) + compare/agent_verdicts.csv

Used by the ``tab-review`` workflow (.claude/workflows/tab-review.js), in between
scripts/compare_tab_audio.py (measurement) and the regeneration of the tab.

pack: tab notes with P(wrong) >= --thr-note and missing-note candidates with P(real) >= --thr-missing
(scripts/compare_tab_audio.py) become review items with their evidence fields and musical context
(bar, beat, 16th, chord symbol, bars that repeat this bar), in time order, ~15 per batch.  The
review folder holds the analysis state the ``probe`` command reads (linked, not copied) and a
notes.csv with bar positions.

collect: every item is judged by an "acoustic" and a "skeptic" reviewer (verdict files
verdicts_<lens>_<batch>.json in the work dir, or --results with a workflow's return value).  The
rule, fixed on the blind benchmark test (reports/tab_compare_agents.md) before the target's verdicts
were read: a tab note is corrected when the acoustic reviewer calls it wrong (its fix is applied);
a missing note is added only when both reviewers call it real.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

TAB = ROOT / "outputs" / "target" / "tab"
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
NOTE_FIELDS = ["post_ens", "votes", "bp", "alt_octave", "overtone", "reattack", "n_chord", "dur", "flux_s", "flux_r", "resid"]
MISS_FIELDS = ["sources", "n_models", "bp", "post_ens", "post_max", "harmonic", "octave", "same_pc", "flux_s", "flux_r", "resid",
               "same_pitch_tab_ms"]
LENSES = ("acoustic", "skeptic")


def note_name(p: int) -> str:
    return NAMES[p % 12] + str(p // 12 - 1)


def repeats(tab: list[dict]) -> dict[int, set[int]]:
    """Bars that repeat each bar: 8-bar windows whose (position, pitch) sets overlap > 0.4 (as verify_tab.py)."""
    bars: dict[int, set] = {}
    for t in tab:
        bars.setdefault(int(t["bar"]), set()).add(((int(t["beat"]) - 1) * 4 + int(t["sixteenth"]) - 1, int(t["midi"])))
    nb = max(bars)
    sim = lambda a, b: len(bars.get(a, set()) & bars.get(b, set())) / max(1, len(bars.get(a, set()) | bars.get(b, set())))  # noqa: E731
    rep: dict[int, set[int]] = {}
    for lag in range(8, nb):
        for a in range(1, nb - lag - 6):
            if np.mean([sim(a + i, a + lag + i) for i in range(8)]) > 0.4:
                for i in range(8):
                    rep.setdefault(a + i, set()).add(a + lag + i)
                    rep.setdefault(a + lag + i, set()).add(a + i)
    return rep


def pack(args) -> dict:
    import render_tab_audio as rta

    tabdir, cmp = Path(args.tab), Path(args.tab) / "compare"
    folder, work = Path(args.folder or cmp / "review"), Path(args.work or cmp / "review" / "work")
    work.mkdir(parents=True, exist_ok=True)
    for f in ("state.npz", "basic_pitch_stem.json"):
        link = folder / f
        if link.is_symlink() or link.exists():
            link.unlink()
        if (cmp / f).exists():
            link.symlink_to((cmp / f).resolve())
    tab = list(csv.DictReader(open(tabdir / f"{args.name}_notes.csv")))
    meta = json.loads((tabdir / f"{args.name}.json").read_text())
    chords = {int(k): v for k, v in meta.get("chord_symbols", {}).items()}
    N = list(csv.DictReader(open(cmp / "notes.csv")))
    M = list(csv.DictReader(open(cmp / "missing.csv")))
    assert len(N) == len(tab), "compare/ is out of date: re-run scripts/compare_tab_audio.py"
    with open(folder / "notes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["onset_s", "offset_s", "pitch", "string", "fret", "bar", "beat", "sixteenth"])
        for r, t in zip(N, tab):
            w.writerow([r["onset_s"], r["offset_s"], r["pitch"], r["string"], r["fret"], t["bar"], t["beat"], t["sixteenth"]])
    rep = repeats(tab)
    beats, cen = np.asarray(meta["beat_times_s"]), np.asarray(meta["swing_centres"])

    def chord_at(bar: int, slot: int) -> str:
        ks = [k for k in chords if k <= (bar - 1) * 8 + slot]
        return chords[max(ks)] if ks else ""

    items = []
    for r, t in zip(N, tab):
        if float(r["p_wrong"]) < args.thr_note:
            continue
        bar, slot = int(t["bar"]), (int(t["beat"]) - 1) * 4 + int(t["sixteenth"]) - 1
        items.append({"id": "n" + hashlib.sha1(f"{r['onset_s']}/{r['pitch']}".encode()).hexdigest()[:7], "type": "tab_note",
                      "t": float(r["onset_s"]), "pitch": int(r["pitch"]), "name": note_name(int(r["pitch"])),
                      "string": int(r["string"]), "fret": int(r["fret"]), "bar": bar, "beat": int(t["beat"]),
                      "sixteenth": int(t["sixteenth"]), "chord": chord_at(bar, slot), "repeats_of_this_bar": sorted(rep.get(bar, [])),
                      "features": {k: r[k] for k in NOTE_FIELDS}, "p_model": float(r["p_wrong"])})
    for c in M:
        if float(c["p_real"]) < args.thr_missing:
            continue
        q = int(round(float(rta.time_to_q(np.array([float(c["onset_s"])]), beats, cen)[0])))
        bar, slot = q // 8 + 1, q % 8
        if bar < 1:
            continue
        items.append({"id": "m" + hashlib.sha1(f"{c['onset_s']}/{c['pitch']}".encode()).hexdigest()[:7], "type": "missing",
                      "t": float(c["onset_s"]), "pitch": int(c["pitch"]), "name": note_name(int(c["pitch"])), "bar": bar,
                      "beat": slot // 4 + 1, "sixteenth": slot % 4 + 1, "chord": chord_at(bar, slot),
                      "repeats_of_this_bar": sorted(rep.get(bar, [])), "features": {k: c[k] for k in MISS_FIELDS},
                      "p_model": float(c["p_real"])})
    items.sort(key=lambda x: x["t"])
    batches, cur = [], []
    for it in items:
        if len(cur) >= args.batch and (it["bar"] != cur[-1]["bar"] or len(cur) >= args.batch + 3):
            batches.append(cur)
            cur = []
        cur.append(it)
    if cur:
        batches.append(cur)
    ids = []
    for i, b in enumerate(batches):
        bid = f"R{i + 1:02d}"
        (work / f"items_{bid}.json").write_text(json.dumps([{k: v for k, v in it.items() if k != "p_model"} for it in b], indent=0))
        ids.append(bid)
    (work / "items_all.json").write_text(json.dumps(items))
    out = {"folder": str(folder), "work": str(work), "ids": ids, "items": len(items),
           "tab_notes": sum(i["type"] == "tab_note" for i in items), "missing": sum(i["type"] == "missing" for i in items)}
    print(json.dumps(out))
    return out


def collect(args) -> dict:
    tabdir = Path(args.tab)
    work = Path(args.work or tabdir / "compare" / "review" / "work")
    items = {it["id"]: it for it in json.loads((work / "items_all.json").read_text())}
    by: dict[str, dict] = {}
    runs = json.loads(Path(args.results).read_text()) if args.results else [
        {"lens": f.stem.split("_")[1], "verdicts": json.loads(f.read_text())} for f in sorted(work.glob("verdicts_*.json"))]
    for r in runs:
        for v in (r["verdicts"]["verdicts"] if isinstance(r["verdicts"], dict) else r["verdicts"]):
            if v["id"] in items:
                by.setdefault(v["id"], {})[r["lens"]] = v
    incomplete = [i for i in items if len(by.get(i, {})) < len(LENSES)]
    if incomplete and not args.partial:
        raise SystemExit(f"{len(incomplete)} of {len(items)} items lack a verdict from both reviewers (use --partial)")
    edits, table = [], []
    for i, it in sorted(items.items(), key=lambda x: x[1]["t"]):
        a, s = by.get(i, {}).get("acoustic"), by.get(i, {}).get("skeptic")
        row = {k: it.get(k, "") for k in ("id", "type", "bar", "beat", "sixteenth", "t", "pitch", "name", "string", "fret", "chord", "p_model")}
        for lens, v in (("acoustic", a), ("skeptic", s)):
            row |= {lens: v["verdict"] if v else "", f"{lens}_p": v["p"] if v else "", f"{lens}_fix": v["fix"] if v else "",
                    f"{lens}_reason": v["reason"] if v else ""}
        row["applied"] = ""
        base = {"t": it["t"], "pitch": it["pitch"], "bar": it["bar"], "beat": it["beat"], "sixteenth": it["sixteenth"],
                "p_acoustic": a["p"] if a else None, "p_skeptic": s["p"] if s else None}
        if it["type"] == "tab_note" and a and a["verdict"] == "wrong":
            fx = a["fix"].strip()
            e = None
            if fx == "remove":
                e = base | {"action": "remove"}
            elif fx in ("octave_up", "octave_down"):
                e = base | {"action": "replace", "new_pitch": it["pitch"] + (12 if fx == "octave_up" else -12)}
            elif fx.startswith("pitch:"):
                e = base | {"action": "replace", "new_pitch": int(fx.split(":")[1])}
            if e:
                e |= {"why": a["reason"], "skeptic_agrees": bool(s and s["verdict"] == "wrong")}
                edits.append(e)
                row["applied"] = e["action"] + (f" -> {e['new_pitch']}" if "new_pitch" in e else "")
            else:
                row["applied"] = f"not applied (fix '{fx}')"
        if it["type"] == "missing" and a and s and a["verdict"] == "real" and s["verdict"] == "real":
            edits.append(base | {"action": "add", "why": a["reason"], "skeptic_agrees": True})
            row["applied"] = "add"
        table.append(row)
    (tabdir / f"{args.name}_edits.json").write_text(json.dumps({
        "source": "scripts/compare_tab_audio.py flags, reviewed by two agents (compare/agent_verdicts.csv)",
        "rule": "tab note: the acoustic reviewer calls it wrong (blind benchmark: 14 of 16 such calls right); "
                "missing note: both reviewers call it real (3 of 4)",
        "edits": edits}, indent=1, ensure_ascii=False))
    with open(tabdir / "compare" / "agent_verdicts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    count = lambda kind, lens, v: sum(1 for r in table if r["type"] == kind and r[lens] == v)  # noqa: E731
    out = {"items": len(items), "judged_by_both": len(items) - len(incomplete),
           "edits": {k: sum(e["action"] == k for e in edits) for k in ("remove", "replace", "add")},
           "skeptic_agrees_on": sum(e["skeptic_agrees"] for e in edits),
           "tab_notes_called_wrong": {lens: count("tab_note", lens, "wrong") for lens in LENSES},
           "missing_called_real": {lens: count("missing", lens, "real") for lens in LENSES}}
    print(json.dumps(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pack", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--tab", default=str(TAB))
        p.add_argument("--name", default="frevo_guitar_tab")
        p.add_argument("--work", default=None, help="default: <tab>/compare/review/work")
    pk = sub.choices["pack"]
    pk.add_argument("--folder", default=None, help="default: <tab>/compare/review")
    pk.add_argument("--thr-note", type=float, default=0.3)
    pk.add_argument("--thr-missing", type=float, default=0.3)
    pk.add_argument("--batch", type=int, default=15)
    co = sub.choices["collect"]
    co.add_argument("--results", default=None, help="a workflow's return value: [{lens, verdicts}, ...]")
    co.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    (pack if args.cmd == "pack" else collect)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
