"""Chord symbols from transcribed notes: per-8th template scores + Viterbi smoothing.

The harmony is labelled on an 8th-note grid so that changes anticipated by an 8th or a 16th (as
in frevo / choro comping, where the thumb or the chord often lands before the beat) are placed
where they happen.  Unit u covers the 16ths 2u-1 and 2u; notes still ringing count with a
lower weight; the bass of a unit is its lowest struck note, else the bass still ringing.
"""
from __future__ import annotations

import numpy as np

ROOTS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
# suffix -> (required intervals, optional intervals); guitar voicings often omit root / 5th
TEMPLATES = {
    "": ((0, 4, 7), ()),
    "m": ((0, 3, 7), ()),
    "7": ((0, 4, 10), (7,)),
    "maj7": ((0, 4, 11), (7,)),
    "m7": ((0, 3, 10), (7,)),
    "m7(b5)": ((0, 3, 6, 10), ()),
    "°7": ((0, 3, 6, 9), ()),
    "6": ((0, 4, 9), (7,)),
    "m6": ((0, 3, 9), (7,)),
    "7(9)": ((0, 4, 10, 2), (7,)),
    "7(b9)": ((0, 4, 10, 1), (7,)),
    "7(#9)": ((0, 4, 10, 3), (7,)),
    "7(13)": ((0, 4, 10, 9), (7,)),
    "m7(9)": ((0, 3, 10, 2), (7,)),
    "7sus4": ((0, 5, 10), (7,)),
    "aug": ((0, 4, 8), ()),
}
RARE = {"maj7": 0.08, "6": 0.05, "m6": 0.05, "aug": 0.1, "7sus4": 0.05}  # uncommon in this idiom
COMPLEXITY = {k: 0.05 * (len(v[0]) - 3) + (0.03 if "(" in k else 0.0) + RARE.get(k, 0.0)
              for k, v in TEMPLATES.items()}


def _score(chroma: np.ndarray, bass: int | None, root: int, suffix: str) -> float:
    req, opt = TEMPLATES[suffix]
    tot = chroma.sum() + 1e-9
    req_pc = [(root + i) % 12 for i in req]
    opt_pc = [(root + i) % 12 for i in opt]
    hit = (sum(chroma[p] for p in req_pc) + 0.5 * sum(chroma[p] for p in opt_pc)) / tot
    extra = sum(chroma[p] for p in range(12) if p not in req_pc and p not in opt_pc) / tot
    missing = sum(chroma[p] < 0.05 * tot for i, p in zip(req, req_pc) if i not in (0, 7))
    s = hit - 1.0 * extra - 0.12 * missing - COMPLEXITY[suffix]
    if bass is not None:
        s += 0.3 if bass == root else (0.1 if bass in req_pc + opt_pc else -0.25)
    elif chroma[root] < 0.05 * tot:
        s -= 0.05
    return s


def evidence(q: np.ndarray, q_end: np.ndarray, pitch: np.ndarray, conf: np.ndarray,
             n_units: int, bass_below: int = 52) -> tuple[np.ndarray, list[int | None]]:
    """Per 8th-note unit: pitch-class weights and bass pitch class (a bass note struck in the
    unit or the unit before -- the thumb often leads the chord by an 8th -- else the most recent
    bass note still ringing, else unknown).  Notes struck in the previous unit count half."""
    chroma = np.zeros((n_units, 12))
    struck = np.full(n_units, 999)
    lead = np.full(n_units, 999)
    ringing: list[tuple[int, int] | None] = [None] * n_units  # (onset q, pitch) latest
    for qq, qe, p, c in sorted(zip(q, q_end, pitch, conf)):
        u = (qq + 1) // 2
        if u >= n_units:
            continue
        chroma[u, p % 12] += 0.5 + c
        if u + 1 < n_units:
            chroma[u + 1, p % 12] += 0.5 * (0.5 + c)
        if p < bass_below:
            struck[u] = min(struck[u], p)
            if u + 1 < n_units:
                lead[u + 1] = min(lead[u + 1], p)
        for uu in range(u + 1, min(int(qe + 1) // 2 + 1, n_units)):  # still ringing
            chroma[uu, p % 12] += 0.5
            if p < bass_below and (ringing[uu] is None or qq >= ringing[uu][0]):
                ringing[uu] = (qq, p)
    bass = [int(a % 12) if a < 999 else int(b % 12) if b < 999 else (int(r[1] % 12) if r else None)
            for a, b, r in zip(struck, lead, ringing)]
    return chroma, bass


def label(chroma: np.ndarray, bass: list[int | None], units_per_bar: int = 4,
          change_cost: float = 0.8, min_units: int = 2) -> list[tuple | None]:
    """One (root pc, suffix, bass pc) per unit, None where nothing sounds.  Changing chord is
    cheapest on the downbeat, then on beats, dearest off the beat; afterwards chords shorter
    than ``min_units`` are absorbed and consecutive chords on the same root are merged."""
    n = len(chroma)
    labels = [(r, s) for r in range(12) for s in TEMPLATES]
    S = np.zeros((n, len(labels)))
    for k in range(n):
        if chroma[k].sum() > 0:
            S[k] = [_score(chroma[k], bass[k], r, s) for r, s in labels]
    cost = -S[0].copy()
    back = np.zeros((n, len(labels)), int)
    for k in range(1, n):
        c = change_cost * (0.25 if k % units_per_bar == 0 else 0.5 if k % 2 == 0 else 1.0)
        best_prev = int(np.argmin(cost))
        move = cost[best_prev] + c
        back[k] = np.where(cost <= move, np.arange(len(labels)), best_prev)
        cost = np.minimum(cost, move) - S[k]
    j = int(np.argmin(cost))
    path = [j]
    for k in range(n - 1, 0, -1):
        j = back[k][j]
        path.append(j)
    path.reverse()
    path = _smooth(path, S, labels, min_units)
    return [None if chroma[k].sum() == 0 else (*labels[j], bass[k]) for k, j in enumerate(path)]


def _runs(path):
    runs, start = [], 0
    for k in range(1, len(path) + 1):
        if k == len(path) or path[k] != path[start]:
            runs.append([start, k, path[start]])
            start = k
    return runs


def _smooth(path: list[int], S: np.ndarray, labels: list, min_units: int) -> list[int]:
    path = list(path)
    for _ in range(3):
        runs = _runs(path)
        changed = False
        for i, (a, b, j) in enumerate(runs):
            nb = [runs[x] for x in (i - 1, i + 1) if 0 <= x < len(runs)]
            same_root = [r for r in nb if labels[r[2]][0] == labels[j][0]]
            if (b - a < min_units or same_root) and nb:
                cand = same_root or nb
                # the label (own or a neighbour's) that best explains the merged span
                span = [r for r in cand] + [[a, b, j]]
                lo, hi = min(r[0] for r in span), max(r[1] for r in span)
                best = max({r[2] for r in span}, key=lambda jj: S[lo:hi, jj].sum())
                for r in span:
                    path[r[0]:r[1]] = [best] * (r[1] - r[0])
                changed = True
                break
        if not changed:
            break
    return path


def changes(per_unit: list[tuple | None]) -> dict[int, str]:
    """Chord symbols at the units where the harmony changes (slash bass taken there)."""
    out, prev = {}, None
    for k, lab in enumerate(per_unit):
        if lab is None:
            continue
        r, s, b = lab
        if (r, s) != prev:
            name = ROOTS[r] + s
            if b is not None and b != r:
                name += "/" + ROOTS[b]
            out[k] = name
            prev = (r, s)
    return out
