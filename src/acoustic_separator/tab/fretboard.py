"""String/fret assignment: Viterbi over (fingering, hand position) states.

A state is one fingering of an onset group (each pitch on a distinct string, higher pitch on a
higher string) plus the fretting-hand position p (index finger at fret p, comfortable window
p..p+3, stretch to p+4).  Costs: position height, stretches, bass notes high up the neck, hand
shifts (cheaper over longer gaps) and cutting notes that are still ringing.  Notes the
transcriber is unsure of may be left out when they would force an awkward or unplayable shape
(cost proportional to their confidence; the bass is kept preferentially).  Weights were tuned
on GuitarSet (performer's actual strings): 67.8 % string accuracy on held-out excerpts vs
43.8 % for "lowest fret".
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

STANDARD = (40, 45, 50, 55, 59, 64)  # string index 0 = low E ... 5 = high E


@dataclass
class Weights:
    height: float = 0.03
    stretch: float = 3.0
    move: float = 0.8
    move_const: float = 1.0
    open_bonus: float = 0.0
    high_low: float = 0.08
    cut: float = 2.0
    lift: float = 0.8
    gap_ref: float = 0.25
    max_fret: int = 19
    drop: float = 4.0  # leaving out a note of confidence c costs drop * c
    weak: float = 0.45  # below this a note may be left out for a better shape


@dataclass
class Group:
    onset: float
    pitches: list[int]
    offsets: list[float]
    idx: list[int] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)  # per pitch; empty = all certain


def candidate_fingerings(pitches, tuning=STANDARD, max_fret=19, max_span=5):
    opts = []
    for p in pitches:
        o = [(s, p - t) for s, t in enumerate(tuning) if 0 <= p - t <= max_fret]
        if not o:
            return []
        opts.append(o)
    out = []
    for combo in itertools.product(*opts):
        strings = [c[0] for c in combo]
        if any(strings[i] >= strings[i + 1] for i in range(len(strings) - 1)):
            continue
        fr = [c[1] for c in combo if c[1] > 0]
        if fr and max(fr) - min(fr) > max_span - 1:
            continue
        out.append(tuple(combo))
    return out


def _positions(fing):
    fr = [f for _, f in fing if f > 0]
    if not fr:
        return list(range(1, 13))
    return list(range(max(1, max(fr) - 4), min(fr) + 1))


def _unary(fing, p, w: Weights):
    c = w.height * p
    for s, f in fing:
        if f == 0:
            c -= w.open_bonus if s <= 2 else 0.0
        else:
            if f > p + 3:
                c += w.stretch
            if s <= 2 and f > 9:
                c += w.high_low * (f - 9)
    return c


def group_notes(onsets, offsets, pitches, keys=None, tol=0.035, conf=None) -> list[Group]:
    """Group notes into chords: same quantised key if given, else onsets within ``tol`` s."""
    order = np.lexsort((pitches, onsets))
    conf = np.ones(len(onsets)) if conf is None else np.asarray(conf, float)
    groups: list[Group] = []
    for i in order:
        p = int(pitches[i])
        same = (groups and (keys[i] == keys[groups[-1].idx[0]] if keys is not None
                            else onsets[i] - groups[-1].onset <= tol))
        if same and p not in groups[-1].pitches:
            g = groups[-1]
            g.pitches.append(p)
            g.offsets.append(float(offsets[i]))
            g.idx.append(int(i))
            g.conf.append(float(conf[i]))
        elif not same:
            groups.append(Group(float(onsets[i]), [p], [float(offsets[i])], [int(i)], [float(conf[i])]))
    for g in groups:
        o = np.argsort(g.pitches)
        g.pitches = [g.pitches[k] for k in o]
        g.offsets = [g.offsets[k] for k in o]
        g.idx = [g.idx[k] for k in o]
        g.conf = [g.conf[k] for k in o]
    return groups


def _keep_sets(g: Group, w: Weights, any_notes: bool = False):
    """Subsets of a group's notes worth trying: all notes, and without one or two weak notes.
    With ``any_notes`` (nothing else is playable) up to three notes of any confidence may go;
    losing the bass (the thumb's note) costs 1.5x."""
    n = len(g.pitches)
    conf = g.conf or [1.0] * n
    pool = list(range(n)) if any_notes else [k for k in range(n) if conf[k] < w.weak]
    cost = lambda k: w.drop * conf[k] * (1.5 if k == 0 and n > 1 else 1.0)  # noqa: E731
    sets = [] if any_notes else [(tuple(range(n)), 0.0)]
    for r in (1, 2, 3) if any_notes else (1, 2):
        if n - r < 1:
            break
        for drop in itertools.combinations(pool, r):
            sets.append((tuple(k for k in range(n) if k not in drop), sum(cost(k) for k in drop)))
    return sets


def _states(g: Group, w: Weights, tuning, any_notes: bool):
    st = []
    for keep, dcost in _keep_sets(g, w, any_notes):
        for f in candidate_fingerings([g.pitches[k] for k in keep], tuning, w.max_fret):
            st += [(f, p, _unary(f, p, w) + dcost, keep) for p in _positions(f)]
    return st


def assign(groups: list[Group], w: Weights | None = None, tuning=STANDARD):
    """Returns per group: (list of (string, fret) or None per pitch, hand position)."""
    w = w or Weights()
    states = []  # per group: list of (fingering, hand position, unary cost, kept note indices)
    for g in groups:
        st = _states(g, w, tuning, False) or _states(g, w, tuning, True)
        states.append(st or [((), 0, 0.0, ())])

    cost = np.array([s[2] for s in states[0]])
    back = [None]
    for i in range(1, len(groups)):
        prev, cur, gp = states[i - 1], states[i], groups[i - 1]
        gap = max(groups[i].onset - gp.onset, 1e-3)
        tf = min(1.0, w.gap_ref / gap)
        # previous-group notes still ringing at this onset, per previous state
        ring = [[pf[n] for n, k in enumerate(keep) if gp.offsets[k] > groups[i].onset + 0.03]
                for pf, _, _, keep in prev]
        pp = np.array([s[1] for s in prev])
        C = np.empty((len(cur), len(prev)))
        for j, (f, p, u, _) in enumerate(cur):
            mv = np.abs(pp - p)
            t = (w.move * mv + w.move_const * (mv > 0)) * tf
            if f:
                used = {s for s, _ in f}
                pen = np.zeros(len(prev))
                for m, rn in enumerate(ring):
                    for s_prev, f_prev in rn:
                        if s_prev in used:
                            pen[m] += w.cut
                        elif f_prev > 0 and not p <= f_prev <= p + 4:
                            pen[m] += w.lift
                t = t + pen
            C[j] = cost + t + u
        b = C.argmin(1)
        cost = C[np.arange(len(cur)), b]
        back.append(b)
    j = int(np.argmin(cost))
    path = [j]
    for i in range(len(groups) - 1, 0, -1):
        j = int(back[i][j])
        path.append(j)
    path.reverse()
    out = []
    for g, st, j in zip(groups, states, path):
        f, p, _, keep = st[j]
        sf = [None] * len(g.pitches)
        for n, k in enumerate(keep):
            if n < len(f):
                sf[k] = f[n]
        out.append((sf, p))
    return out
