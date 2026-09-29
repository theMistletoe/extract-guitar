import xml.dom.minidom

import numpy as np
import pytest

from acoustic_separator.tab import amt, chords, export, fretboard, rhythm


# ------------------------------------------------------------------------------ fretboard
def test_candidate_fingerings_single_note():
    f = fretboard.candidate_fingerings([64])
    assert {c[0] for c in f} == {(5, 0), (4, 5), (3, 9), (2, 14), (1, 19)}


def test_open_g_chord_is_playable_and_uses_distinct_strings():
    g = fretboard.Group(0.0, [43, 47, 50, 55, 59, 67], [1.0] * 6, list(range(6)))
    (sf, pos), = fretboard.assign([g])
    assert all(x is not None for x in sf)
    assert len({s for s, _ in sf}) == 6
    fr = [f for _, f in sf if f > 0]
    assert max(fr) - min(fr) <= 4


def test_weak_note_that_forces_a_stretch_is_dropped():
    # C9 struck while a weak F3 is (mis)detected: x3233x is kept, the F3 goes
    g = fretboard.Group(0.0, [48, 52, 53, 58, 62], [0.3] * 5, list(range(5)), [0.8, 0.8, 0.3, 0.9, 0.9])
    (sf, _), = fretboard.assign([g])
    assert sf[2] is None
    assert sf[0] == (1, 3) and sf[1] == (2, 2) and sf[3] == (3, 3) and sf[4] == (4, 3)


def test_unplayable_chord_keeps_the_bass():
    g = fretboard.Group(0.0, [42, 52, 58, 60, 63, 66], [1.0] * 6, list(range(6)),
                        [0.73, 0.57, 0.87, 0.79, 0.74, 0.34])
    (sf, _), = fretboard.assign([g])
    assert sf[0] == (0, 2)  # F#2 on the low E string
    kept = [x for x in sf if x is not None]
    assert len(kept) >= 4
    fr = [f for _, f in kept if f > 0]
    assert max(fr) - min(fr) <= 4


def test_repeated_shape_keeps_its_fingering():
    gm7 = [43, 53, 58, 62]
    groups = [fretboard.Group(0.2 * i, list(gm7), [0.2 * i + 0.15] * 4, [4 * i + k for k in range(4)])
              for i in range(4)]
    res = fretboard.assign(groups)
    assert all(r[0] == res[0][0] for r in res)


# --------------------------------------------------------------------------------- rhythm
def _grid(n=40, ibi=0.4, t0=1.0):
    return t0 + ibi * np.arange(n)


def test_swing_quantisation():
    beats = _grid()
    centres = np.array([0.0, 0.35, 0.5, 0.7])
    on = np.array([beats[5] + c * 0.4 for c in centres])
    q = rhythm.quantize(on, beats, rhythm.slot_bounds(centres))
    assert list(q) == [20, 21, 22, 23]
    # an onset just before the beat belongs to that beat
    assert rhythm.quantize(np.array([beats[6] - 0.02]), beats, rhythm.slot_bounds(centres))[0] == 24


def test_swing_centres_and_beat_refinement():
    rng = np.random.default_rng(0)
    true = _grid()
    centres = np.array([0.0, 0.35, 0.5, 0.7])
    on = np.sort(np.concatenate([true[k] + centres[j] * 0.4 + rng.normal(0, 0.005, 1)
                                 for k in range(2, 36) for j in rng.choice(4, 2, replace=False)]))
    est = rhythm.swing_centres(on, true)
    assert np.allclose(est, centres, atol=0.03)
    refined = rhythm.refine_beats(on, true + 0.02, centres=centres)
    assert np.median(np.abs(refined[3:35] - true[3:35])) < 0.006


# --------------------------------------------------------------------------------- chords
def _label(notes):
    q = np.array([n[0] for n in notes])
    qe = q + 2.0
    p = np.array([n[1] for n in notes])
    chroma, bass = chords.evidence(q, qe, p, np.ones(len(q)), 8)
    return chords.changes(chords.label(chroma, bass))


def test_chord_labels():
    assert list(_label([(0, 43), (0, 53), (0, 58), (0, 62), (3, 58), (3, 62)]).values())[0] == "Gm7"
    # rootless D7 over a D bass struck an 8th earlier
    assert list(_label([(0, 50), (2, 54), (2, 60), (2, 57)]).values())[0] == "D7"


# ------------------------------------------------------------------------------------ amt
def test_decode_single_note():
    T = 200
    post = {k: np.zeros((T, 88)) for k in ("onset", "offset", "frame", "velocity")}
    k = 60 - amt.BEGIN_NOTE
    post["onset"][48:53, k] = [0.2, 0.6, 0.9, 0.6, 0.2]
    post["frame"][50:80, k] = 0.9
    (on, off, p, conf), = amt.decode(post)
    assert p == 60 and abs(on - 0.50) < 0.011 and abs(off - 0.80) < 0.02 and conf > 0.8


def test_estimate_tuning():
    sr = 16000
    t = np.arange(sr * 2) / sr
    x = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([442.0, 442.0 * 3 / 2 * 2 ** (1.955 / 1200)]))
    assert abs(amt.estimate_tuning(x.astype(np.float32), sr) - 7.85) < 1.5


def test_model_shapes():
    m = amt.RegressCRNN().eval()
    import torch

    with torch.inference_mode():
        out = m(torch.zeros(1, amt.SR))
    assert {k: tuple(v.shape) for k, v in out.items()} == {k: (1, 101, 88) for k in ("onset", "offset", "frame", "velocity")}


# --------------------------------------------------------------------------------- export
def _tab():
    beats = _grid(20)
    notes = []
    for bar in range(3):
        for j, pitches in ((0, [(43, 0, 3), (53, 2, 3), (58, 3, 3), (62, 4, 3)]), (3, [(58, 3, 3), (62, 4, 3)]),
                           (4, [(43, 0, 3)]), (6, [(58, 3, 3), (62, 4, 3)])):
            q = bar * 8 + j
            for p, s, f in pitches:
                t = beats[q // 4] + (q % 4) * 0.1
                notes.append(export.TabNote(t, t + 0.15, p, s, f, 0.9, q, q + 1.5))
    return export.Tab(notes, beats, 2, {0: "Gm7", 8: "Gm7"}, 150.0, "Test", key_fifths=-2,
                      key_mode="minor")


def test_pieces_fill_every_bar():
    tab = _tab()
    for pieces in export._pieces(tab):
        assert sum(l for _, l, *_ in pieces) == tab.spb
        assert all(l in export._TYPES for _, l, *_ in pieces)


def test_ascii_and_musicxml_and_lilypond():
    tab = _tab()
    txt = export.ascii_tab(tab)
    assert txt.count("e|") == 1 and txt.count("\nE|") == 1
    assert "3--------3--" in txt  # Gm7 on beat 1 and the "a"
    xml.dom.minidom.parseString(export.musicxml(tab))
    ly = export.lilypond(tab)
    assert "\\key g \\minor" in ly and "\\time 2/4" in ly and "<g,\\6 f\\4 bes\\3 d'\\2>" in ly


def test_midi_and_gp5(tmp_path):
    tab = _tab()
    path = export.midi(tab.notes, tmp_path / "t.mid")
    assert path.read_bytes()[:4] == b"MThd"
    gp = pytest.importorskip("guitarpro")
    export.guitar_pro(tab, tmp_path / "t.gp5")
    song = gp.parse(str(tmp_path / "t.gp5"))
    assert len(song.tracks[0].measures) == 3
    assert all(sum(b.duration.time for b in m.voices[0].beats) == 2 * 960 for m in song.tracks[0].measures)
