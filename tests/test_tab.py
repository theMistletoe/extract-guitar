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
        assert sum(d for _, d, *_ in pieces) == tab.spb
        assert all(d in export._TYPES for _, d, *_ in pieces)


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


def test_pth_conversion_uses_weights_only(tmp_path):
    import torch

    sd = amt.RegressCRNN().state_dict()
    sampler = {"random_state": np.random.RandomState(0).get_state()[1][:16], "indexes": np.arange(5)}
    torch.save({"iteration": 1, "model": sd, "sampler": sampler}, tmp_path / "m.pth")
    with pytest.raises(Exception):  # plain weights_only refuses the NumPy sampler state
        torch.load(tmp_path / "m.pth", weights_only=True)
    amt._convert_pth(tmp_path / "m.pth", tmp_path / "m.safetensors")
    m = amt.load_model(tmp_path / "m.safetensors")
    assert torch.equal(m.state_dict()["frame_fc.weight"], sd["frame_fc.weight"].float())


def test_check_page_and_synth():
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "make_tab_check", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_tab_check.py")
    mtc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mtc)
    y = mtc.pluck(196.0, 0.5, 1.0, np.random.default_rng(0))
    assert np.isfinite(y).all() and len(y) == int(0.56 * mtc.SR)
    times = np.arange(17) * 0.1
    notes = [{"q": 0, "string": 0, "fret": 3, "midi": 43}, {"q": 3, "string": 3, "fret": 3, "midi": 58}]
    html = mtc.page({"title": "T"}, notes, {0: "Gm7"}, times, 2, {1: ["reason"]})
    assert 'class="bl rv1"' in html and 'title="reason"' in html and "Gm7" in html
    assert '"review": [0]' in html and html.count('class="c q0"') == 7
    assert 'class="u"' not in html and "__LEGEND__" not in html
    html = mtc.page({"title": "T"}, notes, {}, times, 2, uncertain={(3, 2)}, uncertain_correct=0.53)
    assert html.count('<b class="u">3</b>') == 1 and "約 53 %" in html
    html = mtc.page({"title": "T"}, notes, {}, times, 2, uncertain={(3, 2)}, edited={(3, 2)})
    assert '<b class="ed">3</b>' in html and 'class="u"' not in html and "自動レビューで追加・修正" in html


def _script(name):
    import importlib.util
    import pathlib

    import sys

    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def test_render_grid_roundtrip_and_string_damping():
    r = _script("render_tab_audio")
    beats = 1.0 + 0.4 * np.arange(20)
    centres = np.array([-0.006, 0.353, 0.492, 0.7])
    q = np.array([0, 1, 2, 3, 4, 13, 22.5])
    t = r.grid_time(q, 150.0, centres, 0.0)  # 150 BPM: the same 0.4 s beat as ``beats``
    assert np.allclose(r.time_to_q(t + 1.0, beats, centres), q, atol=1e-9)
    notes = [{"string": 2, "midi": 52}, {"string": 2, "midi": 55}, {"string": 0, "midi": 43}]
    ev = r.note_events(notes, np.array([0.1, 0.3, 0.1]), np.array([0.5, 0.6, 0.2]), np.array([80, 90, 70]))
    # the re-plucked string is damped (fade, note-off, sound off) before its second pluck ...
    i_on = ev.index(next(e for e in ev if e[2] == "note_on" and e[4] == 55))
    i_off = ev.index(next(e for e in ev if e[2] == "note_off" and e[4] == 52))
    assert i_off < i_on and ev[i_on][0] == 0.3
    # ... and its volume is back up for it
    assert [e[5] for e in ev[:i_on] if e[3] == 2 and e[2] == "cc" and e[4] == 7][-1] == r.VOLUME
    cut = [e for e in ev if e[3] == 0 and e[2] == "cc" and e[4] == 120]
    assert len(cut) == 1 and abs(cut[0][0] - (0.2 + r.DAMP_S)) < 1e-9


def test_render_lag_and_eq():
    r = _script("render_tab_audio")
    sr = r.SR
    x = np.zeros(2 * sr)
    for t in (0.2, 0.55, 0.9, 1.3, 1.6):
        n = int(t * sr)
        x[n:n + 2000] += np.sin(2 * np.pi * 196 * np.arange(2000) / sr) * np.exp(-np.arange(2000) / 400)
    late = np.r_[np.zeros(441), x[:-441]]  # 10 ms
    assert abs(r.onset_lag(late, x, sr) - 0.010) < 0.002
    fc = np.array([100.0, 1000.0, 5000.0])
    y = np.random.default_rng(0).normal(size=(2, sr)).astype(np.float32)
    assert np.allclose(r.apply_eq(y, sr, fc, np.zeros(3)), y, atol=1e-5)
    _, g = r.match_eq(y[0], y[0], sr)
    assert np.abs(g).max() < 1e-6


def test_compare_matching_flux_and_probe(tmp_path):
    c = _script("compare_tab_audio")
    # one-to-one matching: same pitch, closest onset within 50 ms
    m = c.match(np.array([1.0, 1.02, 2.0]), np.array([60, 60, 62]), np.array([1.01, 2.2]), np.array([60, 62]))
    assert m == {0: 0}
    # onset flux peaks where a pitch row steps up
    S = np.full((72, 200), -60.0)
    S[24, 100:] = -20.0
    fx = c.onset_flux(S)
    assert 94 <= int(np.argmax(fx[24])) <= 100 and fx[24].max() > 30 and abs(fx[30]).max() < 1e-9
    # the probe reads only its folder
    T = 400
    st = {k: np.full((72, T), -60, np.float16) for k in ("Ss", "Sr", "Sx")}
    st["Ss"][55 - c.CQ0, 170:] = -10
    post = np.zeros((T, 88), np.float16)
    post[170, 55 - 21] = 0.9
    np.savez(tmp_path / "state.npz", **st, f=1.0, ens=post, render=post, models=np.array(["a"]), post_a=post)
    (tmp_path / "notes.csv").write_text("onset_s,offset_s,pitch,string,fret\n2.0,2.3,55,4,0\n")
    txt = c.probe(tmp_path, 2.0, 55)
    assert "h1" in txt and "stem, a" in txt and "midi 55 (string 4 fret 0)" in txt


def test_transcribe_apply_edits():
    tt = _script("transcribe_tab")
    notes = np.array([[1.0, 1.3, 55, 0.9], [1.0, 1.3, 43, 0.8], [2.0, 2.2, 60, 0.4]])
    post = {k: np.zeros((400, 88)) for k in ("onset", "frame")}
    post["onset"][150, 62 - amt.BEGIN_NOTE] = 0.6
    post["frame"][150:170, 62 - amt.BEGIN_NOTE] = 0.9
    edits = [{"action": "remove", "t": 2.01, "pitch": 60}, {"action": "replace", "t": 1.0, "pitch": 55, "new_pitch": 67},
             {"action": "add", "t": 1.5, "pitch": 62}, {"action": "remove", "t": 3.0, "pitch": 40}]
    out, rep = tt.apply_edits(notes, post, 1.0, edits)
    assert rep["remove"] == 1 and rep["replace"] == 1 and rep["add"] == 1 and len(rep["not_found"]) == 1
    assert [int(p) for p in out[:, 2]] == [67, 43, 62]
    assert abs(out[2, 1] - 1.7) < 0.011 and abs(out[2, 3] - 0.6) < 1e-9
