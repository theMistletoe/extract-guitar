"""Tab outputs: ASCII tab, MusicXML (notation + TAB staff), LilyPond (engraved PDF), Guitar Pro 5,
MIDI and CSV."""
from __future__ import annotations

import csv
import struct
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .chords import ROOTS

TAB_ROWS = ["e", "B", "G", "D", "A", "E"]  # printed top to bottom = string 1..6
NAMES_FLAT = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class TabNote:
    onset: float  # performance time (s)
    offset: float
    pitch: int  # MIDI
    string: int  # 0 = low E ... 5 = high E
    fret: int
    conf: float  # onset posterior
    q: int  # 16th index from bar 1 beat 1
    q_end: float  # (fractional) 16th position of the offset


@dataclass
class Tab:
    notes: list[TabNote]
    beats: np.ndarray  # beat times, beats[0] = bar 1 beat 1
    beats_per_bar: int
    chords: dict[int, str]  # 16th position -> chord symbol where the harmony changes
    bpm: float
    title: str
    subtitle: str = ""
    composer: str = ""
    tuning: tuple = (40, 45, 50, 55, 59, 64)
    key_fifths: int = 0
    key_mode: str = "major"
    header: list[str] = field(default_factory=list)

    @property
    def spb(self) -> int:  # 16ths per bar
        return 4 * self.beats_per_bar

    @property
    def n_bars(self) -> int:
        return max(n.q for n in self.notes) // self.spb + 1

    def events(self) -> dict[int, list[TabNote]]:
        ev: dict[int, list[TabNote]] = {}
        for n in self.notes:
            ev.setdefault(n.q, []).append(n)
        for v in ev.values():
            v.sort(key=lambda n: n.string)
        return dict(sorted(ev.items()))

    def time_of(self, q: float) -> float:
        k = q / 4.0
        i = int(np.clip(np.floor(k), 0, len(self.beats) - 2))
        return float(self.beats[i] + (k - i) * (self.beats[i + 1] - self.beats[i]))


def key_signature(pitches: np.ndarray, weights: np.ndarray) -> tuple[int, str, str]:
    """Krumhansl-Schmuckler key estimate -> (fifths, mode, name)."""
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    h = np.bincount(np.asarray(pitches) % 12, weights=weights, minlength=12)
    best = max(((np.corrcoef(np.roll(prof, t), h)[0, 1], t, mode)
                for prof, mode in ((maj, "major"), (mnr, "minor")) for t in range(12)))
    _, tonic, _ = best
    mode = "minor" if h[(tonic + 3) % 12] > h[(tonic + 4) % 12] else "major"
    rel = tonic if mode == "major" else (tonic + 3) % 12  # relative major
    fifths = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5}[rel]
    return fifths, mode, f"{NAMES_FLAT[tonic] if fifths < 0 else NAMES_SHARP[tonic]} {mode}"


# ----------------------------------------------------------------------- rhythm pieces
def _pieces(tab: Tab):
    """Single voice: every onset position becomes a chord that lasts until its notes stop
    (quantised offset) or the next onset; gaps become rests.  Returns per bar a list of
    (start16 in bar, length16, notes or None, tie_in, tie_out) split at beats (except whole
    beats / half bars starting on a beat) and at barlines."""
    ev = tab.events()
    qs = list(ev)
    end_q = tab.n_bars * tab.spb
    segs = []  # (start, length, notes)
    for i, q in enumerate(qs):
        nxt = qs[i + 1] if i + 1 < len(qs) else end_q
        ring = int(round(np.median([n.q_end for n in ev[q]]) - q))
        dur = int(np.clip(ring, 1, nxt - q))
        segs.append((q, dur, ev[q]))
        if nxt - q - dur > 0:
            segs.append((q + dur, nxt - q - dur, None))
    if qs and qs[0] > 0:
        segs.insert(0, (0, qs[0], None))
    bars: list[list] = [[] for _ in range(tab.n_bars)]
    for start, length, notes in segs:
        pos, first = start, True
        while length > 0:
            bar, off = divmod(pos, tab.spb)
            take = min(length, tab.spb - off)
            for s, l in _split_in_bar(off, take):
                last = length - (s - off) - l == 0
                bars[bar].append((s, l, notes, notes is not None and not first,
                                  notes is not None and not last))
                first = False
            pos += take
            length -= take
    return bars


_VALID = {1, 2, 3, 4, 6, 8}


def _split_in_bar(off: int, length: int) -> list[tuple[int, int]]:
    out = []
    while length > 0:
        if off % 4 == 0 and length >= 4:  # whole beats from a beat start
            l = 8 if (off == 0 and length >= 8) else (6 if length >= 6 and off % 8 == 0 else 4)
        else:
            to_beat = 4 - off % 4
            l = min(length, to_beat)
            if l not in _VALID:
                l = 2 if l > 2 else 1
            if off % 2 == 1 and l == 3:
                l = 2
        out.append((off, l))
        off += l
        length -= l
    return out


# --------------------------------------------------------------------------- ASCII tab
def ascii_tab(tab: Tab, bars_per_line: int = 4, width: int = 3) -> str:
    ev = tab.events()
    out = list(tab.header)
    for b0 in range(0, tab.n_bars, bars_per_line):
        bars = range(b0, min(tab.n_bars, b0 + bars_per_line))
        head, count = "  ", "  "
        rows = [r + "|" for r in TAB_ROWS]
        chord = [" "] * (2 + len(bars) * (tab.spb * width + 1) + 16)
        cursor = 0
        for bi, bar in enumerate(bars):
            t = tab.time_of(bar * tab.spb)
            label = f"{bar + 1} ({int(t // 60)}:{t % 60:04.1f})"
            cells = tab.spb * width
            head += label.ljust(cells + 1)
            for j in range(tab.spb):
                sym = tab.chords.get(bar * tab.spb + j)
                if sym:
                    pos = max(2 + bi * (cells + 1) + j * width, cursor)
                    chord[pos:pos + len(sym)] = list(sym)
                    cursor = pos + len(sym) + 1
            cnt = ""
            for j in range(tab.spb):
                cnt += (str(j // 4 + 1) if j % 4 == 0 else "e+a"[j % 4 - 1]).ljust(width)
            count += cnt + " "
            for j in range(tab.spb):
                notes = ev.get(bar * tab.spb + j, [])
                for r in range(6):
                    s = 5 - r
                    fr = [n.fret for n in notes if n.string == s]
                    rows[r] += (str(fr[0]) if fr else "").ljust(width, "-")
            rows = [r + "|" for r in rows]
        out += ["", head.rstrip(), "".join(chord).rstrip()] + rows + [count.rstrip()]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------- MusicXML
_TYPES = {1: ("16th", 0), 2: ("eighth", 0), 3: ("eighth", 1), 4: ("quarter", 0),
          6: ("quarter", 1), 8: ("half", 0)}
_KINDS = {"": "major", "m": "minor", "7": "dominant", "maj7": "major-seventh",
          "m7": "minor-seventh", "m7(b5)": "half-diminished", "°7": "diminished-seventh",
          "6": "major-sixth", "m6": "minor-sixth", "7(9)": "dominant-ninth",
          "7(b9)": "dominant", "7(#9)": "dominant", "7(13)": "dominant-13th",
          "m7(9)": "minor-ninth", "7sus4": "suspended-fourth", "aug": "augmented"}


def _step_alter(pc: int, flats: bool) -> tuple[str, int]:
    name = (NAMES_FLAT if flats else NAMES_SHARP)[pc]
    return name[0], (-1 if name.endswith("b") else 1 if name.endswith("#") else 0)


def _harmony_xml(sym: str, flats: bool) -> str:
    main, _, bass = sym.partition("/")
    root = next(r for r in sorted(ROOTS, key=len, reverse=True) if main.startswith(r))
    suffix = main[len(root):]
    st, al = _step_alter(ROOTS.index(root), flats)
    x = f"<harmony><root><root-step>{st}</root-step>"
    x += f"<root-alter>{al}</root-alter>" if al else ""
    x += f'</root><kind text="{escape(suffix)}">{_KINDS.get(suffix, "other")}</kind>'
    if bass:
        bst, bal = _step_alter(ROOTS.index(bass), flats)
        x += f"<bass><bass-step>{bst}</bass-step>" + (f"<bass-alter>{bal}</bass-alter>" if bal else "") + "</bass>"
    return x + "</harmony>"


def musicxml(tab: Tab) -> str:
    flats = tab.key_fifths < 0
    bars = _pieces(tab)
    x = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
         '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
         '"http://www.musicxml.org/dtds/partwise.dtd">',
         '<score-partwise version="4.0">',
         f"<work><work-title>{escape(tab.title)}</work-title></work>",
         "<identification>" + (f'<creator type="composer">{escape(tab.composer)}</creator>' if tab.composer else "")
         + f'<creator type="arranger">{escape(tab.subtitle)}</creator>'
         + "<encoding><software>acoustic-separator tab transcription</software></encoding></identification>",
         '<part-list><score-part id="P1"><part-name>Guitar</part-name>'
         '<score-instrument id="P1-I1"><instrument-name>Acoustic Guitar (nylon)</instrument-name></score-instrument>'
         '<midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>25</midi-program></midi-instrument>'
         "</score-part></part-list>", '<part id="P1">']
    tuning = "".join(
        f'<staff-tuning line="{i + 1}"><tuning-step>{_step_alter(p % 12, False)[0]}</tuning-step>'
        + (f"<tuning-alter>{_step_alter(p % 12, False)[1]}</tuning-alter>" if _step_alter(p % 12, False)[1] else "")
        + f"<tuning-octave>{p // 12 - 1}</tuning-octave></staff-tuning>" for i, p in enumerate(tab.tuning))
    for b, pieces in enumerate(bars):
        m = [f'<measure number="{b + 1}">']
        if b == 0:
            m.append(
                f"<attributes><divisions>4</divisions><key><fifths>{tab.key_fifths}</fifths><mode>{tab.key_mode}</mode></key>"
                f"<time><beats>{tab.beats_per_bar}</beats><beat-type>4</beat-type></time><staves>2</staves>"
                '<clef number="1"><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>'
                '<clef number="2"><sign>TAB</sign><line>5</line></clef>'
                f'<staff-details number="2"><staff-lines>6</staff-lines>{tuning}</staff-details></attributes>'
                '<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                f"<per-minute>{round(tab.bpm)}</per-minute></metronome></direction-type>"
                f'<sound tempo="{round(tab.bpm)}"/></direction>')
        if b > 0 and b % 4 == 0:
            m.append('<print new-system="yes"/>')
        for staff, voice in ((1, 1), (2, 5)):
            if staff == 2:
                m.append(f"<backup><duration>{tab.spb}</duration></backup>")
            for s, l, notes, tie_in, tie_out in pieces:
                typ, dot = _TYPES[l]
                if staff == 1 and not tie_in:
                    sym = tab.chords.get(b * tab.spb + s)
                    if sym:
                        m.append(_harmony_xml(sym, flats))
                if notes is None:
                    m.append(f"<note><rest/><duration>{l}</duration><voice>{voice}</voice><type>{typ}</type>"
                             + "<dot/>" * dot + f"<staff>{staff}</staff></note>")
                    continue
                for i, n in enumerate(sorted(notes, key=lambda n: n.pitch)):
                    st, al = _step_alter(n.pitch % 12, flats)
                    ties = ('<tie type="stop"/>' if tie_in else "") + ('<tie type="start"/>' if tie_out else "")
                    tied = ('<tied type="stop"/>' if tie_in else "") + ('<tied type="start"/>' if tie_out else "")
                    tech = f"<technical><string>{6 - n.string}</string><fret>{n.fret}</fret></technical>"
                    m.append("<note>" + ("<chord/>" if i else "")
                             + f"<pitch><step>{st}</step>" + (f"<alter>{al}</alter>" if al else "")
                             + f"<octave>{n.pitch // 12 - 1}</octave></pitch><duration>{l}</duration>{ties}"
                             f"<voice>{voice}</voice><type>{typ}</type>" + "<dot/>" * dot
                             + (f"<stem>{'up' if staff == 1 else 'none'}</stem>")
                             + f"<staff>{staff}</staff><notations>{tied}{tech}</notations></note>")
        m.append("</measure>")
        x.append("".join(m))
    x += ["</part>", "</score-partwise>"]
    return "\n".join(x) + "\n"


# ----------------------------------------------------------------------------- LilyPond
_LY_NAMES_FLAT = ["c", "cis", "d", "ees", "e", "f", "fis", "g", "aes", "a", "bes", "b"]
_LY_NAMES_SHARP = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
_LY_DUR = {1: "16", 2: "8", 3: "8.", 4: "4", 6: "4.", 8: "2"}


def _ly_pitch(p: int, flats: bool) -> str:
    name = (_LY_NAMES_FLAT if flats else _LY_NAMES_SHARP)[p % 12]
    octv = p // 12 - 1 - 3  # c (no mark) = C3
    return name + ("'" * octv if octv > 0 else "," * -octv)


def lilypond(tab: Tab, bars_per_line: int = 4) -> str:
    """LilyPond source: notation (treble 8vb) + TAB staff from the same fingered notes, chord
    symbols as markup, all bar numbers, recording time every ``bars_per_line`` bars."""
    flats = tab.key_fifths < 0
    tonic = {0: "c", 1: "g", 2: "d", 3: "a", 4: "e", 5: "b", 6: "fis", -1: "f", -2: "bes", -3: "ees",
             -4: "aes", -5: "des"}[tab.key_fifths]
    if tab.key_mode == "minor":
        tonic = _LY_NAMES_FLAT[({"c": 0, "g": 7, "d": 2, "a": 9, "e": 4, "b": 11, "fis": 6, "f": 5,
                                 "bes": 10, "ees": 3, "aes": 8, "des": 1}[tonic] - 3) % 12]
    music = []
    for b, pieces in enumerate(_pieces(tab)):
        bar = []
        for i, (s, l, notes, tie_in, tie_out) in enumerate(pieces):
            marks = ""
            if i == 0 and b % bars_per_line == 0:
                t = tab.time_of(b * tab.spb)
                marks += f'_\\markup {{ \\tiny \\italic "{int(t // 60)}:{t % 60:04.1f}" }}'
            sym = None if tie_in else tab.chords.get(b * tab.spb + s)
            if sym and notes is not None:
                marks += f'^\\markup {{ \\bold "{sym}" }}'
            if notes is None:
                bar.append(f"r{_LY_DUR[l]}{marks}")
                continue
            chord = " ".join(f"{_ly_pitch(n.pitch, flats)}\\{6 - n.string}"
                             for n in sorted(notes, key=lambda n: n.pitch))
            bar.append(f"<{chord}>{_LY_DUR[l]}{'~' if tie_out else ''}{marks}")
        music.append(" ".join(bar) + (" \\break" if (b + 1) % bars_per_line == 0 else "") + f" |  % {b + 1}")
    body = "\n  ".join(music)
    return f"""\\version "2.24.0"
\\header {{
  title = "{tab.title}"
  subtitle = "{tab.subtitle}"
  composer = "{tab.composer}"
  tagline = ##f
}}
\\paper {{ #(set-paper-size "a4") indent = 0 ragged-last = ##t }}
global = {{ \\time {tab.beats_per_bar}/4 \\key {tonic} \\{tab.key_mode} \\tempo 4 = {round(tab.bpm)} }}
music = {{
  {body}
  \\bar "|."
}}
\\score {{
  <<
    \\new Staff \\with {{ \\omit StringNumber }} {{ \\clef "treble_8" \\global \\music }}
    \\new TabStaff \\with {{ \\omit TextScript }} {{ \\global \\music }}
  >>
  \\layout {{
    \\context {{ \\Score \\override BarNumber.break-visibility = ##(#f #t #t) }}
  }}
}}
"""


# --------------------------------------------------------------------------- Guitar Pro
def guitar_pro(tab: Tab, path: Path, artist: str = "") -> Path:
    import guitarpro as gp

    song = gp.Song(title=tab.title, subtitle=tab.subtitle, artist=artist, music=tab.composer,
                   tab="automatic transcription", tempo=int(round(tab.bpm)))
    track = song.tracks[0]
    track.name = "Acoustic Guitar (nylon)"
    track.channel.instrument = 24
    track.fretCount = 20
    track.strings = [gp.GuitarString(i + 1, p) for i, p in enumerate(reversed(tab.tuning))]
    bars = _pieces(tab)
    for b, pieces in enumerate(bars):
        if b > 0:
            song.newMeasure()
        header = song.measureHeaders[b]
        header.timeSignature = gp.TimeSignature(tab.beats_per_bar, gp.Duration(4))
        voice = track.measures[b].voices[0]
        voice.beats = []
        for s, l, notes, tie_in, tie_out in pieces:
            val, dot = {1: (16, 0), 2: (8, 0), 3: (8, 1), 4: (4, 0), 6: (4, 1), 8: (2, 0)}[l]
            beat = gp.Beat(voice, duration=gp.Duration(val, bool(dot)),
                           status=gp.BeatStatus.rest if notes is None else gp.BeatStatus.normal)
            if notes is not None:
                sym = None if tie_in else tab.chords.get(b * tab.spb + s)
                if sym:
                    frets = [-1] * 7
                    for n in notes:
                        frets[5 - n.string] = n.fret
                    fretted = [x for x in frets if x > 0]
                    beat.effect.chord = gp.Chord(
                        6, sharp=False, add=False, name=sym[:22], show=True,
                        firstFret=min(fretted) if fretted and max(fretted) > 4 else 1, strings=frets[:6])
                for n in notes:
                    beat.notes.append(gp.Note(beat, value=n.fret, string=6 - n.string,
                                              type=gp.NoteType.tie if tie_in else gp.NoteType.normal))
            voice.beats.append(beat)
    gp.write(song, str(path))
    return Path(path)


# -------------------------------------------------------------------------------- MIDI
def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def midi(notes: list[TabNote], path: Path, program: int = 24) -> Path:
    """Performance-aligned MIDI (120 BPM, 960 ticks/beat -> 1920 ticks per second)."""
    tps = 1920
    ev = []
    for n in notes:
        vel = int(np.clip(50 + 70 * n.conf, 1, 127))
        ev.append((int(round(n.onset * tps)), 1, 0x90, n.pitch, vel))
        ev.append((int(round(n.offset * tps)), 0, 0x80, n.pitch, 0))
    ev.sort()
    data = b"\x00\xff\x51\x03" + (500000).to_bytes(3, "big") + b"\x00" + bytes([0xC0, program])
    t = 0
    for tick, _, st, p, v in ev:
        data += _vlq(tick - t) + bytes([st, p, v])
        t = tick
    data += b"\x00\xff\x2f\x00"
    Path(path).write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, 960) + b"MTrk"
                           + struct.pack(">I", len(data)) + data)
    return Path(path)


def notes_csv(tab: Tab, path: Path) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bar", "beat", "sixteenth", "onset_s", "offset_s", "midi", "note", "string",
                    "fret", "confidence"])
        for n in sorted(tab.notes, key=lambda n: (n.q, -n.string)):
            bar, r = divmod(n.q, tab.spb)
            name = NAMES_SHARP[n.pitch % 12] + str(n.pitch // 12 - 1)
            w.writerow([bar + 1, r // 4 + 1, r % 4 + 1, f"{n.onset:.3f}", f"{n.offset:.3f}", n.pitch,
                        name, 6 - n.string, n.fret, f"{n.conf:.2f}"])
    return Path(path)
