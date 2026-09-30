"""Guitar tablature transcription from an (extracted) acoustic guitar stem.

amt       note transcription (high-resolution CRNN, guitar checkpoint, safetensors only)
rhythm    tuning, beat grid, bar phase and swing-aware 16th quantisation
fretboard string/fret assignment (Viterbi over fingering x hand position)
chords    chord symbols from the transcribed notes
export    ASCII tab, MusicXML (notation + TAB), Guitar Pro 5, MIDI, CSV
"""
