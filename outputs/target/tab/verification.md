# Tab verification (no ground truth exists for this song)

| check | result |
|---|---|
| notes in the tab | 1255 |
| CRNN onset evidence per note (median) | 0.86 |
| notes with weak onset evidence (< 0.4) | 35 |
| strong activations not in the tab (>= 80 ms) | 20 |
| chroma cosine stem vs. resynthesised tab (median frame / bar) | 0.88 / 0.87 |
| bars with chroma cosine < 0.7 | none |
| agreement with Basic Pitch (onset F1, 50 ms) | 0.64 (pitch class 0.66) |
| max fret span in a chord / max fret | 4 / 10 |
| hand shifts > 5 frets | 0 |

Bars to double-check by ear first (lowest chroma agreement): 62, 155, 127, 32, 84, 15, 139, 112, 27, 97, 5, 38.
