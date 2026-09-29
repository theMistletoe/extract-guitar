# Tab verification (no ground truth exists for this song)

| check | result |
|---|---|
| notes in the tab | 1256 |
| CRNN onset evidence per note (median) | 0.75 |
| notes with weak onset evidence (< 0.4) | 107 |
| strong activations not in the tab (>= 80 ms) | 10 |
| chroma cosine stem vs. resynthesised tab (median frame / bar) | 0.89 / 0.88 |
| bars with chroma cosine < 0.7 | none |
| agreement with Basic Pitch (onset F1, 50 ms) | 0.67 (pitch class 0.69) |
| tab notes detected by k of 3 guitar checkpoints (k: count) | 0: 0, 1: 79, 2: 107, 3: 1070 |
| notes heard by >= 2 checkpoints but not in the tab | 34 |
| repeated passages: note missing at the repeat although heard | 8 |
| max fret span in a chord / max fret | 4 / 7 |
| hand shifts > 5 frets | 0 |

Bars to double-check by ear first (most review reasons, see review.json): 160, 161, 50, 59, 76, 58, 124, 131, 132, 142, 157, 11, 19, 31, 101, 103, 105, 117, 130, 155, 3, 8, 9, 12, 14.
