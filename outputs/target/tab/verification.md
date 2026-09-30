# Tab verification (no ground truth exists for this song)

| check | result |
|---|---|
| notes in the tab | 1250 |
| CRNN onset evidence per note (median) | 0.75 |
| notes with weak onset evidence (< 0.4) | 103 |
| strong activations not in the tab (>= 80 ms) | 11 |
| chroma cosine stem vs. resynthesised tab (median frame / bar) | 0.89 / 0.88 |
| bars with chroma cosine < 0.7 | none |
| agreement with Basic Pitch (onset F1, 50 ms) | 0.67 (pitch class 0.69) |
| tab notes detected by k of 3 guitar checkpoints (k: count) | 0: 0, 1: 67, 2: 113, 3: 1070 |
| expected wrong notes (benchmark precision per agreement level) | about 87 (67-107) of 1250 |
| notes heard by >= 2 checkpoints but not in the tab | 28 |
| repeated passages: note missing at the repeat although heard | 7 |
| max fret span in a chord / max fret | 4 / 7 |
| hand shifts > 5 frets | 0 |

Bars to double-check by ear first (most review reasons, see review.json): 160, 161, 132, 17, 59, 124, 19, 31, 58, 76, 101, 103, 117, 131, 142, 155, 157, 3, 8, 11, 12, 14, 22, 26, 28.
