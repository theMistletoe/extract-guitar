# Tab note model on clean GuitarSet (60 mic excerpts, all players and styles)

Every 6th excerpt of GuitarSet's mono mic recordings (steel-string), onset within 50 ms and same pitch (mir_eval), onset threshold 0.3, mean over excerpts.  `scripts/bench_tab.py guitarset`.

Caveat: the FL checkpoint is documented as zero-shot on GuitarSet, but the GAPS paper reports both supervised (GuitarSet-trained) and zero-shot results and does not say which released checkpoint is which, so `gaps_paper` (and ensembles containing it) may be optimistic here.

| checkpoints | precision | recall | F1 |
|---|---|---|---|
| kroma | 0.870 | 0.749 | 0.798 |
| fl | 0.886 | 0.907 | 0.896 |
| gaps_paper | 0.919 | 0.911 | 0.915 |
| fl+gaps_paper | 0.909 | 0.917 | 0.913 |
