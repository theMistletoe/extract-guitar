# Tab pipeline: end-to-end accuracy on music with known notes

Guitar with note ground truth (GuitarSet bossa-nova comping, 4 players; GAPS classical-guitar test pieces incl. a choro) mixed with violin + clarinet (URMP) + percussion (RawStems) at the target song's balance (guitar about 4.7 dB below the rest), separated with the Champion pipeline (`--quality max`), then transcribed exactly like the target song.  Note F1: onset within 50 ms and same pitch (mir_eval).  String accuracy: share of correctly detected GuitarSet notes placed on the performer's string.  Built by `scripts/bench_tab.py` (129 s, 917 notes).

Caveat: `gaps_paper` may have been trained with GuitarSet (the GAPS paper reports a supervised GuitarSet setting), so its rows on GuitarSet material may be optimistic; the GAPS test pieces (`gaps` rows) are unseen by all checkpoints' documented training data.

| checkpoints | condition | material | precision | recall | F1 | string acc. |
|---|---|---|---|---|---|---|
| kroma | clean guitar | guitarset | 0.929 | 0.787 | 0.852 | 0.786 |
| kroma | clean guitar | gaps | 0.961 | 0.907 | 0.933 | - |
| kroma | clean guitar | all | 0.939 | 0.822 | 0.877 | 0.786 |
| kroma | separated from mix | guitarset | 0.920 | 0.750 | 0.826 | 0.781 |
| kroma | separated from mix | gaps | 0.945 | 0.885 | 0.914 | - |
| kroma | separated from mix | all | 0.928 | 0.790 | 0.853 | 0.781 |
| fl+gaps_paper | clean guitar | guitarset | 0.938 | 0.909 | 0.923 | 0.774 |
| fl+gaps_paper | clean guitar | gaps | 0.961 | 0.907 | 0.933 | - |
| fl+gaps_paper | clean guitar | all | 0.944 | 0.908 | 0.926 | 0.774 |
| fl+gaps_paper | separated from mix | guitarset | 0.923 | 0.872 | 0.897 | 0.771 |
| fl+gaps_paper | separated from mix | gaps | 0.946 | 0.900 | 0.922 | - |
| fl+gaps_paper | separated from mix | all | 0.930 | 0.880 | 0.904 | 0.771 |
