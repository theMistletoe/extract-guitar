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
| fl | clean guitar | guitarset | 0.926 | 0.892 | 0.909 | 0.771 |
| fl | clean guitar | gaps | 0.960 | 0.896 | 0.927 | - |
| fl | clean guitar | all | 0.936 | 0.893 | 0.914 | 0.771 |
| fl | separated from mix | guitarset | 0.919 | 0.841 | 0.878 | 0.805 |
| fl | separated from mix | gaps | 0.921 | 0.867 | 0.893 | - |
| fl | separated from mix | all | 0.920 | 0.848 | 0.883 | 0.805 |
| gaps_paper | clean guitar | guitarset | 0.927 | 0.896 | 0.911 | 0.779 |
| gaps_paper | clean guitar | gaps | 0.949 | 0.904 | 0.926 | - |
| gaps_paper | clean guitar | all | 0.933 | 0.899 | 0.916 | 0.779 |
| gaps_paper | separated from mix | guitarset | 0.923 | 0.876 | 0.899 | 0.783 |
| gaps_paper | separated from mix | gaps | 0.917 | 0.900 | 0.908 | - |
| gaps_paper | separated from mix | all | 0.922 | 0.883 | 0.902 | 0.783 |
| fl+gaps_paper | clean guitar | guitarset | 0.938 | 0.909 | 0.923 | 0.774 |
| fl+gaps_paper | clean guitar | gaps | 0.961 | 0.907 | 0.933 | - |
| fl+gaps_paper | clean guitar | all | 0.944 | 0.908 | 0.926 | 0.774 |
| fl+gaps_paper | separated from mix | guitarset | 0.923 | 0.872 | 0.897 | 0.771 |
| fl+gaps_paper | separated from mix | gaps | 0.946 | 0.900 | 0.922 | - |
| fl+gaps_paper | separated from mix | all | 0.930 | 0.880 | 0.904 | 0.771 |

## How reliable the review signals are (fl+gaps_paper, all material)

`scripts/verify_tab.py` flags tab notes that few of the distinct checkpoints (kroma, fl, gaps_paper; guitar_kroma = guitar-gaps) hear, and notes that >= 2 of them decode but the tab lacks.  The same rules applied here:

| condition | signal | notes | correct / real |
|---|---|---|---|
| clean guitar | tab note heard by 3 of 3 | 756 | 0.97 |
| clean guitar | tab note heard by 2 of 3 | 85 | 0.87 |
| clean guitar | tab note heard by 1 of 3 | 41 | 0.66 |
| clean guitar | heard by >= 2, not in the tab | 13 | 0.38 |
| separated from mix | tab note heard by 3 of 3 | 731 | 0.96 |
| separated from mix | tab note heard by 2 of 3 | 94 | 0.84 |
| separated from mix | tab note heard by 1 of 3 | 43 | 0.53 |
| separated from mix | heard by >= 2, not in the tab | 6 | 0.33 |
