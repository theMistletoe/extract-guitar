# Agent review of tab-vs-stem discrepancies: blind test on the benchmark

The benchmark (`scripts/bench_tab.py`) has known notes.  `scripts/compare_tab_audio.py --bench` flags tab notes that may be
wrong and notes the tab may lack; the items below were judged by agents that saw only the cached analysis (constant-Q levels
of stem / rendering / residual, model posteriors, Basic Pitch; via `compare_tab_audio.py probe`), never the labels.  Each of
10 batches was judged independently from two perspectives: an *acoustic* analyst (harmonic rises, overtones, leakage) and an
adversarial *skeptic* (argues against each suspicion first).  Items: 83 flagged tab notes, 20
random unflagged tab notes (controls), 46 missing-note candidates.  Verdicts: `reports/tab_compare_agents_blind_test.json`.

| tab notes (103, 30 wrong) | AUC | AUC, flagged only | called wrong | of which wrong |
|---|---|---|---|---|
| statistical model (cross-validated P(wrong)) | 0.71 | 0.59 | 25 | 52% |
| acoustic agent | 0.86 | 0.82 | 16 | 88% |
| skeptic agent | 0.81 | 0.75 | 7 | 86% |

Controls called wrong: 0 (acoustic), 0 (skeptic) of 20.
AUC gain of the acoustic agent over the model: +0.15 (bootstrap 95 % CI +0.05 to +0.27).  All 14 correct "wrong" calls of
the acoustic agent also proposed the right fix (13 removals, 1 octave down); its precision 14/16 has a Wilson 95 % CI of 64-97 %.

| missing candidates (46, 8 real) | AUC | called real | of which real |
|---|---|---|---|
| statistical model | 0.55 | 6 | 17% |
| acoustic agent | 0.74 | 4 | 75% |
| skeptic agent | 0.78 | 6 | 67% |
| both agents | - | 4 | 75% |

Rule adopted for the target song before its verdicts were read: correct a tab note when the acoustic agent calls it wrong
(apply its fix); add a missing note only when both agents call it real.  Applied to these benchmark items the rule raises
note F1 from 0.904 to 0.911 (precision 0.930 to 0.943); this is measured on the same items the rule was chosen on.
Samples are small (30 wrong notes, 8 real missing notes), so these rates are rough.
