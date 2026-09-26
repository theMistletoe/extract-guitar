## Findings log (updated as experiments complete)

### Phase 2 — single pretrained models (Strategy A)

* exp001 HTDemucs 6s: val SDR 2.58 dB (real multitracks 0.53, synthetic 4.35), SIR 5.7,
  SAR 6.5, retention 0.61. Strong on plucked/vocal/piano/strings/winds scenarios (8–10 dB),
  collapses when electric guitar is present (distorted −6.4, electric band −2.7) and when the
  acoustic guitar is buried (−3.4) — it is an *all-guitar* model.
* exp002 becruily Mel-RoFormer guitar: val SDR 2.10 dB (real 1.70, synthetic 2.45), SIR 3.2,
  SAR 7.4, retention 0.52. Better than HTDemucs on real multitracks but it misses the nylon
  classical guitar in several synthetic chamber clips (retention down to 0.05) and leaks
  violin on the target (AST strings probability 0.19 vs 0.04 for HTDemucs).
* exp003 HTDemucs 6s + MoisesDB guitar fine-tune: val SDR 3.07 dB (real 1.02, synthetic
  4.84), SIR 6.7, SAR 6.8, retention 0.62 → new Champion (+0.49 dB over stock HTDemucs);
  the fine-tune helps on both families, i.e. it is not a MoisesDB-specific artefact.
