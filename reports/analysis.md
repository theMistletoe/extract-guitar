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
* exp004 X-LANCE guitar: val SDR 3.12 dB (real 1.69, synthetic 4.35), SIR 6.5, **SAR 8.4**,
  lowest target leak proxy (0.03). Very uneven: best model on sax duo / piano / winds /
  nylon-chamber clips (13–19 dB), poor on plucked / strings / clean-electric clips where
  HTDemucs-ft is 10–18 dB better → strong complementarity (ensemble candidate).
* **Finding:** `gtr_mss.pth` (X-LANCE) is bit-identical to SW's trunk + guitar head
  (max |Δw| = 0.0). The two "models" are one; exp005 (SW guitar) reproduces exp004 exactly,
  so SW/X-LANCE are treated as a single ensemble member from here on (confirmed: exp005
  reproduces every exp004 metric exactly — 3.117 dB mean, 0.747 median, SIR 6.47, SAR 8.40).

### Phase 5 — ensembles (Strategy D), first round

* exp006 waveform mean of HTDemucs-ft + SW guitar: val SDR **3.64 dB** (real 1.94, synthetic
  5.10), median 3.59 → Champion (+0.57 dB over the best single model). The two members fail
  on different clips, so averaging mostly removes catastrophic per-clip failures (median
  rises from 2.1/0.7 to 3.6 dB).
* exp007 magnitude mean (estimate phase): 3.61 dB — no gain over the waveform mean.
* Target-song inspection of exp006 (report.html): smooth rising arcs at 300–700 Hz around
  95–115 s look like violin/clarinet glissandi leaking into the guitar stem (a guitar plays
  discrete pitches). Hypothesis for Phase 7: giving a refiner explicit violin/woodwind
  estimates as *negative evidence* should remove this class of leakage.

### Headroom (reports/oracle_bounds.json)

Oracle masks computed from the ground truth on the same 41 clips: mixture-as-estimate
−5.49 dB, ideal ratio mask 9.57 dB, ideal Wiener mask 10.70 dB (real 8.22 / synthetic
12.84). The first ensemble (3.64 dB) is ≈7 dB below the Wiener oracle, so there is real
headroom — the validation set is hard (guitar 7–14 dB below the rest in many clips) but not
saturated.
* exp008 mask mean (mixture phase, masks clipped to [0,1]): 3.30 dB (−0.34) — clipping the
  ratio masks and using the mixture phase loses information the waveform mean keeps.
* exp009–011 adding becruily as a third member: waveform mean 3.59, magnitude median 3.54,
  mask mean 3.26 dB — a weaker member dilutes the average; not every extra model helps.
* exp012 Wiener post-filter on the Champion ensemble: 3.52 dB (−0.12) — re-estimating the
  mask from the ensemble's own target/residual PSDs sharpens it and removes guitar energy.
* exp013/014 weight search (35:65 / 65:35): 3.62 / 3.58 dB — equal weights stay best; the
  optimum is flat, so no fine weight tuning on the validation set is warranted.
* Failure analysis of the Champion (reports/failure_modes.md): 13/41 clips `guitar_removed`
  (retention < 0.4), 9 clips electric-guitar leakage, 3 plucked-instrument leakage, 12 ok.
  The SDR-optimal per-clip gain has median 0.82 and a global gain > 1 lowers SDR, so the
  missing guitar is *local* (time-frequency holes), not a level problem, while other regions
  leak — a TF-local correction (refiner) is needed rather than a gain/threshold change.
* exp016 magnitude **max** of the two members: retention 0.77 (vs 0.60) but leakage −8.4 dB
  (vs −10.4) → SDR 2.86 dB (−0.78). Simply keeping more energy trades the "holes" for leakage;
  the PRD §17 trade-off is real and symmetric here.
* exp017 magnitude **min** (keep only what both members agree on): cleanest output so far
  (leakage −14.0 dB) but retention 0.45 → SDR 3.27 dB (−0.37). Too aggressive for the
  transcription use case (PRD §17), so the waveform mean remains the operating point.
