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
* exp015 Mega-53 acoustic-guitar head: val SDR 3.13 dB (real 1.39, synthetic 4.63) with an
  extreme operating point — **SIR 14.8 dB** (by far the least interference) but SAR 1.2 dB and
  retention 0.33 (much of the guitar removed). The acoustic-specific head is precise but
  conservative: useful as "clean evidence" for an ensemble/refiner, not alone.
* exp018 Mega-53 all-guitar head: 2.55 dB (SIR 8.9, SAR 2.8, retention 0.41) — *lower* than the
  acoustic head on the same trunk (3.13), because electric guitar counts as interference here.

**Phase 2 summary (single models, same settings):** Mega acoustic 3.13 ≈ SW/X-LANCE 3.12 ≈
HTDemucs-6s guitar-FT 3.07 > HTDemucs-6s 2.58 ≈ Mega guitar 2.55 > becruily 2.10 dB. No single
model dominates: their failures are on different clips.

### Phase 5 — ensembles, second round (with the Mega acoustic head)

* exp019 waveform mean of **HTDemucs-ft + SW + Mega acoustic**: val SDR **4.09 dB** (real 2.50,
  synthetic 5.46), SIR 7.2, SAR 7.6, leakage −12.4 dB → Champion (+0.45 dB). The clean but
  conservative acoustic head (SIR 14.8) pulls leakage down while the two fuller models keep
  the guitar — three different operating points average better than any pair.
* exp020 same with the Mega *all-guitar* head instead: 3.71 dB (−0.37 vs exp019) — the
  acoustic-specific head is the better third member.
* exp021 HTDemucs-ft + Mega guitar (pair): 3.77 dB — beats the HTDemucs-ft + SW pair (3.64) but
  not the 3-member Champion.
* Paired statistics (`scripts/evaluate.py compare`, 5000 bootstrap resamples over the 41 clips):
  exp019 vs the best single models: +0.96 dB vs Mega acoustic (95% CI [+0.12, +1.75], wins
  27/41), +0.97 vs SW (CI [−0.06, +1.92], wins 31/41), +1.01 vs HTDemucs-ft (CI [+0.04, +1.92],
  wins 29/41). exp019 vs the 2-member exp006: +0.45 dB (CI [−0.02, +0.89], wins 27/41) — a
  consistent but only borderline-significant step.

### Phase 5 — ensembles, third round (member selection)

* exp022 SW + Mega acoustic: 3.60 dB. exp024 four members (+ stock HTDemucs): 4.03 dB.
* exp023 **HTDemucs-ft + Mega acoustic** (pair): **4.16 dB** (real 2.43, synthetic 5.65), SIR 8.4,
  SAR 6.6, leakage −13.2 dB, retention 0.43 → Champion (+0.07 dB over exp019, i.e. within
  noise, but it is cheaper — no SW pass — and trades a little retention for less leakage,
  which matches the PRD priority order: leakage reduction (2) before loss reduction (3)).
  The two members are the most *dissimilar* pair (hybrid waveform/spectrogram vs. band-split
  transformer; full vs. clean operating point), which is why they combine best.
* exp025/026 weights 60:40 / 40:60 for the pair: 4.10 / 4.14 dB — equal weights (4.16) stay best;
  again a flat optimum, so no weight tuning on the validation set is adopted.
* Band-wise least-squares weights fitted on the 120 disjoint training clips
  (artifacts/band_weights_htft_macou.json): Mega acoustic 0.81–0.98 and HTDemucs-ft 0.01–0.24
  below 9.6 kHz, 0.66 / 0.38 above. Evaluated as challenger `D_bandw_htft_macou` (weights
  never see the validation set).
* `D_bandw_htft_macou` (training-fitted band weights): 3.81 dB (−0.35 vs equal weights). The
  weights learned on the training clips do **not** transfer. Two suspects: (a) training
  candidates were computed with 6 s chunks while inference uses each model's default chunk
  (20 s for Mega), so the members behave differently; (b) the training songs are mostly pop
  with electric guitar, where the acoustic-only head is favoured. The same risk applies to the
  refiner, so its validation result is the real test.

### Phase 7 — learned mask refiner (stacking)

* Candidates on 120 training clips (6 s, 44 min SW + 2 min HTDemucs-ft + 24 min Mega; one Mega
  pass yields all 8 heads). Refiner r1 (69 k params) hold-out SDR on *training-distribution*
  clips: 4.44 dB at init (= mask mean) → 4.70 (epoch 1) → 5.05 (epoch 2)…
* Early transfer check (r1 snapshot, epoch 4, not a recorded experiment): validation SDR 4.00 dB
  vs 3.58 for its own starting point (mask mean of the 4 positives) → **+0.42 dB of learned
  correction transfers to the validation set**, but the mask-mean starting point is weaker than
  the Champion's waveform mean (4.16). Design change → **gain refiner (r3)**: a 0..2 TF gain
  applied to the Champion ensemble's complex STFT (identity at init, so it starts at 4.16 dB),
  trained with the r1 hard-example weights.
* `reports/tradeoff.png` (leakage vs retention, colour = SDR): all experiments lie on one
  diagonal frontier from mag-max (−8.4 dB leak, 0.77 retention) to the Mega acoustic head
  (−17.3 dB, 0.33); the best SDRs sit in the middle (−12…−13 dB leakage, 0.43–0.48 retention).
  Moving *along* the frontier (gains, max/min, Wiener) does not help; the refiner's job is to
  move the frontier itself.
