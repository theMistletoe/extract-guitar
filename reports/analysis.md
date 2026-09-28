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
* r1 finished: best hold-out SDR 5.58 dB at epoch 9 (start 4.44, +1.14 dB on held-out
  training-distribution clips); later epochs fluctuate 5.4–5.5 (early stopping keeps epoch 9).
* Hard-example mining with r1 on the training clips (PRD §23): the worst quartile
  (SDR ≤ 2.4 dB, 30 clips) is 29× `guitar_removed` + 1× electric leakage — the same dominant
  failure as on validation. Next round: 48 new clips drawn with scenario weights
  buried 30 / dense 30 / clean_electric 2 / electric_band 2, and hard clips oversampled 3×.
* Second mining pass over all 168 clips (incl. the 48 new buried/dense clips): mean SDR 7.98 dB,
  worst quartile (≤ 1.92 dB) = 41 `guitar_removed` + 1 electric leak → r2 is trained on
  148 clips (20 held out) with these clips oversampled 3×.
* r1 final (epoch 9) on validation (unrecorded check): 3.89 dB (real 2.24, synthetic 5.32),
  leakage −15.3 dB, retention 0.41 — cleaner but lower than the epoch-4 snapshot (4.00): the
  refiner keeps improving on training-distribution hold-out while validation degrades.
* Chunk-mismatch diagnostic on the 20 training hold-out clips (no validation data):
  refiner gain +1.14 dB with 6 s-chunk candidates (as trained) vs +0.79 dB with the models'
  default chunks (as at inference). The mismatch costs ≈0.35 dB; the larger loss (+0.79 →
  +0.31 dB on validation) comes from the content shift between training songs (mostly pop)
  and the validation songs. Consequences: (1) refine the *Champion* instead of a mask mean
  (gain mode, r3); (2) limit training to 6 epochs as regularisation against that shift — this
  choice uses the r1 validation curve (epoch 4 > 9) and is therefore disclosed here.
* exp028 **R_r1** (mask refiner, recorded benchmark): 3.89 dB (real 2.24, synthetic 5.32), SIR 9.4,
  SAR 4.1, leakage −15.3 dB, retention 0.41 → not promoted (−0.26 dB). The refiner moves the
  operating point toward "clean" (like the Mega head) rather than lifting the guitar holes.
* exp029 **R_r2** (mask refiner + hard-example mining round): 3.88 dB (median 3.02 vs 2.69 for r1),
  SIR 6.8, SAR 5.1 (r1: 9.4 / 4.1), leakage −13.9 dB → not promoted. Mining shifted the refiner
  away from over-cleaning (higher SAR, better median) but the mean did not move: the mask-mode
  refiner is capped by its weak starting point (mask mean of the candidates, 3.58 dB).
* **exp030 R_r3 — gain refiner on the Champion ensemble → new Champion**: validation SDR
  **4.68 dB** (median 3.23; real multitracks 2.93, synthetic 6.19), SI-SDR 1.79, SDRi 10.17,
  **SIR 9.5 / SAR 6.4**, retention **0.51**, leakage −12.9 dB, hard-case SDR 4.84. It raises
  retention (0.43 → 0.51) *and* SIR (8.4 → 9.5) at the same time, i.e. it moves the
  leakage/preservation frontier instead of sliding along it. Paired vs exp023: **+0.53 dB,
  95% CI [+0.18, +0.86], wins 32/41 clips**. Training: 168 disjoint clips (hard-example
  weighted), 6 epochs, best hold-out 6.28 dB from 5.23 at init.
* Target-song spectrogram check, 92–118 s (the passage with violin/clarinet glissandi): the
  smooth arcs (≈0.5–1 kHz at 95–100 s and 108–113 s) are clearly present in exp023 and are
  weaker but **still visible** in exp030; guitar onsets are preserved in both. Residual
  bowed/wind glissandi are therefore a known remaining artifact of the Champion.

### Phase 3 — Strategy B (two-stage) and C (cascades)

* exp031 **B: SW all-guitar → Mega acoustic head on that stem**: 3.05 dB (real 1.63, synthetic
  4.28), SIR 14.4, SAR 1.1, retention 0.32 — the same operating point as the acoustic head on
  the full mix (3.13 dB). Removing vocals/drums/etc. first does not make the head less
  conservative; two-stage brings no gain here.
* exp033 **B: SW all-guitar − Mega electric-guitar estimate of that stem**: **4.20 dB** (real 2.58,
  synthetic 5.60), SIR 9.1, SAR 5.9, retention 0.53 — +1.08 dB over SW alone and on par with the
  plain 2-model ensemble; the best *non-ensemble* pipeline. Subtracting the electric part keeps
  SW's fullness while removing its main confusion (electric guitar). Median is low (1.59): it
  still fails badly on some clips.

### Phase 9 — train/inference condition matching (the "c6" finding)

* exp032 R_r3_c6 = R_r3 with every member computed with 6 s chunks, exactly as the refiner's
  training candidates were. The validation part completed (per_clip.csv): **6.50 dB mean**
  (median 5.33; real 4.06, synthetic 8.62), retention 0.66, SIR 10.9, SAR 9.0 — +1.8 dB over
  R_r3. The run then crashed writing the target output (disk full), so it is re-run from
  scratch as a proper experiment, together with a control (`D_htft_macou_c6`: the same members
  at 6 s chunks *without* the refiner) to attribute the gain.
* **exp037 R_r3_c6 (rerun, complete record): 6.502 dB mean** (median 5.34, SI-SDR 4.38,
  SDRi 12.0; real 4.06 / synthetic 8.62; SIR 10.9 / SAR 8.9; retention 0.655, leakage
  -13.5 dB). Reproduces exp032 to within 0.01 dB. **Promoted to Champion** (+1.82 dB over
  exp030 R_r3). Target-song proxy: residual guitar prob 0.009, max leak prob 0.030.
* **Control exp038 D_htft_macou_c6 (plain ensemble, 6 s chunks, no refiner): 5.47 dB**
  vs 4.16 dB for the same ensemble at default chunks (exp023). Attribution:

  | | default chunks | 6 s chunks |
  |---|---|---|
  | plain HTDemucs-ft + Mega acoustic mean | 4.16 (exp023) | 5.47 (exp038) |
  | + gain refiner r3 | 4.68 (exp030) | **6.50** (exp037) |

  ~1.3 dB of the gain comes from the chunk length itself; the refiner adds +0.53 dB at
  default chunks and +1.03 dB at 6 s chunks (its training condition), so matching the
  training condition roughly doubles the refiner's contribution. Next: chunk-size sweep
  (4 / 8 / 10 s) on the plain ensemble (configs/queues/phase4_chunk_sweep.yaml).
* exp034 (Strategy C cascade) was interrupted by a container restart; re-queued.
* **Chunk sweep, 4 s (exp040 D_htft_macou_c4): 5.56 dB** (median 5.09) vs 5.47 at 6 s (exp038);
  per clip +0.09 dB, wins 19/41 — flat. Shorter than default helps; 4–6 s is a plateau.
  (exp039 was stopped by a container shutdown before any clip finished.)
* **Refiner r4 trained** (base = mean of SW-minus-electric, HTDemucs-ft, Mega acoustic; 5 pos +
  8 neg evidence stems; 6 s chunk candidates; r1 mining weights). Training hold-out SDR:
  base 6.15 → best **6.93 dB** at epoch 5 (r3: base 5.23 → 6.28). Validation benchmark: exp043.
* **exp043 R_r4_c6: 6.901 dB mean** (median 5.89, SI-SDR 5.07, SDRi 12.4; real 4.65 /
  synthetic 8.85; SIR 11.6 / SAR 9.3; retention 0.62, leakage -15.4 dB, hard-case 7.02).
  Per clip vs exp037: +0.40 dB, wins 35/41. **Promoted to Champion.** Cost: 3 members at 6 s
  chunks incl. the SW→Mega cascade — ~6 min per 20 s clip and ~59 min for the 132 s target on
  4 CPU cores (vs ~11 min for exp037).
* **Mega-side chunk sweep** (HTDemucs-ft at its native 7.8 s segment; HTDemucs cannot exceed
  it — exp041/042 crashed): Mega 6 s **5.58 dB** (exp044; +0.11 over both-at-6 s, 26/41 wins),
  Mega 8 s 5.08 dB (exp046; -0.51 vs Mega 6 s, 7/41 wins). Mega default is 20 s. The chunk
  effect is on the Mega side and monotone: shorter is better down to ~6 s. Mega 4 s: exp047.
  **Mega 4 s: 5.79 dB** (exp047; +0.21 vs Mega 6 s, 28/41 wins) — best plain ensemble so far.
  Next candidate: retrain the refiner on candidates with Mega at 4 s (HTDemucs native).
* **Strategy C, exp048 C_xlance_violin_clarinet** (SW guitar → subtract Mega violin → subtract
  Mega clarinet): **3.12 dB** (median 0.75; real 1.70 / synthetic 4.35; retention 0.60, leakage
  -10.4 dB) — identical to plain SW (exp004, 3.12 dB): removing the violin/clarinet
  estimates changes nothing on average, whereas removing the Mega *electric*-guitar estimate
  (exp033) gained +1.08 dB. Violin/clarinet leakage is not what limits SW here; the learned gain
  refiner handles residual leakage better. exp034 (the interrupted first attempt) is superseded by this record.
* **Refiner r5** = r4 recipe with the per-model chunk sizes from the Phase 4 sweep (HTDemucs at its
  native 7.8 s, Mega on the mix at 4 s; SW and the SW→Mega cascade stay at 6 s), in training and
  at inference. Training hold-out: base 6.47 → **7.48 dB** (r4: 6.15 → 6.93).
* **exp050 R_r5_m4: 7.616 dB mean** (median 6.65, SI-SDR 5.79, SDRi 13.1; real 4.89 / synthetic
  9.97; SIR 12.5 / SAR 9.5; retention 0.69, leakage -14.9 dB, hard-case 7.78). Per clip vs exp043:
  +0.72 dB, wins 32/41. **Promoted to Champion.** Target proxy: guitar prob 0.224 (r4 0.179), max
  leak prob 0.029, residual guitar prob 0.008. The recorded runtimes (validation 18 749 s, target
  4 348 s) include one-off recomputation of the SW→Mega cascade caused by a cache-key bug (fixed
  in f4858f5).
* **Strategy C, exp051 C_orch_xlance** (X-LANCE orchestral remover — strings+winds — first, then
  the SW/X-LANCE guitar model on the remainder): **2.92 dB** (median 1.12; real 1.54 / synthetic
  4.12; retention 0.61, leakage -10.2 dB) — 0.2 dB below plain SW (exp004, 3.12). Removing the
  orchestra before guitar extraction also removes guitar energy and adds artifacts the guitar
  model was not trained on.
* **Strategy C, exp052** (SW → strings remover → woodwind subtraction → guitar model): stopped
  after 10/41 clips at -0.42 dB vs plain SW on the same clips (2/10 wins); see its notes.md.
  All three Strategy C cascades are at or below plain SW.
* (exp053 was a 41-clip codec run stopped after its first clip to cut run time; no record.)
* **Codec robustness, exp054_R_r5_m4_aac128_real19**: the Champion pipeline on the 19 real-multitrack clips with the
  mixture passed through AAC 128k (references clean), paired against exp050 on the same clips:
  SDR 4.62 vs 4.89 dB (**-0.28 dB**, AAC better on 5/19), SI-SDR -0.45 dB, retention unchanged
  (0.63), leakage -12.1 vs -13.4 dB (+1.3 dB more leakage). Codec damage costs a little
  separation quality and mostly shows up as extra leakage, not as lost guitar. The target song
  is AAC, so its real quality is probably slightly below the lossless validation numbers.
