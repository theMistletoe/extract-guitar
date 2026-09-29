## What worked

1. **Ensembling models from different families.** No single public model is good everywhere:
   HTDemucs-6s-guitar-FT (hybrid waveform/spectrogram) and the MVSep Mega-53 acoustic head
   (band-split RoFormer, very high SIR, low SAR) fail on different clips. Their plain waveform
   mean (exp023) beats every single model by ≈1 dB (paired 95% CIs exclude 0).
2. **Acoustic-specific evidence.** The Mega acoustic-guitar head is weak alone (3.13 dB, SAR
   1.2 dB) but is the most valuable ensemble member because it rejects electric guitar and
   other instruments (SIR 14.8 dB).
3. **Refining the ensemble instead of replacing it.** A 69 k-parameter gain refiner (exp030)
   that sees the member estimates as positive evidence and SW/Mega estimates of other
   instruments as negative evidence, starts as an identity on the ensemble and learns local
   0..2 gains. Trained only on disjoint clips (with hard-example oversampling), it adds
   +0.53 dB on validation (CI [+0.18, +0.86], 32/41 clips) and raises retention and SIR at
   the same time.
4. **Engineering for a CPU-only box.** bf16 autocast (1.5×, −40 dB difference to fp32), merged
   multi-head checkpoints (one trunk pass → 8 Mega stems), a FLAC stem cache and cached
   candidates made ~35 full experiments possible on 4 CPU cores.

## What failed (or did not help)

* Weight tuning of the ensemble (35:65, 65:35, 60:40, 40:60) — flat optimum, equal weights best.
* Wiener post-filter (−0.12 dB), magnitude-max (−0.78), magnitude-min (−0.37), mask-mean
  (−0.34) — all just slide along the leakage/retention frontier.
* Frequency-dependent least-squares weights fitted on the training clips (−0.35 dB): the
  weights did not transfer to validation (train/validation content shift, 6 s vs default
  chunks).
* Mask-mode refiners r1 / r2 (3.89 / 3.88 dB): they learned to clean, not to restore, and
  were capped by their weak mask-mean starting point; hard-example mining improved their
  median and SAR but not the mean.
* becruily Mel-RoFormer as an extra member (dilutes the average); the Mega all-guitar head
  as the third member (acoustic head is better).
* "X-LANCE fine-tuned guitar model": turned out to be byte-identical to SW's guitar head.

## Remaining artifacts (target song)

* Violin/clarinet glissandi are attenuated but still audible/visible in places (e.g. around
  95–100 s and 108–113 s).
* In dense passages some guitar energy is still missing (retention on validation 0.51 — about
  half of the guitar's time-frequency energy is not fully recovered in hard mixes).
* The target is AAC-encoded with a 16 kHz low-pass; nothing above 16 kHz can be recovered.

## Known failure modes (validation, see reports/failure_modes.md)

* Guitar buried ≥ 12 dB below the rest (`buried`, `dense`): large guitar losses.
* Electric guitar with a clean/crunch tone overlapping the acoustic's register.
* Mandolin/banjo/ukulele (plucked) and bowed-string pizzicato: partly kept as "guitar".

## Why the final model was selected

It has the best mean validation SDR of all experiments, the improvement over the previous
Champion is significant on paired clips, it holds on both validation families (real
multitracks and synthetic scenarios), its refiner was trained only on disjoint data, and on
the target song it has the lowest reference-free leakage proxy among the top candidates
while keeping guitar probability. It also matches the PRD priority order: less leakage
without destroying the guitar (retention went up, not down).
