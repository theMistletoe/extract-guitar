## What worked

1. **Ensembling models from different families.** No single public model is good everywhere.
   The best single models reach 3.07–3.13 dB (HTDemucs-6s-guitar-FT 3.07, SW BS-RoFormer guitar
   3.12, MVSep Mega-53 acoustic head 3.13). The waveform mean of HTDemucs-ft and the Mega acoustic
   head (exp023) reaches 4.16 dB, because the two fail on different clips.
2. **Acoustic-specific evidence.** The Mega acoustic-guitar head is weak alone (retention 0.33)
   but it rejects electric guitar and other instruments, which makes it the most valuable
   ensemble member. The same idea used by subtraction also works: SW guitar minus the Mega
   *electric*-guitar estimate (Strategy B, exp033) scores 4.20 dB, +1.08 over SW alone.
3. **Inference chunk length (Phase 4).** This was the largest single gain in the project. Mega's
   default chunk is 20 s. With Mega at 4 s and HTDemucs at its native 7.8 s segment, the plain
   2-model mean goes from 4.16 dB (exp023) to **5.79 dB** (exp047). The effect sits entirely on
   the Mega side and is monotone: 8 s 5.08 → 6 s 5.58 → 4 s 5.79. HTDemucs cannot run chunks
   longer than 7.8 s, and running it shorter (6 s or 4 s) is slightly worse.
4. **Refining the ensemble instead of replacing it.** A ~70 k-parameter *gain* refiner starts
   as an identity on the ensemble mean and learns local 0..2 gains. It sees the member estimates
   as positive evidence and SW/Mega estimates of the other instruments as negative evidence. It
   is trained only on disjoint clips, with hard-example oversampling.
   * r3 on 2 members: +0.53 dB at default chunks (exp030, 4.68 dB).
   * r3 with every member computed exactly as in training (6 s chunks): +1.03 dB over its own
     base (exp037, 6.50 dB). Matching the train and inference conditions doubled the refiner's
     contribution.
   * r4 on a 3-member base (adds the Strategy-B stem): **6.90 dB** (exp043), +0.40 over exp037,
     better on 35 of 41 clips. It raises SIR (11.6 dB) and lowers leakage (−15.4 dB).
5. **Engineering for a CPU-only box.** The main tools were bf16 autocast (1.5× faster, −40 dB
   difference from fp32), merged multi-head checkpoints (one trunk pass gives all Mega stems),
   and a content-addressed FLAC stem cache that makes interrupted runs resumable. Together they
   made ~45 full experiments possible on 4 CPU cores.

## What failed (or did not help)

* Weight tuning of the ensemble (35:65 … 65:35): flat optimum, equal weights best.
* Wiener post-filter (−0.12 dB), magnitude max/min, mask mean. These only move along the
  leakage/retention frontier.
* Frequency-dependent least-squares ensemble weights fitted on the training clips (−0.35 dB).
  They did not transfer to validation.
* Mask-mode refiners r1/r2 (3.89/3.88 dB). They learned to clean, not to restore, and were capped
  by their weak mask-mean starting point.
* Strategy C, subtracting Mega violin and clarinet estimates after SW (exp048): 3.12 dB, i.e.
  no change from plain SW. Violin/clarinet leakage is not what limits the guitar estimate.
* Strategy B with the Mega *acoustic* head applied to the SW guitar output (exp031, 3.05 dB).
  The Mega acoustic head is useful on the mix, not as a second stage.
* becruily Mel-RoFormer as an extra member (dilutes the average). The Mega all-guitar head as a
  member (the acoustic head is better).
* The "X-LANCE fine-tuned guitar model" turned out to be byte-identical to SW's guitar head.

## Remaining artifacts (target song)

* Violin/clarinet glissandi are attenuated but still audible in places (e.g. around 95–100 s
  and 108–113 s).
* In dense passages some guitar energy is still missing. Validation retention is 0.62: in hard
  mixes about a third of the guitar's time-frequency energy is not fully recovered.
* The target is AAC-encoded with a 16 kHz low-pass, so nothing above 16 kHz can be recovered.

## Known failure modes (validation; see reports/failure_modes.md)

* Guitar buried ≥ 12 dB below the rest (`buried`, `dense`): large guitar losses.
* Electric guitar with a clean/crunch tone overlapping the acoustic's register. This is the
  most common leak class on the real multitracks.
* Mandolin/banjo/ukulele (plucked) and pizzicato strings are partly kept as "guitar".

## Why the final model was selected

It has the best mean validation SDR of all experiments, and the gain over the previous Champion
holds clip by clip, not just on average. It improves both validation families (real multitracks
and synthetic scenarios). Its refiner was trained only on disjoint data. On the target song it
keeps the reference-free leakage proxy at the level of the best candidates while raising guitar
probability. It also follows the PRD priority order: less leakage (SIR up, leakage down) without
destroying the guitar.
