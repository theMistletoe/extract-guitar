# Final report — acoustic guitar extraction

```text
Best architecture:   Phase 9 gain refiner (r5): r4 members/evidence, HTDemucs at native chunk, Mega on the mix at 4 s, SW and the SW->Mega cascade at 6 s (all exactly as in training)
Best checkpoint(s):  xlance_gtr (chenxie95/xlance-msr-ckpt:gtr_mss.pth, sha256 813a92bd210e1fc7); mega_multi (noblebarkrr/BS-Roformer-MVSep-Mega-53-stems:multihead, sha256 1ce0838f692246a3); htdemucs6s_gtrft (None:htdemucs_6s, sha256 demucs:htdemucs_); sw6 (enerjazzer/BS-ROFO-SW-Fixed:BS-Rofo-SW-Fixed.ckpt, sha256 24e7d35ee9c64415); learned refiner artifacts/refiner/r5/model.pt (in git)
Pipeline:            R_r5_m4 (experiment exp050_R_r5_m4)
Dataset:             datasets/validation — 41 clips x 12 s (19 real multitrack + 22 scenario), exact GT
Validation SDR:      7.62 dB mean / 6.65 dB median (real multitracks 4.89, synthetic 9.97)
Validation SI-SDR:   5.79 dB (SDRi 13.10 dB)
SIR / SAR (BSS):     12.49 / 9.45 dB
Leakage:             -14.91 dB excess energy; target retention 0.693
Runtime:             target song 4348 s for 132 s of audio (search setting); validation 18749 s (wall clock incl. any member not yet in the stem cache)
Hardware:            x86_64, 4 cores, torch 2.14.0+cpu, CUDA=False
Number of experiments: 44
```

Best single pretrained model (Strategy A): `exp015_A_mega_acoustic` — validation SDR 3.13 dB; Champion improves on it by +4.49 dB.

## Champion pipeline

```text
  1. {"id": "g", "model": "xlance_gtr", "input": "mix", "params": {"num_overlap": 2, "precision": "bf16", "chunk_size": 264600}}
  2. {"id": "swme", "model": "mega_multi", "input": "g", "take": "~electric-guitar", "params": {"num_overlap": 2, "precision": "bf16", "chunk_size": 264600}}
  3. {"id": "ht", "model": "htdemucs6s_gtrft", "input": "mix", "params": {"num_overlap": 2, "precision": "bf16"}}
  4. {"id": "mm", "model": "mega_multi", "input": "mix", "params": {"num_overlap": 2, "precision": "bf16", "chunk_size": 176400}}
  5. {"id": "sw", "model": "sw6", "input": "mix", "params": {"num_overlap": 2, "precision": "bf16", "chunk_size": 264600}}
  6. {"id": "ref", "refiner": "artifacts/refiner/r5/model.pt", "positives": ["swme", "ht.guitar", "mm.acoustic-guitar", "sw.guitar", "mm.guitar"], "negatives": ["sw.other", "sw.piano", "sw.vocals", "sw.bass", "sw.drums", "mm.violin", "mm.woodwind", "mm.electric-guitar"]}
```

## All experiments

See `docs/experiments.md` (full table and per-experiment Hypothesis / Change / Result / Conclusion / Next), `experiments/results.csv`, and the chronological lab notebook `reports/analysis.md`.

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
   * r4 on a 3-member base (adds the Strategy-B stem): 6.90 dB (exp043), +0.40 over exp037,
     better on 35 of 41 clips. It raises SIR (11.6 dB) and lowers leakage (−15.4 dB).
   * r5 = r4 with the per-model chunk sizes from item 3, in training and at inference:
     **7.62 dB** (exp050, the final Champion), +0.72 over r4, better on 32 of 41 clips. SIR
     12.5 dB, SAR 9.5 dB, retention 0.69 (r4 0.62), leakage −14.9 dB. Over the best single
     pretrained model (3.13 dB) this is +4.49 dB.
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
* Strategy C cascades. Subtracting Mega violin and clarinet estimates after SW (exp048): 3.12 dB,
  i.e. no change from plain SW. Removing strings and winds with the X-LANCE orchestral model
  *before* the guitar model (exp051): 2.92 dB, 0.2 dB worse than SW alone. Violin/clarinet
  leakage is not what limits the guitar estimate.
* Strategy B with the Mega *acoustic* head applied to the SW guitar output (exp031, 3.05 dB).
  The Mega acoustic head is useful on the mix, not as a second stage.
* becruily Mel-RoFormer as an extra member (dilutes the average). The Mega all-guitar head as a
  member (the acoustic head is better).
* The "X-LANCE fine-tuned guitar model" turned out to be byte-identical to SW's guitar head.

## Remaining artifacts (target song)

* Violin/clarinet glissandi are attenuated but still audible in places (e.g. around 95–100 s
  and 108–113 s).
* In dense passages some guitar energy is still missing. Validation retention is 0.69: in hard
  mixes roughly 30% of the guitar's time-frequency energy is not fully recovered.
* The target is AAC-encoded with a 16 kHz low-pass, so nothing above 16 kHz can be recovered.

## Known failure modes (validation; see reports/failure_modes.md)

With the final Champion, 25 of 41 clips have no failure label (mean SDR 11.5 dB). The rest:

* Electric guitar overlapping the acoustic's register (5 clips labelled `electric_guitar_leak` /
  `electric_clean_leak`, mean SDR −1.1 / 0.7 dB). This is the worst failure mode: the models keep
  a clean or crunch electric as "guitar".
* Guitar far below the rest of the mix (5 clips labelled `guitar_removed`, mean SDR 2.1 dB:
  both `buried` scenarios, `band_pop`, and two real multitracks): the guitar is partly removed
  along with the band.
* Bass leakage (3 clips: one real multitrack, the `dense` and `distorted_electric` scenarios),
  plus single cases of vocal, violin and plucked-instrument (mandolin/banjo) leakage.

## Why the final model was selected

It has the best mean validation SDR of all experiments, and the gain over the previous Champion
holds clip by clip, not just on average. It improves both validation families (real multitracks
and synthetic scenarios). Its refiner was trained only on disjoint data. On the target song it
keeps the reference-free leakage proxy at the level of the best candidates while raising guitar
probability. It also follows the PRD priority order: less leakage (SIR up, leakage down) without
destroying the guitar.

