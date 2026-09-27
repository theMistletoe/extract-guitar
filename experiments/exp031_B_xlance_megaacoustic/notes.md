# exp031_B_xlance_megaacoustic

## Hypothesis
Removing vocals/drums/bass etc. first with an all-guitar model makes the acoustic-vs-rest decision easier for the Mega acoustic head (PRD §11 Strategy B).

## Change
Stage 1 xlance_gtr (all guitar) -> stage 2 mega_acoustic on the guitar stem.

## Result
- validation SDR mean/median: 3.051 / 1.746 dB, SI-SDR mean -1.751, SDRi 8.536
- SIR/SAR (BSS): 14.40 / 1.09 dB
- by family: ms_* (real multitracks) SDR 1.625, syn_* (synthetic) SDR 4.282
- target retention 0.323, leakage -18.26 dB, hard-case SDR 3.238
- target-song proxy: guitar prob 0.243, max leak prob 0.044, residual guitar prob 0.025
- runtime: validation 5905s, target 2345s

## Conclusion
Not promoted. champion before: exp030_R_r3 (sdr_mean=4.683); delta = -1.632 dB

## Next experiment
Same with SW as stage 1; guitar minus electric variant.
