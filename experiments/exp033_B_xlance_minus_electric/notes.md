# exp033_B_xlance_minus_electric

## Hypothesis
Subtracting an electric-guitar estimate from the all-guitar stem removes electric leakage while keeping acoustic energy the acoustic head might drop.

## Change
Stage 1 xlance_gtr -> subtract mega_electric estimate of that stem.

## Result
- validation SDR mean/median: 4.199 / 1.590 dB, SI-SDR mean 0.597, SDRi 9.684
- SIR/SAR (BSS): 9.07 / 5.94 dB
- by family: ms_* (real multitracks) SDR 2.577, syn_* (synthetic) SDR 5.599
- target retention 0.525, leakage -12.56 dB, hard-case SDR 4.354
- target-song proxy: guitar prob 0.141, max leak prob 0.030, residual guitar prob 0.014
- runtime: validation 5905s, target 2345s

## Conclusion
Not promoted. champion before: exp030_R_r3 (sdr_mean=4.683); delta = -0.484 dB

## Next experiment
Cascades (C).
