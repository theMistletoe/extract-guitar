# exp002_A_becruily

## Hypothesis
The small becruily Mel-Band RoFormer guitar model (22M params) is a fast RoFormer baseline.

## Change
Model = becruily_guitar.

## Result
- validation SDR mean/median: 2.100 / 1.143 dB, SI-SDR mean -1.841, SDRi 7.586
- SIR/SAR (BSS): 3.24 / 7.35 dB
- by family: ms_* (real multitracks) SDR 1.698, syn_* (synthetic) SDR 2.447
- target retention 0.521, leakage -8.26 dB, hard-case SDR 1.973
- target-song proxy: guitar prob 0.259, max leak prob 0.192, residual guitar prob 0.017
- runtime: validation 1567s, target 1s

## Conclusion
Not promoted. champion before: exp001_A_htdemucs6s (sdr_mean=2.579); delta = -0.479 dB

## Next experiment
Large BS-RoFormer guitar models.
