# exp022_D_xl_macou_wave_mean

## Hypothesis
Combining A_xlance, A_mega_acoustic with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 3.601 / 2.311 dB, SI-SDR mean -0.394, SDRi 9.086
- SIR/SAR (BSS): 8.32 / 5.21 dB
- by family: ms_* (real multitracks) SDR 2.092, syn_* (synthetic) SDR 4.904
- target retention 0.443, leakage -12.67 dB, hard-case SDR 3.766
- target-song proxy: guitar prob 0.142, max leak prob 0.032, residual guitar prob 0.019
- runtime: validation 3462s, target 1453s

## Conclusion
Not promoted. champion before: exp019_D_htft_xl_macou_wave_mean (sdr_mean=4.085); delta = -0.485 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
