# exp021_D_htft_mgtr_wave_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_mega_guitar with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 3.765 / 3.135 dB, SI-SDR mean 0.712, SDRi 9.251
- SIR/SAR (BSS): 7.22 / 6.94 dB
- by family: ms_* (real multitracks) SDR 2.029, syn_* (synthetic) SDR 5.265
- target retention 0.476, leakage -11.94 dB, hard-case SDR 3.855
- target-song proxy: guitar prob 0.137, max leak prob 0.043, residual guitar prob 0.017
- runtime: validation 2581s, target 922s

## Conclusion
Not promoted. champion before: exp019_D_htft_xl_macou_wave_mean (sdr_mean=4.085); delta = -0.320 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
