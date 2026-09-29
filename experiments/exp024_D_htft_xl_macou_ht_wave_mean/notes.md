# exp024_D_htft_xl_macou_ht_wave_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance, A_mega_acoustic, A_htdemucs6s with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 4.032 / 3.927 dB, SI-SDR mean 1.145, SDRi 9.517
- SIR/SAR (BSS): 6.64 / 8.04 dB
- by family: ms_* (real multitracks) SDR 2.381, syn_* (synthetic) SDR 5.458
- target retention 0.502, leakage -11.77 dB, hard-case SDR 4.139
- target-song proxy: guitar prob 0.143, max leak prob 0.031, residual guitar prob 0.015
- runtime: validation 4017s, target 1552s

## Conclusion
Not promoted. champion before: exp023_D_htft_macou_wave_mean (sdr_mean=4.157); delta = -0.125 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
