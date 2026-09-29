# exp035_D_swme_htft_macou_wave_mean

## Hypothesis
Combining B_xlance_minus_electric, A_htdemucs6s_ft, A_mega_acoustic with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 4.445 / 3.770 dB, SI-SDR mean 1.659, SDRi 9.930
- SIR/SAR (BSS): 8.03 / 7.35 dB
- by family: ms_* (real multitracks) SDR 2.817, syn_* (synthetic) SDR 5.851
- target retention 0.451, leakage -13.32 dB, hard-case SDR 4.559
- target-song proxy: guitar prob 0.132, max leak prob 0.032, residual guitar prob 0.016
- runtime: validation 8485s, target 3268s

## Conclusion
Not promoted. champion before: exp030_R_r3 (sdr_mean=4.683); delta = -0.238 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
