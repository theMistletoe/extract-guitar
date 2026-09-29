# exp020_D_htft_xl_mgtr_wave_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance, A_mega_guitar with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 3.712 / 2.834 dB, SI-SDR mean 0.728, SDRi 9.198
- SIR/SAR (BSS): 6.67 / 7.72 dB
- by family: ms_* (real multitracks) SDR 2.168, syn_* (synthetic) SDR 5.046
- target retention 0.506, leakage -11.67 dB, hard-case SDR 3.825
- target-song proxy: guitar prob 0.131, max leak prob 0.032, residual guitar prob 0.016
- runtime: validation 3726s, target 1502s

## Conclusion
Not promoted. champion before: exp019_D_htft_xl_macou_wave_mean (sdr_mean=4.085); delta = -0.373 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
