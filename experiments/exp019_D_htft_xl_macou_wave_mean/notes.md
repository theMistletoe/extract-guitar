# exp019_D_htft_xl_macou_wave_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance, A_mega_acoustic with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 4.085 / 3.769 dB, SI-SDR mean 1.155, SDRi 9.571
- SIR/SAR (BSS): 7.23 / 7.59 dB
- by family: ms_* (real multitracks) SDR 2.499, syn_* (synthetic) SDR 5.455
- target retention 0.477, leakage -12.38 dB, hard-case SDR 4.215
- target-song proxy: guitar prob 0.132, max leak prob 0.032, residual guitar prob 0.016
- runtime: validation 3726s, target 1502s

## Conclusion
PROMOTED to Champion. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = +0.449 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
