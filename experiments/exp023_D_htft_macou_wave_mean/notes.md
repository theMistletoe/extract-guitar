# exp023_D_htft_macou_wave_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_mega_acoustic with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 4.157 / 3.955 dB, SI-SDR mean 1.220, SDRi 9.643
- SIR/SAR (BSS): 8.43 / 6.62 dB
- by family: ms_* (real multitracks) SDR 2.433, syn_* (synthetic) SDR 5.646
- target retention 0.434, leakage -13.21 dB, hard-case SDR 4.255
- target-song proxy: guitar prob 0.137, max leak prob 0.042, residual guitar prob 0.017
- runtime: validation 2581s, target 922s

## Conclusion
PROMOTED to Champion. champion before: exp019_D_htft_xl_macou_wave_mean (sdr_mean=4.085); delta = +0.072 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
