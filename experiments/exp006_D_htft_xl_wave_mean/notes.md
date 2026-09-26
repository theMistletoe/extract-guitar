# exp006_D_htft_xl_wave_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 3.637 / 3.594 dB, SI-SDR mean 0.980, SDRi 9.122
- SIR/SAR (BSS): 6.12 / 8.59 dB
- by family: ms_* (real multitracks) SDR 1.944, syn_* (synthetic) SDR 5.098
- target retention 0.583, leakage -10.75 dB, hard-case SDR 3.702
- target-song proxy: guitar prob 0.133, max leak prob 0.031, residual guitar prob 0.012
- runtime: validation 1410s, target 628s

## Conclusion
PROMOTED to Champion. champion before: exp003_A_htdemucs6s_ft (sdr_mean=3.072); delta = +0.565 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
