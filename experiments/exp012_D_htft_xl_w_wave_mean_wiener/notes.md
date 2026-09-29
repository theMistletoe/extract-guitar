# exp012_D_htft_xl_w_wave_mean_wiener

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance with wave_mean and a Wiener refinement averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal, wiener post-filter

## Result
- validation SDR mean/median: 3.516 / 3.068 dB, SI-SDR mean 0.429, SDRi 9.001
- SIR/SAR (BSS): 7.39 / 7.48 dB
- by family: ms_* (real multitracks) SDR 1.821, syn_* (synthetic) SDR 4.979
- target retention 0.592, leakage -11.26 dB, hard-case SDR 3.622
- target-song proxy: guitar prob 0.215, max leak prob 0.035, residual guitar prob 0.009
- runtime: validation 1410s, target 628s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.121 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
