# exp007_D_htft_xl_mag_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance with mag_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=mag_mean, weights=equal

## Result
- validation SDR mean/median: 3.608 / 3.621 dB, SI-SDR mean 0.962, SDRi 9.093
- SIR/SAR (BSS): 6.02 / 8.60 dB
- by family: ms_* (real multitracks) SDR 1.890, syn_* (synthetic) SDR 5.091
- target retention 0.587, leakage -10.63 dB, hard-case SDR 3.671
- target-song proxy: guitar prob 0.133, max leak prob 0.031, residual guitar prob 0.012
- runtime: validation 1410s, target 628s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.029 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
