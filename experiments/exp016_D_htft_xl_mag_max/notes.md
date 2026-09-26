# exp016_D_htft_xl_mag_max

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance with mag_max averages out model-specific errors and beats the best single member.

## Change
ensemble method=mag_max, weights=equal

## Result
- validation SDR mean/median: 2.856 / 1.193 dB, SI-SDR mean 0.990, SDRi 8.342
- SIR/SAR (BSS): 4.88 / 10.05 dB
- by family: ms_* (real multitracks) SDR 0.660, syn_* (synthetic) SDR 4.753
- target retention 0.769, leakage -8.42 dB, hard-case SDR 2.779
- target-song proxy: guitar prob 0.128, max leak prob 0.048, residual guitar prob 0.009
- runtime: validation 1410s, target 628s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.780 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
