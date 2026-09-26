# exp010_D_htft_xl_bec_mag_median

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance, A_becruily with mag_median averages out model-specific errors and beats the best single member.

## Change
ensemble method=mag_median, weights=equal

## Result
- validation SDR mean/median: 3.540 / 2.635 dB, SI-SDR mean 0.512, SDRi 9.025
- SIR/SAR (BSS): 5.91 / 8.43 dB
- by family: ms_* (real multitracks) SDR 2.107, syn_* (synthetic) SDR 4.777
- target retention 0.584, leakage -10.94 dB, hard-case SDR 3.540
- target-song proxy: guitar prob 0.156, max leak prob 0.065, residual guitar prob 0.012
- runtime: validation 2954s, target 915s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.097 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
