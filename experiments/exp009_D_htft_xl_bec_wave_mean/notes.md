# exp009_D_htft_xl_bec_wave_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance, A_becruily with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 3.586 / 3.040 dB, SI-SDR mean 0.818, SDRi 9.071
- SIR/SAR (BSS): 4.89 / 9.41 dB
- by family: ms_* (real multitracks) SDR 2.310, syn_* (synthetic) SDR 4.688
- target retention 0.554, leakage -10.12 dB, hard-case SDR 3.594
- target-song proxy: guitar prob 0.134, max leak prob 0.070, residual guitar prob 0.013
- runtime: validation 2954s, target 915s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.051 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
