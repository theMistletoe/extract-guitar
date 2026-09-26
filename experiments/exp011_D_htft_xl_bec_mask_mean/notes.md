# exp011_D_htft_xl_bec_mask_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance, A_becruily with mask_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=mask_mean, weights=equal

## Result
- validation SDR mean/median: 3.258 / 2.933 dB, SI-SDR mean 0.130, SDRi 8.743
- SIR/SAR (BSS): 3.42 / 9.72 dB
- by family: ms_* (real multitracks) SDR 2.038, syn_* (synthetic) SDR 4.311
- target retention 0.546, leakage -9.85 dB, hard-case SDR 3.297
- target-song proxy: guitar prob 0.146, max leak prob 0.078, residual guitar prob 0.014
- runtime: validation 2954s, target 915s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.379 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
