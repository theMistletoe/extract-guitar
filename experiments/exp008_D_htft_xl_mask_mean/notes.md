# exp008_D_htft_xl_mask_mean

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance with mask_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=mask_mean, weights=equal

## Result
- validation SDR mean/median: 3.295 / 3.187 dB, SI-SDR mean 0.281, SDRi 8.780
- SIR/SAR (BSS): 4.22 / 8.99 dB
- by family: ms_* (real multitracks) SDR 1.689, syn_* (synthetic) SDR 4.681
- target retention 0.571, leakage -10.51 dB, hard-case SDR 3.376
- target-song proxy: guitar prob 0.146, max leak prob 0.038, residual guitar prob 0.012
- runtime: validation 1410s, target 628s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.342 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
