# exp017_D_htft_xl_mag_min

## Hypothesis
Combining A_htdemucs6s_ft, A_xlance with mag_min averages out model-specific errors and beats the best single member.

## Change
ensemble method=mag_min, weights=equal

## Result
- validation SDR mean/median: 3.269 / 2.109 dB, SI-SDR mean -0.711, SDRi 8.754
- SIR/SAR (BSS): 8.68 / 5.02 dB
- by family: ms_* (real multitracks) SDR 1.722, syn_* (synthetic) SDR 4.605
- target retention 0.445, leakage -13.99 dB, hard-case SDR 3.467
- target-song proxy: guitar prob 0.162, max leak prob 0.031, residual guitar prob 0.014
- runtime: validation 1410s, target 628s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.368 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
