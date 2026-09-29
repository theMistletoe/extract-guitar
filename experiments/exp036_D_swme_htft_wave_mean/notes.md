# exp036_D_swme_htft_wave_mean

## Hypothesis
Combining B_xlance_minus_electric, A_htdemucs6s_ft with wave_mean averages out model-specific errors and beats the best single member.

## Change
ensemble method=wave_mean, weights=equal

## Result
- validation SDR mean/median: 4.259 / 3.798 dB, SI-SDR mean 1.577, SDRi 9.744
- SIR/SAR (BSS): 6.92 / 8.29 dB
- by family: ms_* (real multitracks) SDR 2.521, syn_* (synthetic) SDR 5.759
- target retention 0.545, leakage -11.59 dB, hard-case SDR 4.317
- target-song proxy: guitar prob 0.133, max leak prob 0.031, residual guitar prob 0.012
- runtime: validation 6169s, target 2394s

## Conclusion
Not promoted. champion before: exp030_R_r3 (sdr_mean=4.683); delta = -0.424 dB

## Next experiment
Compare methods; keep the best as Champion candidate.
