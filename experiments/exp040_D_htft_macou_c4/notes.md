# exp040_D_htft_macou_c4

## Hypothesis
exp038 showed 6 s chunks give +1.3 dB over default on the plain ensemble; 4 s tests which direction of chunk length helps.

## Change
exp038 with chunk_size 176400 (4 s)

## Result
- validation SDR mean/median: 5.558 / 5.085 dB, SI-SDR mean 3.253, SDRi 11.043
- SIR/SAR (BSS): 9.02 / 8.89 dB
- by family: ms_* (real multitracks) SDR 3.294, syn_* (synthetic) SDR 7.513
- target retention 0.631, leakage -12.64 dB, hard-case SDR 5.643
- target-song proxy: guitar prob 0.156, max leak prob 0.045, residual guitar prob 0.009
- runtime: validation 6208s, target 1478s

## Conclusion
Not promoted. champion before: exp037_R_r3_c6 (sdr_mean=6.502); delta = -0.944 dB

## Next experiment
pick the chunk size for the final members / refiner retraining
