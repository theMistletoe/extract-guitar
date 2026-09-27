# exp038_D_htft_macou_c6

## Hypothesis
Control: if the plain ensemble at 6 s chunks is itself much better than at default chunks, the R_r3_c6 gain comes from chunking, not from matching the refiner's training condition.

## Change
exp023 pipeline with chunk_size 264600 for both members; no refiner

## Result
- validation SDR mean/median: 5.472 / 5.156 dB, SI-SDR mean 2.998, SDRi 10.958
- SIR/SAR (BSS): 8.99 / 8.83 dB
- by family: ms_* (real multitracks) SDR 3.309, syn_* (synthetic) SDR 7.341
- target retention 0.579, leakage -13.21 dB, hard-case SDR 5.556
- target-song proxy: guitar prob 0.153, max leak prob 0.047, residual guitar prob 0.011
- runtime: validation 1866s, target 284s

## Conclusion
Not promoted. champion before: exp037_R_r3_c6 (sdr_mean=6.502); delta = -1.030 dB

## Next experiment
attribute the c6 gain
