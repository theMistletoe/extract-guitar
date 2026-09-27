# exp037_R_r3_c6

## Hypothesis
The refiner was trained on member outputs computed with 6 s chunks; computing the members the same way at inference removes the train/inference mismatch that cost ~0.35 dB on training hold-out.

## Change
R_r3 with chunk_size 264600 (6 s) for HTDemucs-ft, Mega and SW

## Result
- validation SDR mean/median: 6.502 / 5.336 dB, SI-SDR mean 4.378, SDRi 11.988
- SIR/SAR (BSS): 10.91 / 8.94 dB
- by family: ms_* (real multitracks) SDR 4.055, syn_* (synthetic) SDR 8.615
- target retention 0.655, leakage -13.53 dB, hard-case SDR 6.590
- target-song proxy: guitar prob 0.167, max leak prob 0.030, residual guitar prob 0.009
- runtime: validation 3624s, target 672s

## Conclusion
PROMOTED to Champion. champion before: exp030_R_r3 (sdr_mean=4.683); delta = +1.819 dB

## Next experiment
final render
