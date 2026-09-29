# exp043_R_r4_c6

## Hypothesis
The 3-member ensemble with the Strategy-B stem (SW minus electric) is 0.29 dB better than the pair used by r3; the same gain refiner on this stronger base should add a similar gain on top.

## Change
gain refiner r4: base = mean of SW-minus-electric, HTDemucs-ft, Mega acoustic; 5 positive + 8 negative evidence stems; r1 mining weights; 6 epochs; members at 6 s chunks as in training (the c6 finding)

## Result
- validation SDR mean/median: 6.901 / 5.894 dB, SI-SDR mean 5.065, SDRi 12.387
- SIR/SAR (BSS): 11.59 / 9.33 dB
- by family: ms_* (real multitracks) SDR 4.648, syn_* (synthetic) SDR 8.848
- target retention 0.618, leakage -15.41 dB, hard-case SDR 7.019
- target-song proxy: guitar prob 0.179, max leak prob 0.030, residual guitar prob 0.009
- runtime: validation 15059s, target 3516s

## Conclusion
PROMOTED to Champion. champion before: exp037_R_r3_c6 (sdr_mean=6.502); delta = +0.399 dB

## Next experiment
final render
