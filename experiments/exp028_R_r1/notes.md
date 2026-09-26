# exp028_R_r1

## Hypothesis
A small mask refiner trained on disjoint training clips, fed with the positive (guitar) and negative (violin/woodwind/electric/other) estimates of the pretrained models, removes leakage the plain average keeps.

## Change
refiner round 1 (mask mode, uniform sampling, best hold-out epoch 9), 4 positive + 8 negative evidence stems

## Result
- validation SDR mean/median: 3.894 / 2.693 dB, SI-SDR mean 0.154, SDRi 9.380
- SIR/SAR (BSS): 9.43 / 4.06 dB
- by family: ms_* (real multitracks) SDR 2.243, syn_* (synthetic) SDR 5.320
- target retention 0.412, leakage -15.28 dB, hard-case SDR 4.073
- target-song proxy: guitar prob 0.190, max leak prob 0.025, residual guitar prob 0.014
- runtime: validation 4744s, target 1590s

## Conclusion
Not promoted. champion before: exp023_D_htft_macou_wave_mean (sdr_mean=4.157); delta = -0.263 dB

## Next experiment
hard-example mining round (r2); gain-mode refiner on the Champion (r3)
