# exp026_W_htft_macou_weights0.4-0.6

## Hypothesis
Mega acoustic is clean but conservative (SAR 1.2); HTDemucs-ft keeps more guitar. Weighting may move the pair to a better leakage/retention point.

## Change
{"ens.weights": [0.4, 0.6]}

## Result
- validation SDR mean/median: 4.141 / 3.635 dB, SI-SDR mean 1.157, SDRi 9.626
- SIR/SAR (BSS): 8.98 / 6.11 dB
- by family: ms_* (real multitracks) SDR 2.439, syn_* (synthetic) SDR 5.610
- target retention 0.402, leakage -13.96 dB, hard-case SDR 4.255
- target-song proxy: guitar prob nan, max leak prob nan, residual guitar prob nan
- runtime: validation 2581s, target nans

## Conclusion
Not promoted. champion before: exp023_D_htft_macou_wave_mean (sdr_mean=4.157); delta = -0.017 dB

## Next experiment
Learned frequency-dependent weights (training clips) and the refiner.
