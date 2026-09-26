# exp025_W_htft_macou_weights0.6-0.4

## Hypothesis
Mega acoustic is clean but conservative (SAR 1.2); HTDemucs-ft keeps more guitar. Weighting may move the pair to a better leakage/retention point.

## Change
{"ens.weights": [0.6, 0.4]}

## Result
- validation SDR mean/median: 4.100 / 4.131 dB, SI-SDR mean 1.173, SDRi 9.585
- SIR/SAR (BSS): 7.98 / 6.94 dB
- by family: ms_* (real multitracks) SDR 2.326, syn_* (synthetic) SDR 5.632
- target retention 0.470, leakage -12.50 dB, hard-case SDR 4.177
- target-song proxy: guitar prob nan, max leak prob nan, residual guitar prob nan
- runtime: validation 2581s, target nans

## Conclusion
Not promoted. champion before: exp023_D_htft_macou_wave_mean (sdr_mean=4.157); delta = -0.058 dB

## Next experiment
Learned frequency-dependent weights (training clips) and the refiner.
