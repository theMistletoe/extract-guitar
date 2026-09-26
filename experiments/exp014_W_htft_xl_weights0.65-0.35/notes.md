# exp014_W_htft_xl_weights0.65-0.35

## Hypothesis
Unequal weights may help because the two members have different error profiles (HTDemucs-ft better SI-SDR/retention, SW better SAR).

## Change
{"ens.weights": [0.65, 0.35]}

## Result
- validation SDR mean/median: 3.583 / 3.170 dB, SI-SDR mean 0.939, SDRi 9.068
- SIR/SAR (BSS): 6.20 / 8.27 dB
- by family: ms_* (real multitracks) SDR 1.808, syn_* (synthetic) SDR 5.115
- target retention 0.591, leakage -10.57 dB, hard-case SDR 3.617
- target-song proxy: guitar prob nan, max leak prob nan, residual guitar prob nan
- runtime: validation 1410s, target nans

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.054 dB

## Next experiment
Frequency-dependent weights learned on training clips.
