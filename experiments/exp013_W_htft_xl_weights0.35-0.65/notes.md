# exp013_W_htft_xl_weights0.35-0.65

## Hypothesis
Unequal weights may help because the two members have different error profiles (HTDemucs-ft better SI-SDR/retention, SW better SAR).

## Change
{"ens.weights": [0.35, 0.65]}

## Result
- validation SDR mean/median: 3.615 / 2.946 dB, SI-SDR mean 0.836, SDRi 9.100
- SIR/SAR (BSS): 6.10 / 8.67 dB
- by family: ms_* (real multitracks) SDR 1.985, syn_* (synthetic) SDR 5.023
- target retention 0.581, leakage -10.81 dB, hard-case SDR 3.719
- target-song proxy: guitar prob nan, max leak prob nan, residual guitar prob nan
- runtime: validation 1410s, target nans

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.022 dB

## Next experiment
Frequency-dependent weights learned on training clips.
