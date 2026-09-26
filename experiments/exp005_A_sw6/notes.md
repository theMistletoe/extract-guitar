# exp005_A_sw6

## Hypothesis
BS-RoFormer SW (9.01 on MVSep all-guitar) is the strongest public guitar separator.

## Change
Model = sw6, guitar stem.

## Result
- validation SDR mean/median: 3.117 / 0.747 dB, SI-SDR mean -0.525, SDRi 8.603
- SIR/SAR (BSS): 6.47 / 8.40 dB
- by family: ms_* (real multitracks) SDR 1.686, syn_* (synthetic) SDR 4.353
- target retention 0.595, leakage -10.35 dB, hard-case SDR 3.294
- target-song proxy: guitar prob 0.141, max leak prob 0.030, residual guitar prob 0.014
- runtime: validation 2164s, target 668s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.520 dB

## Next experiment
Acoustic-specific Mega-53 head.
