# exp018_A_mega_guitar

## Hypothesis
Mega-53 all-guitar head (8.25 on MVSep) - same trunk as the acoustic head, isolates the effect of the head.

## Change
Model = mega_guitar.

## Result
- validation SDR mean/median: 2.550 / 1.500 dB, SI-SDR mean -2.520, SDRi 8.036
- SIR/SAR (BSS): 8.88 / 2.80 dB
- by family: ms_* (real multitracks) SDR 0.827, syn_* (synthetic) SDR 4.038
- target retention 0.406, leakage -12.13 dB, hard-case SDR 2.728
- target-song proxy: guitar prob 0.245, max leak prob 0.045, residual guitar prob 0.023
- runtime: validation 2316s, target 874s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -1.087 dB

## Next experiment
Strategy B/C pipelines and ensembles of the best direct models.
