# exp004_A_xlance

## Hypothesis
X-LANCE MSR guitar BS-RoFormer (8.60 SDR on MVSep all-guitar) is among the best public guitar models. Caveat: trained on RawStems, so its ms_* validation clips may be contaminated -> compare syn_* too.

## Change
Model = xlance_gtr.

## Result
- validation SDR mean/median: 3.117 / 0.747 dB, SI-SDR mean -0.525, SDRi 8.603
- SIR/SAR (BSS): 6.47 / 8.40 dB
- by family: ms_* (real multitracks) SDR 1.686, syn_* (synthetic) SDR 4.353
- target retention 0.595, leakage -10.35 dB, hard-case SDR 3.294
- target-song proxy: guitar prob 0.141, max leak prob 0.030, residual guitar prob 0.014
- runtime: validation 1146s, target 579s

## Conclusion
Not promoted. champion before: exp003_A_htdemucs6s_ft (sdr_mean=3.072); delta = +0.045 dB

## Next experiment
SW 6-stem guitar output.
