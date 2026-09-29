# exp048_C_xlance_violin_clarinet

## Hypothesis
For the target (guitar + violin + clarinet), residual violin/clarinet leakage in the guitar stem can be removed by dedicated heads applied after guitar extraction.

## Change
xlance_gtr -> subtract mega_violin estimate -> subtract mega_clarinet estimate.

## Result
- validation SDR mean/median: 3.122 / 0.747 dB, SI-SDR mean -0.524, SDRi 8.607
- SIR/SAR (BSS): 6.47 / 8.39 dB
- by family: ms_* (real multitracks) SDR 1.695, syn_* (synthetic) SDR 4.354
- target retention 0.595, leakage -10.36 dB, hard-case SDR 3.299
- target-song proxy: guitar prob 0.141, max leak prob 0.030, residual guitar prob 0.014
- runtime: validation 10917s, target 4052s

## Conclusion
Not promoted. champion before: exp043_R_r4_c6 (sdr_mean=6.901); delta = -3.780 dB

## Next experiment
Full removal cascade before guitar extraction.
