# exp051_C_orch_xlance

## Hypothesis
Removing orchestral instruments (strings + winds) from the mix before guitar extraction reduces violin/clarinet confusion.

## Change
mix -> remove xlance_orch estimate -> xlance_gtr.

## Result
- validation SDR mean/median: 2.923 / 1.123 dB, SI-SDR mean -0.616, SDRi 8.408
- SIR/SAR (BSS): 6.29 / 8.55 dB
- by family: ms_* (real multitracks) SDR 1.538, syn_* (synthetic) SDR 4.118
- target retention 0.609, leakage -10.17 dB, hard-case SDR 3.075
- target-song proxy: guitar prob 0.151, max leak prob 0.035, residual guitar prob 0.014
- runtime: validation 5790s, target 2942s

## Conclusion
Not promoted. champion before: exp050_R_r5_m4 (sdr_mean=7.616); delta = -4.693 dB

## Next experiment
Longer cascade with SW + strings + woodwind removal.
