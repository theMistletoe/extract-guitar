# exp050_R_r5_m4

## Hypothesis
The Mega-side chunk sweep showed Mega at 4 s and HTDemucs at its native 7.8 s beat all-6 s by 0.32 dB on the plain ensemble (exp047 vs exp038); a refiner trained on candidates computed that way should keep that gain.

## Change
gain refiner r5: r4 recipe; HTDemucs members at native chunk, Mega-on-mix members at 4 s chunks (train and inference); SW and the SW-to-Mega cascade stay at 6 s

## Result
- validation SDR mean/median: 7.616 / 6.646 dB, SI-SDR mean 5.785, SDRi 13.101
- SIR/SAR (BSS): 12.49 / 9.45 dB
- by family: ms_* (real multitracks) SDR 4.892, syn_* (synthetic) SDR 9.969
- target retention 0.693, leakage -14.91 dB, hard-case SDR 7.782
- target-song proxy: guitar prob 0.224, max leak prob 0.029, residual guitar prob 0.008
- runtime: validation 18749s, target 4348s

## Conclusion
PROMOTED to Champion. champion before: exp043_R_r4_c6 (sdr_mean=6.901); delta = +0.715 dB

## Next experiment
final render
