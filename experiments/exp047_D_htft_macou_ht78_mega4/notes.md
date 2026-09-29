# exp047_D_htft_macou_ht78_mega4

## Hypothesis
Mega at 8 s is clearly worse than at 6 s (exp045 interim -0.44 dB, 1/10 wins); test the shorter side (4 s) instead of 10 s.

## Change
HTDemucs default chunk, Mega chunk_size 176400 (4 s)

## Result
- validation SDR mean/median: 5.791 / 5.633 dB, SI-SDR mean 3.565, SDRi 11.276
- SIR/SAR (BSS): 9.95 / 8.92 dB
- by family: ms_* (real multitracks) SDR 3.510, syn_* (synthetic) SDR 7.761
- target retention 0.608, leakage -13.58 dB, hard-case SDR 5.872
- target-song proxy: guitar prob 0.181, max leak prob 0.034, residual guitar prob 0.009
- runtime: validation 5421s, target 1270s

## Conclusion
Not promoted. champion before: exp043_R_r4_c6 (sdr_mean=6.901); delta = -1.110 dB

## Next experiment
choose per-model chunk sizes
