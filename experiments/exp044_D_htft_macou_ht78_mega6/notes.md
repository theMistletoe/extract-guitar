# exp044_D_htft_macou_ht78_mega6

## Hypothesis
HTDemucs cannot exceed its 7.8 s training segment (exp041/042 crashed), and Mega defaults to 20 s; the 6 s gain is probably Mega-side. Mega at 6 s with HTDemucs native isolates it.

## Change
HTDemucs default chunk, Mega chunk_size 264600 (6 s)

## Result
- validation SDR mean/median: 5.583 / 5.155 dB, SI-SDR mean 3.175, SDRi 11.068
- SIR/SAR (BSS): 9.38 / 8.73 dB
- by family: ms_* (real multitracks) SDR 3.398, syn_* (synthetic) SDR 7.470
- target retention 0.564, leakage -13.59 dB, hard-case SDR 5.674
- target-song proxy: guitar prob 0.171, max leak prob 0.035, residual guitar prob 0.012
- runtime: validation 1899s, target 274s

## Conclusion
Not promoted. champion before: exp043_R_r4_c6 (sdr_mean=6.901); delta = -1.319 dB

## Next experiment
choose per-model chunk sizes
