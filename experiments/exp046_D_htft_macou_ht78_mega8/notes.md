# exp046_D_htft_macou_ht78_mega8

## Hypothesis
HTDemucs cannot exceed its 7.8 s training segment (exp041/042 crashed), and Mega defaults to 20 s; the 6 s gain is probably Mega-side. Mega at 8 s with HTDemucs native isolates it.

## Change
HTDemucs default chunk, Mega chunk_size 352800 (8 s)

## Result
- validation SDR mean/median: 5.075 / 4.561 dB, SI-SDR mean 2.519, SDRi 10.561
- SIR/SAR (BSS): 8.81 / 8.20 dB
- by family: ms_* (real multitracks) SDR 3.026, syn_* (synthetic) SDR 6.846
- target retention 0.511, leakage -13.43 dB, hard-case SDR 5.176
- target-song proxy: guitar prob 0.166, max leak prob 0.038, residual guitar prob 0.014
- runtime: validation 7038s, target 1683s

## Conclusion
Not promoted. champion before: exp043_R_r4_c6 (sdr_mean=6.901); delta = -1.826 dB

## Next experiment
choose per-model chunk sizes
