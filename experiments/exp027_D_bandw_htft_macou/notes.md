# exp027_D_bandw_htft_macou

## Hypothesis
Frequency-dependent least-squares weights fitted on the disjoint TRAINING clips (not on validation) exploit that each member is better in different bands.

## Change
band_weighted ensemble of HTDemucs-ft + Mega acoustic, 8 bands, weights from scripts/fit_band_weights.py (artifacts/band_weights_htft_macou.json)

## Result
- validation SDR mean/median: 3.811 / 3.035 dB, SI-SDR mean 0.568, SDRi 9.296
- SIR/SAR (BSS): 11.77 / 4.07 dB
- by family: ms_* (real multitracks) SDR 2.193, syn_* (synthetic) SDR 5.208
- target retention 0.357, leakage -16.08 dB, hard-case SDR 3.940
- target-song proxy: guitar prob 0.214, max leak prob 0.035, residual guitar prob 0.020
- runtime: validation 2581s, target 922s

## Conclusion
Not promoted. champion before: exp023_D_htft_macou_wave_mean (sdr_mean=4.157); delta = -0.346 dB

## Next experiment
Refiner (Phase 7) as the non-linear generalisation of per-band weights.
