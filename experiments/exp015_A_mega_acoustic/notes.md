# exp015_A_mega_acoustic

## Hypothesis
The only public acoustic-guitar-specific head (Mega-53) should reject electric guitar better, possibly at the cost of acoustic retention.

## Change
Model = mega_acoustic.

## Result
- validation SDR mean/median: 3.130 / 1.917 dB, SI-SDR mean -1.444, SDRi 8.616
- SIR/SAR (BSS): 14.84 / 1.21 dB
- by family: ms_* (real multitracks) SDR 1.389, syn_* (synthetic) SDR 4.634
- target retention 0.330, leakage -17.28 dB, hard-case SDR 3.294
- target-song proxy: guitar prob 0.254, max leak prob 0.059, residual guitar prob 0.024
- runtime: validation 2316s, target 874s

## Conclusion
Not promoted. champion before: exp006_D_htft_xl_wave_mean (sdr_mean=3.637); delta = -0.506 dB

## Next experiment
Mega-53 all-guitar head; then two-stage (B) pipelines.
