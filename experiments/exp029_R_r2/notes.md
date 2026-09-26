# exp029_R_r2

## Hypothesis
Oversampling the worst training clips and adding clips of their failure types (hard-example mining: buried/dense scenarios) improves the refiner on hard cases.

## Change
refiner round 2: 48 new buried/dense clips + 3x oversampling of mined hard clips (stopped at epoch 4, best hold-out epoch 3)

## Result
- validation SDR mean/median: 3.880 / 3.018 dB, SI-SDR mean 0.137, SDRi 9.366
- SIR/SAR (BSS): 6.80 / 5.12 dB
- by family: ms_* (real multitracks) SDR 2.252, syn_* (synthetic) SDR 5.287
- target retention 0.409, leakage -13.88 dB, hard-case SDR 4.001
- target-song proxy: guitar prob 0.153, max leak prob 0.034, residual guitar prob 0.018
- runtime: validation 4744s, target 1590s

## Conclusion
Not promoted. champion before: exp023_D_htft_macou_wave_mean (sdr_mean=4.157); delta = -0.277 dB

## Next experiment
gain-mode refiner on the Champion (r3)
