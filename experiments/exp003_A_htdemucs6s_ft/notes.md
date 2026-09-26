# exp003_A_htdemucs6s_ft

## Hypothesis
The MoisesDB guitar fine-tune of htdemucs_6s improves guitar SDR over stock htdemucs_6s.

## Change
Model = htdemucs_6s + adityalakhani guitar fine-tune weights.

## Result
- validation SDR mean/median: 3.072 / 2.138 dB, SI-SDR mean 0.239, SDRi 8.558
- SIR/SAR (BSS): 6.71 / 6.77 dB
- by family: ms_* (real multitracks) SDR 1.020, syn_* (synthetic) SDR 4.844
- target retention 0.623, leakage -9.86 dB, hard-case SDR 3.082
- target-song proxy: guitar prob 0.139, max leak prob 0.051, residual guitar prob 0.009
- runtime: validation 276s, target 4s

## Conclusion
PROMOTED to Champion. champion before: exp001_A_htdemucs6s (sdr_mean=2.579); delta = +0.493 dB

## Next experiment
Large BS-RoFormer guitar models.
