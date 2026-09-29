# exp001_A_htdemucs6s

## Hypothesis
HTDemucs 6s (hybrid waveform/spectrogram, MIT) gives a first baseline from a different architecture family than the RoFormers.

## Change
Baseline: htdemucs_6s guitar stem, overlap 2, bf16.

## Result
- validation SDR mean/median: 2.579 / 1.840 dB, SI-SDR mean -0.676, SDRi 8.064
- SIR/SAR (BSS): 5.70 / 6.51 dB
- by family: ms_* (real multitracks) SDR 0.527, syn_* (synthetic) SDR 4.351
- target retention 0.614, leakage -9.60 dB, hard-case SDR 2.615
- target-song proxy: guitar prob 0.169, max leak prob 0.041, residual guitar prob 0.008
- runtime: validation 1s, target 1s

## Conclusion
PROMOTED to Champion. no champion yet

## Next experiment
Run the RoFormer guitar models with identical settings.
