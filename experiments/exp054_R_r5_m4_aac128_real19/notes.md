# exp054_R_r5_m4_aac128_real19

## Hypothesis
The target is AAC with a 16 kHz low-pass; the Champion's validation score is on lossless mixtures. Scoring it on the 19 real-multitrack clips passed through AAC 128k (clean references) measures how much codec damage costs.

## Change
exp050 pipeline on datasets/validation_aac, first 19 clips (= all real multitracks); mixtures AAC-128k, references clean

## Result
- validation SDR mean/median: 4.616 / 3.957 dB, SI-SDR mean 2.156, SDRi 11.298
- SIR/SAR (BSS): 7.63 / 6.97 dB
- by family: ms_* (real multitracks) SDR 4.616, syn_* (synthetic) SDR nan
- target retention 0.627, leakage -12.06 dB, hard-case SDR 4.792
- target-song proxy: guitar prob nan, max leak prob nan, residual guitar prob nan
- runtime: validation 10180s, target nans

## Conclusion
Not promoted. champion before: exp050_R_r5_m4 (sdr_mean=7.616); delta = -3.000 dB

## Next experiment
report the codec penalty in the final report
