# exp030_R_r3

## Hypothesis
Refining the Champion ensemble itself (0..2 TF gain, identity at init) keeps its strengths and learns local fixes: lift guitar holes, cut violin/clarinet/electric leakage; trained on disjoint clips with hard-example oversampling.

## Change
gain-mode refiner r3 on HTDemucs-ft + Mega acoustic mean; 4 positive + 8 negative evidence stems; mining weights from r1

## Result
- validation SDR mean/median: 4.683 / 3.234 dB, SI-SDR mean 1.786, SDRi 10.168
- SIR/SAR (BSS): 9.48 / 6.44 dB
- by family: ms_* (real multitracks) SDR 2.934, syn_* (synthetic) SDR 6.193
- target retention 0.510, leakage -12.86 dB, hard-case SDR 4.838
- target-song proxy: guitar prob 0.146, max leak prob 0.029, residual guitar prob 0.013
- runtime: validation 4744s, target 1590s

## Conclusion
PROMOTED to Champion. champion before: exp023_D_htft_macou_wave_mean (sdr_mean=4.157); delta = +0.526 dB

## Next experiment
final render / inference tuning
