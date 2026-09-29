# exp052_C_sw6_strings_winds_xlance (stopped early — partial result)

4-stage cascade: SW keeps guitar+other → gilliaan bowed-strings remover → Mega woodwind
subtraction → SW/X-LANCE guitar. ~6 min per clip on 4 CPU cores.

Stopped after 10/41 clips (partial_log.txt): mean SDR 3.42 dB vs 3.84 dB for plain SW (exp004)
on the same clips, -0.42 dB, better on only 2/10. Consistent with exp048 (3.12, = SW) and exp051
(2.92, < SW): cascaded removal of the orchestral instruments does not help the guitar estimate.
The remaining ~3.5 h were spent on the codec-robustness check instead.
