# Failure modes — exp006_D_htft_xl_wave_mean

Per-clip labels on the ground-truth validation set: `guitar_removed` (retention < 0.4), `<class>_leak` (largest attributed leakage > −12 dB re. target energy), `artifacts/other` (SDR < 3 dB without a dominant leak), else `ok`.

| clip | SDR | retention | worst leak class | leak dB | label |
|---|---|---|---|---|---|
| syn_clean_electric_1 | -6.80 | 0.83 | electric_clean | 6.2 | electric_clean_leak |
| ms_Rod_Alexander_Tears_In_The_R_1 | -6.62 | 0.79 | bass | 3.2 | bass_leak |
| syn_distorted_electric_0 | -5.89 | 0.69 | electric_dist | 2.9 | electric_dist_leak |
| syn_buried_1 | -1.80 | 0.25 | electric_clean | -1.7 | guitar_removed |
| ms_Swing_Bazar_Fleche_DOr_0 | -1.67 | 0.63 | electric_guitar | -2.0 | electric_guitar_leak |
| ms_Andres_Guazzelli_Attention_0 | -1.36 | 0.35 | electric_guitar | -3.1 | guitar_removed |
| syn_plucked_1 | -1.15 | 0.92 | plucked_other | 0.3 | plucked_other_leak |
| ms_Perdidos_Na_Zona_Sul_Meu_Bem_1 | -1.13 | 0.37 | electric_guitar | -6.6 | guitar_removed |
| ms_Swing_Bazar_Fleche_DOr_1 | -1.03 | 0.60 | electric_guitar | -4.5 | electric_guitar_leak |
| syn_dense_0 | -0.99 | 0.19 | bass | -4.2 | guitar_removed |
| syn_electric_band_0 | -0.65 | 0.72 | electric_clean | -1.4 | electric_clean_leak |
| ms_Perdidos_Na_Zona_Sul_Meu_Bem_0 | -0.64 | 0.47 | electric_guitar | -5.5 | electric_guitar_leak |
| ms_Egda_Carolyn_Saudade_Do_Teu__0 | -0.19 | 0.31 | electric_guitar | -5.4 | guitar_removed |
| syn_buried_0 | -0.05 | 0.22 | electric_clean | -6.9 | guitar_removed |
| ms_Catfolkin_Sant_Jordi_v2_0_1 | 1.39 | 0.35 | electric_guitar | -11.1 | guitar_removed |
| ms_Leslie_Mendelson_The_Hardest_0 | 1.64 | 0.56 | vocals | -11.2 | vocals_leak |
| ms_Rod_Alexander_Tears_In_The_R_0 | 1.91 | 0.81 | electric_guitar | -5.5 | electric_guitar_leak |
| ms_Catfolkin_Sant_Jordi_v2_0_0 | 2.11 | 0.42 | electric_guitar | -10.7 | electric_guitar_leak |
| syn_band_pop_0 | 2.22 | 0.28 | electric_clean | -11.3 | guitar_removed |
| ms_Spektakulatius_What_Child_Is_1 | 2.84 | 0.34 | piano | -15.3 | guitar_removed |
| syn_strings_1 | 3.59 | 0.33 | violin | -10.6 | guitar_removed |
| ms_Adam_Buckley_Drag_me_Down_0 | 3.71 | 0.38 | electric_guitar | -15.9 | guitar_removed |
| syn_chamber_frevo_2 | 3.88 | 0.70 | bass | -7.6 | bass_leak |
| ms_Wolfs_Head_Vixen_Morris_Band_0 | 4.19 | 0.82 | plucked_other | -10.6 | plucked_other_leak |
| ms_Spektakulatius_What_Child_Is_0 | 4.42 | 0.46 | winds | -19.8 | ok |
| ms_Enda_Reilly_An_Nasc_Nua_0 | 4.97 | 0.38 | bass | -25.7 | guitar_removed |
| syn_cymbal_drums_0 | 5.64 | 0.39 | drums | -26.4 | guitar_removed |
| ms_Enda_Reilly_An_Nasc_Nua_1 | 6.39 | 0.53 | vocals | -16.8 | ok |
| syn_plucked_0 | 6.51 | 0.97 | plucked_other | -7.0 | plucked_other_leak |
| syn_chamber_frevo_0 | 7.11 | 0.66 | bass | -14.0 | ok |
| ms_Bolz_Knecht_Summertime_1 | 7.28 | 0.74 | winds | -10.1 | winds_leak |
| syn_clean_electric_0 | 8.21 | 0.99 | electric_clean | -8.5 | electric_clean_leak |
| ms_Bolz_Knecht_Summertime_0 | 8.71 | 0.55 | winds | -26.6 | ok |
| syn_piano_1 | 9.00 | 0.60 | piano | -27.6 | ok |
| syn_piano_band_0 | 9.38 | 0.70 | bass | -24.2 | ok |
| syn_winds_0 | 10.21 | 0.62 | winds | -28.9 | ok |
| syn_chamber_frevo_1 | 10.39 | 0.77 | bass | -21.2 | ok |
| syn_strings_0 | 11.58 | 0.76 | violin | -25.2 | ok |
| syn_female_vocal_0 | 12.21 | 0.78 | vocals | -26.1 | ok |
| syn_winds_1 | 13.03 | 0.76 | winds | -31.1 | ok |
| syn_piano_0 | 16.49 | 0.93 | piano | -24.6 | ok |

## Summary

| label | clips | mean SDR |
|---|---|---|
| guitar_removed | 13 | 1.45 |
| ok | 12 | 9.91 |
| electric_guitar_leak | 5 | 0.14 |
| electric_clean_leak | 3 | 0.25 |
| plucked_other_leak | 3 | 3.19 |
| bass_leak | 2 | -1.37 |
| electric_dist_leak | 1 | -5.89 |
| vocals_leak | 1 | 1.64 |
| winds_leak | 1 | 7.28 |
