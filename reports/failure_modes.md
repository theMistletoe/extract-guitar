# Failure modes — exp050_R_r5_m4

Per-clip labels on the ground-truth validation set: `guitar_removed` (retention < 0.4), `<class>_leak` (largest attributed leakage > −12 dB re. target energy), `artifacts/other` (SDR < 3 dB without a dominant leak), else `ok`.

| clip | SDR | retention | worst leak class | leak dB | label |
|---|---|---|---|---|---|
| ms_Swing_Bazar_Fleche_DOr_1 | -3.86 | 0.85 | electric_guitar | -0.7 | electric_guitar_leak |
| syn_clean_electric_1 | -2.23 | 0.81 | electric_clean | 1.2 | electric_clean_leak |
| ms_Swing_Bazar_Fleche_DOr_0 | -0.94 | 0.79 | electric_guitar | -1.1 | electric_guitar_leak |
| ms_Rod_Alexander_Tears_In_The_R_1 | 0.49 | 0.53 | bass | -6.8 | bass_leak |
| syn_band_pop_0 | 1.22 | 0.21 | bass | -11.1 | guitar_removed |
| syn_buried_0 | 1.41 | 0.33 | electric_clean | -8.5 | guitar_removed |
| ms_Leslie_Mendelson_The_Hardest_0 | 1.44 | 0.56 | vocals | -10.5 | vocals_leak |
| ms_Perdidos_Na_Zona_Sul_Meu_Bem_0 | 1.45 | 0.45 | electric_guitar | -11.9 | electric_guitar_leak |
| syn_dense_0 | 2.08 | 0.49 | bass | -7.0 | bass_leak |
| ms_Perdidos_Na_Zona_Sul_Meu_Bem_1 | 2.12 | 0.36 | bass | -18.5 | guitar_removed |
| ms_Catfolkin_Sant_Jordi_v2_0_1 | 2.50 | 0.34 | electric_guitar | -19.3 | guitar_removed |
| syn_distorted_electric_0 | 3.01 | 0.73 | bass | -7.4 | bass_leak |
| syn_buried_1 | 3.09 | 0.34 | drums | -18.9 | guitar_removed |
| syn_electric_band_0 | 3.66 | 0.76 | electric_clean | -7.2 | electric_clean_leak |
| syn_strings_1 | 3.80 | 0.56 | violin | -8.5 | violin_leak |
| ms_Andres_Guazzelli_Attention_0 | 3.82 | 0.51 | electric_guitar | -13.8 | ok |
| ms_Egda_Carolyn_Saudade_Do_Teu__0 | 4.07 | 0.52 | electric_guitar | -19.3 | ok |
| ms_Catfolkin_Sant_Jordi_v2_0_0 | 4.33 | 0.53 | electric_guitar | -18.6 | ok |
| ms_Spektakulatius_What_Child_Is_1 | 5.03 | 0.51 | bass | -21.9 | ok |
| ms_Spektakulatius_What_Child_Is_0 | 6.35 | 0.71 | bass | -16.5 | ok |
| ms_Adam_Buckley_Drag_me_Down_0 | 6.65 | 0.68 | bass | -21.2 | ok |
| syn_plucked_1 | 6.70 | 0.92 | plucked_other | -10.0 | plucked_other_leak |
| ms_Wolfs_Head_Vixen_Morris_Band_0 | 7.96 | 0.85 | plucked_other | -13.7 | ok |
| syn_chamber_frevo_2 | 8.54 | 0.77 | bass | -17.9 | ok |
| ms_Rod_Alexander_Tears_In_The_R_0 | 8.70 | 0.72 | bass | -22.6 | ok |
| ms_Bolz_Knecht_Summertime_0 | 9.24 | 0.65 | winds | -31.8 | ok |
| ms_Enda_Reilly_An_Nasc_Nua_1 | 9.91 | 0.82 | vocals | -17.5 | ok |
| ms_Enda_Reilly_An_Nasc_Nua_0 | 9.98 | 0.72 | vocals | -23.8 | ok |
| syn_chamber_frevo_0 | 10.12 | 0.76 | bass | -18.5 | ok |
| syn_cymbal_drums_0 | 10.41 | 0.67 | drums | -25.5 | ok |
| syn_piano_band_0 | 11.58 | 0.85 | bass | -23.3 | ok |
| syn_chamber_frevo_1 | 13.35 | 0.81 | bass | -25.6 | ok |
| ms_Bolz_Knecht_Summertime_1 | 13.72 | 0.81 | winds | -29.7 | ok |
| syn_piano_1 | 14.14 | 0.85 | piano | -25.8 | ok |
| syn_female_vocal_0 | 16.01 | 0.91 | vocals | -24.8 | ok |
| syn_strings_0 | 17.39 | 0.94 | violin | -21.1 | ok |
| syn_plucked_0 | 17.41 | 0.97 | plucked_other | -21.8 | ok |
| syn_winds_0 | 17.78 | 0.95 | winds | -23.0 | ok |
| syn_piano_0 | 19.80 | 0.95 | piano | -29.0 | ok |
| syn_winds_1 | 19.99 | 0.95 | winds | -27.1 | ok |
| syn_clean_electric_0 | 20.07 | 0.99 | electric_clean | -23.8 | ok |

## Summary

| label | clips | mean SDR |
|---|---|---|
| ok | 25 | 11.45 |
| guitar_removed | 5 | 2.06 |
| electric_guitar_leak | 3 | -1.12 |
| bass_leak | 3 | 1.86 |
| electric_clean_leak | 2 | 0.71 |
| vocals_leak | 1 | 1.44 |
| violin_leak | 1 | 3.80 |
| plucked_other_leak | 1 | 6.70 |
