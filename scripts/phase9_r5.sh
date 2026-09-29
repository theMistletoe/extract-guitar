#!/usr/bin/env bash
# Phase 9 (round 5): r4 with the per-model chunk sizes from the Phase 4 sweep (HTDemucs native 7.8 s, Mega on the mix at 4 s).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
POS="sw6:guitar>mega_multi:~electric-guitar htdemucs6s_gtrft@native:guitar mega_multi@176400:acoustic-guitar sw6:guitar mega_multi@176400:guitar"
NEG="sw6:other sw6:piano sw6:vocals sw6:bass sw6:drums mega_multi@176400:violin mega_multi@176400:woodwind mega_multi@176400:electric-guitar"
W=datasets/train_clips/mining_r1b.json
$PY scripts/train.py candidates --pos $POS --neg $NEG
$PY scripts/train.py fit --mode gain --base-members 0 1 2 --pos $POS --neg $NEG --epochs ${EPOCHS:-6} \
    --out artifacts/refiner/r5 --weights $W
$PY scripts/benchmark.py --pipeline R_r5_m4 --slug R_r5_m4 --strategy "B+D+gain-refiner+HEM" \
    --hypothesis "The Mega-side chunk sweep showed Mega at 4 s and HTDemucs at its native 7.8 s beat all-6 s by 0.32 dB on the plain ensemble (exp047 vs exp038); a refiner trained on candidates computed that way should keep that gain." \
    --change "gain refiner r5: r4 recipe; HTDemucs members at native chunk, Mega-on-mix members at 4 s chunks (train and inference); SW and the SW-to-Mega cascade stay at 6 s" \
    --next "final render" || true
