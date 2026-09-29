#!/usr/bin/env bash
# Phase 9 (round 4): gain refiner on the improved ensemble (SW minus electric + HTDemucs-ft + Mega acoustic).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
POS="sw6:guitar>mega_multi:~electric-guitar htdemucs6s_gtrft:guitar mega_multi:acoustic-guitar sw6:guitar mega_multi:guitar"
NEG="sw6:other sw6:piano sw6:vocals sw6:bass sw6:drums mega_multi:violin mega_multi:woodwind mega_multi:electric-guitar"
W=datasets/train_clips/mining_r1b.json
$PY scripts/train.py candidates --pos $POS --neg $NEG
$PY scripts/train.py fit --mode gain --base-members 0 1 2 --pos $POS --neg $NEG --epochs ${EPOCHS:-6} \
    --out artifacts/refiner/r4 --weights $W
$PY scripts/benchmark.py --pipeline R_r4_c6 --slug R_r4_c6 --strategy "B+D+gain-refiner+HEM" \
    --hypothesis "The 3-member ensemble with the Strategy-B stem (SW minus electric) is 0.29 dB better than the pair used by r3; the same gain refiner on this stronger base should add a similar gain on top." \
    --change "gain refiner r4: base = mean of SW-minus-electric, HTDemucs-ft, Mega acoustic; 5 positive + 8 negative evidence stems; r1 mining weights; 6 epochs; members at 6 s chunks as in training (the c6 finding)" \
    --next "final render" || true
