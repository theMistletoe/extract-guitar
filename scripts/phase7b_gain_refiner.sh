#!/usr/bin/env bash
# Phase 7-9 (round 3): gain refiner on top of the Champion ensemble, hard-example weighted.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
POS="htdemucs6s_gtrft:guitar mega_multi:acoustic-guitar sw6:guitar mega_multi:guitar"
NEG="sw6:other sw6:piano sw6:vocals sw6:bass sw6:drums mega_multi:violin mega_multi:woodwind mega_multi:electric-guitar"
W=datasets/train_clips/mining_r1b.json
[ -f "$W" ] && WARG="--weights $W" || WARG=""
$PY scripts/train.py candidates --pos $POS --neg $NEG
$PY scripts/train.py fit --mode gain --base-members 0 1 --pos $POS --neg $NEG --epochs ${EPOCHS:-16} \
    --out artifacts/refiner/r3 $WARG
$PY scripts/benchmark.py --pipeline R_r3 --slug R_r3 --strategy "D+gain-refiner+HEM" \
    --hypothesis "Refining the Champion ensemble itself (0..2 TF gain, identity at init) keeps its strengths and learns local fixes: lift guitar holes, cut violin/clarinet/electric leakage; trained on disjoint clips with hard-example oversampling." \
    --change "gain-mode refiner r3 on HTDemucs-ft + Mega acoustic mean; 4 positive + 8 negative evidence stems; mining weights from r1" \
    --next "final render / inference tuning" || true
