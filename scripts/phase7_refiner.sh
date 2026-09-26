#!/usr/bin/env bash
# Phase 7-9: learned mask refiner on top of the pretrained separators.
#   1. training clips (disjoint from validation)       -> datasets/train_clips
#   2. candidate stems of every input model (6 s chunks, cached)
#   3. fit round 1 (uniform sampling)                   -> artifacts/refiner/r1
#   4. hard-example mining on the training clips        -> mining_r1.json
#   5. new clips for the failure types + fit round 2    -> artifacts/refiner/r2
# Every round is then benchmarked on the untouched validation set by scripts/benchmark.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
POS=${POS:-"sw6:guitar xlance_gtr mega_acoustic"}
NEG=${NEG:-"sw6:other sw6:piano sw6:vocals mega_violin mega_woodwind"}
EPOCHS=${EPOCHS:-25}

[ -d datasets/train_clips ] || $PY scripts/train.py prepare --clips-per-song 4 --n-syn 64 --seconds 6
$PY scripts/train.py candidates --pos $POS --neg $NEG
$PY scripts/train.py fit --pos $POS --neg $NEG --epochs "$EPOCHS" --out artifacts/refiner/r1
$PY scripts/train.py mine --pos $POS --neg $NEG --refiner artifacts/refiner/r1/model.pt \
    --out datasets/train_clips/mining_r1.json
$PY scripts/train.py prepare --clips-per-song 0 --n-syn 48 --seconds 6 --tag r2 \
    --scenario-weights datasets/train_clips/mining_r1_scenario_weights.json
$PY scripts/train.py candidates --pos $POS --neg $NEG
$PY scripts/train.py mine --pos $POS --neg $NEG --refiner artifacts/refiner/r1/model.pt \
    --out datasets/train_clips/mining_r1b.json
$PY scripts/train.py fit --pos $POS --neg $NEG --epochs "$EPOCHS" --out artifacts/refiner/r2 \
    --weights datasets/train_clips/mining_r1b.json
