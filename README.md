# extract-guitar — acoustic guitar stem extraction

完成済みのステレオ楽曲から **アコースティックギターだけ** を高精度に抽出するローカルツールです。
既存の最新 Music Source Separation モデル (BS-RoFormer / Mel-Band RoFormer / HTDemucs ほか) を
Direct・Two-stage・Cascade・Ensemble の各方式で組み合わせ、正解 stem 付きの合成検証セットで
客観評価して最良のパイプライン (Champion) を選びます。

* PRD: [`docs/PRD.md`](docs/PRD.md)
* 調査: [`docs/research.md`](docs/research.md)
* 設計: [`docs/architecture.md`](docs/architecture.md)
* 実験履歴: [`docs/experiments.md`](docs/experiments.md), [`experiments/results.csv`](experiments/results.csv)
* 最終レポート: [`reports/final_report.md`](reports/final_report.md)

## Results (ground-truth validation set, 41 clips)

| Pipeline | Val. SDR | SIR / SAR | Retention | Notes |
|---|---|---|---|---|
| Best single model (Mega-53 acoustic head) | 3.13 dB | 14.8 / 1.2 | 0.33 | clean but loses guitar |
| HTDemucs-6s guitar-FT + Mega acoustic, waveform mean | 4.16 dB | 8.4 / 6.6 | 0.43 | Strategy D, default chunks |
| same, Mega at 4 s chunks (HTDemucs native 7.8 s) | 5.79 dB | 9.9 / 8.9 | 0.61 | Phase 4 inference tuning |
| 3-member ensemble + learned gain refiner r4 | 6.90 dB | 11.6 / 9.3 | 0.62 | Phase 9 |
| **+ refiner r5 with tuned chunks (Champion, `--quality max`)** | **7.62 dB** | **12.5 / 9.5** | **0.69** | +0.72 dB over r4, 32/41 clips |
| Oracle ideal Wiener mask (upper bound) | 10.70 dB | – | – | headroom |

Details: [`reports/final_report.md`](reports/final_report.md), per-experiment log
[`docs/experiments.md`](docs/experiments.md). No ground truth exists for the target song, so
"perfect" is never claimed; selection is by validation metrics, paired statistics and
target-song diagnostics (`outputs/target/report.html`, generated locally).

## Quick start (1 command)

```bash
# 1. install (uv, Python 3.10–3.12). Choose the PyTorch build:
uv sync --extra cpu        # CPU / Apple Silicon (MPS)
# uv sync --extra cu126    # NVIDIA CUDA 12.6

# 2. separate (file or URL). --quality max = current Champion pipeline, fp32
uv run --no-sync python -m acoustic_separator --input "<youtube-url-or-audio-file>" --quality max
```

`--quality max` runs 5 separation passes (4 distinct checkpoints) plus the learned refiner (weights ship in
`artifacts/refiner/`). On the 4-core reference CPU a 2-minute song takes roughly 1–1.5 hours; a GPU
is much faster. `--quality standard` (best single model, 3.13 dB) and `--quality fast`
(HTDemucs-ft, 3.07 dB, ~50 s per song) are the cheap alternatives.

Outputs (float32 WAV, original sample rate):

```text
outputs/<track-id>/
  original.wav
  acoustic_guitar.wav        # extracted acoustic guitar
  non_acoustic_guitar.wav    # everything else (= original − acoustic_guitar, exact)
  run.json                   # models, checkpoint sha256, parameters, runtime
```

Options: `--output DIR`, `--pipeline NAME|PATH.yaml` (any file in `configs/pipelines/`),
`--device auto|cpu|cuda|mps`, `--save-candidates` (write every intermediate stem),
`--no-cache`.

URL input uses `yt-dlp` and is optional/decoupled (`src/acoustic_separator/fetch.py`).
If a site blocks the download (terms of use, DRM, region, bot checks), obtain the audio
legally and pass the local file (WAV / MP3 / M4A / FLAC / MP4 …).

Model weights are downloaded once from Hugging Face into `~/.cache/acoustic-separator/`
(override with `ACOUSTIC_SEPARATOR_CACHE`) and their sha256 is recorded. Weights, datasets
and audio are never committed to git.

## Guitar tablature of the target song

[`outputs/target/tab/`](outputs/target/tab/README.md) holds a tab transcribed from the extracted
guitar stem: engraved [PDF](outputs/target/tab/frevo_guitar_tab.pdf) (notation + TAB),
[ASCII tab](outputs/target/tab/frevo_guitar_tab.txt), Guitar Pro 5, MusicXML, recording-aligned
MIDI and a note CSV, with a [verification report](outputs/target/tab/verification.md).

```bash
uv sync --extra cpu --extra tab
uv run --no-sync python scripts/transcribe_tab.py   # --stem x.wav --out dir --title ... for other songs
uv run --no-sync python scripts/verify_tab.py
```

Pipeline (`src/acoustic_separator/tab/`): tuning estimate -> high-resolution onset/offset
regression CRNN (Kong et al. architecture, Riley's guitar checkpoint loaded from safetensors) ->
beat grid from the guitar's own onsets with swing-aware 16th quantisation -> string/fret Viterbi
over fingering x hand position -> chord symbols. On GuitarSet (not used for training) the note
model reaches onset F1 0.80 and the fingering picks the performer's string for 67.8 % of notes.

## Reproducing the experiments

```bash
uv sync --extra cpu --extra dev
python scripts/download_models.py                 # fetch + checksum every catalog model
python scripts/fetch_datasets.py                  # raw sources (HF / Zenodo) -> data/raw/
python scripts/build_manifest.py                  # datasets/manifest.csv (licence, split)
python scripts/prepare_dataset.py --clean         # 41-clip ground-truth validation set
python scripts/make_codec_valset.py               # AAC-128k variant (codec robustness)
python scripts/run_queue.py configs/queues/phase2_baselines.yaml   # etc. for every queue
python scripts/sweep.py configs/sweeps/<sweep>.yaml
bash   scripts/phase7_refiner.sh                  # training clips, refiner, hard-example mining
bash   scripts/phase9_r5.sh                       # final refiner r5 (candidates, fit, benchmark)
python scripts/run_target.py --render-best        # Champion in fp32 -> outputs/target/best + report.html
python scripts/update_presets.py && python scripts/failure_analysis.py
python scripts/summarize_experiments.py && python scripts/make_final_report.py
```

The raw multitrack songs are deleted after the clips are rendered (disk budget); the
scripts above re-download and re-render them deterministically (fixed seeds).

See `docs/experiments.md` for the full list of commands that produced every experiment.

## Repository layout

```text
configs/models.yaml        model catalog (HF repo, file, sha256, licence)
configs/pipelines/         direct / two-stage / cascade / ensemble pipelines
src/acoustic_separator/    cli, audio, inference, pipeline, ensemble, evaluation, proxy,
                           mixing/augment (synthetic data), report, tracking, models/,
                           tab/ (guitar tablature: amt, rhythm, fretboard, chords, export)
scripts/                   benchmark, download_models, prepare_dataset, train, evaluate, run_target,
                           transcribe_tab, verify_tab
datasets/manifest.csv      every source file used, with licence and split
experiments/               one folder per experiment + results.csv
artifacts/champion/        current Champion (history.jsonl keeps every past Champion)
artifacts/refiner/rN/      learned gain refiners (model.pt ~300 KB + train_log.json)
outputs/target/            target-song candidates, best/, report.html, tab/ (guitar tab)
reports/final_report.md    final evaluation
```
