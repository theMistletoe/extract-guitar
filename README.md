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
| HTDemucs-6s guitar-FT + Mega acoustic, waveform mean | 4.16 dB | 8.4 / 6.6 | 0.43 | Strategy D |
| **+ learned gain refiner (Champion, `--quality max`)** | **4.68 dB** | **9.5 / 6.4** | **0.51** | paired +0.53 dB, CI [+0.18, +0.86] |
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
python scripts/run_target.py --render-best        # Champion in fp32 -> outputs/target/best + report.html
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
                           mixing/augment (synthetic data), report, tracking, models/
scripts/                   benchmark, download_models, prepare_dataset, train, evaluate, run_target
datasets/manifest.csv      every source file used, with licence and split
experiments/               one folder per experiment + results.csv
artifacts/champion/        current Champion (history.jsonl keeps every past Champion)
outputs/target/            target-song candidates, best/, report.html
reports/final_report.md    final evaluation
```
