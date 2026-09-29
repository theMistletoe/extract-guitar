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
MIDI and a note CSV.  [`check/index.html`](outputs/target/tab/check/index.html) plays the extracted
guitar, a rendering of the tab, or both (left/right) while highlighting the current 16th, with
the bars flagged by the automatic checks marked for listening first.  It has not been checked
against the recording by ear yet.

```bash
uv sync --extra cpu --extra tab
uv run --no-sync python scripts/transcribe_tab.py   # --stem x.wav --out dir --title ... for other songs
uv run --no-sync python scripts/verify_tab.py       # consistency checks + bars to review
uv run --no-sync python scripts/make_tab_check.py   # listening aids
```

Pipeline (`src/acoustic_separator/tab/`): tuning estimate -> high-resolution onset/offset
regression CRNN (Kong et al. architecture) averaging two of X. Riley's guitar checkpoints (FL and
GAPS paper version; `.pth` files are read with `torch.load(weights_only=True)` and a NumPy-only
allowlist, then kept as safetensors) -> beat grid from the guitar's own onsets with swing-aware
16th quantisation -> string/fret Viterbi over fingering x hand position -> chord symbols.

End-to-end accuracy on music with known notes ([`reports/tab_benchmark.md`](reports/tab_benchmark.md),
`scripts/bench_tab.py`): GuitarSet bossa-nova comping and GAPS choro guitar mixed with URMP
violin/clarinet and percussion at the target's balance, separated with the Champion pipeline,
then transcribed: note precision 0.930, recall 0.880, F1 0.904 (the earlier single checkpoint:
0.853; FL alone 0.883, GAPS paper version alone 0.902); strings match the performer's for 77 % of
correctly detected notes.  On clean GuitarSet (60 excerpts,
[`reports/tab_benchmark_guitarset.md`](reports/tab_benchmark_guitarset.md)) F1 is 0.913 (earlier:
0.798).  The GAPS-paper checkpoint may have seen GuitarSet in training, so the GuitarSet-based
numbers can be optimistic; on the benchmark's GAPS pieces (270 notes, unseen by every checkpoint)
the gain is smaller (0.914 -> 0.922; FL alone 0.893, GAPS paper alone 0.908).  The same benchmark
calibrates the review signals: tab notes that all three distinct checkpoints hear are right 96 %
of the time, two 84 %, only one 53 %; applied to the target's counts this predicts about 90 wrong
notes out of 1256 ([`verification.md`](outputs/target/tab/verification.md)), and the checker marks the
79 single-checkpoint notes.

`scripts/render_tab_audio.py` plays the tab back as audio (FluidSynth, FluidR3_GM nylon guitar, MIT):
one MIDI channel per string, each string damped at the transcribed end of its note, per-note
velocity measured in the stem, and an EQ (within +-6 dB) toward the stem's long-term spectrum.
[`outputs/target/tab/audio/frevo_tab_guitar.mp3`](outputs/target/tab/audio/frevo_tab_guitar.mp3)
keeps the recording's timing and pitch (aligned with the stem to within the 1.5 ms measurement
step); `frevo_tab_guitar_practice_110.mp3` is the tab's grid at a steady 110 BPM with a count-in and
a click.

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
                           transcribe_tab, verify_tab, make_tab_check, bench_tab
datasets/manifest.csv      every source file used, with licence and split
experiments/               one folder per experiment + results.csv
artifacts/champion/        current Champion (history.jsonl keeps every past Champion)
artifacts/refiner/rN/      learned gain refiners (model.pt ~300 KB + train_log.json)
outputs/target/            target-song candidates, best/, report.html, tab/ (guitar tab)
reports/final_report.md    final evaluation
```
