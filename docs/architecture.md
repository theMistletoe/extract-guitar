# Architecture

```
            ┌──────────── acoustic_separator (src/) ────────────┐
input  ──►  │ fetch.py  (URL → audio, optional, yt-dlp)          │
(file|URL)  │ audio.py  (decode once → float32 stereo PCM)       │
            │                                                    │
            │ pipeline.py  declarative DAG of steps:             │
            │   model step   → inference.Separator (chunked OLA, │
            │                  TTA, shifts, bf16/fp32)           │
            │   ensemble     → ensemble.METHODS                  │
            │   post         → wiener / gate                     │
            │   op sub/sum   → cascades (mix − stem, …)          │
            │   cache: ~/.cache/acoustic-separator/stems/<key>   │
            │                                                    │
            │ models/loader.py  catalog (configs/models.yaml),   │
            │   HF download + sha256, MSST architectures         │
            │   (models/msst/, MIT, vendored), demucs package    │
            └──────────────┬─────────────────────────────────────┘
                           ▼
        outputs/<track>/acoustic_guitar.wav      (float32 WAV)
        outputs/<track>/non_acoustic_guitar.wav  (= mix − acoustic, exact)
        outputs/<track>/run.json                 (models, checksums, params)
```

## Separation core

* **Decoding** – `audio.load_audio` decodes WAV/FLAC natively and MP3/M4A/MP4/OGG via
  ffmpeg (system or the `imageio-ffmpeg` wheel) straight to float32 at the original rate;
  resampling (soxr VHQ) happens only when a model needs another rate, and results are
  resampled back. Nothing is re-encoded; every output is float32 WAV.
* **Inference** – `inference.Separator.separate` runs the model over overlapping chunks
  (`chunk_size`, `num_overlap`, linear-fade or Hann window, reflect padding at the track
  edges), optionally with test-time augmentation (`swap` L/R, `invert` polarity) and random
  time shifts, averaging all passes. `precision: bf16` uses CPU/GPU autocast (≈1.5× faster
  on AMX CPUs, output within −40 dB of fp32); final renders use fp32.
* **Pipelines** – `configs/pipelines/*.yaml` describe Strategy A (direct), B (two-stage),
  C (instrument-removal cascade) and D (ensemble) with the same step vocabulary, so every
  strategy is evaluated by identical code. Separator outputs are cached by
  (checkpoint sha256, parameters, input hash); ensembles and post-processing re-use them.
* **Mix consistency** – the residual written as `non_acoustic_guitar.wav` is always
  `mixture − acoustic_guitar`, so `mixture == acoustic + non_acoustic` holds exactly.

## Evaluation

* `evaluation.py` – ground-truth metrics (SDR, SI-SDR, SDRi, BSS-Eval SIR/SAR with a 512-tap
  distortion filter, target retention, leakage, per-class leakage, MR-STFT error).
* `proxy.py` – reference-free proxies for the real song (AST AudioSet classifier
  probabilities for guitar and each interfering class on 5 s windows). Used only as a
  sanity check, never as the selection criterion.
* `mixing.py` + `augment.py` – synthetic validation/training mixtures with exact ground
  truth (EQ, compression, saturation, band-limiting, pan/width, Haas delay, echo, synthetic
  reverb per stem; linked bus compression/limiting so `mix == Σ stems` still holds).
* `scripts/benchmark.py` – one experiment = validation set + target song → `experiments/<id>/`
  (`config.yaml`, `metrics.json`, `notes.md`, `per_clip.csv`, `output.wav`),
  `experiments/results.csv`, `outputs/target/candidates/<id>.wav`, and the
  Champion/Challenger update (`artifacts/champion/`, history never deleted).
* `report.py` – HTML report with synchronized original/target/residual players,
  click-to-seek spectrograms and loudest-segment hints.

## Reproducibility

Every experiment stores: git commit, command, timestamp, seed, hardware, package list
(`environment.txt`), model names, checkpoint sha256, pipeline YAML and parameters.
Weights are cached under `~/.cache/acoustic-separator/` (never committed). Audio, weights
and datasets are git-ignored.
