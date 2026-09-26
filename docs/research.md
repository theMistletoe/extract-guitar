# Phase 0 — Research (as of 2026-09-26)

This file summarises the survey that drove model and data selection. Everything marked
*verified* was checked from the project container (HF file listing, HTTP 206 range
request, and for key checkpoints the state-dict keys). Full notes with every source URL:
[`research_models_raw.md`](research_models_raw.md) (models) and
[`research_datasets_raw.md`](research_datasets_raw.md) (datasets).

Environment constraint that shaped the survey: the build container reaches Hugging Face,
PyPI and Zenodo but **not** GitHub releases or `dl.fbaipublicfiles.com`, has no GPU (4-core
Sapphire Rapids CPU, AMX-BF16, 15 GB RAM), and YouTube downloads are refused (HTTP 429 /
"video not available"); the target audio was supplied by the user as a file.

## 0. Target song context

*Frevo!* by **ko-ko-ya** (コーコーヤ, album *Frevo!*, 2011): a trio of
笹子重治 Shigeharu Sasago (nylon/gut-string acoustic guitar, choro style), 江藤有希 Yuki Eto
(violin) and 黒川紗恵子 Saeko Kurokawa (clarinet); instrumental. An AudioSet classifier
(AST) run over the track confirms violin as the dominant instrument, clarinet in places,
light percussion, no vocals and a comparatively quiet guitar.

Consequences:
* the hard interferers are **violin (incl. pizzicato) and clarinet**, not vocals/drums;
* the guitar is nylon-string — all public "guitar" models were trained mostly on
  steel-string acoustic + electric guitar;
* no electric guitar is present, so an *all-guitar* model is a legitimate core, but the
  validation set must still punish electric-guitar / piano / strings / winds leakage so the
  pipeline stays general (PRD §2, §12).

## 1. Architectures (2023–2026)

| Family | Paper | Notes for this task |
|---|---|---|
| **BS-RoFormer** | Lu et al., SDX23 winner, [arXiv 2309.02612](https://arxiv.org/abs/2309.02612) | Band-split + RoPE transformers over time & band axes; strongest public guitar/piano checkpoints (SW, X-LANCE, MVSep Mega) use it. |
| **Mel-Band RoFormer** | Wang, Lu, Won, [arXiv 2310.01809](https://arxiv.org/abs/2310.01809) | Overlapping mel bands; dominant for vocals; becruily guitar model (small). |
| **HTDemucs / Demucs v4** | Rouard et al. 2022 | Hybrid waveform+spectrogram; `htdemucs_6s` has guitar & piano stems (weaker, but a *different family* → ensemble diversity). `demucs` 4.1.0 now loads weights from HF. |
| **SCNet** | [arXiv 2401.13276](https://arxiv.org/abs/2401.13276) | Sparse compression; strong 4-stem models; no public guitar head. |
| **BandIt / BandIt v2** | [2309.02539](https://arxiv.org/abs/2309.02539), [2407.07275](https://arxiv.org/abs/2407.07275) | Cinematic (dialogue/music/effects) — not useful for instruments. |
| **Banquet** (query BandIt) | [arXiv 2406.18747](https://arxiv.org/abs/2406.18747) | Stem-agnostic, query-by-example; trained on MoisesDB incl. *acoustic guitar*; reported acoustic-guitar SNR only ≈2–3 dB. Weights Zenodo 13694558 (CC-BY-NC-SA). |
| **Apollo** | [arXiv 2409.08514](https://arxiv.org/abs/2409.08514) | Restoration of lossy / band-limited music (optional post-process). |
| **BS-Mamba2 / TS-BSMamba2** | [arXiv 2409.06245](https://arxiv.org/abs/2409.06245) | Vocals only public. |
| **MDX23C (TFC-TDF-v3)** | [arXiv 2308.06979](https://arxiv.org/abs/2308.06979) | SDX23; drumsep / 4-stem checkpoints. |
| **Moises-Light** | [arXiv 2510.06785](https://arxiv.org/abs/2510.06785) | Lightweight band-split U-Net. |
| **SAM-Audio** (Meta) | [arXiv 2512.18099](https://arxiv.org/abs/2512.18099) | Text-prompted generative separator (48 kHz); can alter timbre; dependencies GitHub-only → not usable here. |
| **MSR Challenge 2025** (X-LANCE winner) | [arXiv 2602.09042](https://arxiv.org/abs/2602.09042) | Cascade of SW-Fixed + fine-tuned single-stem BS-RoFormers (Gtr, Key, Orch…). |

Frameworks: **MSST** (ZFTurbo/Music-Source-Separation-Training, MIT; now also `msst` on
PyPI) — we vendor its architecture code (`src/acoustic_separator/models/msst/`, commit
`050cae7`); **UVR / python-audio-separator** (downloads from GitHub releases, blocked here);
**MVSep** (many best guitar/acoustic/strings/wind models are *site-only*, weights not public).

## 2. Independent benchmark: MVSep guitar leaderboard

Ground truth there is *all guitars* (acoustic + electric); acoustic-only models are
penalised and not comparable.

| Model | Guitar SDR | Weights public |
|---|---|---|
| BS-RoFormer SW (6 stems) | **9.01** | yes (HF) |
| Logic Pro 11.2 Stem Splitter | 9.00 | – |
| X-LANCE `gtr_mss` (BS-RoFormer) | 8.60 | yes (HF, MIT) |
| MVSep Mega-53 `guitar` head | 8.25 | yes (HF) |
| MVSep Guitar ensemble / BS / Mel | 7.51 / 7.13 / 7.02 | no |
| becruily Mel-RoFormer guitar | 5.05–5.63 | yes (HF) |
| HTDemucs 6s | 5.25 | yes (HF, MIT) |
| MVSep Acoustic Guitar / Mel [Acoustic] | 3.95 / 4.70 (acoustic-only) | no |
| Mega-53 `acoustic-guitar` head | 3.35 (acoustic-only) | yes (HF) |
| anvuew MDX-Net acoustic (ONNX) | 2.38 (acoustic-only) | yes |

Source: <https://mvsep.com/quality_checker/leaderboard/guitar>. README / leaderboard numbers
were **not** used to pick the final model — every candidate was re-run on our own
validation set and on the target song (PRD §10).

## 3. Downloadable checkpoints used (configs/models.yaml)

| Catalog name | HF repo : file | Arch | Stems | Params | Licence / provenance |
|---|---|---|---|---|---|
| `sw6` | enerjazzer/BS-ROFO-SW-Fixed : BS-Rofo-SW-Fixed.ckpt | bs_roformer d256×12 | bass, drums, other, vocals, guitar, piano | 175 M | **unknown**, anonymous; very likely derived from Logic Pro's splitter → private use only, never redistributed |
| `xlance_gtr` | chenxie95/xlance-msr-ckpt : gtr_mss.pth (+ config from noblebarkrr/mvsepless_resources) | bs_roformer d256×12 | guitar | 51 M | MIT on HF; fine-tuned from SW on RawStems + MoisesDB (non-commercial data) |
| `mega_acoustic` / `mega_guitar` / `mega_electric` | noblebarkrr/BS-Roformer-MVSep-Mega-53-stems : v1/bs_mega_53stem_*.ckpt | bs_roformer d256×12 (shared trunk + one head) | acoustic-guitar / guitar / electric-guitar | 39 M each | unspecified (ZFTurbo/MVSep release v1.0.21, private MVSep data) |
| `mega_violin`, `mega_bowed`, `mega_clarinet`, `mega_woodwind`, `mega_percussion` | same repo | same | one stem each | 39 M each | same |
| `becruily_guitar` | becruily/mel-band-roformer-guitar | mel_band_roformer d256×4 | guitar | 22 M | none stated |
| `gilliaan_strings` | gilliaan/Stem-Separation-Models : BowedStrings v2 | bs_roformer d256×12 | strings, other | 76 M | none stated; best public bowed-strings (4.69) |
| `xlance_orch` | chenxie95/xlance-msr-ckpt : orch_mss.pth | bs_roformer | orch (strings+winds) | 51 M | MIT on HF |
| `htdemucs6s` | adefossez/HTDemucs-6s (via `demucs` 4.1.0) | htdemucs | drums, bass, other, vocals, guitar, piano | 27 M | MIT |
| `htdemucs6s_gtrft` | adityalakhani/htdemucs-6s-guitar-ft | htdemucs | same (guitar fine-tuned on MoisesDB) | 27 M | Apache-2.0 |

Checksums (sha256) are verified/recorded on download (`~/.cache/acoustic-separator/checksums.json`)
and stored in each experiment's `config.yaml`.

Not usable / not public: MVSep's own Guitar, Acoustic Guitar, Strings, Wind and Piano models
(site-only); ViperX acoustic (paid); SAM-Audio (GitHub-only deps); UVR model_repo (GitHub
releases blocked).

## 4. Datasets (ground truth for evaluation and training)

No public dataset has an isolated acoustic-guitar stem for the target song, so evaluation
uses **separate datasets with exact acoustic-guitar ground truth** (PRD §14).
Everything used is listed per file in `datasets/manifest.csv` (id, source, licence,
instrument, type, path, split). Nothing below is redistributed in this repository.

| Source | What we use | Licence / terms | Role |
|---|---|---|---|
| **RawStems** (Mixing Secrets / Cambridge-MT multitracks, HF `kwatcharasupat/mixing-secrets-rawstems`) | 13 complete songs with acoustic guitar (`Gtr/AG`) + all other stems; separate single stems (violin, winds, percussion, mandolin/banjo/ukulele, vocals, electric guitar, piano, drum overheads, bass) from other songs | non-commercial research/education only, no redistribution | **Coherent real-song validation** (`ms_*` clips); hard negatives; training songs (disjoint) |
| **GuitarSet** (Zenodo 3371780) | mono mic recordings, 6 players | CC BY 4.0 | steel-string acoustic targets: player 05 → validation, players 00–04 → training |
| **GAPS** (HF `xavriley/GAPS`) | 12 test-split pieces (val), 16 train-split pieces (train) | research use (audio from YouTube performances) | **nylon-string classical guitar** targets — closest timbre to the target song |
| **URMP** (HF `Eredis02/URMP`) | separated violin / clarinet / flute tracks, disjoint pieces for val and train | research use only | the target's own interferer instruments |
| **GuitarJam** (HF `Julian-br/GuitarJam`) | 30 clean Stratocaster DI clips (even → val, odd → train) | CC0 | clean-electric hard negatives |

Not usable: MoisesDB (form-gated), MedleyDB full (restricted), MVSep benchmark (hidden GT),
Slakh (100 GB; BabySlakh 16 kHz), Medley-solos-DB (3 s clips), MSR sets (targets are
unprocessed sources, not mixture-consistent).

### Validation set (`datasets/validation`, `scripts/prepare_dataset.py`, seed 20260926)

41 clips × 12 s, 44.1 kHz stereo float32, each with `mixture.wav`, `acoustic_guitar.wav`,
`stems/<class>.wav` and `meta.json`; `mixture == acoustic_guitar + Σ stems` exactly.

* **19 `ms_*` clips** – excerpts of 12 real multitrack songs (gypsy-jazz with violin and
  accordion, folk with violin/flute, bass clarinet + piano, sax duo, Brazilian pop, Celtic
  with mandolin/bouzouki/whistles, pop/rock bands…). Stems in `Gtr/AG` that are mandolin,
  banjo, ukulele, bouzouki, dobro etc. are moved to the interferers; room/bleed/click stems are
  dropped so the ground truth cannot contain other instruments. Per-stem EQ/compression/
  reverb/pan and a linked bus compressor+limiter are applied; the window is chosen where the
  guitar and many other classes are active. Guitar-to-rest ratio −11…+1 dB.
* **22 `syn_*` clips** – scenario mixtures (PRD §15): chamber/“frevo-like” (violin + winds +
  bass + percussion), strings, winds, piano, clean electric, plucked (mandolin/banjo/ukulele),
  pop band, electric band, distorted electric, female vocal, cymbal-heavy drums, guitar buried
  at −12…−14 dB, dense 7-instrument mix, piano band. Nylon (GAPS) and steel (GuitarSet)
  targets are balanced ≈ 50/50. Randomised gain, EQ, compression, saturation, band-limiting,
  reverb (synthetic RIR), stereo position/width, Haas delay, echo, bus mastering.

**Contamination caveat (PRD §28):** X-LANCE's guitar model was fine-tuned on RawStems, and
Mixing Secrets songs also appear in MUSDB18/DSD100, so the `ms_*` family may be partly seen
by some models. Results are therefore always reported for `ms_*` and `syn_*` separately as well
as combined, and model choices must hold on `syn_*` (GAPS/GuitarSet targets, which none of the
separators is documented to have used).

### Training set (only for Phase 7; never overlaps validation)

RawStems songs not in the validation list (live-bleed sessions excluded), GuitarSet players
00–04, GAPS train split, URMP pieces not used for validation, odd GuitarJam clips and
negatives drawn from songs not used anywhere in validation.
