# Acoustic-guitar extraction: model and weights survey (as of 2026-09-26)

Scope: separate **only the acoustic guitar** from a commercial stereo mix of the ko-ko-ya track (album *Frevo!*, 2011).
Constraints: weights must come from HuggingFace, PyPI or Zenodo. GitHub releases and dl.fbaipublicfiles.com are unreachable.

Everything marked **[verified]** was checked from this container:
- the HF API file listing (sizes, LFS sha256),
- a ranged `curl -r 0-1023` that returned HTTP 206,
- and, for the key checkpoints, the `data.pkl` state-dict key names and shapes, read through zip range requests.

---

## 0. Target-track context (it changes the strategy)

- ko-ko-ya (コーコーヤ) is a trio formed in 2004:
  - 笹子重治 Shigeharu Sasago: acoustic guitar. He is known for gut/nylon-string (ガットギター) choro-style playing.
  - 江藤有希 Yuki Eto: violin.
  - 黒川紗恵子 Saeko Kurokawa: clarinet.
- *Frevo!* (2nd album, 2011-01-26) is mostly instrumental.
- The main things to remove are therefore **violin** and **clarinet**, plus any guest percussion, bass or piano.
- Vocals are probably absent.
- The guitar is probably a nylon-string classical guitar. Every public "guitar" model was trained on mixed acoustic and electric data.
- Sources: https://ototoy.jp/_/default/p/17615 , https://www.cdjapan.co.jp/product/XQJT-1002 , https://ja.wikipedia.org/wiki/笹子重治

Practical implication: a strong generic "guitar" model is appropriate, because the track has no electric guitar that needs rejecting. Useful cascade/cleanup steps are:
- subtract violin with the bowed-strings or violin models,
- subtract clarinet with the woodwind, wind or clarinet models.

---

## 1. Runtime code available without GitHub

| Package (PyPI) | Version / date | What it gives | Notes |
|---|---|---|---|
| **`msst`** | 0.1.0 (2026-09-09), MIT, py3.10–3.13, torch<2.12 | **Official ZFTurbo Music-Source-Separation-Training as a library.** The wheel (284 KB) contains `msst/models/`: `bs_roformer` (+ experimental, bs_conformer, PoPE), `mel_band_roformer` (+ experimental, mel_band_conformer), `mdx23c_tfc_tdf_v3`, `demucs4ht` (htdemucs), `scnet` / `scnet_masked` / `scnet_tran` / `scnet_unofficial`, `bandit`, `bandit_v2`, `bs_mamba2`, `look2hear` (Apollo), `conformer`, `moises_light`, segm/torchseg/swin_upernet. | API: `msst.inference(model_type=..., config_path=..., checkpoint_path=..., input_folder=..., output_folder=..., force_cpu=, use_tta=, extract_instrumental=)` and `msst.Separator(...)`. CLI: `msst-inference`. Install extras per arch, e.g. `pip install "msst[bs-roformer,mel-band-roformer]"`. It has **no model downloader**, so checkpoints and configs are supplied manually from HF. **[verified wheel contents]** The `BSRoformer.__init__` kwargs include `mlp_expansion_factor`, `skip_connection`, `use_torch_checkpoint` and `use_pope`, but **not** `use_shared_bias`. That means the original "SW" and "Logic" configs with `use_shared_bias: true` will not load; the **SW-Fixed** variant does. |
| `audio-separator` (python-audio-separator) | 0.47.0 (2026-08-27), MIT | Contains roformer (bs/mel), MDXC (tfc_tdf_v3), MDX-Net ONNX, VR arch and Demucs code (`uvr_lib_v5`). Its model list includes "BS Roformer SW by jarredou" (`BS-Roformer-SW.ckpt` + `BS-Roformer-SW.yaml`). | **It downloads from GitHub releases** (`TRvlvr/model_repo`, `nomadkaraoke/python-audio-separator/releases/model-configs`), which are blocked here. It skips downloading when the file already exists in `model_file_dir`, so HF-mirrored files can be pre-placed; `lainlives/audio-separator-models` mirrors its filenames. It is useful mainly for the **VR arch** (e.g. `17_HP-Wind_Inst-UVR.pth`) and **MDX-Net ONNX** models, which msst lacks. |
| `demucs` | 4.1.0 (2026-07-11), MIT | Official Demucs now **loads pretrained bags from HuggingFace** (`demucs/hf.py`, namespace `adefossez`: `htdemucs` -> `adefossez/HTDemucs`, `htdemucs_6s` -> `adefossez/HTDemucs-6s`, `htdemucs_ft` -> `adefossez/HTDemucs-ft`). It falls back to AWS only if HF fails. | **[verified]** `get_model('htdemucs_6s')` works through HF. |
| `demucs-infer` (openmirlab) | 4.2.2 | Inference-only Demucs fork | Downloads from "official Demucs repositories" (probably fbaipublicfiles). Prefer `demucs` 4.1.0. |
| `BS-RoFormer` (lucidrains) | 1.2.4 | Reference BS/Mel RoFormer modules | Current version depends on `hyper-connections`, `pope-pytorch` and `einx`. Its key names may drift from MSST; use `msst` instead. |
| `bs-roformer-infer`, `melband-roformer-infer` (openmirlab) | 0.1.5 (2026-07-12) | Inference-only BS/Mel RoFormer with sha256-verified auto-download | The default is BS-RoFormer-SW from `huggingface.co/enerjazzer/BS-ROFO-SW-Fixed` (sha256 `24e7d35e…`). The Mel variant defaults to Kim vocals from `KimberleyJSN/melbandroformer`. This is a good fallback if `msst` has issues. |
| `openunmix` | 1.3.0 | Open-Unmix | Legacy. |
| `sam-audio-infer` | 0.1.1 | SAM-Audio wrapper | Still needs `git+https://github.com/facebookresearch/sam-audio`, which is blocked. |
| `torch-scnet`, `mel-band-roformer`, `sam-audio`, `query-bandit`, `banquet` (MSS) | not on PyPI | – | Use `msst` for SCNet. |

Other code sources on HF, which are reachable:
- HF Spaces `noblebarkrr/mvsepless_*`: a full MSST-derived inference app with MDX-Net, VR and every model listed below.
- `chenxie95/xlance-msr`: the X-LANCE MSR code.
- `chenxie95/Language-Audio-Banquet`: a query-bandit fork with CLAP text queries.
- `akhaliq/sam-audio-large`: vendors the `sam_audio` package, but its requirements pull dacvae, sam3, ImageBind and CLAP from GitHub.

---

## 2. Guitar / acoustic-guitar separation weights

### Independent benchmark: MVSep Guitar leaderboard
- Source: https://mvsep.com/quality_checker/leaderboard/guitar (30-track guitar set; the ground truth is **all guitars, acoustic + electric**).

| Rank | Entry | SDR guitar | SDR other | Weights public? |
|---|---|---|---|---|
| 1 | BS Roformer SW (UVR5.6) | **9.012** | 15.98 | yes (HF, see A) |
| 2 | Logic Pro 11.2 Stem Splitter (Apple) | 9.002 | 15.94 | n/a |
| 7 | **bs_gtr_xlancer.ckpt** (X-LANCE MSR guitar) | **8.597** | 12.51 | yes (HF, see B) |
| 9 | **bs_mega_53stem_guitar_mvsep.ckpt** | **8.245** | 12.05 | yes (HF, see C) |
| 11 | MVSep Guitar Ensemble (BSRoformer + MelBandRoformer) | 7.511 | 14.48 | **no (mvsep.com only)** |
| 16 / 17 | MVSep BS-RoFormer [Guitar] / MVSep Guitar MelBand Roformer (2024.06.25) | 7.13 / 7.02 | – | **no** |
| 18 | Moises Guitar (2025) | 6.94 | – | no (commercial) |
| 24 / 26 / 28 | Mel-RoFormer Guitar (becruily) / mbr_guitar_becruily.ckpt | 5.63 / 5.29 / 5.05 | – | yes (HF, see D) |
| 27 | Demucs 6s | 5.25 | 11.94 | yes (HF, see E) |
| 33 | Mel-RoFormer [Acoustic Guitar] (MVSep) | 4.70 | 11.57 | **no** |
| 35 | MVSep Acoustic Guitar | 3.95 | 10.92 | **no** (the community says it separates acoustic from electric "very well"; https://msst-bible.pages.dev/separating-electric-and-acoustic-guitar) |
| 36 | bs_mega_53stem_electric-guitar_mvsep.ckpt | 3.81 | 8.48 | yes |
| 39 | **bs_mega_53stem_acoustic-guitar_mvsep.ckpt** | 3.35 | 8.64 | yes (see C) |
| 40 | mdx_6s_acoustic_guitar_anvuew.onnx | 2.38 | 7.96 | yes (see G) |

The acoustic-only models (rows 33, 35, 39, 40) are penalised on this set because its ground truth also contains electric guitar. Their numbers are **not** comparable to all-guitar models. For an all-acoustic trio track, an all-guitar model is a fair choice.

### A. BS-RoFormer "SW" (6 stems: bass, drums, other, vocals, guitar, piano)
- **Arch:** MSST `model_type: bs_roformer`. dim 256, depth 12, 62 bands (`freqs_per_bands` tuple), `mlp_expansion_factor` 4, `num_stems` 6. 44.1 kHz stereo. Inference `chunk_size` 588800 (config `dim_t` 1101, `num_overlap` 2).
- **Stem order:** `['bass','drums','other','vocals','guitar','piano']`.
- **Checkpoint:** `BS-Rofo-SW-Fixed.ckpt`, 699.4 MB, sha256 `24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e` **[verified identical across mirrors]**:
  - `enerjazzer/BS-ROFO-SW-Fixed` : `BS-Rofo-SW-Fixed.ckpt` + `BS-Rofo-SW-Fixed.yaml` (license: unknown; README: "Edited version of that checkpoint to allow its use with UVR and MSST").
  - `Blakus/bs_roformer_sw_6stem` : `BS-Rofo-SW-Fixed.ckpt` + `BS-Rofo-SW-Fixed.yaml`. Tagged MIT by the re-uploader; the original `jarredou/BS-ROFO-SW-Fixed` account no longer exists.
  - `cdjmix1991/bandsplit-roformer-sw-by-jarredou` : `model_BandSplit-Roformer_SW_by-jarredou.ckpt` + `config_BandSplit-Roformer_SW_by-jarredou.yaml`.
  - `lainlives/audio-separator-models` : `BS-Roformer-SW.ckpt` + `BS-Roformer-SW.yaml` (these are the audio-separator filenames).
  - `ChanTrail/BS-RoFormer` : `BS-Rofo-SW-Fixed.ckpt`, and a *different* `logic_bs_roformer.ckpt` / `logic_roformer.pt` (sha `57c4a987…`) with `logic_pro_config_v1.yaml`.
  - `noblebarkrr/mvsepless_resources` : `bs_roformer/bs_6stem_fixed.ckpt`.
- The state-dict uses standard MSST keys (`layers.N.N.layers.N.N.to_qkv.weight`, `to_gates`, `mask_estimators.0..5`) **[verified]**. It loads in stock `msst`.
- The "logic" ckpt has extra `to_qkv.bias`, `to_out.bias` and `linear_62_bias_0`-style keys and needs `use_shared_bias`. That is **not** stock MSST.
- **Scores:**
  - MVSep (https://mvsep.com/algorithms/77): vocals 11.30, instrum 17.50, bass 14.62, drums 14.11, **guitar 9.05**, **piano 7.83**, other 8.71.
  - MVSep piano leaderboard: 7.80, rank 1.
- **Provenance / licence: unclear.** Nobody discloses who trained it.
  - The MVSep guitar and piano leaderboards list "Logic Pro 11.2 Stem Splitter by Apple" with near-identical scores (9.002 vs 9.012; 7.789 vs 7.803).
  - ChanTrail's repo names the files "logic_roformer.pt" and "logic_pro_config_v1.yaml", and the key names look like a converted graph.
  - This strongly suggests the weights derive from Apple Logic Pro's Stem Splitter. The HF "MIT" tags are the re-uploaders' own, so treat the licence as unknown. It is fine for private experimentation; do not redistribute it.
- The X-LANCE MSR paper uses SW-Fixed as its frozen backbone: https://arxiv.org/html/2602.09042v1

### B. X-LANCE (SJTU) BS-RoFormer single-stem "Gtr" (MSR Challenge 2025 winner system)
- **Repo:** `chenxie95/xlance-msr-ckpt` (MIT): `gtr_mss.pth`, 204.4 MB, sha256 `813a92bd…`. The same file is mirrored as `noblebarkrr/mvsepless_resources` : `bs_roformer/bs_gtr_xlancer.ckpt` **[verified same sha]**.
- **Configs:**
  - MSST-format: `noblebarkrr/mvsepless_resources` : `bs_roformer/bs_gtr_xlancer_config.yaml` (instruments `[guitar, other]`, target `guitar`, sample_rate 44100, chunk 588800).
  - Original MSRKit config: HF Space `chenxie95/xlance-msr` : `configs/bsroformer/gtr.yaml`.
- **Arch:** `bs_roformer`. dim 256, depth 12, 62 bands, `mlp_expansion_factor` 4, `num_stems` 1. The keys are MSST-standard **[verified]**.
- **Training:**
  - initialised from BS-Rofo-SW-Fixed,
  - fine-tuned on **RawStems** (cleaned) + **MoisesDB**, target stem "Gtr" (all guitars),
  - trained on 48 kHz data; the paper notes the pretrained models run at 44.1 kHz with resampling,
  - L1 + multi-STFT loss.
- **Score:** 8.597 SDR guitar on MVSep (rank 7, the #2 public model).
- The same repo also holds `key_mss.pth` (keys; 7.777 SDR on the MVSep piano leaderboard), `vox_mss.pth`, `bass_mss.pth`, `drums_mss.pth`/`drums_mss1.pth`, `perc_mss*.pth`, `syn_mss*.pth`, `orch_mss*.pth` (orchestral: strings + winds), `denoise.pth` and `dereverb.pth` (913 MB mel-roformers).
- The HF licence is MIT. Note that MoisesDB's own licence is non-commercial research.
- Paper: https://arxiv.org/abs/2602.09042 ; code: https://github.com/ModistAndrew/xlance-msr (mirrored in the HF Space).

### C. MVSep **Mega 53 stems** BS-RoFormer (ZFTurbo, MSST release v1.0.21): the only public model with explicit acoustic-guitar and electric-guitar stems
- **Full model:** `oulianov/mvsep_mega_53` : `mvsep_mega_model_bs_roformer_53_stems_v1.ckpt` (1368.9 MB) + `mvsep_mega_model_bs_roformer_53_stems.yaml`.
  - Identical sha `c6282089…` in `lainlives/MelBand_BS_Roformers` (`mvsep_mega_model/…`) and `noblebarkrr/mvsepless_resources` (`bs_roformer/bs_mega_53stem_full_mvsep.ckpt`) **[verified]**.
  - The release notes say it needs ≥16 GB VRAM even at batch 1.
- **Per-stem split checkpoints** (shared trunk + one mask estimator, `num_stems: 1`, 77.6 MB each): `noblebarkrr/BS-Roformer-MVSep-Mega-53-stems` : `v1/bs_mega_53stem_<stem>_mvsep.ckpt` + `v1/bs_mega_53stem_<stem>_mvsep_config.yaml`. Relevant stems:
  - **`acoustic-guitar`**, `guitar`, `electric-guitar`,
  - `violin`, `bowed_strings`, `strings`, `viola`, `cello`, `double-bass`,
  - `clarinet`, `woodwind`, `wind`, `flute`,
  - `piano`, `keys`, `digital-piano`,
  - `bass`, `drums`, `percussion`, `congas`, `tambourine`, and others.
  - The acoustic-guitar checkpoint has 699 keys, a single `mask_estimators.0`, and MSST-standard names **[verified]**. The config sets `training.instruments: [acoustic-guitar, other]`.
- **Arch:** `bs_roformer`. dim 256, depth 12, `mlp_expansion_factor` 2, 44.1 kHz, audio chunk 441000, inference chunk 882000, `num_overlap` 2.
- **Scores:**
  - MVSep guitar leaderboard: guitar 8.245, electric 3.81, acoustic 3.35.
  - Piano 6.85, keys 6.36.
  - **wind 8.63** (rank 2 on the wind leaderboard, best public).
  - ZFTurbo warns that "individual stems may be lower than specialised models" and that the stems don't sum to the mix.
- **Licence:** the MSST code is MIT; no licence is stated for the weights. They were trained on MVSep's private data.
- Sources:
  - https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21
  - https://huggingface.co/noblebarkrr/BS-Roformer-MVSep-Mega-53-stems
  - https://vi-control.net/community/threads/new-open-source-53-stems-audio-splitting-model-available.171774/

### D. becruily Mel-Band RoFormer Guitar
- **Repo:** `becruily/mel-band-roformer-guitar` : `becruily_guitar.ckpt` (45.1 MB, sha `83472bbf…`) + `config_guitar_becruily.yaml`. It is mirrored in `noblebarkrr/mvsepless_resources` (`mel_band_roformer/mbr_guitar_becruily.ckpt`), `Politrees/UVR_resources` (`models/Roformer/MelBand/melband_roformer_guitar_becruily.ckpt`) and `xavriley/source_separation_mirror`.
- **Arch:** `mel_band_roformer`. dim 256, depth 4, 60 bands, `mlp_expansion_factor` 1, hop 441, 44.1 kHz, chunk 485100, instruments `[Guitar, Other]`. The keys are MSST-standard **[verified]**.
- **Score:** 5.05–5.63 SDR on the MVSep guitar leaderboard.
- **Licence:** there is no README or licence. Training data is undisclosed.
- It is small and fast, so it is a good ensemble member.

### E. HTDemucs 6s (Meta) and its guitar fine-tune
- **Official weights on HF:** `adefossez/HTDemucs-6s` : `5c90dfd2.safetensors` (54.9 MB) + `5c90dfd2.json` + `htdemucs_6s.yaml`.
  - Sources: `[drums, bass, other, vocals, guitar, piano]`, 44.1 kHz.
  - Loaded natively by `demucs==4.1.0` **[verified]**. MIT.
  - Trained on Meta's internal set (MUSDB-HQ + ~800 extra songs).
- **Scores:** MVSep guitar 5.25. The community says its piano is weak.
- **Legacy MSST-format copies** (`msst` model_type `htdemucs`):
  - `KitsuneX07/Music_Source_Sepetration_Models` : `multi_stem_models/HTDemucs4_6stems.th`,
  - `Sucial/MSST-WebUI` : `All_Models/multi_stem_models/HTDemucs4_6stems.th`,
  - `noblebarkrr/mvsepless_resources` : `htdemucs/demucs4_6stem.ckpt` + `demucs4_6stem_config.yaml`.
  - Plus `AEmotionStudio/htdemucs-models` and `set-soft/audio_separation` : `Demucs/htdemucs_6s.safetensors`.
- **Fine-tune:** `adityalakhani/htdemucs-6s-guitar-ft` : `guitar_htdemucs_6s.pt` (329.7 MB) + `config.yaml` + `inference.py`, Apache-2.0.
  - Fine-tuned on MoisesDB (240 songs).
  - Reports guitar SDR 8.73 vs 8.42 for stock htdemucs_6s, and SIR +0.89 dB, on a 24-track MoisesDB split. These are self-reported and not comparable to MVSep.
  - Load pattern: `get_model('htdemucs_6s')`, then `state_dict={'models.0.'+k: v}`.
- Other guitar fine-tunes (low value):
  - `barvarvara/demucs-guitar-lora` (2 MB PEFT LoRA on htdemucs, "acoustic guitar", MIT, no metrics),
  - `barvarvara/openunmix-guitar-ft` (`openunmix_guitar.pt` 106.8 MB, MIT, "acoustic guitar", self-reported SDR 10.7 on a custom set).

### F. Mel-Band RoFormer Guitar by chenCFD
- **Checkpoint:** `noblebarkrr/mvsepless_resources` : `mel_band_roformer/mbr_guitar_chencfd.ckpt` (276.7 MB) + `mbr_guitar_chencfd_config.yaml`.
- **Arch:** `mel_band_roformer`. dim 192, depth 8, 60 bands, `mlp_expansion_factor` 4, hop 512, chunk 131584, instruments `[guitar, others]`.
- No public metrics, provenance or licence.

### G. anvuew MDX-Net "6 stems" acoustic_guitar (UVR MDX-Net ONNX, not MSST)
- **Checkpoint:** `noblebarkrr/mvsepless_resources` : `mdxnet/mdx_6s_acoustic_guitar_anvuew.onnx` (27.1 MB) + `mdx_6s_acoustic_guitar_anvuew_config.yaml`.
  - Config: `dim_f` 2048, `dim_t` 256, `n_fft` 4096, hop 1024, compensation 1.03, primary stem `acoustic_guitar`.
- Siblings in the same set: `mdx_6s_electric_guitar_anvuew.onnx`, `…piano…`, `…bass…`, `…drum…`, `…vocals…`.
- **Score:** 2.38 on the MVSep guitar set, which includes electric guitar.
- Running it needs audio-separator's MDX runner (a hash lookup may fail, so parameters would have to be passed manually), mvsepless, or about 50 lines of onnxruntime code.

### H. Other guitar-related weights (not directly useful)
- Lead/rhythm guitar splitters:
  - `noblebarkrr/mvsepless_resources` : `mel_band_roformer/mbr_lead_rhythm_guitar_listra92.ckpt`,
  - `htdemucs/demucs4_lead_rhythm_guitar_drypaint.ckpt`.
- `yongyizang/MSRChallengeBaseline` : `melrnn/Guitar.pth` (22.5 MB) and `melunet/Guitar.pth` (648.8 MB). These are MSR baselines (Apache-2.0) and much weaker than B.
- MVSep-only (weights **not** public):
  - MVSep Guitar (MDX23C / MelRoformer / BSRoformer ensembles),
  - MVSep Acoustic Guitar,
  - MVSep "Mel-RoFormer [Acoustic Guitar]",
  - MVSep Guitar SW, Bowed Strings BSRoformer (2025.09), Wind BS Roformer (2025.08), Violin, Piano, Keys and similar.
  - ViperX's acoustic model is available only on uvronline.app premium.

---

## 3. Models useful in cascades (removing everything that is not guitar)

| Purpose | Repo : file (+ config) | Arch (msst model_type) | Size | Metric / notes |
|---|---|---|---|---|
| **Violin / bowed strings** | `gilliaan/Stem-Separation-Models` : `BowedStrings/BandSplitRoformer/gilliaan_bsroformer_bowedstrings_v2.ckpt` + `.yaml` (also v1; mirror `oulianov/BS-Roformer-BowedStrings-Duality`) | bs_roformer, 2 stems `['strings','other']`, 44.1k | 303 MB | MVSep strings SDR 4.69 (best public; MVSep's private model scores 5.42). The community calls it "ridiculously good". No licence stated. |
| Violin (specific) | Mega 53 split: `v1/bs_mega_53stem_violin_mvsep.ckpt` (+ `bowed_strings`, `strings`) | bs_roformer | 77.6 MB each | – |
| **Clarinet / woodwind** | Mega 53 split: `v1/bs_mega_53stem_clarinet_mvsep.ckpt`, `…_woodwind_…`, `…_wind_…` | bs_roformer | 77.6 MB | Mega "wind" 8.63 SDR on the MVSep wind leaderboard (#2 overall, best public). |
| Woodwind (VR arch) | `Politrees/UVR_resources` : `models/VR_Arch/17_HP-Wind_Inst-UVR.pth` (also `Sucial/MSST-WebUI`, `nomadkaraoke/public_uvr_models`, `seanghay/uvr_models`) | UVR VR arch (audio-separator) | 223.7 MB | Older model. |
| Orchestra (strings + winds) | `chenxie95/xlance-msr-ckpt` : `orch_mss.pth`, `orch_mss1.pth`; configs `noblebarkrr/mvsepless_resources` : `bs_roformer/bs_orch_xlancer_config.yaml`, `bs_orch2_xlancer_config.yaml` | bs_roformer 1-stem | 204 MB | MSR challenge. |
| **Piano / keys** | SW (A) piano stem; `chenxie95/xlance-msr-ckpt` : `key_mss.pth` (= `bs_keys_xlancer.ckpt`, config `bs_roformer/bs_keys_xlancer_config.yaml`); Mega 53 `piano` / `keys`; `jorisvaneyghen/SCNet` : `model_jazz_scnet_xl.ckpt` / `…_large.ckpt` + `config_jazz_scnet.yaml` (SCNet jazz: bass/drums/other/piano, MIT); `noblebarkrr/mvsepless_resources` : `mdxnet/mdx_6s_piano_anvuew.onnx`; `vr/wip-piano-4band-129605kb.ckpt`; `tjpurdy/Piano-Separation-Model-small` (CC-BY-NC) | – | – | Piano SDR: SW 7.80, xlance keys 7.78, Mega piano 6.85, MVSep piano ensemble 6.21 (private). |
| Drums / percussion | SW drums (14.11); `chenxie95/xlance-msr-ckpt` : `drums_mss.pth`, `perc_mss.pth`; `gilliaan/...Drums/…drums_v1.ckpt`; `noblebarkrr/mvsepless_resources` : `mdx23c/mdx23c_drumsep_6stem_aufr33_jarredou.ckpt` (drumsep); Mega 53 `percussion`, `congas`, `tambourine` | – | – | Frevo percussion (pandeiro, surdo, caixa) may land in "percussion" rather than "drums". |
| Bass | SW bass (14.62); `chenxie95/xlance-msr-ckpt` : `bass_mss.pth`; Mega 53 `bass`, `double-bass` | – | – | – |
| 4-stem general | `Aname-Tommy/Huge-SCNet-4stems` : `huge_scnet_4stems_v1.2.ckpt` (+ bleedless / fullness variants) + `config.yaml` (Apache-2.0); `SYH99999/bs_roformer_4stems_ft` : `bs_roformer_4stems_ft.pth` + `config.yaml` (Apache-2.0); `noblebarkrr/mvsepless_resources` : `bs_roformer/bs_4stem_zfturbo.ckpt`, `scnet/scnet_xl_ihf_4stem_zfturbo.ckpt`, `mdx23c/mdx23c_4stem_zfturbo.ckpt` | scnet / bs_roformer / mdx23c | 0.2–0.5 GB | The "other" stem is a good pre-filter. |
| Vocals / instrumental (if any vocals) | `KimberleyJSN/melbandroformer` : `MelBandRoformer.ckpt` (913 MB, MIT; Kim vocals, config `config_vocals_mel_band_roformer_kj.yaml` from MSST docs; the mvsepless copy is `mel_band_roformer/mbr_vocals_kim_config.yaml`); viperx BS-Roformer 1297/1296 (`noblebarkrr/mvsepless_resources` : `bs_roformer/bs_vocals_1297_viperx.ckpt`, `…1296…`; `Sucial/MSST-WebUI` : `All_Models/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt`); unwa big beta (`pcunwa/Mel-Band-Roformer-big` : `big_beta5e.ckpt` + `big_beta5e.yaml`, … `big_beta7`); becruily vocals / instrumental / deux (`becruily/mel-band-roformer-deux` : `becruily_deux.ckpt`, CC-BY-NC-4.0); `anvuew/BS-RoFormer` (GPL-3.0) | – | – | Probably not needed for this instrumental track. |
| Restoration after separation | `JusperLee/Apollo` : `pytorch_model.bin` (66.5 MB, CC-BY-SA-4.0); `Sucial/MSST-WebUI` : `All_Models/single_stem_models/apollo_model_uni.ckpt`; `baicai1145/Apollo-vocal-msst` | msst `apollo` | – | Band-limited / MP3 restoration. Optional. |
| Dereverb / denoise | `anvuew/dereverb_mel_band_roformer`, `anvuew/dereverb_bs_roformer`, `chenxie95/xlance-msr-ckpt` : `denoise.pth` | – | – | Optional post-processing. |

Suggested chained order from the community (https://msst-bible.pages.dev/chained-separation-order): instrumental, then drums, then piano/guitar, then strings/horns. A variant runs instrumental, drums, piano, strings/horns, bass, then guitars. Separate the stems with the highest SDR first.

---

## 4. Query-based / universal separators

| Model | Where | Notes |
|---|---|---|
| **Banquet** (query-bandit; Watcharasupat & Lerch, ISMIR 2024, arXiv 2406.18747) | Zenodo **record 13694558** (CC-BY-NC-SA-4.0): `ev-pre-aug.ckpt` (645.5 MB, recommended), `ev-pre.ckpt`, `vdbgp-*`, `bandit-vdbo.ckpt` **[verified reachable, HTTP 206]**. Code is on GitHub only (`kwatcharasupat/query-bandit`, blocked). | A BandIt/BSRNN decoder queried by a **PaSST embedding of a reference audio clip** (32 kHz mono), 44.1 kHz, 24.9 M params. Trained on MoisesDB and can target "clean acoustic guitar". Reported acoustic-guitar SNR is only about 2.2–3.3 dB. An idea for this track is to use a solo-guitar passage from the song itself as the query. |
| **Language-Audio Banquet** (SJTU fork, CLAP text/audio queries, RoFormer instead of RNN) | HF `chenxie95/Language-Audio-Banquet-ckpt` (MIT): `ev-pre-aug.ckpt` / `ev-pre.ckpt` (1050.4 MB each), `bandit-vdbo-roformer.ckpt` (717.8 MB). **Code is in HF Space `chenxie95/Language-Audio-Banquet`** (downloadable). It also needs the CLAP checkpoint `lukewys/laion_clap` : `music_speech_epoch_15_esc_89.25.pt`. | Supports a text query such as "acoustic guitar". It is experimental and has no published metrics. |
| **SAM-Audio** (Meta, arXiv 2512.18099, Dec 2025) | `facebook/sam-audio-{small,base,large}` (gated, "SAM license"). **Ungated mirrors:** `mrfakename/sam-audio-small` (5.1 GB), `-base` (7.7 GB), `-large` (14.9 GB); `AEmotionStudio/sam-audio-models` (bf16/fp8 safetensors). | A generative flow-matching DiT on a DAC-VAE codec at **48 kHz**, prompted by text such as "acoustic guitar". It is heavy and can hallucinate or change timbre (AudioShake: listeners preferred targeted models for instrument separation). The `sam_audio` code is vendored in HF Space `akhaliq/sam-audio-large`, but its dependencies (dacvae, sam3, ImageBind, CLAP) are GitHub-only. **Low priority.** |
| AudioSep / FlowSep | `JusperLee/AudioSep-hive`, `AlayaLab/FlowSep-hive` | General sound separation; weak on music instruments. |
| BandIt (cinematic, OJSP 2023) / BandIt v2 (DnR v3) | `kwatcharasupat/bandit-ojsp2023-musical64`; `noblebarkrr/mvsepless_resources` : `bandit_v2/bandit_v2_multi.ckpt` | Dialogue / music / effects only. **Not useful** for instruments. |

---

## 5. Papers and competitions (2023–2026) relevant here

- **BS-RoFormer**: Lu et al., ByteDance, SDX23 MSS-track winner. https://arxiv.org/abs/2309.02612
- **Mel-Band RoFormer**: Wang, Lu, Won. https://arxiv.org/abs/2310.01809 . Mel-RoFormer for vocals: https://arxiv.org/abs/2409.04702
- **SCNet** (sparse compression network; `msst` model_type `scnet`, plus `scnet_masked` and `scnet_tran` variants): https://arxiv.org/abs/2401.13276
- **SDX23 / MDX23 music demixing track** (MDX23C = TFC-TDF-v3, KUIELab / ZFTurbo): https://arxiv.org/abs/2308.06979 ; leaderboards and benchmarks: https://arxiv.org/abs/2305.07489
- **BandIt** (cinematic): https://arxiv.org/abs/2309.02539 . BandIt v2 / Divide-and-Remaster v3: https://arxiv.org/abs/2407.07275
- **Banquet** (stem-agnostic, query-based; separates fine classes such as clean acoustic guitar): https://arxiv.org/abs/2406.18747 ; weights https://zenodo.org/records/13694558
- **Apollo** (band-sequence restoration of lossy music; msst `apollo`): https://arxiv.org/abs/2409.08514
- **TS-BSMamba2 / BSMamba2** (Mamba-2 band-split; msst `bs_mamba2`): https://arxiv.org/abs/2409.06245
- **Moises-Light** (WASPAA 2025, lightweight band-split U-Net; msst `moises_light`): https://arxiv.org/abs/2510.06785
- **ACMID** (7-stem dataset with separate Acoustic Guitar / Electric Guitar / Strings / Wind-Brass stems; code on GitHub): https://arxiv.org/abs/2510.07840
- **Music Source Restoration (MSR) Challenge 2025** and the X-LANCE winning system (cascade of SW-Fixed + fine-tuned single-stem BS-RoFormers including Gtr and Key): https://arxiv.org/abs/2602.09042 ; baseline weights in `yongyizang/MSRChallengeBaseline`. Follow-ups include https://arxiv.org/pdf/2603.04032 and DTT-BSR+ https://arxiv.org/pdf/2606.24127
- **SAM Audio**: https://arxiv.org/abs/2512.18099
- Community guides (current state of the art, including MVSep-only models):
  - MSST Bible: https://msst-bible.pages.dev/ (see `acoustic-guitar`, `separating-electric-and-acoustic-guitar`, `strings`, `piano`, `chained-separation-order`)
  - deton24's UVR/MDX guide: https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c

---

## 6. Prioritised download list for acoustic-guitar extraction

All files are on huggingface.co, except #8, which is on Zenodo. Load them with `pip install "msst[bs-roformer,mel-band-roformer]"` unless noted otherwise.

| # | Repo : checkpoint | Config | msst model_type | Stems | Size | Licence | Why |
|---|---|---|---|---|---|---|---|
| 1 | `enerjazzer/BS-ROFO-SW-Fixed` : `BS-Rofo-SW-Fixed.ckpt` (or `Blakus/bs_roformer_sw_6stem`, same sha) | `BS-Rofo-SW-Fixed.yaml` (same repo) | bs_roformer | bass, drums, other, vocals, **guitar**, **piano** | 699.4 MB | unknown; likely Logic-Pro-derived | Best public guitar SDR (9.01) and piano SDR (7.80). Gives all context stems in one pass. |
| 2 | `chenxie95/xlance-msr-ckpt` : `gtr_mss.pth` (= `noblebarkrr/mvsepless_resources` : `bs_roformer/bs_gtr_xlancer.ckpt`) | `noblebarkrr/mvsepless_resources` : `bs_roformer/bs_gtr_xlancer_config.yaml` | bs_roformer | guitar (+ other = residual) | 204.4 MB | MIT (HF); trained on RawStems + MoisesDB | Guitar SDR 8.60. An independent fine-tune, good for ensembling with #1. |
| 3 | `noblebarkrr/BS-Roformer-MVSep-Mega-53-stems` : `v1/bs_mega_53stem_acoustic-guitar_mvsep.ckpt` | `v1/bs_mega_53stem_acoustic-guitar_mvsep_config.yaml` | bs_roformer | acoustic-guitar | 77.6 MB | unspecified (ZFTurbo/MVSep release) | The only public HF model trained specifically on acoustic guitar. |
| 4 | same repo : `v1/bs_mega_53stem_guitar_mvsep.ckpt` (+ `…violin…`, `…clarinet…`, `…woodwind…`, `…bowed_strings…`, `…piano…`, `…percussion…`) | matching `v1/bs_mega_53stem_<stem>_mvsep_config.yaml` | bs_roformer | 1 stem each | 77.6 MB each | unspecified | Guitar 8.25 SDR. The violin and clarinet stems allow a subtract-the-others cascade for this trio. |
| 5 | `becruily/mel-band-roformer-guitar` : `becruily_guitar.ckpt` | `config_guitar_becruily.yaml` | mel_band_roformer | Guitar / Other | 45.1 MB | none stated | Small and fast; an ensemble or sanity-check member (SDR about 5.3). |
| 6 | `gilliaan/Stem-Separation-Models` : `BowedStrings/BandSplitRoformer/gilliaan_bsroformer_bowedstrings_v2.ckpt` | `…/gilliaan_bsroformer_bowedstrings_v2.yaml` | bs_roformer | strings / other | 303.4 MB | none stated | Best public bowed-strings model (4.69). Use it to remove the violin before or after guitar extraction. |
| 7 | `adefossez/HTDemucs-6s` : `5c90dfd2.safetensors` (via `demucs==4.1.0` `get_model('htdemucs_6s')`); optional fine-tune `adityalakhani/htdemucs-6s-guitar-ft` : `guitar_htdemucs_6s.pt` | `htdemucs_6s.yaml` / `config.yaml` | htdemucs (demucs pkg) | drums, bass, other, vocals, guitar, piano | 54.9 MB / 329.7 MB | MIT / Apache-2.0 | A different architecture family (waveform plus spectrogram hybrid), useful as a diverse ensemble member. It is weaker (5.25). |
| 8 | Zenodo `13694558` : `ev-pre-aug.ckpt`, or HF `chenxie95/Language-Audio-Banquet-ckpt` : `ev-pre-aug.ckpt` (code in HF Space `chenxie95/Language-Audio-Banquet`) | configs in the Space's `config/` | Banquet (query-bandit, not msst) | any stem given an audio or text query | 645 MB / 1050 MB | CC-BY-NC-SA-4.0 / MIT | Query-based. It could be given an isolated guitar passage of the song itself as the query. Experimental and low SDR. |

Honourable mentions:
- `noblebarkrr/mvsepless_resources` : `mel_band_roformer/mbr_guitar_chencfd.ckpt` (276.7 MB, mel_band_roformer, unknown quality).
- `…/mdxnet/mdx_6s_acoustic_guitar_anvuew.onnx` (27 MB, MDX-Net ONNX, acoustic-only, weak).
- `chenxie95/xlance-msr-ckpt` : `key_mss.pth` / `orch_mss.pth` for removing piano or orchestral parts.
- Mega 53 full: `oulianov/mvsep_mega_53` (1.37 GB, all 53 stems in one pass, ≥16 GB VRAM).
