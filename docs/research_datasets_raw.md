# Datasets for acoustic-guitar separation: ground-truth validation and small fine-tuning set

Researched 2026-09-26 from inside the container. All sizes and URLs below were checked from this container with curl, the HF API, the Zenodo API, and HTTP Range requests, unless marked "not verified".

## 0. Environment facts found during the research (these affect the plan)

| Thing | Result |
|---|---|
| zenodo.org | Reachable. **HTTP Range works** (206) on `https://zenodo.org/api/records/<id>/files/<name>/content`. Throughput is about **1–2.6 MB/s**: an 883 MB stream took 338 s. |
| huggingface.co | Reachable. LFS/xet files redirect to `us.aws.cdn.hf.co` (works; `cdn-lfs.huggingface.co` once gave a 502 at the proxy). Throughput is about **15 MB/s**. Range works. |
| GitHub release assets | `github.com/.../releases/download/...` redirects to `release-assets.githubusercontent.com` and **did work** (206) for the sigsep MUSDB18-7 zip. |
| download.magenta.tensorflow.org | Reachable (NSynth). |
| Partial zip extraction | Because Range works, you can pull single members out of huge zips (MUSDB18-HQ 22.7 GB, IDMT, AG-PT-set) with a remote-zip reader. A working helper is `scratchpad/dsr/rzip.py` (`open_remote(url)` returns a `zipfile.ZipFile` over HTTP ranges). |
| ffmpeg | Not installed. `pip install imageio-ffmpeg` (pypi, 29.5 MB wheel) ships a static ffmpeg 7.0.2. It is already installed at `scratchpad/pylib` (use `PYTHONPATH=scratchpad/pylib`). The binary is `scratchpad/pylib/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`. numpy is also in `scratchpad/pylib`. No soundfile, librosa or torchaudio is installed system-wide (they are all on pypi). |

---

## D. Target track: "Frevo!" by ko-ko-ya (コーコーヤ)

### Track facts
- **Track:** "Frevo!", track 1 of the album **"Frevo!"** (ko-ko-ya's 2nd album).
- **Release:** 2011-01-26, **Happiness Records XQJT-1002**, UPC 4538182107584, ISRC **JPB461030201**.
- **Duration:** **2:12** (iTunes `trackTimeMillis` = 132147; Deezer = 132 s; OTOTOY = 02:12).
- **Composer:** 笹子重治 (Shigeharu Sasago).
- **Personnel** (Happiness Records page):
  - Guitar: **笹子重治 Shigeharu Sasago**, leader of Choro Club. His Japanese Wikipedia page says his work centres on **ガットギター**, i.e. **nylon-string (classical) guitar**. The target is therefore a nylon-string acoustic guitar, not a steel-string one.
  - Violin: **江藤有希 Yuki Eto**.
  - Clarinet: **黒川紗恵子 Saeko Kurokawa**.
  - Guest percussion: **岡部洋一 Yoichi Okabe**, on tracks **1**, 2, 3, 5, 6, 10, 11 and 15. Frevo is a fast Brazilian (Recife) march, so the percussion is probably pandeiro or snare-like.
  - Guest guitar: ゴンザレス三上 (Gontiti), on track 13 only.
  - No piano, bass or vocals are credited. **The hard negatives that matter most for this target are violin, clarinet and hand percussion.**
- **Genre tags:** iTunes "ラテン" (Latin); Deezer "Latin Music"; OTOTOY "Classical/Traditional".
- **Sources:**
  - https://happiness-records.com/discography/frevo/
  - https://www.kurokawasaeko.com/discography/340
  - https://ja.wikipedia.org/wiki/%E7%AC%B9%E5%AD%90%E9%87%8D%E6%B2%BB
  - Deezer album API: https://api.deezer.com/album/278223162
  - https://ototoy.jp/_/default/p/17615

### Legal previews (verified and downloaded into the scratchpad)

| Source | URL | Format | Length | Saved as |
|---|---|---|---|---|
| iTunes Search API (JP; the US store also has it) | `https://itunes.apple.com/search?term=ko-ko-ya+frevo&entity=song&country=jp`, which gives previewUrl `https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview115/v4/a6/aa/a9/a6aaa9ae-15e0-bab8-d51e-bf51db0533cf/mzaf_11970545144905918689.plus.aac.p.m4a` | AAC-LC 256 kbps (264 kb/s), 44.1 kHz stereo | 29.98 s | `scratchpad/target_preview.m4a` (1,020,039 B) |
| Deezer API | `https://api.deezer.com/search?q=ko-ko-ya%20frevo`, track id 1579567072; the `preview` URL carries an expiring hdnea token, so fetch it fresh | MP3 128 kbps 44.1 kHz | 30 s | `scratchpad/target_preview_deezer.mp3` |
| **OTOTOY trial** | `https://ototoy.jp/api/trial/op/115047.mp3` (send the Referer header of the album page) | MP3 256 kbps, **48 kHz** stereo | **45.0 s** | `scratchpad/target_preview_ototoy.mp3` (1,440,768 B) |

- All three previews start at **the same point in the song**. Cross-correlation lag between iTunes and Deezer is 0.025 s with peak 0.99; the OTOTOY lag is about 0. OTOTOY runs 15 s longer.
- **OTOTOY gives the most audio (45 of 132 s); iTunes has the best codec.** Where the preview sits inside the song is unknown without the full track.
- **Full lossless track, legally:** OTOTOY sells track 1 alone as FLAC/WAV/ALAC 16-bit/44.1 kHz for **¥157** (album ¥1,257). iTunes JP sells it as AAC for ¥153. If the user buys it and uploads the file, you get the full 132 s.
- Other tracks on the same album have iTunes previews in `scratchpad/itunes_jp.json` (15 tracks). They share the same guitar, violin and clarinet timbre, so they can serve as unlabeled in-domain checks. Tracks 4, 7, 8, 9, 12, 13 and 14 have no percussion.

---

## A + C. Isolated acoustic-guitar ground truth, and multitracks with an acoustic-guitar stem

### A1. The best find: Mixing Secrets raw multitracks, per-file on HF
- **Repo:** HF dataset `kwatcharasupat/mixing-secrets-rawstems`. It is a renamed and reorganized mirror of **RawStems** by Yongyi Zang (`yongyizang/RawStems`), which is itself a crawl of the Cambridge-MT "Mixing Secrets" library.
- **Size:** 14,607 FLAC files, 166.7 GB in total. **Every stem is a separate file, so you can download selectively.** Audio is **48 kHz, 24-bit, stereo FLAC**; stems are full-length and sample-aligned (checked from FLAC STREAMINFO).
- **Layout:** `dev/<Artist - Song>/<Category>/<NN_Name>.flac` for 462 songs, plus `musdb18hq/test/<Artist - Song>/...` for the 24 MUSDB18-HQ test songs as raw stems.
- **Category tree (files / GB / songs):**
  - `Gtr/AG` 342 / 4.75 / **132 songs**; `Gtr/EG` 2336 / 22.2 / 342
  - `Orch/STR` 282 / 4.16 / 91; `Orch/WW` 115 / 1.41 / 51; `Orch/BR` 159 / 1.71
  - `Kbs/PN` 210 / 4.28 / 145; `Kbs/OR`; `Kbs/EP`
  - `Rhy/DK` 3897 / 64 / 449; `Rhy/PERC` 593 / 5.2 / 200
  - `Voc/LV` 958; `Voc/BV` 1972; `Voc/GRP` 238
  - `Bass` 736; `Synth` 990; `Misc` 618
- **Download URL pattern:** `https://huggingface.co/datasets/kwatcharasupat/mixing-secrets-rawstems/resolve/main/<url-encoded path>`. Alternatively use `huggingface_hub.snapshot_download(repo_id="kwatcharasupat/mixing-secrets-rawstems", repo_type="dataset", allow_patterns=["dev/Swing_Bazar - Fleche_DOr/*", ...])`.
- **Full list of the 132 songs with AG stems:** `scratchpad/dsr/ms_ag_songs.txt`. The full file tree is `scratchpad/dsr/tree_kwatcharasupat__mixing-secrets-rawstems.json`.
- **License and terms:**
  - The HF README says: "THIS DATASET SHOULD STRICTLY BE USED ONLY FOR NON-COMMERCIAL RESEARCH AND EDUCATIONAL PURPOSES."
  - Cambridge-MT terms say downloads are "provided free of charge for educational purposes only… not… for any commercial purpose". Their FAQ says ML research is within the spirit of educational use but has not been agreed with the contributors.
  - **Use internally only. Do not redistribute the stems or the mixtures made from them.** (Sources: https://cambridge-mt.com/ms3/mtk-faq/ and https://cambridge-mt.com/ms3/mtk/, which were 403 to WebFetch; the terms quoted come from search snippets.)
- **Caveats:**
  - `Gtr/AG` also holds **mandolin, banjo, ukulele/"Ukelele", guitalele, bouzouki ("Bazouki"), dobro, autoharp and sitar**. Filter by filename. Those files are good hard negatives.
  - `Timo_Carlier - The_Road_Ahead` AG files are "AcGtrPlusVox" and contain voice. Exclude them.
  - Sessions with room or Blumlein mics (Pretty_Saro, Mipso, Barnstar, Chad_Hollister) are live, so the stems contain **bleed**.
  - Stems are **raw, unprocessed** tracks, and often several mics per instrument (Mic1/Mic2/DI). Build the mixture yourself by summing the stems, optionally with light gain, EQ or reverb applied the same way to the GT. Then GT = the sum of the AG stems, and the separation target is exactly consistent with the mixture.

**Recommended RawStems songs for this target** (instrumental, folk or Latin; acoustic guitar with violin, clarinet or winds). Sizes are for all stems of the song.

| Song path (under `dev/`) | Total | Stems |
|---|---|---|
| `Swing_Bazar - Fleche_DOr` | 113 MB | AG + **Violin** + Accordion + EG + Bass. Gypsy-jazz instrumental, the closest analogue to the target. |
| `Bolz__Knecht - Summertime` | 99 MB | AG close/far/DI + Saxophone (3 mics). Duo. |
| `Spektakulatius - What_Child_Is_This` | 190 MB | AG + **Bass clarinet** + Piano + Bass + drums + vox |
| `Catfolkin - Sant_Jordi_v2.0` | 339 MB | AG×2 + **Violin** + **Flute** + Accordion + EG + drums + vox |
| `Wolfs_Head__Vixen_Morris_Band - Lament` | 149 MB | AcGuitar1/2 + Mandolin + Bouzouki + Hurdy-gurdy + Tin whistles/Recorders + drum |
| `Pretty_Saro - Carolina_In_The_Pines` | 152 MB | AG + Mandolin + **Violin** + Bass + room mics (bleed) |
| `Mipso - Wallpaper_Baby` | 160 MB | AG + Mando + **Violin** + upright bass + **percussion** + group vox (room mics, bleed) |
| `Rod_Alexander - Tears_In_The_Rain` | 129 MB | AG×2 + **14_NylonGtr** (the only nylon stem in RawStems) + EG + drums + tambourine |
| `Perdidos_Na_Zona_Sul - Meu_Bem` | 243 MB | Brazilian. 3 AGs × 2 mics + piano + EG + drums + vox |
| `Egda_Carolyn - Saudade_Do_Teu_Beijo` | 195 MB | Brazilian. AG + organ + EG + shaker/cowbell + vox |
| `Enda_Reilly - An_Nasc_Nua` | 118 MB | AG (2 mics) + **Fiddles** + bass + vox |
| `Eddie_Garrido - Africa` | 258 MB | Guitar + **Flute** + brass + piano + djembe/percussion |
| `Leslie_Mendelson - The_Hardest_Part` | 97 MB | 2 AG + 2 vocals (duo) |
| `Nikola_Stajic_feat._Vlasis_Kostas - Nalim` | 100 MB | 2 AG (rhythm and lead) + room |
| `Chad_Hollister_Band - Eyes` | 456 MB | AG DI+mic + mandolin + sax + horns + congas + drums + vox |
| `musdb18hq/test/Raft_Monk - Tiring` | 241 MB | AG + strings + organ + EG + drums |
| `musdb18hq/test/Enda_Reilly - Cur_An_Long_Ag_Seol` | 139 MB | AG + Mandolin + strings + organ + vox |

The first 15 rows total about **2.8 GB**; all 17 total about **3.2 GB**. At about 15 MB/s from HF that is roughly 4 minutes.

### A2. MedleyDB Sample on Zenodo (small, openly licensed)
- **Record:** https://zenodo.org/records/1438309. **License: CC BY-SA 4.0**.
- **File:** `https://zenodo.org/api/records/1438309/files/MedleyDB_Sample.tar.gz/content`, **415.5 MB**. I listed its contents by streaming it.
- **`Phoenix_ScotchMorris`:** World/Folk instrumental. STEM_01 **acoustic guitar**, STEM_02 **flute**, STEM_03 **violin**, STEM_04 "Main System" (room). has_bleed = yes. 44.1 kHz, about 177 s. Very close to the target's instrumentation.
- **`LizNelson_Rainfall`:** Singer/Songwriter. STEM_01–03 **female singer**, STEM_04 **acoustic guitar**, STEM_05 clean electric guitar. has_bleed = no. About 285 s.
- Includes MIX, STEMS and RAW, plus metadata YAML. Metadata is also on HF `jzgdev/medleydb_sample`.
- The full MedleyDB (v1 and v2, zenodo 1649325 and 1715175) is **restricted** and needs a manual access request. Not feasible here.

### A3. GuitarSet on Zenodo (solo acoustic guitar; CC BY 4.0)
- **Record:** https://zenodo.org/records/3371780, v1.1.0, **CC BY 4.0**. Redistributable with attribution.
- **Files:**

| File | Size | URL |
|---|---|---|
| **`audio_mono-mic.zip`** | **656.9 MB** | `https://zenodo.org/api/records/3371780/files/audio_mono-mic.zip/content` |
| `audio_mono-pickup_mix.zip` | 683.1 MB | `.../audio_mono-pickup_mix.zip/content` |
| `audio_hex-pickup_debleeded.zip` | 3607 MB | |
| `audio_hex-pickup_original.zip` | 3211 MB | |
| `annotation.zip` | 39 MB | |

- **What is in the mic zip:** 360 WAVs such as `00_BN1-129-Eb_comp_mic.wav`. They are mono, 44.1 kHz, 16-bit, about 30 s each (up to 46 s), 0.97 GB unpacked. The reference mic is a **Neumann U-87**.
- **Players and styles:** 6 players. Styles are **BN (bossa nova)**, Funk, Jazz, Rock and SS (singer-songwriter), each as comp and solo.
- **The guitar is steel-string** (a magnetic hexaphonic pickup needs steel strings). It sounds different from the target's nylon guitar.
- **Use `mic`, not `pickup_mix`,** for a realistic acoustic sound.
- **HF mirrors** (faster download, parquet format):
  - `taohu/guitarset`: 5 parquet files, 1.86 GB, `audio_mic` + `audio_mix` columns, no license tag.
  - `jhartquist/guitarset`: 18 parquet files, 8.7 GB, all 4 captures, CC BY 4.0.

### A4. GAPS: classical (nylon) guitar solo recordings, on HF
- **Repo:** HF dataset **`xavriley/GAPS`** (v1.1, now includes audio). The Zenodo record 17152440 has annotations only (7 MB, CC BY-NC-SA 4.0, YouTube links).
- **Audio:** 404 WAVs under `audio/<id>.wav`, **48 kHz stereo 16-bit**, median 35 MB (about 3 min), 16.2 GB in total. The metadata CSV `gaps_metadata_with_splits.csv` gives splits of **train 270 / test 30** / blank 101.
- **The 30-piece test split is 1.22 GB and 99 minutes.** Example: `https://huggingface.co/datasets/xavriley/GAPS/resolve/main/audio/019_Vpswc.wav`.
- **This is the closest timbre match to the target (nylon-string guitar).** Recording conditions vary (home recordings, room reverb).
- **License:** the HF card says "mit", but the audio comes from YouTube performances by over 200 performers. Treat it as **research-only and do not redistribute**.

### A5. IDMT-SMT-Guitar V2 on Zenodo
- **Record:** https://zenodo.org/records/7544110. **License: CC BY-NC-ND 4.0** (no derivatives, so do not share mixtures).
- **File:** `https://zenodo.org/api/records/7544110/files/IDMT-SMT-GUITAR_V2.zip/content`, **1,329.9 MB** (1.97 GB unpacked). It is range-extractable.
- **`dataset4/acoustic_mic/{slow,fast}/{classical,country_folk,jazz,latin,metal,pop,reggae_ska,rock_blues}/audio/*.wav`:** about 128 rhythm-pattern recordings of an **acoustic guitar (mic)**. Mono, 44.1 kHz, 16-bit, **14–59 s**. About **231 MB compressed**; extract just this subfolder. `acoustic_pickup` is a parallel 236 MB set.
- **Electric guitar folders (negatives):**
  - `dataset1/*` clean Fender Strat and Ibanez Power Strat: notes and chords, about 60 MB.
  - `dataset4/Career SG` (201 MB) and `dataset4/Ibanez 2820` (206 MB).

### A6. Smaller or secondary acoustic-guitar sources

| Dataset | Where | Size | License | Notes |
|---|---|---|---|---|
| Five guitar dataset | zenodo 4988354, per-file WAV (e.g. `https://zenodo.org/api/records/4988354/files/Mountain_Larrivee_104_DI.wav/content`) | 90 files, about 8 MB each, 0.73 GB total | CC BY 4.0 | Larrivée OM-40 and Eastman E10OM **acoustic** guitars, plus Telecaster, Ibanez and Epiphone. Each is recorded 3 ways: DI, phone mic, laptop mic. The mics are low quality. |
| SonicSets Stems Evaluation Kit | HF `sonicsets-data/Stems-Evaluation-Kit`; `Demo5/AG Guitar.wav` + Guitar/Piano/Vocals/Drums/Bass/Mix | 6 demos × about 7 files × 4.3 MB (15 s, 48 kHz / 24-bit) | CC BY-NC 4.0 | A vendor sample. One short coherent song with an AG stem. |
| BabySlakh | zenodo 4603870, `babyslakh_16k.tar.gz` | 882.8 MB | CC BY 4.0 | 20 synthetic Slakh tracks at **16 kHz**. Metadata lists Acoustic Guitar (steel) in tracks 3, 5, 10, 11, 13, 14, 17 and (nylon) in track 18. Synthetic, 16 kHz. Not recommended. |
| NSynth test | `http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz` (reachable), or HF `jg583/NSynth` `data/test/test.parquet` (406 MB) | 349.5 MB | CC BY 4.0 | 4-s single notes at 16 kHz; includes `guitar_acoustic`. Low value for separation. |
| AG-PT-set | zenodo 10159492, `aGPTset_z.zip` | 6.75 GB (a nested `audio.zip`) | CC BY 4.0 | Monophonic playing-technique notes on steel-string guitar. Not musical. Skip. |
| MSR Challenge 2025 test set | HF `MSRChallenge/2025TestSet` | 2.7 GB | CC BY-NC 4.0 | 10-s 48 kHz clips; 189 "Guitars" (AG and EG not distinguished). **Targets are unprocessed sources, not mixture-consistent.** |
| MSRBench | HF `yongyizang/MSRBench`, `Guitars.zip` | 3.49 GB | CC BY-NC 4.0 | Same caveat as the MSR test set. |

### A7. Checked and not usable

| Dataset | Why not |
|---|---|
| **MoisesDB** (has an `acoustic guitar` sub-stem) | CC BY-NC-SA 4.0, but download is **form-gated** at https://music.ai/research/ (developer.moises.ai) and was not reachable. The HF repo `wearemusicai/moisesdb` holds only code/CSV. `arjunbahuguna/zipped-moises-musdb` `moisesdb.zip` (56 GB) holds only degraded mixtures `mixture_DT0..12` for 240 tracks, with **no stems**; I listed it via range requests. `choihy/moisesdb_before_pd` holds codec tokens, not audio. |
| MVSep guitar benchmark (HF `MusicDemixingBenchmarks/guitar`, and `https://mvsep.com/storage/public/guitar_validation.zip`) | **Mixtures only** (29 × 120 s); ground truth is hidden. The same applies to `/strings`, `/wind` and `/piano`. |
| Slakh2100 | Full set is 104 GB (zenodo 4599666). HF `DreamyWanderer/Slakh2100-FLAC-Redux-Reduced` holds mixes and MIDI only. |
| `f1oth3r0/guitar-stems` | One 11.9 GB tar with no README or license. |
| `Isaac105/melodix-stems` | m4a files with no license. |
| `danjacobellis/musdb18HQ`, `roro128/musdb18-hq-flac` | Parquet mirrors of MUSDB18-HQ. |
| DadaGP | Symbolic only. |
| EGDB (original) | On Google Drive, which is blocked. |

---

## B. Isolated non-target stems (hard negatives)

| Need | Recommended source | Details |
|---|---|---|
| **Violin, clarinet, flute** (the target's own instruments) | **URMP**, per-file from the HF mirror `Eredis02/URMP` | `AuSep_*` separated tracks are mono **48 kHz** 16-bit WAV. Violin/clarinet/flute: 62 files, **0.97 GB**; all AuSep: 149 files, 2.38 GB; full repo with videos: 12.5 GB. Good pieces: `17_Nocturne_vn_fl_cl` (3 × 13.8 MB), `19_Pavane_cl_vn_vc`, `37_Rondeau_fl_vn_va_cl`, `29_Fugue_fl_fl_ob_cl`, `28_Fugue_fl_ob_cl_bn`, `14_Waltz_fl_fl_cl`, `12_Spring_vn_vn_vc`. URL example: `https://huggingface.co/datasets/Eredis02/URMP/resolve/main/17_Nocturne_vn_fl_cl/AuSep_3_cl_17_Nocturne.wav`. **License:** the original (https://labsites.rochester.edu/air/projects/URMP.html) is research use via a Google form with no explicit CC license; the mirror states no license. Internal use only. |
| Violin, clarinet, flute, percussion from real mixes | RawStems (repo above) | Violins: 75 files, 1.28 GB (e.g. `dev/Swing_Bazar - Fleche_DOr/Orch/STR/08_Violin.flac`, `dev/Catfolkin - Odi_A_La_Barretina/Orch/STR/29_Violin.flac`, Barnstar Violin-C12). Clarinets: `dev/Maurizio_Pagnutti_Sextet - Bess/Orch/WW/14_Clarinet.flac` (jazz, 15 MB), `dev/Trybes - Running_Out/Orch/WW/20_Clarinets.flac`, `dev/Ethan_Winer - Cello_Concerto_in_A_Minor/Orch/WW/12_Clarinets.flac`, bass clarinets. Flutes: 21 files. Sax: 72. Accordion: 16. Hand percussion (shaker, conga, cajon, tambourine, djembe): 192 files, 2.2 GB. |
| Ukulele, mandolin, banjo, bouzouki, dobro, autoharp, sitar | RawStems `Gtr/AG` non-guitar files | 51 files, 0.73 GB. Examples: `Anna_Blanton - Rachel/Gtr/AG/04_UkeleleMic.flac`, `James_May - Eliza_Jane/Gtr/AG/10_Mandolin1.flac`, `Rovers_Ahead - Its_In_These_Times/Gtr/AG/12_BanjoMic1.flac` and `22_Autoharp.flac`, `Wolfs_Head.../06_Bazouki.flac`, `Uncle_Dad - Who_I_Am/Gtr/AG/DOBRO-*.flac`. No harp stem was found in RawStems ("harp" matches are synth arps). |
| Vocals (male/female) | RawStems `Voc/LV` (958 files; gender inferred from artist); MedleyDB Sample `LizNelson_Rainfall` (female); MUSDB18-7 or MUSDB18-HQ `vocals` | |
| Drums, cymbal-heavy | RawStems `Rhy/DK`: overhead, cymbal, ride, hihat (1049 files, 23 GB; pick a few); MUSDB `drums` | |
| Bass | RawStems `Bass` (736); MUSDB `bass` | |
| Piano | RawStems `Kbs/PN` (240 piano files, 4.85 GB; e.g. `Maurizio_Pagnutti_Sextet - Bess/Kbs/PN/12_PianoMics1.flac`) | MAESTRO is overkill for this budget. |
| Clean electric guitar | **HF `Julian-br/GuitarJam`**: **CC0**, 580 × ~15 s Strat DI, 44.1 kHz, 0.80 GB, per-file; IDMT `dataset1` clean Strat; RawStems `Gtr/EG` | |
| Distorted electric guitar | RawStems `Gtr/EG` (2336 files, 22 GB; pick ~30); EGDB BIAS FX2 (zenodo 12674910, `EGDB_BIAS_FX2_random.zip`, 1,044.8 MB, CC BY 4.0, 4-s wet/dry clips, range-extractable); Guitar-TECHS (zenodo 14963133, CC BY 4.0, per-part zips 109 MB–1.15 GB); IDMT `dataset4/Career SG` | |
| Generic 4-stem songs | **MUSDB18-7-STEMS.zip** via the GitHub release (see below); **MUSDB18-HQ** via partial remote-zip extraction (see below) | |

**MUSDB18-7-STEMS.zip**
- This is what `musdb.DB(download=True)` fetches: `https://github.com/sigsep/sigsep-mus-db/releases/download/v0.4.0/MUSDB18-7-STEMS.zip`, **147.2 MB**. Verified reachable (206).
- It holds 144 × 7-s `.stem.mp4` files (AAC, 5 streams: mix, drums, bass, other, vocals). Decoding needs ffmpeg (imageio-ffmpeg; `stempeg` also wants ffprobe, so demux with `ffmpeg -map 0:N` instead).
- MUSDB license: non-commercial research (zenodo "other-nc").

**MUSDB18-HQ**
- Zenodo 3338373, `musdb18hq.zip` is **22,656.7 MB**, but you can remote-extract single tracks. Each track is about 95–285 MB compressed and has 6 WAVs (44.1 kHz).
- URL: `https://zenodo.org/api/records/3338373/files/musdb18hq.zip/content`.
- Songs whose `other` stem is acoustic-heavy: `test/Enda Reilly - Cur An Long Ag Seol`, `test/Cristina Vane - So Easy`, `test/Raft Monk - Tiring`. The raw per-instrument versions of these test songs are in RawStems under `musdb18hq/test/`.

**Medley-solos-DB** (violin, clarinet, flute, female singer, piano, distorted EG, trumpet, sax)
- CC BY 4.0, but **not recommended**:
  - v1.0 (zenodo 1344103) `Medley-solos-DB.zip` is 6.66 GB and **corrupt/truncated**: it holds 17,497 of 21,571 files and many local-header offsets are broken.
  - v1.2 (zenodo 3464194) is a 7.92 GB `tar.gz`. It can only be stream-extracted (about 1 h at zenodo speeds).
  - Clips are only 2.97 s (mono, 44.1 kHz float32).
  - The metadata CSV is 1.3 MB: `https://zenodo.org/api/records/3464194/files/Medley-solos-DB_metadata.csv/content`.

---

## Recommended concrete download plan (about 8–9 GB, within 20 GB)

Priority order. Times assume HF at about 15 MB/s and Zenodo at about 1.5 MB/s.

1. **Target previews:** already downloaded (`scratchpad/target_preview*.{m4a,mp3}`). Use OTOTOY's 45 s clip as the main in-domain qualitative test; its first 30 s equal the iTunes clip.
2. **RawStems coherent multitracks with AG ground truth** (HF, about 3.2 GB, 4 min). Get the 17 songs in the table in A1 with `snapshot_download(allow_patterns=[f"dev/{s}/*" for s in songs] + [...musdb18hq/test/...])`. Build validation mixtures of 10–30 s by summing all stems. GT = the sum of the real-guitar AG files (filter out mandolin, banjo and similar). This gives realistic, coherent validation, and some tracks are also usable for a small fine-tune (non-commercial, internal).
3. **MedleyDB Sample** (Zenodo, 415.5 MB, about 5 min, CC BY-SA): `Phoenix_ScotchMorris` (AG + flute + violin) and `LizNelson_Rainfall`.
4. **GAPS test split** (HF `xavriley/GAPS`, 30 WAVs, 1.22 GB, about 1.5 min). Nylon-guitar GT; mix with violin/clarinet/percussion negatives.
5. **GuitarSet `audio_mono-mic.zip`** (Zenodo, 656.9 MB, about 7 min, CC BY 4.0). Steel-string GT, including bossa nova.
6. **URMP violin/clarinet/flute AuSep** (HF `Eredis02/URMP`, 0.25–0.97 GB). Plus RawStems negatives: the clarinet list above, about 20 violin files, about 20 percussion files, mandolin/banjo/ukulele/bouzouki, about 20 lead vocals, about 20 EG, about 10 piano, about 10 overheads. Roughly 1–1.5 GB.
7. **IDMT-SMT-Guitar `dataset4/acoustic_mic/`**: about 231 MB, remote-extracted from the 1.33 GB zip with `scratchpad/dsr/rzip.py`. CC BY-NC-ND. Internal only.
8. **Optional:**
   - GuitarJam, CC0 clean EG: download a subset.
   - MUSDB18-7-STEMS.zip (147 MB).
   - Individual MUSDB18-HQ tracks by remote-zip.
   - SonicSets Demo5 (about 30 MB).

**Licensing summary for the validation set:**

| Status | Datasets |
|---|---|
| Redistributable with attribution | GuitarSet (CC BY), MedleyDB Sample (CC BY-SA), Five-guitar (CC BY), BabySlakh (CC BY), NSynth (CC BY), GuitarJam (CC0), EGDB-BIAS (CC BY), Guitar-TECHS (CC BY) |
| Non-commercial or research-only; do not redistribute mixtures | Mixing Secrets / RawStems (educational/NC), MUSDB18/-HQ (NC), MoisesDB (BY-NC-SA, not accessible anyway), IDMT (BY-NC-ND), URMP (research, no explicit license), GAPS audio (YouTube-sourced), MSR sets (BY-NC), SonicSets (BY-NC) |
| The target song | Commercial (Happiness Records). Previews may be used for private evaluation only. |

## Helper artifacts in the scratchpad
- `scratchpad/dsr/rzip.py`: remote zip over HTTP Range (tested on Zenodo, HF and GitHub release assets).
- `scratchpad/dsr/hfinfo.py`: HF dataset tree and size summarizer. `scratchpad/dsr/zen.py`: Zenodo record file lister.
- `scratchpad/dsr/tree_*.json`: file trees of the inspected HF repos (RawStems, GAPS, URMP, and others).
- `scratchpad/dsr/ms_ag_songs.txt`: 132 RawStems songs with AG stems and their other categories.
- `scratchpad/dsr/babyslakh_meta/`: BabySlakh `metadata.yaml` files. `scratchpad/dsr/gaps_meta.csv`, `msdb_meta.csv`, `msr_meta.jsonl`, `medleydb_sample_list.txt`.
- `scratchpad/pylib/`: numpy + imageio-ffmpeg (ffmpeg 7.0.2 static).
