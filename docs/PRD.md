# Acoustic Guitar Source Separation — Product Requirements Document

## 0. Claude Codeへの最重要指示

このプロジェクトの目的は「コードを書くこと」ではない。

**実際の楽曲からアコースティックギターだけを可能な限り高精度に抽出した、最終的な音源ファイルを生成すること**が目的である。

実装が動いた時点で終了してはいけない。

以下を自律的に繰り返すこと。

**Research → Implement → Run → Generate Audio → Evaluate → Compare → Diagnose → Improve → Re-run**

最終成果物として、少なくとも実際に再生可能な

`acoustic_guitar.wav`

を生成すること。

モデル候補、前処理、後処理、ensemble、fine-tuning、データセット、損失関数、推論パラメータ等を固定観念なく比較し、精度向上が可能である限り改善を続ける。

「とりあえず動いた」「一般的には高性能」「READMEに書いてある」という理由で終了してはいけない。

---

# 1. Product Goal

市販楽曲などの完成済みステレオ音源から、

**アコースティックギター成分だけ**

を可能な限り高精度に抽出するローカルツールを構築する。

Primary Goal:

> 元楽曲に含まれるアコースティックギターを最大限保持しつつ、ボーカル、ドラム、ベース、ピアノ、エレキギター、シンセ、ストリングスその他の音を極力除去した音声を生成する。

速度、モデルサイズ、処理時間、実装の単純さは二次的要素である。

**精度を最優先する。**

リアルタイム処理は不要。

1曲の処理に数十分かかっても、明確な品質改善が得られるなら許容する。

---

# 2. Initial Target / Definition of Victory

最初のターゲット楽曲:

https://www.youtube.com/watch?v=saKb8Z7ocVY&list=RDsaKb8Z7ocVY&start_radio=1

この楽曲からアコースティックギターだけを高品質に抽出する。

MVPの勝利条件:

> この楽曲を入力し、アコースティックギターが明瞭に残り、それ以外の楽器・ボーカルの混入が大幅に抑制された `acoustic_guitar.wav` を生成できる。

ただし、この1曲だけへの過学習を最終成果とはしない。

アーキテクチャとしては他の一般的な楽曲にも適用可能な状態を維持する。

---

# 3. Future Goal

将来的には抽出したアコースティックギター音声から

- Pitch detection
- Note onset detection
- Chord recognition
- Polyphonic transcription
- Guitar fingering estimation
- TAB譜生成

を行う予定。

ただし、**本プロジェクトではTAB生成を実装しない。**

今回の責務は、

**TAB生成に利用できるほどクリーンなアコースティックギターStemを作ること**

までとする。

そのため単純な「聴いてそれっぽい音」ではなく、

- ピッキングアタック
- サステイン
- コード構成音
- 倍音
- ゴーストノート
- ストローク
- フレットノイズ

等を可能な範囲で保持すること。

---

# 4. Input

以下をサポートする。

## 4.1 Local audio

最低限:

- WAV
- MP3
- M4A
- FLAC

内部処理は原則としてlossless PCMへ変換する。

推奨内部フォーマット:

- WAV
- float32
- stereo
- original sample rate または 44.1/48 kHz

不必要な再エンコードを繰り返さないこと。

---

## 4.2 URL

YouTube URLを入力できるようにしてよい。

例:

```bash
python separate.py \
  "https://www.youtube.com/watch?v=saKb8Z7ocVY"
```

可能であれば音質劣化が最小になる音声ソースを取得する。

URL取得が利用規約、DRM、環境制約等により利用できない場合は、ユーザーが合法的に取得したローカル音源を入力できること。

URL入力機能はSource Separation本体と疎結合にする。

---

# 5. Output

必須:

```text
outputs/
  <track-id>/
    acoustic_guitar.wav
    non_acoustic_guitar.wav
```

`acoustic_guitar.wav`

には可能な限りアコースティックギターのみを含める。

`non_acoustic_guitar.wav`

には残りを含める。

可能であれば、

```text
mixture ≈ acoustic_guitar + non_acoustic_guitar
```

となるmix consistencyを維持する。

さらに実験用として候補Stemを保存してよい。

例:

```text
candidates/
  model_a.wav
  model_b.wav
  model_c.wav
  ensemble_01.wav
  ensemble_02.wav
```

最終採用されたファイルだけでなく、比較可能な候補を残すこと。

---

# 6. Priority

優先順位は以下。

1. Acoustic guitar isolation accuracy
2. 他楽器・ボーカルのleakage低減
3. Acoustic guitar本体の欠損低減
4. 位相・音質・transient保持
5. 再現性
6. ローカル実行可能性
7. 処理速度
8. UI/UX

速度改善のために分離品質を犠牲にしない。

---

# 7. Supported Environments

最低1環境で完全に動けばMVP達成とするが、可能な限り以下をサポートする。

### macOS

Apple Siliconを想定。

可能なら:

- PyTorch MPS
- CoreML
- ONNX Runtime

等を利用。

ただしMPS対応のために精度を落とさない。

### Windows

NVIDIA GPUがある場合はCUDAを第一候補とする。

### Linux / Codespaces

GPUが存在すればCUDA。

CPUのみの場合でも低速推論可能であることが望ましい。

---

# 8. Fundamental Engineering Policy

## 精度 > 計算量

モデルサイズやVRAM使用量が増えても、品質改善が明確なら採用する。

## Existing models first

最初から独自ニューラルネットワークをゼロから作らない。

まず最新のMusic Source Separationモデル・公開weights・学習frameworkを調査する。

その後、

1. Existing pretrained model
2. Cascaded models
3. Ensemble
4. Test-time augmentation
5. Post-processing
6. Fine-tuning
7. Domain-specific training
8. New architecture

の順に検討する。

新規モデル開発は既存モデルを超える根拠がある場合のみ行う。

---

# 9. Phase 0 — Research

実装前に最新状況を調査する。

調査結果を

`docs/research.md`

に残す。

最低限調査するもの:

- BS-RoFormer
- Mel-Band RoFormer
- HTDemucs / Demucs
- SCNet
- BandIt / BandIt v2
- Apollo
- BSMamba / BSMamba2
- MDX23C
- Music Source Separation Training / MSST
- Ultimate Vocal Remover ecosystem
- MVSep guitar models
- Acoustic-guitar-specific models
- guitar / acoustic guitar datasets
- recent Music Source Separation papers
- recent SDX / demixing competition approaches

2024年以前の記事だけで技術選定しない。

**現在利用可能なweights、GitHub、Hugging Face、papers、leaderboardsを確認すること。**

モデルのライセンス、training data provenanceも記録する。

---

# 10. Baseline Experiments

最初に最低3系統以上を実際にターゲット曲へ適用する。

READMEの数値比較だけでモデルを決めてはいけない。

各モデルについて実際に

```text
target song
↓
separator
↓
candidate wav
```

を生成する。

結果を

`experiments/`

以下に保存する。

例:

```text
experiments/
  exp001_bs_roformer/
  exp002_melband_roformer/
  exp003_scnet/
  exp004_demucs/
```

各experimentに:

```text
config.yaml
metrics.json
notes.md
output.wav
```

を保存する。

---

# 11. Separation Strategies

最低限、以下を比較する。

## Strategy A — Direct Acoustic Guitar Separation

```text
Full Mix
   ↓
Acoustic Guitar Separator
   ↓
Acoustic Guitar
```

---

## Strategy B — Two-stage Separation

```text
Full Mix
   ↓
General Guitar Separator
   ↓
All Guitar
   ↓
Acoustic-vs-Rest Separator
   ↓
Acoustic Guitar
```

これは重要な候補とする。

アコギと音響的に大きく異なる

- Vocal
- Kick
- Snare
- Bass

などを最初に除去することで、第2モデルの問題を単純化できる可能性がある。

---

## Strategy C — Instrument-removal Cascade

例:

```text
Mix
 ↓
Vocal removal
 ↓
Drum/Bass/Piano removal
 ↓
Guitar separator
 ↓
Acoustic/Electric separator
```

モデル誤差が蓄積する可能性もあるためA/B方式と必ず比較する。

---

## Strategy D — Ensemble

複数モデルの結果を組み合わせる。

単純平均だけでなく、

- frequency-dependent weighting
- magnitude mask averaging
- confidence-weighted ensemble
- median ensemble
- stem-specific weighting
- phase-aware reconstruction

を検討する。

単一モデルより高品質なら積極的に採用する。

---

# 12. Acoustic vs Electric Guitar

このプロジェクトでは「guitar stem」では不十分。

以下を区別する。

Target:

**Acoustic Guitar**

Non-target:

- Distorted electric guitar
- Clean electric guitar
- Bass guitar
- Ukulele
- Mandolin
- Banjo
- Harp
- Piano
- Strings
- Other plucked instruments

特に

**clean electric guitar vs acoustic guitar**

はhard negativeとして重点評価する。

---

# 13. Evaluation

感覚だけで「良くなった」と判断しない。

Objective evaluationとTarget-song evaluationを両方行う。

---

# 14. Ground Truth Evaluation Dataset

指定曲にはacoustic guitarの正解stemが存在しない可能性が高い。

したがって評価用に、

**正解acoustic guitar stemを持つdataset**

を別途用意する。

利用可能なpublic datasetを調査し、ライセンスを確認する。

必要ならsynthetic mixture datasetを構築する。

例:

```text
isolated acoustic guitar
+
vocal
+
drums
+
bass
+
piano
+
electric guitar
+
other instruments
=
synthetic mixture
```

この場合、acoustic guitarのground truthが完全に分かる。

---

# 15. Synthetic Dataset

必要なら独自validation setを生成する。

以下をランダム化する。

- instrument volume
- EQ
- compression
- reverb
- room impulse
- stereo position
- delay
- saturation
- sample rate
- mastering compression

特にhard casesを増やす。

例:

- acoustic + piano
- acoustic + clean electric
- acoustic + ukulele
- acoustic + cymbal-heavy drums
- acoustic + female vocal
- acoustic + strings
- acoustic guitar buried -15 dB below mix

---

# 16. Metrics

最低限以下を計測する。

- SDR
- SI-SDR
- SDR improvement
- SIR
- SAR
- leakage energy
- target retention
- multi-resolution STFT error

可能ならinstrument recognition modelなどを利用し、

```text
Acoustic Guitar Probability
Vocal Leakage
Drum Leakage
Bass Leakage
Piano Leakage
Electric Guitar Leakage
```

についてもproxy scoreを作る。

ただしclassifier score単独を品質判定に使用しない。

---

# 17. Leakage vs Preservation

重要なトレードオフ:

### Too aggressive

```text
他楽器: 少ない
ギター: 欠損
```

### Too conservative

```text
ギター: 完全
他楽器: 多い
```

本プロジェクトでは、

**後段でTAB transcriptionを行えること**

を考慮し、ギター音そのものを破壊する過度なnoise suppressionを避ける。

ただし明確なボーカル・ドラム等の混入も可能な限り減らす。

---

# 18. Target Song Evaluation

Target:

```text
https://www.youtube.com/watch?v=saKb8Z7ocVY
```

各主要experimentについてtarget曲のStemを生成する。

以下を比較できるようにする。

```text
outputs/target/
  original.wav

  candidates/
    001.wav
    002.wav
    003.wav
    ...

  best/
    acoustic_guitar.wav
    non_acoustic_guitar.wav

  report.html
```

reportには最低限:

- model
- checkpoint
- inference parameters
- objective proxy metrics
- waveform
- spectrogram
- runtime
- experiment ID

を記録する。

---

# 19. Spectrogram Diagnostics

以下を生成する。

- original spectrogram
- extracted acoustic spectrogram
- residual spectrogram

可能なら

```text
original
target
residual
```

を同期表示できるHTML reportを生成する。

問題のある時間帯を特定できるようにする。

---

# 20. Fine-Tuning

既存モデルだけでは不十分な場合、fine-tuningを行う。

第一候補:

**最も性能の良かったpretrained architectureをacoustic guitar専用にfine-tuneする。**

ゼロからtrainingするよりpretrained weightsを優先する。

Training target:

```text
input = full music mixture
target = acoustic guitar
```

または2-stageなら:

```text
input = guitar-containing stem
target = acoustic guitar
```

を比較する。

---

# 21. Training Data

品質向上に必要であればtraining dataを構築する。

ただし、

- 使用ライセンス
- 出典
- dataset terms
- redistribution条件

を記録する。

`datasets/manifest.csv`

等に管理する。

例:

```csv
id,source,license,instrument,type,path
```

---

# 22. Data Augmentation

Acoustic guitar stemに対して:

- gain
- EQ
- compression
- reverb
- stereo width
- room simulation
- pitch shift
- time stretch

を必要に応じて実施。

ただしacoustic guitarとして不自然になるaugmentationは避ける。

Mix側では

- accompaniment gain
- stem dropout
- random instrument combinations
- hard negative oversampling

を利用する。

---

# 23. Hard Example Mining

初回モデルの失敗例を自動収集する。

例:

```text
piano leaked
electric guitar leaked
vocal leaked
acoustic guitar removed
```

失敗タイプごとにdatasetを作る。

次回trainingでこれらを重点sampleする。

Iteration:

```text
train
↓
evaluate
↓
find worst examples
↓
add/oversample hard examples
↓
train
```

を繰り返す。

---

# 24. Loss Function Research

最低限検討する。

- waveform L1/L2
- STFT loss
- multi-resolution STFT loss
- mask loss
- SI-SDR loss
- multi-domain loss

必要なら

target preservation

と

interference rejection

に異なるweightを与える。

例:

```text
Loss =
    target reconstruction
  + interference penalty
  + spectral loss
```

実験により決定する。

---

# 25. Test-Time Optimization

Trainingせず改善可能なものも調査する。

- segment size
- overlap
- shift / TTA
- window
- chunk size
- stereo processing
- model overlap
- phase reconstruction
- Wiener filtering
- mixture consistency
- ensemble weights

同一checkpointでも推論パラメータによる品質差を探索する。

---

# 26. Automated Experiment Search

可能ならパラメータ探索を自動化する。

例:

```text
models × chunk_size × overlap × ensemble_weight
```

すべてのexperimentをDBまたはCSVへ保存する。

例:

`experiments/results.csv`

```text
experiment
model
checkpoint
overlap
chunk
sdr
sir
sar
si_sdr
target_retention
leakage
runtime
```

---

# 27. Champion / Challenger System

常に現在のbest modelを

**Champion**

として保存する。

新しいexperimentを

**Challenger**

として評価する。

Challengerがvalidation metricでChampionを上回った場合のみ更新する。

```text
artifacts/champion/
```

を常に最新bestとして維持する。

過去Championは削除しない。

---

# 28. Prevent False Progress

以下は禁止。

### 禁止1

Lossが下がっただけで品質向上と判断する。

### 禁止2

training datasetだけで評価する。

### 禁止3

1曲だけ良かったモデルをSOTA扱いする。

### 禁止4

classifier probabilityだけで分離精度を判断する。

### 禁止5

新モデルを作ったという理由だけで既存モデルより優れていると判断する。

### 禁止6

「これ以上改善できそうにない」と根拠なく終了する。

---

# 29. Iteration Loop

Claude Codeは以下を繰り返す。

```text
1. Generate candidate
2. Run validation benchmark
3. Generate target-song result
4. Compute metrics
5. Compare against Champion
6. Analyze failure cases
7. Form hypothesis
8. Change one or a controlled set of variables
9. Repeat experiment
```

各loopで

`experiments/<id>/notes.md`

に

```text
Hypothesis
Change
Result
Conclusion
Next experiment
```

を残す。

---

# 30. Stopping Conditions

「完璧」という抽象的理由では終了しない。

以下の条件を利用する。

### Required

- 実際のtarget曲から`acoustic_guitar.wav`が生成されている
- reproducible pipelineになっている
- automated evaluationが存在する
- ground-truth validation datasetで評価済み
- baselineモデルより改善している、またはbaselineが最善であることを実験で示している
- target曲について複数方式を比較している

### Convergence

直近の十分な数のexperimentで、

- architecture
- cascade
- ensemble
- inference tuning
- fine-tuning

の主要候補を試しても意味のある改善が得られない場合、convergedと判断できる。

ただし改善余地が明確に残っている場合は終了しない。

---

# 31. Important Reality Check

Target曲についてground truth stemが存在しない場合、

**「100%完全にアコギだけ」と数学的に証明することはできない。**

したがって最終報告では

「perfect」

と無根拠に表現しない。

代わりに、

- validation metrics
- baseline comparison
- leakage analysis
- spectrogram analysis
- target retention
- candidate comparison

に基づいてbest resultを決める。

---

# 32. Architecture

最初はCLIでよい。

例:

```bash
python -m acoustic_separator \
    --input song.wav \
    --output outputs/song/
```

URL:

```bash
python -m acoustic_separator \
    --input "https://youtube.com/watch?v=..." \
    --output outputs/song/
```

高品質モード:

```bash
python -m acoustic_separator \
    --input song.wav \
    --quality max
```

`--quality max`では処理時間より品質を優先する。

---

# 33. Repository Structure

推奨:

```text
.
├── README.md
├── PRD.md
├── pyproject.toml
├── configs/
│
├── src/
│   └── acoustic_separator/
│       ├── cli.py
│       ├── audio.py
│       ├── inference.py
│       ├── ensemble.py
│       ├── evaluation.py
│       └── models/
│
├── scripts/
│   ├── benchmark.py
│   ├── download_models.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_target.py
│
├── datasets/
│   └── manifest.csv
│
├── experiments/
│
├── outputs/
│   └── target/
│
├── artifacts/
│   └── champion/
│
├── reports/
│
└── docs/
    ├── research.md
    ├── architecture.md
    └── experiments.md
```

モデルweights、datasets、大容量WAVをGitへ直接commitしない。

---

# 34. Reproducibility

全experimentで保存するもの:

- git commit
- dependency versions
- model name
- checkpoint hash
- config
- random seed
- hardware
- command
- metrics

同じcommandから同等のoutputを再生成できること。

---

# 35. Model Cache

weightsを毎回downloadしない。

例:

```text
~/.cache/acoustic-separator/
```

またはproject内の

```text
models/
```

にcacheする。

checksumも可能なら保存する。

---

# 36. Final Deliverables

プロジェクト完了時、最低限以下が存在すること。

## Actual audio

```text
outputs/target/best/acoustic_guitar.wav
outputs/target/best/non_acoustic_guitar.wav
```

これが最重要成果物。

## Code

再実行可能なsource separation pipeline。

## Model

必要ならfine-tuned checkpoint。

## Evaluation

```text
reports/final_report.md
```

## Experiments

何を試し、何が効き、何が効かなかったかの履歴。

## Usage

READMEに1コマンドで再現できる手順を書く。

例:

```bash
uv run python -m acoustic_separator \
  --input "<youtube-or-file>" \
  --quality max
```

---

# 37. Final Report

最終reportには最低限以下を書く。

```text
Best architecture:
Best checkpoint:
Pipeline:
Dataset:
Validation SDR:
Validation SI-SDR:
Leakage:
Runtime:
Hardware:
Number of experiments:
```

さらに、

### What worked

### What failed

### Remaining artifacts

### Known failure modes

### Why the final model was selected

を記載する。

---

# 38. Phase Plan

## Phase 1

環境構築・target audio準備。

## Phase 2

既存SOTA separatorを最低3系統benchmark。

## Phase 3

Direct / two-stage / cascadeを比較。

## Phase 4

Inference parameter optimization。

## Phase 5

Ensemble。

## Phase 6

Ground-truth evaluation dataset構築。

## Phase 7

必要ならfine-tuning。

## Phase 8

Hard-example mining。

## Phase 9

再training / ensemble optimization。

## Phase 10

Target曲のbest stem生成。

## Phase 11

Final validationとreport。

---

# 39. First Tasks for Claude Code

まず以下を実行する。

1. Repositoryを初期化する。

2. `docs/research.md` を作る。

3. 2026年時点のacoustic guitar / guitar Music Source Separation手法を調査する。

4. 利用可能なpretrained weightsを列挙する。

5. Target曲をWAV入力として準備する方法を作る。

6. 少なくとも3つの有力モデルでtarget曲を処理する。

7. 各モデルのcandidate WAVを保存する。

8. Direct acoustic separationとtwo-stage separationを比較する。

9. 自動evaluation frameworkを構築する。

10. Ground-truthを持つvalidation setを作る。

11. Championを決定する。

12. Championを改善するexperimentを開始する。

**この時点で終了してはいけない。**

改善loopへ進む。

---

# 40. Core Principle

このプロジェクトでは、

**software completion ≠ project completion**

である。

CLIが完成しても、テストが通っても、モデルが動いても終了ではない。

Project completionとは、

> 実際のtarget曲から、現時点で合理的に到達可能な最高品質のアコースティックギターStemを生成し、それが他の候補より優れていることを実験結果によって説明できる状態

を指す。

最優先成果物はコードではない。

**音源である。**
