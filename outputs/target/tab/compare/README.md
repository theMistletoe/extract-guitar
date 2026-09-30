# タブの音と元のギターの比較（scripts/compare_tab_audio.py）

説明と結果は [../README.md](../README.md) の「タブ譜の音と元のギターの比較チェック」にあります。

| ファイル | 内容 |
|---|---|
| `summary.json` | 再現度・ズレ・和音の一致・誤りの見込み（全体） |
| `bars.csv` | 小節ごと：再現 F1、同じ採譜器での一致、8 分ごとの響きの一致（クロマ）、時間のずれ |
| `notes.csv` | タブの音ごと：各モデルの確率、Basic Pitch、倍音の立ち上がり（元／タブ）、漏れ、誤りの確率 `p_wrong` |
| `missing.csv` | 元の音にあってタブに無い音の候補と、本物である確率 `p_real` |
| `evidence/bar_XXX.png` | 疑いの強い小節の画像（上から元のギター・タブの音・残りの楽器。赤枠＝誤りの疑い、水色の丸＝抜けの疑い） |
| `agent_verdicts.csv` | AI レビュー（音響分析担当・反証担当）の判定と理由。修正前のタブに対するもの |
| `basic_pitch_stem.json` | Basic Pitch の採譜結果（キャッシュ） |

`state.npz` と `posteriors_render.npz`（計算のキャッシュ、リポジトリ外）がある状態で、次のように 1 か所の証拠を表示できます。

    python scripts/compare_tab_audio.py probe outputs/target/tab/compare --t 26.50 --pitch 51 --image x.png
