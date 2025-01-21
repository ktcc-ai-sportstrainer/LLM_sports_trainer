# Baseball Swing AI Coach (2-Video Version)

本リポジトリは、野球のスイング動画をAIで解析・コーチングするシステムです。  
**ユーザーが実際に打ったスイング動画**と、**理想とするスイング動画**の2本を入力し、3D姿勢推定や分析を行い、それらを比較した結果をもとに目標設定・練習計画を生成します。

---

## 機能概要

1. **ペルソナ情報と指導方針の入力**  
   - 選手（ユーザー）の基本情報（年齢・ポジション・身長体重・野球経験など）  
   - 指導者が重点的に鍛えたいポイントや目標

2. **2本のスイング動画の入力**  
   - ユーザーが実際に打ったスイング動画 (`user_video`)  
   - 理想とするスイング動画 (`ideal_video`)  
     - 例: プロ選手のスイングや自分のベストスイングの動画

3. **InteractiveAgent**  
   - 選手のペルソナや指導方針を読み、追加で聞いておきたい質問を2～3個生成  
   - （簡易的な対話形式）

4. **ModelingAgent**  
   - 2本の動画を MotionAGFormer で3D姿勢推定 → `JsonAnalist.py` で分析  
   - スイングのフェーズ・重心移動・バットスピードなどを抽出  
   - さらにユーザー動画と理想動画を比較し、差分と改善点をまとめる

5. **GoalSettingAgent**  
   - InteractiveAgentから得た新情報やModelingAgentの出力を踏まえ、主目標・サブ目標・達成指標を設定

6. **PlanAgent**  
   - 設定された目標・課題から、段階的な練習タスクを具体化

7. **SearchAgent**  
   - PlanAgentが作成したタスクに対して、インターネット検索等を活用し具体的なドリルや練習メニューを提示

8. **SummarizeAgent**  
   - 全体の出力を統合し、コーチングレポートとして最終サマリーを生成

---

## 前提条件・インストール

1. **Python 3.9+**  
2. **CUDA対応GPU**（推奨、MotionAGFormerによる推論のため）  
3. `requirements.txt` に記載されたパッケージのインストール  
   ```bash
   pip install -r requirements.txt
   ```
4. **MotionAGFormerの事前学習モデル**をダウンロードし、`checkpoint/motionagformer-b-h36m.pth.tr` のように保存  
5. **OpenAI APIキー** 等の設定（ `.env` や環境変数で指定）

---

## 使い方

### 1. CLIモード

```bash
python main.py \
    --user_video user_swing.mp4 \
    --ideal_video ideal_swing.mp4 \
    [その他オプション...]
```

主なオプション：

| オプション         | 説明                                   | 例                     |
|--------------------|----------------------------------------|------------------------|
| `--user_video`     | ユーザーが打ったスイング動画のパス     | `--user_video mybat.mp4` |
| `--ideal_video`    | 理想のスイング動画のパス               | `--ideal_video pro.mp4`  |
| `--age`            | プレイヤーの年齢                       | `--age 18`             |
| `--experience`     | 野球経験 (文字列)                      | `--experience "3年"`   |
| `--level`          | 現在の実力レベル                       | `--level "中級者"`       |
| `--height`         | 身長 (cm)                              | `--height 175.0`       |
| `--weight`         | 体重 (kg)                              | `--weight 68.0`        |
| `--position`       | 選手のポジション                       | `--position "内野手"`    |
| `--batting_style`  | 打席 (右打ち/左打ち/両打ち)            | `--batting_style "右打ち"`|
| `--focus_points`    | 指導で重視するポイント（複数可）       | `--focus_points 重心移動 タイミング` |
| `--goal`            | おおまかな目標                        | `--goal "ミート力アップ"` |

実行完了後、標準出力に各ステップの結果が表示されます。

### 2. Streamlit WebUI

```bash
streamlit run app.py
```

- サイドバーで年齢やポジション、指導方針を入力し、  
- **「あなたのスイング動画」** と **「理想のスイング動画」** をアップロード  
- 「コーチング開始」ボタンを押すと、処理が行われ、分析レポートが画面上に表示されます。

---

## ファイル構成

```
.
├─ app.py                    # Streamlitアプリ (WebUI)
├─ main.py                   # CLI実行スクリプト
├─ requirements.txt          # 必要ライブラリ一覧
├─ config/
│   ├─ config.yaml           # システム設定
│   └─ load_config.py        # 設定ロードスクリプト
├─ core/
│   ├─ system.py             # SwingCoachingSystemの主要ロジック
│   ├─ state.py              # SystemState (動画2本対応に修正)
│   ├─ graph.py              # ワークフローDAG管理
│   └─ logger.py             # ログ出力管理
├─ agents/
│   ├─ base.py               # BaseAgent
│   ├─ interactive_agent/
│   ├─ modeling_agent/
│   │   ├─ agent.py          # 2本の動画を扱い、比較
│   │   └─ metrics/          # スイング指標計算
│   ├─ goal_setting_agent/
│   ├─ plan_agent/
│   ├─ search_agent/
│   └─ summarize_agent/
├─ MotionAGFormer/
│   ├─ run/vis.py            # 動画→3D姿勢推定 (HRNet + MotionAGFormer)
│   ├─ JsonAnalist.py        # 3D姿勢JSONの分析スクリプト
│   └─ (モデル関連ファイル多数)
└─ utils/
    ├─ video.py
    ├─ json_handler.py
    └─ validators.py
```
## `example_input.json` のイメージ

ユーザー（選手）の情報と指導方針をまとめたJSONの例です。  
`main.py` や `app.py` の内部でこれを参照して、`persona_data` / `policy_data` として扱うことを想定します。

```json5
{
  "persona_data": {
    "age": 16,
    "experience": "3年（中学野球）",
    "level": "中級者",
    "height": 170.0,
    "weight": 65.0,
    "position": "外野手",
    "batting_style": "右打ち"
  },
  "policy_data": {
    "focus_points": ["重心移動", "バットコントロール"],
    "teaching_style": "基礎重視",
    "goal": "試合での安定したミート力"
  }
}
```

- `persona_data`  
  - 選手の基本プロフィール  
  - `"level"` は `"初心者"|"中級者"|"上級者"` 等  
  - `"experience"` はどれくらいの年数・チーム歴か  
- `policy_data`  
  - 指導者が重点的に見たいポイント（"重心移動" 等）  
  - 指導スタイルや最終目標

---

## 参考・注意点

1. **動画フォーマット**  
   - 現在は `.mp4`, `.mov`, `.avi` を想定  
   - フレームレートが低い・暗いなどの場合、3D姿勢推定の精度が落ちる可能性

2. **MotionAGFormerのセットアップ**  
   - `checkpoint/` に事前学習モデルを配置  
   - `MotionAGFormer/run/vis.py` の引数に `--video` と `--output` が渡される仕組み。  
   - 解析結果は `3d_result.json` として保存される想定

3. **2本の動画の差分**  
   - `ModelingAgent` が `JsonAnalist.py` の結果を読み、各指標 (バットスピードなど) の差を計算  
   - LLMへのプロンプトにより「どの部分が大きく違うのか」「どう改善すべきか」などをまとめる

4. **OpenAI等のAPI**  
   - `.env` ファイルなどで APIキーの設定が必要  
   - 実行環境によってはトークン数やAPI呼び出し回数に注意

5. **検索機能 (SearchAgent)**  
   - Google APIキーなどが必要  
   - 日本語/英語クエリで練習ドリルを検索し統合する想定

6. **依存関係**  
   - `langchain`, `torch`, `opencv-python`, `timm` などのライブラリが必要  
   - GPUメモリが少ないと推論に失敗する可能性がある

---

## ライセンス

- 本リポジトリのコードは MIT License で公開しています。  
- 詳細は [LICENSE](LICENSE) ファイルを参照してください。

---

## 今後の拡張

- **InteractiveAgent**：ユーザー回答をもう少し受け取り、追加の対話を行う機能  
- **MotionAGFormer**：Webカメラやスマホ動画への対応最適化  
- **SearchAgent**：練習メニュー検索の多言語対応、より高度なフィルタリング  
- **SummarizeAgent**：より洗練された最終レポートのテンプレート生成

---

以下に追加情報として、**簡単なJSONファイルの例**を2種類提示します。

1. **ペルソナ・ポリシー入力用の例**（`example_input.json`）  
2. **3D解析結果の例**（`3d_result_example.json`）  

実際のシステムでは、これらのフォーマットやフィールドは要件に応じて変わります。あくまでイメージとしてご参考ください。

---

