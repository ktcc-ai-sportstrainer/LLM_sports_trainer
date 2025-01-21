# Baseball Swing AI Coach

野球のバッティングフォームをAIで分析し、個別化されたコーチングを提供するシステムです。選手の情報と指導者の方針を考慮し、3D姿勢推定技術とLLMを組み合わせることで、技術的な分析と具体的な改善提案を行います。

## 必要環境

- Python 3.9+
- CUDA対応GPU（推奨）
- OpenAI API キー

## インストール

1. リポジトリのクローン:
```bash
git clone https://github.com/yourusername/baseball-swing-ai-coach.git
cd baseball-swing-ai-coach
```

2. 依存パッケージのインストール:
```bash
pip install -r requirements.txt
```

3. MotionAGFormerの事前学習済みモデルをダウンロード:
```bash
mkdir -p checkpoint
wget https://example.com/motionagformer-b-h36m.pth.tr -O checkpoint/motionagformer-b-h36m.pth.tr
```

4. 環境変数の設定:
```bash
cp .env.example .env
# .envファイルを編集してOpenAI APIキーを設定
```

## 使用方法

### WebUI モード（推奨）

```bash
streamlit run app.py
```

ブラウザで http://localhost:8501 を開き、以下の情報を入力します：

#### 1. 基本情報
- 名前
- 年齢・学年
- ポジション（現在または希望）
- 利き手（投球・打撃）
- 身長・体重
- 野球歴（経験年数や所属歴）
- 目標
- 普段の練習時間
- 個人的な課題
- その他の情報

#### 2. 指導方針
- 監督の指導方針
- 監督が見たプレーヤーの強み
- 監督が見たプレーヤーの弱み

#### 3. スイング動画
- アップロード可能形式: .mp4, .mov, .avi
- 推奨: 30fps以上、明るい環境での撮影
- スイング全体が映るように撮影

### CLI モード（開発者向け）

CLI モードでは、入力情報をJSONファイルとして提供します：

```bash
python main.py --input player_info.json --video swing_video.mp4
```

player_info.jsonの形式:
```json
{
  "basic_info": {
    "name": "山田太郎",
    "age": 16,
    "grade": "高校1年",
    "position": "外野手",
    "dominant_hand": {
      "batting": "右",
      "throwing": "右"
    },
    "height": 170,
    "weight": 65,
    "experience": {
      "years": 8,
      "history": "少年野球(6年)→中学野球(3年)"
    },
    "goal": "レギュラー獲得",
    "practice_time": "平日2時間",
    "personal_issues": ["タイミングが合わない", "内角が苦手"],
    "additional_info": "部活動以外で自主練習も行っている"
  },
  "coaching_policy": {
    "philosophy": "基礎技術の徹底",
    "player_strengths": ["練習熱心", "集中力がある"],
    "player_weaknesses": ["力に頼りすぎ", "焦りがち"]
  }
}
```

## 出力情報

システムは以下の情報を生成します：

1. インタラクティブな追加質問
   - 選手情報の補完や詳細の確認

2. スイング解析結果
   - フェーズごとの動作分析
   - 技術的な課題点
   - 良い点

3. 目標設定
   - 主目標
   - サブ目標
   - 達成指標

4. 練習計画
   - 具体的な練習メニュー
   - 段階的な上達ステップ
   - 時間配分

## トラブルシューティング

### よくある問題と解決方法

1. 動画アップロードの問題
   - ファイルサイズの制限: 500MB以下を推奨
   - 対応フォーマットの確認
   - 動画の長さ: 10秒以内を推奨

2. 姿勢推定の精度が低い場合
   - 動画の撮影環境を改善（明るさ、角度など）
   - フレームレートの確認
   - バックグラウンドのノイズを減らす

3. システムエラー
   - ログの確認
   - GPUメモリ使用量の確認
   - APIキーの有効性確認

## ライセンス

MIT License

## サポート

- 技術的な問題: Issues
- 使用方法の質問: [サポートメール]