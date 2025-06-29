# LLM Baseball Swing Coach

![](fig/fig.jpg)

本プロジェクトは、AKATSUKIプロジェクト採択「関西テッククリエイターチャレンジ」にて開発された，AIを活用して野球選手のバッティングフォームを分析し，個別化されたコーチングを提供するシステムです．3D姿勢推定技術とLLMを組み合わせることで，選手のスイングを詳細に分析し，具体的な改善アドバイスを提供します．

## 特徴
- 3D姿勢推定による詳細なスイング分析
- インタラクティブな対話による選手理解
- 個別化された目標設定と練習プラン作成
- AIによる総合的なコーチングレポート生成
- StreamlitベースのWebインターフェース

## インストール方法

### 方法1: Docker開発環境（推奨）

#### 前提条件
- Docker
- Docker Compose
- Make (推奨)

#### クイックスタート
```bash
# 1. リポジトリのクローン
git clone [リポジトリURL]
cd llm-baseball-swing-coach

# 2. 環境変数ファイルの作成
make env

# 3. 開発環境のビルド・起動
make build
make up
```

#### アクセス
- **Streamlit**: http://localhost:8502
- **Jupyter Lab**: http://localhost:8889
- **デバッグ**: localhost:5679

#### 利用可能なコマンド
```bash
make help      # ヘルプ表示
make build     # Dockerイメージをビルド
make up        # 開発環境を起動
make down      # 開発環境を停止
make shell     # 開発コンテナにシェル接続
make jupyter   # Jupyter環境を起動
make debug     # デバッグ環境を起動
make streamlit # Streamlit環境を起動
make test      # テストを実行
make format    # コードフォーマット
make lint      # コードリント
make clean     # コンテナとイメージを削除
```

### 方法2: ローカル環境

#### 前提条件
- Python 3.11+
- GPU推奨（動画分析用）

#### インストール手順
```bash
# 1. リポジトリのクローン
git clone [リポジトリURL]
cd llm-baseball-swing-coach

# 2. 依存パッケージのインストール
pip install -r requirements.txt

# 3. 事前学習モデルのダウンロード
mkdir -p MotionAGFormer/checkpoint
mkdir -p MotionAGFormer/run/lib/checkpoint

# [YOLOv3とHRNetの事前学習モデル](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing)をダウンロードし、`MotionAGFormer/run/lib/checkpoint/`に配置
# [MotionAGFormerのベースモデル](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view)をダウンロードし、`MotionAGFormer/checkpoint/`に配置

# 4. 環境変数の設定
touch .env
# .envファイルを編集し、APIキーを設定
```

#### 環境変数設定例
```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## 使用方法

### WebUIでの実行
```bash
# Docker環境の場合
make streamlit

# ローカル環境の場合
streamlit run app.py
```

### コマンドラインでの実行
```bash
# Docker環境の場合
make shell
python main.py --json info.json --user_video path/to/video.mp4

# ローカル環境の場合
python main.py --json info.json --user_video path/to/video.mp4
```

#### オプション
- `--json`: 選手情報を含むJSONファイルのパス（必須）
- `--user_video`: 分析対象の動画ファイルパス
- `--user_pose_json`: 3D姿勢データ（JSONファイル）のパス
- `--ideal_video`: 理想フォームの動画ファイルパス（オプション）
- `--ideal_pose_json`: 理想フォームの3D姿勢データ（オプション）

## 主な機能

![](fig/LLM_sports_trainer-all.png)

1. **3D姿勢推定と分析**
   - スイング動画からの3D姿勢推定
   - バットスピード、回転の連動性などの定量的分析
   - 理想フォームとの比較分析

2. **インタラクティブコーチング**
   - 選手との対話による情報収集
   - 個別の課題やニーズの把握
   - カスタマイズされたアドバイス提供

3. **トレーニングプラン作成**
   - 分析結果に基づく具体的な練習メニュー
   - 段階的な目標設定
   - 実行可能な練習スケジュール

4. **総合レポート生成**
   - スイング分析結果のまとめ
   - 改善ポイントの提示
   - 具体的なアクションプラン

## システム構成
```
llm-baseball-swing-coach/
├── agents/           # 各種AIエージェント
├── core/            # システムコア機能
├── models/          # データモデル定義
├── utils/           # ユーティリティ関数
├── config/          # 設定ファイル
├── MotionAGFormer/  # 3D姿勢推定モデル
├── Dockerfile       # Docker開発環境
├── docker-compose.yml # Docker Compose設定
├── docker-compose.dev.yml # 開発環境設定
├── Makefile         # 開発環境操作
└── docker-entrypoint.sh # コンテナ起動スクリプト
```

## Docker開発環境の詳細

### 主な機能
- 🐍 Python 3.11 開発環境
- 📊 Streamlit アプリケーション
- 📓 Jupyter Lab 開発環境
- 🐛 デバッグ環境 (debugpy)
- 🧪 テスト環境
- 🎨 コードフォーマット・リント
- 🔄 ホットリロード対応

### 個別サービスの起動
```bash
# Jupyter環境のみ起動
docker-compose -f docker-compose.dev.yml up -d dev-jupyter

# Streamlit環境のみ起動
docker-compose -f docker-compose.dev.yml up -d dev-streamlit

# デバッグ環境のみ起動
docker-compose -f docker-compose.dev.yml up -d dev-debug
```

### ファイルの同期
プロジェクトファイルは自動的にホストとコンテナ間で同期されます：
- ソースコードの変更は即座に反映
- `data/`, `logs/`, `models/`, `config/` ディレクトリも同期

### トラブルシューティング

#### よくある問題

1. **ポートが既に使用されている**
```bash
# 使用中のポートを確認
lsof -i :8501
lsof -i :8889
lsof -i :5679

# 既存のコンテナを停止
docker-compose -f docker-compose.dev.yml down
```

2. **権限エラー**
```bash
# Dockerグループにユーザーを追加
sudo usermod -aG docker $USER
# ログアウト・ログイン後に再試行
```

3. **メモリ不足**
```bash
# Dockerのメモリ制限を増やす
# Docker Desktop > Settings > Resources > Memory
```

4. **ビルドエラー**
```bash
# キャッシュをクリアして再ビルド
make clean
make build
```

### ログの確認
```bash
# 全サービスのログ
make logs

# 特定サービスのログ
docker-compose -f docker-compose.dev.yml logs dev-app
docker-compose -f docker-compose.dev.yml logs dev-jupyter
```

### 開発ワークフロー

#### 1. 新機能開発
```bash
# 開発環境を起動
make up

# コードを編集（ホスト側）
# 変更は自動的にコンテナに反映

# テスト実行
make test

# コードフォーマット
make format
```

#### 2. デバッグ
```bash
# デバッグ環境を起動
make debug

# VS Codeでデバッグ接続
# launch.json設定例:
{
    "name": "Docker Debug",
    "type": "python",
    "request": "attach",
    "connect": {
        "host": "localhost",
        "port": 5679
    }
}
```

#### 3. Jupyter開発
```bash
# Jupyter環境を起動
make jupyter

# ブラウザで http://localhost:8889 にアクセス
```

### パフォーマンス最適化

#### GPU使用（オプション）
```bash
# nvidia-dockerがインストールされている場合
# docker-compose.dev.ymlのservicesに以下を追加:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

#### メモリ最適化
```bash
# 不要なサービスを停止
docker-compose -f docker-compose.dev.yml stop dev-jupyter dev-debug

# 使用していないコンテナを削除
docker container prune
```

### セキュリティ

#### ベストプラクティス
- `.env`ファイルをGitにコミットしない
- 本番環境では異なる設定を使用
- 定期的にDockerイメージを更新

#### セキュリティスキャン
```bash
# イメージの脆弱性スキャン
docker scan llm_sports_trainer_dev
```

## 参考文献
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
- その他関連論文やプロジェクト

## 注意事項
- このシステムはアシスタントとして機能することを目的としており、プロのコーチの判断を完全に代替するものではありません。
- 動画分析には高性能なGPUを推奨します。
- APIキーの利用には別途OpenAIのアカウントとクレジットが必要です。
- Docker環境を使用する場合は、十分なメモリとストレージ容量を確保してください。

## 貢献

### 開発環境の改善
1. 新しい依存関係を追加する場合は`requirements.txt`を更新
2. Dockerfileの変更は必要最小限に
3. セキュリティアップデートを定期的に適用

### 問題の報告
- GitHub Issuesで問題を報告
- ログとエラーメッセージを含める
- 再現手順を詳細に記載
