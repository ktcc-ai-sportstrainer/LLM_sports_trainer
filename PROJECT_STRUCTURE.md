# LLM Sports Trainer プロジェクト構造

## 概要
LLM Baseball Swing Coachプロジェクトの完全なファイル構造と、各ファイルの役割を説明します。

## プロジェクト構造

```
LLM_sports_trainer/
├── 📁 コアファイル
│   ├── main.py                    # メインエントリーポイント
│   ├── app.py                     # Streamlit Webアプリケーション
│   ├── requirements.txt           # Python依存関係
│   ├── README.md                  # プロジェクトドキュメント
│   └── LICENSE                    # ライセンスファイル
│
├── 🐳 Docker開発環境
│   ├── Dockerfile                 # Dockerイメージ定義
│   ├── docker-compose.yml         # 基本的なDocker Compose設定
│   ├── docker-compose.dev.yml     # 開発専用設定
│   ├── docker-entrypoint.sh       # コンテナ起動スクリプト
│   ├── .dockerignore              # Dockerビルド除外ファイル
│   └── Makefile                   # 開発環境操作コマンド
│
├── 🤖 AIエージェント
│   ├── agents/
│   │   ├── base.py                # エージェント基底クラス
│   │   ├── interactive_agent/     # 対話型エージェント
│   │   ├── modeling_agent/        # モデリングエージェント
│   │   ├── plan_agent/            # 計画作成エージェント
│   │   ├── search_agent/          # 検索エージェント
│   │   ├── goal_setting_agent/    # 目標設定エージェント
│   │   └── summarize_agent/       # 要約エージェント
│
├── 🏗️ システムコア
│   ├── core/
│   │   ├── base/                  # 基底クラス
│   │   ├── cli/                   # CLIシステム
│   │   └── webui/                 # WebUIシステム
│
├── 📊 データモデル
│   ├── models/
│   │   ├── input/                 # 入力データモデル
│   │   ├── internal/              # 内部データモデル
│   │   └── output/                # 出力データモデル
│
├── ⚙️ 設定ファイル
│   ├── config/
│   │   ├── config.yaml            # メイン設定ファイル
│   │   └── load_config.py         # 設定読み込み
│
├── 🎯 3D姿勢推定
│   ├── MotionAGFormer/            # 3D姿勢推定モデル
│   │   ├── model/                 # モデル定義
│   │   ├── configs/               # モデル設定
│   │   └── run/                   # 実行スクリプト
│
├── 🛠️ ユーティリティ
│   ├── utils/
│   │   ├── json_handler.py        # JSON処理
│   │   ├── validators.py          # データ検証
│   │   └── video.py               # 動画処理
│
├── 📁 データ・アセット
│   ├── fig/                       # 画像ファイル
│   │   ├── fig.jpg                # プロジェクト画像
│   │   └── LLM_sports_trainer-all.png
│   └── info.json                  # サンプルデータ
│
└── 📝 開発ファイル
    ├── __pycache__/               # Pythonキャッシュ（自動生成）
    └── .git/                      # Gitリポジトリ（自動生成）
```

## 必須ファイル一覧

### 🎯 プロジェクト実行に必須
1. **main.py** - メインエントリーポイント
2. **app.py** - Streamlit Webアプリケーション
3. **requirements.txt** - Python依存関係
4. **config/config.yaml** - 設定ファイル
5. **config/load_config.py** - 設定読み込み

### 🤖 AIエージェント（必須）
1. **agents/base.py** - エージェント基底クラス
2. **agents/interactive_agent/** - 対話型エージェント
3. **agents/modeling_agent/** - モデリングエージェント
4. **agents/plan_agent/** - 計画作成エージェント
5. **agents/search_agent/** - 検索エージェント
6. **agents/goal_setting_agent/** - 目標設定エージェント
7. **agents/summarize_agent/** - 要約エージェント

### 🏗️ システムコア（必須）
1. **core/base/** - 基底クラス
2. **core/cli/system.py** - CLIシステム
3. **core/webui/system.py** - WebUIシステム

### 📊 データモデル（必須）
1. **models/input/** - 入力データモデル
2. **models/internal/** - 内部データモデル
3. **models/output/** - 出力データモデル

### 🎯 3D姿勢推定（必須）
1. **MotionAGFormer/** - 3D姿勢推定モデル全体
   - **model/MotionAGFormer.py** - メインモデル
   - **run/lib/** - 実行ライブラリ
   - **configs/** - モデル設定

### 🛠️ ユーティリティ（必須）
1. **utils/json_handler.py** - JSON処理
2. **utils/validators.py** - データ検証
3. **utils/video.py** - 動画処理

### 🐳 Docker開発環境（推奨）
1. **Dockerfile** - Dockerイメージ定義
2. **docker-compose.yml** - 基本的なDocker Compose設定
3. **docker-compose.dev.yml** - 開発専用設定
4. **docker-entrypoint.sh** - コンテナ起動スクリプト
5. **.dockerignore** - Dockerビルド除外ファイル
6. **Makefile** - 開発環境操作コマンド

### 📝 ドキュメント（推奨）
1. **README.md** - プロジェクトドキュメント
2. **LICENSE** - ライセンスファイル

### 📁 データ・アセット（オプション）
1. **fig/** - 画像ファイル
2. **info.json** - サンプルデータ

## ファイルの重要度

### 🔴 最重要（プロジェクト実行に必須）
- main.py
- app.py
- requirements.txt
- agents/（全ディレクトリ）
- core/（全ディレクトリ）
- models/（全ディレクトリ）
- MotionAGFormer/（全ディレクトリ）
- utils/（全ファイル）

### 🟡 重要（推奨）
- config/
- Dockerfile
- docker-compose.yml
- docker-compose.dev.yml
- Makefile
- README.md

### 🟢 オプション
- fig/
- info.json
- LICENSE

## 開発環境セットアップ時の注意点

### 1. 事前学習モデルのダウンロード
```
MotionAGFormer/checkpoint/          # MotionAGFormerベースモデル
MotionAGFormer/run/lib/checkpoint/  # YOLOv3とHRNet事前学習モデル
```

### 2. 環境変数の設定
```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. ディレクトリ構造の確認
- 必要なディレクトリが存在することを確認
- 権限設定が適切であることを確認
- パス設定が正しいことを確認

## トラブルシューティング

### よくある問題
1. **モジュールが見つからない**
   - PYTHONPATHの設定確認
   - 依存関係のインストール確認

2. **モデルファイルが見つからない**
   - 事前学習モデルのダウンロード確認
   - ファイルパスの確認

3. **権限エラー**
   - ファイル・ディレクトリの権限確認
   - Docker環境での権限設定確認

## 開発ワークフロー

### 新機能開発時
1. 必要なファイルの特定
2. 依存関係の確認
3. テスト環境での動作確認
4. ドキュメントの更新

### デバッグ時
1. ログファイルの確認
2. 設定ファイルの確認
3. 環境変数の確認
4. 依存関係の確認 
