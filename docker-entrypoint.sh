#!/bin/bash
# LLM Sports Trainer Docker Entrypoint Script
#
# 概要: Dockerコンテナ起動時の初期化処理を実行
# 主な仕様: 環境変数チェック、ディレクトリ作成、アプリケーション起動
# 制限事項: 必要な環境変数が設定されている必要がある

set -e

echo "🚀 LLM Sports Trainer 開発環境を起動中..."

# 作業ディレクトリに移動
cd /app

# 必要なディレクトリを作成
echo "📁 必要なディレクトリを作成中..."
mkdir -p logs
mkdir -p data
mkdir -p models
mkdir -p config

# 環境変数のチェック
echo "🔍 環境変数をチェック中..."
required_vars=("OPENAI_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ 以下の環境変数が設定されていません:"
    printf '%s\n' "${missing_vars[@]}"
    echo "💡 .envファイルまたは環境変数で設定してください"
    exit 1
fi

echo "✅ 環境変数のチェック完了"

# Pythonパッケージの確認
echo "📦 Pythonパッケージを確認中..."
python -c "import streamlit, langchain, opencv-python, torch; print('✅ 主要パッケージの読み込み成功')" || {
    echo "❌ 必要なPythonパッケージが不足しています"
    echo "💡 requirements.txtを再インストールしてください"
    exit 1
}

# 権限設定
echo "🔐 ファイル権限を設定中..."
chmod -R 755 /app
chown -R root:root /app

# ヘルスチェック
echo "🏥 システムヘルスチェック中..."
python -c "
import sys
import os
print(f'Python version: {sys.version}')
print(f'Working directory: {os.getcwd()}')
print(f'PYTHONPATH: {os.environ.get(\"PYTHONPATH\", \"Not set\")}')
print('✅ システムヘルスチェック完了')
"

# コマンドライン引数の処理
if [ $# -eq 0 ]; then
    echo "📋 利用可能なコマンド:"
    echo "  python main.py --help          # メインアプリケーションのヘルプ"
    echo "  streamlit run main.py          # Streamlitサーバー起動"
    echo "  jupyter lab                    # Jupyter Lab起動"
    echo "  python -m pytest               # テスト実行"
    echo "  python -m black .              # コードフォーマット"
    echo ""
    echo "🔄 デフォルトでbashシェルを起動します"
    exec /bin/bash
else
    echo "🚀 指定されたコマンドを実行: $@"
    exec "$@"
fi 
