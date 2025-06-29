# LLM Sports Trainer Docker開発環境用Makefile
#
# 概要: Docker開発環境の操作を簡単にするためのMakefile
# 主な仕様: ビルド、起動、停止、クリーンアップなどの操作を提供
# 制限事項: Makeがインストールされている必要がある

.PHONY: help build up down clean logs shell jupyter debug streamlit test format lint

# デフォルトターゲット
help:
	@echo "🚀 LLM Sports Trainer Docker開発環境"
	@echo ""
	@echo "利用可能なコマンド:"
	@echo "  make build      - Dockerイメージをビルド"
	@echo "  make up         - 開発環境を起動"
	@echo "  make down       - 開発環境を停止"
	@echo "  make clean      - コンテナとイメージを削除"
	@echo "  make logs       - ログを表示"
	@echo "  make shell      - 開発コンテナにシェル接続"
	@echo "  make jupyter    - Jupyter環境を起動"
	@echo "  make debug      - デバッグ環境を起動"
	@echo "  make streamlit  - Streamlit環境を起動"
	@echo "  make test       - テストを実行"
	@echo "  make format     - コードフォーマット"
	@echo "  make lint       - コードリント"
	@echo "  make help       - このヘルプを表示"

# Dockerイメージのビルド
build:
	@echo "🔨 Dockerイメージをビルド中..."
	docker-compose -f docker-compose.dev.yml build
	@echo "✅ ビルド完了"

# 開発環境の起動
up:
	@echo "🚀 開発環境を起動中..."
	docker-compose -f docker-compose.dev.yml --profile development up -d
	@echo "✅ 開発環境が起動しました"
	@echo "📊 アクセス先:"
	@echo "  - Streamlit: http://localhost:8502"
	@echo "  - Jupyter: http://localhost:8889"
	@echo "  - デバッグ: localhost:5679"

# 開発環境の停止
down:
	@echo "🛑 開発環境を停止中..."
	docker-compose -f docker-compose.dev.yml down
	@echo "✅ 開発環境が停止しました"

# クリーンアップ
clean:
	@echo "🧹 クリーンアップ中..."
	docker-compose -f docker-compose.dev.yml down -v --rmi all
	docker system prune -f
	@echo "✅ クリーンアップ完了"

# ログの表示
logs:
	@echo "📋 ログを表示中..."
	docker-compose -f docker-compose.dev.yml logs -f

# 開発コンテナにシェル接続
shell:
	@echo "🐚 開発コンテナに接続中..."
	docker-compose -f docker-compose.dev.yml exec dev-app /bin/bash

# Jupyter環境の起動
jupyter:
	@echo "📓 Jupyter環境を起動中..."
	docker-compose -f docker-compose.dev.yml up -d dev-jupyter
	@echo "✅ Jupyter環境が起動しました"
	@echo "🌐 アクセス先: http://localhost:8889"

# デバッグ環境の起動
debug:
	@echo "🐛 デバッグ環境を起動中..."
	docker-compose -f docker-compose.dev.yml up -d dev-debug
	@echo "✅ デバッグ環境が起動しました"
	@echo "🔍 デバッグポート: localhost:5679"

# Streamlit環境の起動
streamlit:
	@echo "📊 Streamlit環境を起動中..."
	docker-compose -f docker-compose.dev.yml up -d dev-streamlit
	@echo "✅ Streamlit環境が起動しました"
	@echo "🌐 アクセス先: http://localhost:8502"

# テストの実行
test:
	@echo "🧪 テストを実行中..."
	docker-compose -f docker-compose.dev.yml run --rm dev-test
	@echo "✅ テスト完了"

# コードフォーマット
format:
	@echo "🎨 コードフォーマットを実行中..."
	docker-compose -f docker-compose.dev.yml run --rm dev-app black .
	docker-compose -f docker-compose.dev.yml run --rm dev-app isort .
	@echo "✅ フォーマット完了"

# コードリント
lint:
	@echo "🔍 コードリントを実行中..."
	docker-compose -f docker-compose.dev.yml run --rm dev-app black --check .
	docker-compose -f docker-compose.dev.yml run --rm dev-app isort --check-only .
	docker-compose -f docker-compose.dev.yml run --rm dev-app mypy .
	@echo "✅ リント完了"

# 環境変数ファイルの作成
env:
	@echo "📝 .envファイルを作成中..."
	@if [ ! -f .env ]; then \
		echo "# LLM Sports Trainer 環境変数" > .env; \
		echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env; \
		echo "GOOGLE_API_KEY=your_google_api_key_here" >> .env; \
		echo "TAVILY_API_KEY=your_tavily_api_key_here" >> .env; \
		echo "✅ .envファイルを作成しました"; \
		echo "⚠️  実際のAPIキーを設定してください"; \
	else \
		echo "ℹ️  .envファイルは既に存在します"; \
	fi

# 開発環境の状態確認
status:
	@echo "📊 開発環境の状態:"
	docker-compose -f docker-compose.dev.yml ps
	@echo ""
	@echo "🌐 アクセス可能なサービス:"
	@echo "  - Streamlit: http://localhost:8502"
	@echo "  - Jupyter: http://localhost:8889"
	@echo "  - デバッグ: localhost:5679" 
