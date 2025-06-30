# LLM Sports Trainer 開発環境用Dockerfile
# 
# 概要: LLMを使用したスポーツトレーニングシステムの開発環境
# 主な仕様: Python 3.11, CUDA対応, 開発ツール含む
# 制限事項: GPU使用時はnvidia-dockerが必要

# ベースイメージ: Python 3.11 + CUDA 12.1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# メタデータ
LABEL maintainer="LLM Sports Trainer Team"
LABEL description="LLM Sports Trainer Development Environment"
LABEL version="1.0.0"

# 環境変数設定
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# 作業ディレクトリ設定
WORKDIR /app

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    # 基本的な開発ツール
    build-essential \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    # 画像・動画処理関連
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # その他の依存関係
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 開発用ツールの追加インストール
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    jupyterlab \
    debugpy \
    pytest-cov \
    black[jupyter] \
    pre-commit

COPY . .

COPY download_weights.py ./
RUN python download_weights.py

# ポート設定
EXPOSE 8080

# Streamlit の起動コマンド（Cloud RunのPORTに対応）
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
