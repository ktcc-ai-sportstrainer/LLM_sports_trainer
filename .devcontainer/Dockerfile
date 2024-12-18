# Ubuntu - DevContiner公式イメージ
# * GitHub | devcontainer/cpp : https://github.com/devcontainers/images/tree/main/src/base-ubuntu
FROM mcr.microsoft.com/devcontainers/base:ubuntu

# [オプション] 追加パッケージのインストール
#
# 使い方 :
#  * 「&& apt-get -y install --no-install-recommends <install packages>\」
#    のコメントアウトを外し、<install packages>の部分にパッケージを追加
#
# RUN apt-get update \
#     && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get upgrade -y \
#     # && apt-get -y install --no-install-recommends <install packages> \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        openssh-client \
        less \
        iproute2 \
        procps \
        wget \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*