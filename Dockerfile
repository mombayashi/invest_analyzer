FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 日本語フォントとビルドツールをインストール
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    fontconfig \
    fonts-noto-cjk \
    fonts-ipaexfont \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt をコピーしてパッケージインストール
COPY app/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# アプリケーションのソースをコピー
COPY app /app
COPY app/analyzer/japan_stocks.csv /app/analyzer/japan_stocks.csv
