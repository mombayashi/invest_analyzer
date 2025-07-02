from django.shortcuts import render
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf
from .stock_predictor import predict  # 予測関数をインポート
import numpy as np
import pandas as pd  # 追加：pandasをインポート
import os
import matplotlib.dates as mdates
from .models import StockPrice

# CSVから企業名→証券コード辞書を作成
def load_japan_stock_dicts():
    csv_path = os.path.join(os.path.dirname(__file__), "japan_stocks.csv")
    df = pd.read_csv(csv_path, dtype=str)
    name_to_code = {row["銘柄名"]: row["コード"] + ".T" for _, row in df.iterrows()}
    code_to_name = {row["コード"] + ".T": row["銘柄名"] for _, row in df.iterrows()}
    return name_to_code, code_to_name

JAPAN_STOCK_DICT, CODE_TO_NAME_DICT = load_japan_stock_dicts()

def predict_and_plot(symbol):
    prediction, history = predict(symbol)
    latest_close = history['Close'].iloc[-1] if len(history) > 0 else None
    latest_date = history['Date'].iloc[-1].strftime("%Y年%m月%d日") if len(history) > 0 else ""
    # 翌日の日付を計算
    next_day = history['Date'].iloc[-1] + pd.Timedelta(days=1) if len(history) > 0 else None
    next_date = next_day.strftime("%Y年%m月%d日") if next_day is not None else ""

    plt.figure(figsize=(15, 5))
    plt.plot(history['Date'], history['Close'], label='History', color='blue', marker='o')
    if len(history) > 0:
        plt.plot([history['Date'].iloc[-1], next_day], [history['Close'].iloc[-1], prediction],
                 color='red', linestyle='-', marker='o', label='Prediction')
        ax = plt.gca()
        # 予測日だけを補助目盛り（minor ticks）として追加
        ax.set_xticks([next_day], minor=True)
        # 予測日の補助目盛りラベル（rotation=45で斜め表示）
        ax.set_xticklabels([next_day.strftime('%Y-%m-%d')], minor=True, fontsize=10, color='red', rotation=45)
    plt.legend()
    plt.title(f"{symbol} Stock Price")
    plt.ylabel("YEN")
    plt.xlabel("DATE")
    plt.xticks(fontsize=10, rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    graph = base64.b64encode(buf.getvalue()).decode('utf-8')

    return prediction, history, graph, latest_close, latest_date, next_date

def convert_to_symbol(query):
    query = query.strip()
    # まずCSV辞書で検索
    if query in JAPAN_STOCK_DICT:
        return JAPAN_STOCK_DICT[query]
    # すでに .T が付いていればそのまま
    if query.endswith(".T"):
        return query
    # 「A」を含む場合も .T を付けて返す
    if "A" in query:
        return query + ".T"
    # 数字だけなら .T を付けて返す
    if query.isdigit():
        return query + ".T"
    return None

def index(request):
    query = request.GET.get("query", "")
    error = None
    prediction = None
    graph = None
    symbol = None
    latest_close = None
    latest_date = ""
    next_date = ""
    company_name = ""
    company_code = ""

    if query:
        symbol = convert_to_symbol(query)
        # 企業名を取得（コードから逆引き）
        company_name = CODE_TO_NAME_DICT.get(symbol, query)
        # 企業コードを取得（企業名から変換 or 入力値そのまま）
        if symbol:
            company_code = symbol
        else:
            company_code = query
        if symbol is None:
            error = "会社名または証券コードが正しくありません。"
        else:
            try:
                prediction, history, graph, latest_close, latest_date, next_date = predict_and_plot(symbol)
                # 予測や株価取得後に保存
                latest_date_obj = history['Date'].iloc[-1] if len(history) > 0 else None
                if latest_date_obj is not None:
                    StockPrice.objects.update_or_create(
                        code=symbol,
                        date=latest_date_obj,  # datetime.date型
                        defaults={
                            "name": company_name,
                            "close": latest_close,
                            "prediction": prediction,
                        }
                    )
            except Exception as e:
                error = f"予測中にエラーが発生しました: {e}"
    return render(request, "index.html", {
        "query": query,
        "error": error,
        "prediction": prediction,
        "graph": graph,
        "symbol": symbol,
        "latest_close": latest_close,
        "latest_date": latest_date,
        "next_date": next_date,
        "company_name": company_name,
        "company_code": company_code,
    })

# 正しい例
input_data = np.array([[1, 2, 3, 4, 5]])  # shape: (1, 5)
input_data = input_data.reshape((1, 5, 1))  # shape: (1, 5, 1)
input_data = np.expand_dims(input_data, axis=0)  # 余計な次元追加
input_data = np.expand_dims(input_data, axis=-1) # さらに追加
# shape: (1, 1, 5, 1) ← 4次元
