import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# LSTMモデル定義（hidden_sizeやnum_layersはチューニング可能）
class StockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):  # hidden_size を変更可
        super().__init__()
        # LSTM層の追加。num_layersを増やすと深くできる
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 株価データ取得関数
def get_stock_data(symbol, period="1y"):
    df = yf.download(symbol, period=period)
    prices = df["Close"].values
    return prices

# 時系列データセット作成関数
def create_dataset(data, look_back=10):  # look_back の値を変えて過去何日使うか変更可能
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        Y.append(data[i+look_back])
    return np.array(X), np.array(Y)

# 予測処理本体
def predict(symbol):
    print(f"symbol: {symbol}")
    data = get_stock_data(symbol)
    print(f"data length: {len(data)}")
    if len(data) == 0:
        raise ValueError("株価データが取得できませんでした。")

    # 正規化
    data = (data - data.min()) / (data.max() - data.min())

    # 学習する日数の設定
    look_back = 10
    X, y = create_dataset(data, look_back=look_back)

    # テンソル化
    if X.ndim == 2:
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    elif X.ndim == 3:
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 隠れ層ユニット数を増減して性能確認可能
    model = StockPredictor(hidden_size=64)

    # 損失関数（MSE以外にもL1Loss, HuberLossなど試せる）
    criterion = nn.MSELoss()

    # 最適化手法（SGD, RMSpropも試して比較できる）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 学習率を0.001などに変更可能

    # 学習回数（epoch）
    for epoch in range(100):  # ← epoch数を変更して精度を比較
        output = model(X_tensor)
        loss = criterion(output.squeeze(), y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 10エポックごとに損失を出力して学習状態を確認
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # 直近データで予測
    pred = model(X_tensor[-1:])
    pred_value = pred.item()

    # 最新1ヶ月の株価を取得し、正規化を戻す
    df = yf.download(symbol, period="1mo")
    if isinstance(df.columns, pd.MultiIndex):
        close_col = ('Close', symbol)
        if close_col not in df.columns:
            raise ValueError("1ヶ月分の株価データが取得できませんでした。")
        close_series = df[close_col]
        df_simple = pd.DataFrame({'Date': df.index, 'Close': close_series.values})
    else:
        if 'Close' not in df.columns:
            raise ValueError("1ヶ月分の株価データが取得できませんでした。")
        df_simple = pd.DataFrame({'Date': df.index, 'Close': df['Close'].values})

    # 予測値を正規化から戻す
    close_min = df_simple["Close"].min()
    close_max = df_simple["Close"].max()
    pred_value_denorm = pred_value * (close_max - close_min) + close_min
    pred_value_denorm = round(pred_value_denorm, 1)

    return pred_value_denorm, df_simple
