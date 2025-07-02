import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class StockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def get_stock_data(symbol, period="1y"):
    df = yf.download(symbol, period=period)
    prices = df["Close"].values
    return prices

def create_dataset(data, look_back=10):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        Y.append(data[i+look_back])
    return np.array(X), np.array(Y)

def predict(symbol):
    print(f"symbol: {symbol}")
    data = get_stock_data(symbol)
    print(f"data length: {len(data)}")
    if len(data) == 0:
        raise ValueError("株価データが取得できませんでした。")
    data = (data - data.min()) / (data.max() - data.min())
    X, y = create_dataset(data)
    if X.ndim == 2:
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    elif X.ndim == 3:
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = StockPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(100):
        output = model(X_tensor)
        loss = criterion(output.squeeze(), y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pred = model(X_tensor[-1:])
    pred_value = pred.item()

    df = yf.download(symbol, period="1mo")
    if isinstance(df.columns, pd.MultiIndex):
        # カラムがMultiIndexの場合
        close_col = ('Close', symbol)
        if close_col not in df.columns:
            raise ValueError("1ヶ月分の株価データが取得できませんでした。")
        close_series = df[close_col]
        df_simple = pd.DataFrame({'Date': df.index, 'Close': close_series.values})
    else:
        if 'Close' not in df.columns:
            raise ValueError("1ヶ月分の株価データが取得できませんでした。")
        df_simple = pd.DataFrame({'Date': df.index, 'Close': df['Close'].values})

    close_min = df_simple["Close"].min()
    close_max = df_simple["Close"].max()
    pred_value_denorm = pred_value * (close_max - close_min) + close_min
    pred_value_denorm = round(pred_value_denorm, 1)  # 小数第一位までに丸める
    return pred_value_denorm, df_simple
