from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf


class StockPredictor(nn.Module):
    """LSTMベースの株価予測モデル."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def _extract_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """マルチインデックス/単一インデックス両対応でOHLCVを抽出する."""

    required_cols = ["Close", "Open", "High", "Low", "Volume"]
    if df.empty:
        return pd.DataFrame(columns=required_cols)

    if isinstance(df.columns, pd.MultiIndex):
        data = {col: df[(col, symbol)] for col in required_cols if (col, symbol) in df.columns}
    else:
        data = {col: df[col] for col in required_cols if col in df.columns}

    result = pd.DataFrame(data).dropna()
    missing = set(required_cols) - set(result.columns)
    if missing:
        raise ValueError(f"必須列が不足しています: {missing}")
    return result


def get_stock_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """株価データを取得し、OHLCV列のみを返す."""

    df = yf.download(symbol, period=period, progress=False)
    ohlcv = _extract_ohlcv(df, symbol)
    if ohlcv.empty:
        raise ValueError("株価データが取得できませんでした。")
    return ohlcv


def create_dataset(data: np.ndarray, look_back: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """過去look_back日分を説明変数、翌日の終値を目的変数とするデータセットを生成."""

    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        Y.append(data[i + look_back, 0])  # 終値を予測対象とする
    return np.array(X), np.array(Y)

# 予測処理本体
def predict(symbol: str) -> tuple[float, pd.DataFrame]:
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"symbol: {symbol}")
    ohlcv = get_stock_data(symbol)
    print(f"data length: {len(ohlcv)}")

    look_back = 30
    if len(ohlcv) <= look_back:
        raise ValueError("学習に必要な十分な日数の株価データがありません。")

    feature_min = ohlcv.min()
    feature_max = ohlcv.max()
    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1.0

    normalized = (ohlcv - feature_min) / feature_range

    X, y = create_dataset(normalized.values, look_back=look_back)
    if len(X) < 2:
        raise ValueError("学習と検証に必要なデータが不足しています。")

    train_size = max(1, int(len(X) * 0.8))
    if train_size >= len(X):
        train_size = len(X) - 1

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=min(64, len(train_dataset)), shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockPredictor(input_size=X_train_tensor.size(-1)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_state = None
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    model.train()
    for epoch in range(200):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X.to(device)).squeeze(-1)
            loss = criterion(output, batch_y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)

        epoch_loss /= len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor.to(device)).squeeze(-1)
            val_loss = criterion(val_output, y_val_tensor.to(device)).item()

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: train_loss = {epoch_loss:.6f}, val_loss = {val_loss:.6f}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        latest_window = torch.tensor(
            normalized.values[-look_back:], dtype=torch.float32
        ).unsqueeze(0)
        pred = model(latest_window.to(device))
    pred_value = pred.item()

    df_recent = yf.download(symbol, period="1mo", progress=False)
    df_simple = _extract_ohlcv(df_recent, symbol).reset_index()
    if "Date" not in df_simple.columns:
        df_simple = df_simple.rename(columns={"index": "Date"})
    if "Date" not in df_simple.columns:
        first_col = df_simple.columns[0]
        df_simple = df_simple.rename(columns={first_col: "Date"})

    close_min = feature_min["Close"]
    close_max = feature_max["Close"]
    pred_value_denorm = pred_value * (close_max - close_min) + close_min
    pred_value_denorm = round(float(pred_value_denorm), 1)

    return pred_value_denorm, df_simple[["Date", "Close"]]
