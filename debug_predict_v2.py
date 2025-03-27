import os
import json
import torch
import pickle
import numpy as np
import yfinance as yf
import pandas as pd

MODEL_DIR = "models"
symbol = "BTC"

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def fetch_latest_data(symbol, features):
    df = yf.download(f"{symbol}-USD", period="90d").dropna()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    RS = gain / loss
    df["RSI"] = 100 - (100 / (1 + RS))
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["BB_upper"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
    df["BB_lower"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df = df.dropna()
    return df[features].tail(60)

def predict_sequence(symbol, days):
    model_path = os.path.join(MODEL_DIR, f"lstm_{symbol}.pt")
    scaler_X_path = os.path.join(MODEL_DIR, f"scaler_X_{symbol}.pkl")
    scaler_y_path = os.path.join(MODEL_DIR, f"scaler_y_{symbol}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"meta_{symbol}.json")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_X_path, "rb") as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)

    features = meta["features"]
    model = LSTMModel(input_size=len(features))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    data = fetch_latest_data(symbol, features)
    print("✅ Data shape:", data.shape)
    if data.empty:
        raise Exception("❌ DataFrame gol")

    scaled_input = scaler_X.transform(data.values)
    sequence = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)

    preds = []
    for _ in range(days):
        with torch.no_grad():
            out = model(sequence)
        next_val = out.item()
        preds.append(next_val)
        next_row = torch.tensor(scaled_input[-1:], dtype=torch.float32)
        sequence = torch.cat((sequence[:, 1:, :], next_row.unsqueeze(0)), dim=1)

    result = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return result.tolist()

# === EXECUTĂ ===
try:
    result = predict_sequence("BTC", 7)
    print("✅ Predicții generate:")
    for i, val in enumerate(result, 1):
        print(f"Ziua {i}: {val:.2f} USD")
except Exception as e:
    print("❌ Eroare:", e)
