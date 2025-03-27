import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# === CONFIG ===
symbol = "BTC"
MODEL_DIR = "models"

# === MODELUL ===
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# === FETCH DATE ===
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
    df = df.dropna()
    return df[features].tail(60)

# === LOAD MODEL & SCALER ===
def load_model_and_scaler(symbol):
    model_path = os.path.join(MODEL_DIR, f"lstm_{symbol}.pt")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{symbol}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"meta_{symbol}.json")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = LSTMModel(input_size=len(meta["features"]))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, scaler, meta

# === PREDICȚIE ===
def predict_sequence(symbol, days):
    model, scaler, meta = load_model_and_scaler(symbol)
    data = fetch_latest_data(symbol, meta["features"])
    if data.empty:
        print("❌ DataFrame gol la fetch_latest_data")
        return []
    scaled_input = scaler.transform(data.values)
    sequence = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)

    preds = []
    for _ in range(days):
        with torch.no_grad():
            out = model(sequence)
        next_val = out.item()
        preds.append(next_val)
        next_row = torch.tensor(scaler.transform([sequence[0, -1, :].numpy()]), dtype=torch.float32)
        sequence = torch.cat((sequence[:, 1:, :], next_row.unsqueeze(0)), dim=1)

    result = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return result.tolist()

# === RULEAZĂ TESTUL ===
if __name__ == "__main__":
    try:
        predictions = predict_sequence("BTC", 7)
        print("✅ Predicții generate cu succes:")
        for i, p in enumerate(predictions, 1):
            print(f"Ziua {i}: {p:.2f} USD")
    except Exception as e:
        print("❌ Eroare la predicție:", e)
