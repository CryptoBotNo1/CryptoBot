import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pickle
import json
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression  # Meta-model
from xgboost import XGBRegressor
from flaml import AutoML
from joblib import Parallel, delayed
from datetime import datetime
import argparse
import hashlib
import schedule
import time

# === CONFIG ===
CRYPTO_LIST = [
    "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "MATIC", "LTC"
]  # Lista redusă pentru testare rapidă
SEQ_LENGTH = 30
EPOCHS = 10
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
NUM_LAYERS = 2
MODEL_DIR = "models"
META_FILE = os.path.join(MODEL_DIR, "metadata.json")
LOG_FILE = os.path.join(MODEL_DIR, "training_log.csv")
N_JOBS = 1
ROLLING_WINDOW_DAYS = 90
ENABLE_STACKING = True
NOISE_LEVEL = 0.001
PATIENCE = 5
DRIFT_THRESHOLD = 0.2
EPSILON = 1e-8
TIMESNET_EPOCHS = 5
FLAML_TIME_BUDGET = 10
RANDOM_STATE = 42

# === LOGGING ===
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# === DEVICE ===
device = torch.device("cpu")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# === MODELE ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# === FUNCȚII ===
def download_data(symbol, years=10):
    try:
        logging.info(f"Încep descărcarea datelor pentru {symbol}...")
        df = yf.download(f"{symbol}-USD", period=f"{years}y").dropna()
        logging.info(f"Date descărcate pentru {symbol}.")

        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["EMA_10"] = df["Close"].ewm(span=10).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / (loss + EPSILON))))
        df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
        df["BB_upper"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
        df["BB_lower"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

        return df.dropna()
    except Exception as e:
        logging.error(f"Eroare la descărcarea datelor pentru {symbol}: {e}")
        return pd.DataFrame()

def create_sequences(data, target, seq_length, noise_level=NOISE_LEVEL):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        x += np.random.normal(0, noise_level, x.shape)
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(X_train, y_train, input_size, model_type="LSTM", epochs=EPOCHS):
    if model_type == "LSTM":
        model = LSTMModel(input_size).to(device)
    else:
        raise ValueError("Tip de model necunoscut")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        model.train()
        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
        output = model(X_t)
        loss = criterion(output, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        losses.append(loss.item())
        logging.debug(f"Epoca {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model, losses

def run_pipeline(symbol):
    logging.info(f"Pornesc pipeline-ul pentru {symbol}...")
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    df = download_data(symbol)
    if df.empty:
        return

    df = df[df.index > (df.index[-1] - pd.Timedelta(days=ROLLING_WINDOW_DAYS))]
    features = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "EMA_10", "RSI", "MACD", "BB_upper", "BB_lower", "OBV"]
    target = df["Close"].values

    scaler_X = MinMaxScaler()
    scaled_features = scaler_X.fit_transform(df[features])
    scaler_y = MinMaxScaler()
    scaled_target = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()

    selector = SelectKBest(score_func=f_regression, k='all')
    try:
        selector.fit(scaled_features, scaled_target)
        selected_features = [features[i] for i in selector.get_support(indices=True)]
        selected = selector.transform(scaled_features)
    except ValueError as e:
        logging.error(f"Eroare la selectarea caracteristicilor: {e}")
        selected = scaled_features
        selected_features = features

    X, y = create_sequences(selected, scaled_target, SEQ_LENGTH)
    tscv = TimeSeriesSplit(n_splits=2)

    for train_idx, test_idx in tscv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model, losses = train_model(X_train, y_train, X.shape[2], model_type="LSTM")
        model.eval()
        with torch.no_grad():
            preds_lstm = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()

        logging.info(f"Evaluare pentru {symbol} finalizată.")

        # === Salvare fișiere pentru bot ===
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"lstm_{symbol}.pt"))
        with open(os.path.join(MODEL_DIR, f"scaler_X_{symbol}.pkl"), "wb") as f:
            pickle.dump(scaler_X, f)
        with open(os.path.join(MODEL_DIR, f"scaler_y_{symbol}.pkl"), "wb") as f:
            pickle.dump(scaler_y, f)
        with open(os.path.join(MODEL_DIR, f"meta_{symbol}.json"), "w") as f:
            json.dump({"features": selected_features}, f, indent=2)

        logging.info(f"✅ Fișiere salvate pentru {symbol}")

def main():
    for symbol in CRYPTO_LIST:
        try:
            run_pipeline(symbol)
        except Exception as e:
            logging.error(f"Eroare la rularea pipeline-ului pentru {symbol}: {e}")

if __name__ == "__main__":
    main()
