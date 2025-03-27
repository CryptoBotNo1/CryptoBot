# ai_utils.py
import os
import logging
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Dict, Any, Tuple

# Importă configurările necesare și funcția de date
import config
from data_utils import fetch_latest_data # Atenție: import circular posibil dacă data_utils importă din ai_utils. Verificăm.

logger = logging.getLogger(__name__)

# Dicționar global pentru a stoca modelele preîncărcate
loaded_models: Dict[str, Tuple[nn.Module, MinMaxScaler, MinMaxScaler, Dict]] = {}

# --- Definiție Model LSTM ---
class LSTMModel(nn.Module):
    """Clasa modelului LSTM folosit pentru predicții."""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Folosim output-ul ultimului pas

# --- Funcții de Încărcare și Predicție ---

def load_model_and_scaler(symbol: str) -> Optional[Tuple[nn.Module, MinMaxScaler, MinMaxScaler, Dict]]:
    """Încarcă modelul AI, scalerele și metadatele pentru un simbol dat."""
    model_path = os.path.join(config.MODEL_DIR, f"lstm_{symbol}.pt")
    scaler_X_path = os.path.join(config.MODEL_DIR, f"scaler_X_{symbol}.pkl")
    scaler_y_path = os.path.join(config.MODEL_DIR, f"scaler_y_{symbol}.pkl")
    meta_path = os.path.join(config.MODEL_DIR, f"meta_{symbol}.json")

    if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path, meta_path]):
        logger.error(f"Fișiere model/scaler/meta lipsă pentru {symbol} în '{config.MODEL_DIR}'")
        return None
    try:
        with open(meta_path, "r") as f: meta = json.load(f)
        features = meta.get("features")
        if not features:
            logger.error(f"Fișier meta pentru {symbol} ('{meta_path}') nu conține 'features'.")
            return None

        with open(scaler_X_path, "rb") as f: scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f: scaler_y = pickle.load(f)

        model = LSTMModel(input_size=len(features))
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        logger.debug(f"Model, scalere și meta încărcate pentru {symbol}.") # Schimbat în debug pt a reduce zgomotul
        return model, scaler_X, scaler_y, meta

    except Exception as e:
        logger.error(f"Eroare la încărcarea modelului/scalerelor pentru {symbol}: {e}", exc_info=True)
        return None

def preload_models():
    """Funcție sincronă pentru a preîncărca toate modelele definite în config."""
    logger.info(">>> Începe preîncărcarea modelelor AI...")
    loaded_count = 0
    for symbol in config.CRYPTO_LIST:
        logger.debug(f"Încercare preîncărcare model pentru {symbol}...")
        load_result = load_model_and_scaler(symbol)
        if load_result:
            loaded_models[symbol] = load_result
            logger.info(f"Modelul pentru {symbol} a fost preîncărcat.")
            loaded_count += 1
        else:
            logger.error(f"!!! Eșec la preîncărcarea modelului pentru {symbol}. Predicțiile AI vor fi indisponibile.")
    logger.info(f">>> Preîncărcare modele AI finalizată. {loaded_count}/{len(config.CRYPTO_LIST)} modele încărcate.")

def predict_sequence(symbol: str, days: int) -> List[float]:
    """Generează predicții AI folosind modelul LSTM preîncărcat."""
    if symbol not in loaded_models:
        logger.error(f"Modelul pentru {symbol} nu este disponibil (nu a fost preîncărcat sau a eșuat).")
        return []

    model, scaler_X, scaler_y, meta = loaded_models[symbol]
    features = meta["features"]

    data = fetch_latest_data(symbol, features) # Folosește funcția importată
    if data.empty or len(data) < config.SEQUENCE_LENGTH:
        logger.error(f"Date insuficiente ({len(data)} rânduri) pentru predicție {symbol} după fetch.")
        return []

    try:
        scaled_input = scaler_X.transform(data.values)
        sequence = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)

        predictions_scaled = []
        current_sequence = sequence.clone()

        for _ in range(days):
            with torch.no_grad():
                prediction_scaled_next = model(current_sequence)
                predictions_scaled.append(prediction_scaled_next.item())

                next_input_features_scaled = current_sequence[:, -1:, :].clone()
                # Asumăm că 'Close' (sau feature-ul prezis) este primul (index 0)
                next_input_features_scaled[0, 0, 0] = prediction_scaled_next.item()
                current_sequence = torch.cat((current_sequence[:, 1:, :], next_input_features_scaled), dim=1)

        final_predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        logger.info(f"Predicții generate cu succes pentru {symbol} ({days} zile).")
        return final_predictions.tolist()

    except Exception as e:
        logger.error(f"Eroare în timpul generării predicțiilor pentru {symbol}: {e}", exc_info=True)
        return []