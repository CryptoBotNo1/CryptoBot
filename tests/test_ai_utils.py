# tests/test_ai_utils.py
import pytest
import pytest_asyncio # Chiar dacă funcțiile din ai_utils sunt sincrone, unele fixtures pot fi async
import os
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importăm modulele/funcțiile testate și config
import config
from ai_utils import LSTMModel, load_model_and_scaler, predict_sequence, loaded_models
# Atenție: Vom avea nevoie să mock-uim fetch_latest_data, deci îl importăm direct
from data_utils import fetch_latest_data

# --- Fixture pentru creare fișiere model/scaler/meta temporare ---
@pytest.fixture
def mock_model_files(tmp_path, monkeypatch):
    """
    Creează directoare și fișiere AI temporare (valide și invalide)
    și configurează config.MODEL_DIR să pointeze spre ele.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # Modifică temporar calea către modele
    monkeypatch.setattr(config, 'MODEL_DIR', str(models_dir))

    # --- Fișiere pentru un simbol VALID ---
    valid_symbol = "VALID"
    features_valid = ["Close", "Volume"]
    # Meta valid
    meta_valid_path = models_dir / f"meta_{valid_symbol}.json"
    with open(meta_valid_path, "w") as f:
        json.dump({"features": features_valid}, f)
    # Scaler valid (creăm obiecte dummy și le salvăm cu pickle)
    scaler_x_valid = MinMaxScaler()
    scaler_y_valid = MinMaxScaler()
    # Simulare fit pe date dummy pentru a avea atribute interne
    scaler_x_valid.fit(np.random.rand(10, len(features_valid)))
    scaler_y_valid.fit(np.random.rand(10, 1))
    with open(models_dir / f"scaler_X_{valid_symbol}.pkl", "wb") as f: pickle.dump(scaler_x_valid, f)
    with open(models_dir / f"scaler_y_{valid_symbol}.pkl", "wb") as f: pickle.dump(scaler_y_valid, f)
    # Model valid (creăm un model dummy și salvăm state_dict gol)
    model_valid = LSTMModel(input_size=len(features_valid))
    torch.save(model_valid.state_dict(), models_dir / f"lstm_{valid_symbol}.pt")

    # --- Fișiere pentru un simbol cu META INVALID ---
    invalid_meta_symbol = "INVALIDMETA"
    # Meta invalid (lipsește cheia 'features')
    meta_invalid_path = models_dir / f"meta_{invalid_meta_symbol}.json"
    with open(meta_invalid_path, "w") as f: json.dump({"other_key": "value"}, f)
    # Creăm și celelalte fișiere pentru a nu eșua din cauza lipsei lor
    with open(models_dir / f"scaler_X_{invalid_meta_symbol}.pkl", "wb") as f: pickle.dump(scaler_x_valid, f)
    with open(models_dir / f"scaler_y_{invalid_meta_symbol}.pkl", "wb") as f: pickle.dump(scaler_y_valid, f)
    torch.save(model_valid.state_dict(), models_dir / f"lstm_{invalid_meta_symbol}.pt")

    # --- Fișiere pentru un simbol cu fișier LIPSĂ (ex: modelul .pt) ---
    missing_file_symbol = "MISSING"
    meta_missing_path = models_dir / f"meta_{missing_file_symbol}.json"
    with open(meta_missing_path, "w") as f: json.dump({"features": features_valid}, f)
    with open(models_dir / f"scaler_X_{missing_file_symbol}.pkl", "wb") as f: pickle.dump(scaler_x_valid, f)
    with open(models_dir / f"scaler_y_{missing_file_symbol}.pkl", "wb") as f: pickle.dump(scaler_y_valid, f)
    # NU creăm fișierul lstm_MISSING.pt

    # Returnăm informații utile, dacă e nevoie (nu e obligatoriu pentru autouse=False)
    # return {"models_dir": models_dir, "valid_symbol": valid_symbol}

# --- Teste pentru load_model_and_scaler ---

def test_load_model_valid(mock_model_files):
    """Verifică încărcarea reușită pentru fișiere valide."""
    result = load_model_and_scaler("VALID")
    assert result is not None
    model, scaler_x, scaler_y, meta = result
    assert isinstance(model, LSTMModel)
    assert isinstance(scaler_x, MinMaxScaler)
    assert isinstance(scaler_y, MinMaxScaler)
    assert "features" in meta
    assert meta["features"] == ["Close", "Volume"]

def test_load_model_invalid_meta(mock_model_files):
    """Verifică eșecul la încărcare dacă meta.json e invalid."""
    result = load_model_and_scaler("INVALIDMETA")
    assert result is None

def test_load_model_missing_file(mock_model_files):
    """Verifică eșecul la încărcare dacă lipsește un fișier (ex: model.pt)."""
    result = load_model_and_scaler("MISSING")
    assert result is None

def test_load_model_nonexistent_symbol(mock_model_files):
    """Verifică eșecul la încărcare pentru un simbol care nu are fișiere."""
    result = load_model_and_scaler("NOSUCHSYMBOL")
    assert result is None

# --- Teste pentru predict_sequence ---

# Definim niște date de test reutilizabile
TEST_SYMBOL = "TESTBTC"
TEST_FEATURES = ["Close", "Volume"]
TEST_DAYS = 5

@pytest.fixture
def mock_loaded_models_fixture(monkeypatch):
    """Fixture care populează temporar dicționarul loaded_models."""
    # Creăm obiecte dummy pe care să le punem în loaded_models
    dummy_model = LSTMModel(input_size=len(TEST_FEATURES))
    dummy_scaler_x = MinMaxScaler().fit(np.random.rand(10, len(TEST_FEATURES)))
    dummy_scaler_y = MinMaxScaler().fit(np.random.rand(10, 1))
    dummy_meta = {"features": TEST_FEATURES}
    dummy_data = (dummy_model, dummy_scaler_x, dummy_scaler_y, dummy_meta)

    # Folosim monkeypatch pentru a modifica dicționarul global temporar
    # Atenție: Trebuie să știm calea exactă a variabilei globale
    monkeypatch.setitem(loaded_models, TEST_SYMBOL, dummy_data)
    # Putem șterge alte chei dacă vrem izolare completă
    # monkeypatch.setattr(ai_utils, "loaded_models", {TEST_SYMBOL: dummy_data})

    yield # Rulează testul

    # Curățenie (opțională, monkeypatch face rollback de obicei)
    if TEST_SYMBOL in loaded_models:
         del loaded_models[TEST_SYMBOL]


def test_predict_sequence_model_not_loaded():
    """Testează cazul în care modelul nu e preîncărcat."""
    # Ne asigurăm că nu există cheia (deși fixture nu rulează aici)
    if "NOSUCHMODEL" in loaded_models: del loaded_models["NOSUCHMODEL"]
    result = predict_sequence("NOSUCHMODEL", TEST_DAYS)
    assert result == []

def test_predict_sequence_insufficient_data(mocker, mock_loaded_models_fixture):
    """Testează cazul în care fetch_latest_data returnează date insuficiente."""
    # Mock-uim fetch_latest_data să returneze un DataFrame gol
    mocker.patch("ai_utils.fetch_latest_data", return_value=pd.DataFrame())
    result = predict_sequence(TEST_SYMBOL, TEST_DAYS)
    assert result == []

    # Mock-uim fetch_latest_data să returneze un DataFrame cu prea puține rânduri
    dummy_df_short = pd.DataFrame(np.random.rand(config.SEQUENCE_LENGTH - 1, len(TEST_FEATURES)), columns=TEST_FEATURES)
    mocker.patch("ai_utils.fetch_latest_data", return_value=dummy_df_short)
    result = predict_sequence(TEST_SYMBOL, TEST_DAYS)
    assert result == []

def test_predict_sequence_valid(mocker, mock_loaded_models_fixture):
    """Testează rularea normală a predicției (fără a verifica acuratețea)."""
    # Mock-uim fetch_latest_data să returneze date valide suficiente
    dummy_df_valid = pd.DataFrame(np.random.rand(config.SEQUENCE_LENGTH, len(TEST_FEATURES)), columns=TEST_FEATURES)
    mocker.patch("ai_utils.fetch_latest_data", return_value=dummy_df_valid)

    result = predict_sequence(TEST_SYMBOL, TEST_DAYS)

    assert isinstance(result, list) # Verifică tipul returnat
    assert len(result) == TEST_DAYS # Verifică dacă avem predicții pt nr. corect de zile
    # Verifică dacă toate elementele sunt numere (float)
    assert all(isinstance(p, (float, np.floating)) for p in result)