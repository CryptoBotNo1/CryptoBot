# tests/test_data_utils.py
import pytest
import pytest_asyncio
import os
import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
# Corecție: Re-adăugăm importul pentru MagicMock (și AsyncMock)
from unittest.mock import AsyncMock, MagicMock
import feedparser
from aioresponses import aioresponses # Importăm aioresponses

# Importăm modulele/funcțiile testate și config
import config
from data_utils import fetch_latest_data, fetch_price_for_portfolio, parse_rss_feed

# --- Teste pentru fetch_latest_data ---

# Fixture pentru a crea un DataFrame dummy similar cu cel de la yfinance
@pytest.fixture
def sample_yf_df():
    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    data = {
        'Open': np.random.rand(180) * 100,
        'High': np.random.rand(180) * 110,
        'Low': np.random.rand(180) * 90,
        'Close': np.random.rand(180) * 100,
        'Volume': np.random.randint(10000, 1000000, size=180)
    }
    return pd.DataFrame(data, index=dates)

def test_fetch_data_calculates_indicators(mocker, sample_yf_df):
    """Verifică dacă indicatorii tehnici sunt calculați (verificare simplă rulare)."""
    mocker.patch("data_utils.yf.download", return_value=sample_yf_df.copy())
    mocker.patch("os.path.exists", return_value=False)
    df_result = fetch_latest_data("BTC", ["Close", "Volume"])

    assert not df_result.empty
    assert len(df_result) == config.SEQUENCE_LENGTH
    assert "Close" in df_result.columns

@pytest.mark.parametrize("cache_exists, cache_fresh, yf_should_be_called", [
    (False, True, True),   # Cache nu există -> yf.download e apelat
    (True, False, True),   # Cache există, dar e vechi -> yf.download e apelat
    (True, True, False),   # Cache există și e proaspăt -> yf.download NU e apelat
])
def test_fetch_data_caching_logic(mocker, tmp_path, monkeypatch, sample_yf_df, cache_exists, cache_fresh, yf_should_be_called):
    """Testează logica de caching (există, proaspăt, expirat)."""
    symbol = "CACHE_TEST"
    features = ["Close"]
    cache_file = tmp_path / f"cache_{symbol}.pkl"

    monkeypatch.setattr(config, 'CACHE_DIR', str(tmp_path))
    mock_os_path_exists = mocker.patch("os.path.exists", return_value=cache_exists)

    if cache_exists:
        try:
            with open(cache_file, "wb") as f: pickle.dump(sample_yf_df, f)
            now_ts = datetime.now().timestamp()
            mtime = now_ts - config.CACHE_EXPIRY_SECONDS / 2 if cache_fresh else now_ts - config.CACHE_EXPIRY_SECONDS * 2
            mocker.patch("os.path.getmtime", return_value=mtime)
        except Exception as e:
             pytest.fail(f"Setup error for existing cache: {e}")

    mock_yf_download = mocker.patch("data_utils.yf.download", return_value=sample_yf_df.copy())

    fetch_latest_data(symbol, features)

    mock_os_path_exists.assert_called_with(str(cache_file))

    if yf_should_be_called:
        mock_yf_download.assert_called_once()
    else:
        mock_yf_download.assert_not_called()

def test_fetch_data_handles_yf_error(mocker):
    """Verifică returnarea unui DataFrame gol dacă yf.download eșuează."""
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("data_utils.yf.download", return_value=pd.DataFrame()) # Simulare yf eșec

    result = fetch_latest_data("FAIL", ["Close"])
    assert result.empty

# --- Teste pentru fetch_price_for_portfolio (Rescrise cu aioresponses) ---

@pytest.mark.asyncio
async def test_fetch_price_success(): # Nu mai avem nevoie de mocker aici
    """Testează obținerea cu succes a prețului folosind aioresponses."""
    symbol = "BTC"
    amount = 0.5
    mock_price = "90000.00"
    expected_value = 45000.0
    expected_line_part = f"× ${float(mock_price):,.2f} = *${expected_value:,.2f}*"
    url = f"{config.BINANCE_API_BASE}/ticker/price?symbol={symbol}USDT"

    with aioresponses() as m:
        m.get(url, payload={"price": mock_price}, status=200)
        line, value = await fetch_price_for_portfolio(None, symbol, amount)

    assert value == pytest.approx(expected_value)
    assert expected_line_part in line

@pytest.mark.asyncio
async def test_fetch_price_api_error():
    """Testează gestionarea unei erori API (ex: status 404) cu aioresponses."""
    symbol = "ERR"
    amount = 1.0
    status_code = 404
    url = f"{config.BINANCE_API_BASE}/ticker/price?symbol={symbol}USDT"

    with aioresponses() as m:
        m.get(url, status=status_code)
        line, value = await fetch_price_for_portfolio(None, symbol, amount)

    assert value == 0.0
    assert f"Eroare preț ({status_code})" in line

@pytest.mark.asyncio
async def test_fetch_price_timeout():
    """Testează gestionarea unui TimeoutError cu aioresponses."""
    symbol = "TMOUT"
    amount = 1.0
    url = f"{config.BINANCE_API_BASE}/ticker/price?symbol={symbol}USDT"

    with aioresponses() as m:
        m.get(url, exception=asyncio.TimeoutError())
        line, value = await fetch_price_for_portfolio(None, symbol, amount)

    assert value == 0.0
    assert "Eroare (Timeout)" in line


# --- Teste pentru parse_rss_feed (Folosesc MagicMock importat acum) ---
@pytest.mark.asyncio
async def test_parse_rss_success(mocker):
    """Testează parsarea unui feed RSS valid."""
    feed_url = "http://valid.url/rss"; source_name = "ValidSource"
    # Acum MagicMock este definit
    mock_feed_data = MagicMock(); mock_feed_data.bozo = 0
    mock_entry1 = {"title": "Title 1", "link": "link1"}; mock_entry2 = {"title": "Title 2", "link": "link2"}
    mock_feed_data.entries = [mock_entry1, mock_entry2]
    mock_parse = mocker.patch("data_utils.feedparser.parse", return_value=mock_feed_data)
    result = await parse_rss_feed({"name": source_name, "url": feed_url})
    mock_parse.assert_called_once_with(feed_url, request_headers=mocker.ANY)
    assert result is not None; assert len(result.entries) == 2
    assert result.entries[0]['source_name'] == source_name; assert result.entries[1]['source_name'] == source_name

@pytest.mark.asyncio
async def test_parse_rss_bozo_error(mocker):
    """Testează gestionarea unei erori de parsare (bozo=1)."""
    feed_url = "http://invalid.url/rss"; source_name = "InvalidSource"
    # Acum MagicMock este definit
    mock_feed_data = MagicMock(); mock_feed_data.bozo = 1
    mock_feed_data.bozo_exception = Exception("Parsing failed badly")
    mock_parse = mocker.patch("data_utils.feedparser.parse", return_value=mock_feed_data)
    result = await parse_rss_feed({"name": source_name, "url": feed_url})
    mock_parse.assert_called_once(); assert result is None

@pytest.mark.asyncio
async def test_parse_rss_exception(mocker):
    """Testează gestionarea unei excepții în timpul parsării."""
    feed_url = "http://exception.url/rss"; source_name = "ExceptionSource"
    mock_parse = mocker.patch("data_utils.feedparser.parse", side_effect=Exception("Network error"))
    result = await parse_rss_feed({"name": source_name, "url": feed_url})
    mock_parse.assert_called_once(); assert result is None