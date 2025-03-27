# data_utils.py
import os
import logging
import pickle
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio
import aiohttp
import feedparser
import time
# Modificat: Adăugăm Optional pentru tipul sesiunii
from typing import List, Optional, Dict, Tuple

# Importă configurările necesare
import config

logger = logging.getLogger(__name__)

# --- Funcție Preluare Date Istorice (yfinance) cu Cache ---
def fetch_latest_data(symbol: str, features: List[str]) -> pd.DataFrame:
    """
    Descarcă date yfinance (sau încarcă din cache) și calculează indicatorii.
    Returnează ultimele `config.SEQUENCE_LENGTH` rânduri cu `features` specificate.
    """
    cache_path = os.path.join(config.CACHE_DIR, f"cache_{symbol}.pkl")
    try:
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() < config.CACHE_EXPIRY_SECONDS:
                try:
                    with open(cache_path, "rb") as f: df_cached = pickle.load(f)
                    logger.debug(f"Date din cache utilizate pentru {symbol}")
                    if all(feature in df_cached.columns for feature in features):
                        return df_cached[features].tail(config.SEQUENCE_LENGTH).copy()
                    else:
                        logger.warning(f"Cache {symbol} incomplet (lipsesc features). Redescarcare.")
                except (pickle.UnpicklingError, EOFError) as e:
                    logger.error(f"Eroare la citirea cache-ului {symbol}: {e}. Fișierul va fi suprascris.")
                    # Continuă pentru a descărca date noi

        logger.info(f"Descărcare date noi yfinance pentru {symbol}...")
        df = yf.download(f"{symbol}-USD", period="180d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            logger.error(f"yf download gol pentru {symbol}-USD")
            return pd.DataFrame()

        # Calculează indicatori
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
        rolling_mean = df["Close"].rolling(window=20).mean()
        rolling_std = df["Close"].rolling(window=20).std()
        df["BB_upper"] = rolling_mean + 2 * rolling_std
        df["BB_lower"] = rolling_mean - 2 * rolling_std
        df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
        df = df.dropna()

        if len(df) < config.SEQUENCE_LENGTH:
            logger.error(f"Insuficiente date {symbol} ({len(df)}) după procesare pt secvența {config.SEQUENCE_LENGTH}.")
            return pd.DataFrame()

        try:
            with open(cache_path, "wb") as f: pickle.dump(df, f)
            logger.info(f"Date noi salvate în cache pentru {symbol}")
        except Exception as e:
             logger.error(f"Nu s-a putut salva cache-ul pentru {symbol}: {e}")

        return df[features].tail(config.SEQUENCE_LENGTH).copy()

    except Exception as e:
        logger.error(f"Eroare majoră la fetch_latest_data pentru {symbol}: {e}", exc_info=True)
        return pd.DataFrame()

# --- Funcție Preluare Preț Live (Binance) pentru Portofoliu (MODIFICATĂ) ---
async def fetch_price_for_portfolio(session: Optional[aiohttp.ClientSession], symbol: str, amount: float) -> Tuple[str, float]:
    """
    Obține prețul curent de pe Binance pentru un simbol și calculează valoarea deținerii.
    Creează o sesiune nouă dacă nu este furnizată una.
    """
    # Creează o sesiune locală DACA nu primește una externă
    # Folosim un context manager pentru a ne asigura că sesiunea e închisă
    local_session: Optional[aiohttp.ClientSession] = None
    if session is None:
        local_session = aiohttp.ClientSession()
        current_session = local_session # Folosim sesiunea locală
        logger.debug(f"fetch_price_for_portfolio: Created new aiohttp session for {symbol}")
    else:
        current_session = session # Folosim sesiunea primită

    try:
        url = f"{config.BINANCE_API_BASE}/ticker/price?symbol={symbol}USDT"
        # Acum folosim 'current_session' care fie a fost primită, fie a fost creată local
        async with current_session.get(url, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                price = float(data.get("price", 0))
                if price > 0:
                    total_value = price * amount
                    line = f"• {symbol}: {amount:.8f} × ${price:,.2f} = *${total_value:,.2f}*"
                    return line, total_value
                else:
                    logger.warning(f"Preț invalid (0) de la Binance pentru {symbol} în portfolio.")
                    return f"• {symbol}: {amount:.8f} × $? = Preț invalid (0)", 0.0
            else:
                logger.warning(f"Eroare API Binance ({response.status}) la preț portfolio {symbol}USDT.")
                return f"• {symbol}: {amount:.8f} × $? = Eroare preț ({response.status})", 0.0
    except asyncio.TimeoutError:
        logger.warning(f"Timeout Binance la preț portfolio {symbol}USDT.")
        return f"• {symbol}: {amount:.8f} × $? = Eroare (Timeout)", 0.0
    except aiohttp.ClientError as e:
        logger.error(f"Eroare rețea la preț portfolio {symbol}USDT: {e}")
        return f"• {symbol}: {amount:.8f} × $? = Eroare (Rețea)", 0.0
    except Exception as e:
        logger.error(f"Eroare necunoscută la fetch_price_for_portfolio pentru {symbol}: {e}", exc_info=True)
        return f"• {symbol}: {amount:.8f} × $? = Eroare generală", 0.0
    finally:
        # Închidem sesiunea DOAR dacă am creat-o local (local_session nu e None)
        if local_session:
            await local_session.close()
            logger.debug(f"fetch_price_for_portfolio: Closed internally created aiohttp session for {symbol}")


# --- Funcție Parsare Flux RSS ---
async def parse_rss_feed(feed_info: Dict[str, str]) -> Optional[feedparser.FeedParserDict]:
    """Descarcă și parsează asincron un flux RSS individual."""
    url = feed_info["url"]
    name = feed_info["name"]
    logger.info(f"Preluare RSS: {name} ({url})")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 Python CryptoBot'}
        feed_data = await asyncio.to_thread(feedparser.parse, url, request_headers=headers)

        if not feed_data or feed_data.bozo:
            bozo_exception = getattr(feed_data, 'bozo_exception', 'Necunoscută')
            logger.warning(f"Eroare parsare RSS {name}: {bozo_exception}")
            return None

        for entry in feed_data.entries: entry['source_name'] = name
        logger.info(f"Preluare RSS OK: {name} ({len(feed_data.entries)} intrări)")
        return feed_data

    except Exception as e:
        logger.error(f"Excepție la preluarea RSS pentru {name} ({url}): {e}", exc_info=True)
        return None