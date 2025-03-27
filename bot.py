# -*- coding: utf-8 -*-

# === IMPORTURI ===
import os
import logging
import json
import aiohttp
import pickle
import yfinance as yf
import matplotlib
matplotlib.use('Agg') # Set backend non-interactiv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from io import BytesIO
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from sklearn.preprocessing import MinMaxScaler
from matplotlib.dates import DateFormatter
import csv
import asyncio
import time # Pentru formatare data RSS
import feedparser # Pentru RSS
import re # Pentru re.sub în formatare știri RSS
from typing import List, Optional, Dict, Any, Tuple

# === ÎNCĂRCARE .env ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
COINSTATS_API_KEY = os.getenv("COINSTATS_API_KEY")

# === CONFIGURAȚII & CONSTANTE ===
MODEL_DIR = "models"
PORTFOLIO_FILE = "portfolio.csv"
CACHE_DIR = "cache"
CACHE_EXPIRY_SECONDS = 86400 # 24 ore
SEQUENCE_LENGTH = 60      # Lungimea secvenței pentru LSTM
TREND_DAYS = 7            # Zile pentru calcul trend (/trend, /summary)
CHART_DAYS = 30           # Zile pentru grafice (/grafic, /compare)
PREDICT_MAX_DAYS = 30     # Număr maxim zile pentru predicție
NEWS_COUNT = 7            # Număr știri de afișat în /news
BINANCE_API_BASE = "https://api.binance.com/api/v3"
COINSTATS_API_BASE = "https://openapi.coinstats.app/public/v1"

CRYPTO_SYMBOLS_CONFIG = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance Coin",
    "XRP": "XRP", "ADA": "Cardano", "SOL": "Solana",
    "DOGE": "Dogecoin", "DOT": "Polkadot", "MATIC": "Polygon",
    "LTC": "Litecoin"
}
CRYPTO_LIST = list(CRYPTO_SYMBOLS_CONFIG.keys())

# Lista cu fluxuri RSS gratuite
RSS_FEEDS = [
    {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"name": "CryptoSlate", "url": "https://cryptoslate.com/feed/"},
    {"name": "Decrypt", "url": "https://decrypt.co/feed"},
]

# Creare directoare dacă nu există
if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR); print(f"Directorul {MODEL_DIR} a fost creat.")

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING) # Suprimă loguri matplotlib
logger = logging.getLogger(__name__)

# === STOCARE MODELE PREÎNCĂRCATE ===
# Dicționar pentru a stoca modelele, scalerele și metadatele încărcate
# Structura: { "SIMBOL": (model, scaler_X, scaler_y, meta), ... }
loaded_models: Dict[str, Tuple[nn.Module, MinMaxScaler, MinMaxScaler, Dict]] = {}

# === MODEL AI (LSTM) ===
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Folosim doar output-ul ultimului pas temporal pentru predicție
        return self.fc(out[:, -1, :])

# === FUNCȚII PRELUARE DATE ===

def fetch_latest_data(symbol: str, features: List[str]) -> pd.DataFrame:
    """
    Descarcă date yfinance (sau încarcă din cache) și calculează indicatorii necesari.
    Returnează ultimele `SEQUENCE_LENGTH` rânduri cu `features` specificate.
    """
    cache_path = os.path.join(CACHE_DIR, f"cache_{symbol}.pkl")
    try:
        # Verifică cache
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() < CACHE_EXPIRY_SECONDS:
                with open(cache_path, "rb") as f: df_cached = pickle.load(f)
                logger.info(f"Date din cache pentru {symbol}")
                # Verifică dacă toate feature-urile necesare sunt în cache
                if all(feature in df_cached.columns for feature in features):
                    # Asigură-te că returnezi suficiente date, dar nu mai mult decât e necesar
                    return df_cached[features].tail(SEQUENCE_LENGTH).copy()
                else:
                    logger.warning(f"Cache {symbol} incomplet (lipsesc features). Redescarcare.")

        # Descarcă date noi
        logger.info(f"Descărcare date noi yfinance pentru {symbol}...")
        # Adăugăm auto_adjust=True
        df = yf.download(f"{symbol}-USD", period="180d", interval="1d", progress=False, auto_adjust=True) # Perioadă mai mare pt. a avea date după dropna
        if df.empty:
            logger.error(f"yf download gol pentru {symbol}-USD")
            return pd.DataFrame()

        # Calculează indicatori tehnici
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / (loss + 1e-8) # Adăugăm epsilon pentru a evita împărțirea la zero
        df["RSI"] = 100 - (100 / (1 + rs))
        df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
        rolling_mean = df["Close"].rolling(window=20).mean()
        rolling_std = df["Close"].rolling(window=20).std()
        df["BB_upper"] = rolling_mean + 2 * rolling_std
        df["BB_lower"] = rolling_mean - 2 * rolling_std
        # Calcul OBV corectat
        df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()

        df = df.dropna() # Elimină rândurile cu NaN generate de indicatori

        # Verifică dacă avem suficiente date *după* procesare
        if len(df) < SEQUENCE_LENGTH:
            logger.error(f"Insuficiente date {symbol} ({len(df)} rânduri) după procesare pentru secvența {SEQUENCE_LENGTH}.")
            return pd.DataFrame()

        # Salvează în cache (DataFrame complet procesat)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
            logger.info(f"Date salvate în cache pentru {symbol}")

        # Returnează doar ultimele rânduri necesare cu feature-urile specificate
        return df[features].tail(SEQUENCE_LENGTH).copy()

    except Exception as e:
        logger.error(f"Eroare majoră la fetch_latest_data pentru {symbol}: {e}", exc_info=True)
        return pd.DataFrame()

# === FUNCȚII UTILITARE ===
def is_valid_symbol(symbol: str) -> bool:
    """Verifică dacă simbolul este în lista de criptomonede suportate."""
    return symbol.upper() in CRYPTO_LIST

def is_valid_days(days_str: str) -> bool:
    """Verifică dacă numărul de zile este un întreg valid între 1 și PREDICT_MAX_DAYS."""
    try:
        days = int(days_str)
        return 1 <= days <= PREDICT_MAX_DAYS
    except ValueError:
        return False

# === FUNCȚII AI ===

def load_model_and_scaler(symbol: str) -> Optional[Tuple[nn.Module, MinMaxScaler, MinMaxScaler, Dict]]:
    """
    Încarcă modelul AI, scalerele și metadatele dintr-un director specific.
    Rulează sincron la pornire.
    """
    model_path = os.path.join(MODEL_DIR, f"lstm_{symbol}.pt")
    scaler_X_path = os.path.join(MODEL_DIR, f"scaler_X_{symbol}.pkl")
    scaler_y_path = os.path.join(MODEL_DIR, f"scaler_y_{symbol}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"meta_{symbol}.json")

    # Verifică existența fișierelor esențiale
    if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path, meta_path]):
        logger.error(f"Fișiere model/scaler/meta lipsă pentru {symbol} în directorul '{MODEL_DIR}'")
        return None
    try:
        # Încarcă metadatele (features)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        features = meta.get("features")
        if not features:
            logger.error(f"Fișierul meta pentru {symbol} ('{meta_path}') nu conține cheia 'features'.")
            return None

        # Încarcă scalerele
        with open(scaler_X_path, "rb") as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)

        # Inițializează și încarcă modelul
        model = LSTMModel(input_size=len(features))
        # Încarcă starea modelului (asigură compatibilitatea CPU/GPU)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval() # Setează modelul în modul de evaluare

        logger.info(f"Model, scalere și meta încărcate cu succes pentru {symbol}.")
        return model, scaler_X, scaler_y, meta

    except FileNotFoundError as fnf_error:
         logger.error(f"Eroare la încărcare fișier pentru {symbol}: {fnf_error}")
         return None
    except pickle.UnpicklingError as pickle_error:
         logger.error(f"Eroare la deserializarea scaler-ului pentru {symbol}: {pickle_error}")
         return None
    except json.JSONDecodeError as json_error:
         logger.error(f"Eroare la citirea fișierului meta JSON pentru {symbol}: {json_error}")
         return None
    except Exception as e:
        # Captură generală pentru alte erori neprevăzute
        logger.error(f"Eroare necunoscută la încărcarea modelului {symbol}: {e}", exc_info=True)
        return None

def preload_models():
    """Funcție sincronă pentru a încărca toate modelele la pornirea botului."""
    logger.info(">>> Începe preîncărcarea modelelor AI...")
    for symbol in CRYPTO_LIST:
        logger.info(f"Încercare preîncărcare model pentru {symbol}...")
        load_result = load_model_and_scaler(symbol)
        if load_result:
            loaded_models[symbol] = load_result
            logger.info(f"Modelul pentru {symbol} a fost preîncărcat.")
        else:
            logger.error(f"!!! Eșec la preîncărcarea modelului pentru {symbol}. Predicțiile AI pentru acesta vor fi indisponibile.")
    logger.info(">>> Preîncărcarea modelelor AI finalizată.")


def predict_sequence(symbol: str, days: int) -> List[float]:
    """
    Generează predicții AI folosind modelul LSTM preîncărcat.
    """
    # Verifică dacă modelul pentru simbolul cerut a fost preîncărcat
    if symbol not in loaded_models:
        logger.error(f"Modelul pentru {symbol} nu este disponibil (nu a fost preîncărcat sau a eșuat la încărcare).")
        return []

    model, scaler_X, scaler_y, meta = loaded_models[symbol]
    features = meta["features"]

    # Obține datele istorice recente
    data = fetch_latest_data(symbol, features)
    if data.empty or len(data) < SEQUENCE_LENGTH:
        logger.error(f"Date insuficiente ({len(data)} rânduri) pentru predicție {symbol} după fetch.")
        return []

    try:
        # Scalează datele de intrare folosind scaler-ul preîncărcat
        scaled_input = scaler_X.transform(data.values)
        # Transformă în tensor PyTorch și adaugă dimensiunea batch-ului (1)
        sequence = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)

        predictions_scaled = []
        current_sequence = sequence.clone() # Folosim o copie pentru a nu modifica originalul

        # Generează predicții iterative pentru numărul de zile specificat
        for _ in range(days):
            with torch.no_grad(): # Dezactivăm calculul gradienților pentru inferență
                # Obține predicția pentru următorul pas
                prediction_scaled_next = model(current_sequence)
                predictions_scaled.append(prediction_scaled_next.item())

                # --- Pregătire secvență pentru următoarea iterație ---
                # Creează un nou tensor pentru următorul pas temporal.
                # Aici folosim o abordare simplă: repetăm ultimul vector de feature-uri cunoscut.
                next_input_features_scaled = current_sequence[:, -1:, :].clone()
                # Actualizăm valoarea 'Close' (sau oricare e prima coloană - prezisă) cu predicția făcută
                # Presupunem că 'Close' este primul feature (index 0)
                next_input_features_scaled[0, 0, 0] = prediction_scaled_next.item() # Asumând Close e primul

                # Construiește noua secvență: elimină primul pas temporal și adaugă noul pas la sfârșit
                current_sequence = torch.cat((current_sequence[:, 1:, :], next_input_features_scaled), dim=1)


        # Inversează scalarea pentru a obține valorile predicțiilor în USD
        final_predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

        logger.info(f"Predicții generate cu succes pentru {symbol} ({days} zile).")
        return final_predictions.tolist()

    except Exception as e:
        logger.error(f"Eroare în timpul generării predicțiilor pentru {symbol}: {e}", exc_info=True)
        return []

# === COMENZI TELEGRAM ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mesaj de bun venit."""
    await update.message.reply_text("👋 Salut! Sunt botul tău AI pentru crypto.\nFolosește /help pentru lista completă de comenzi.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Afișează comenzile disponibile."""
    user_id = update.effective_user.id
    logger.info(f"/help command requested by user {user_id}")
    try:
        symbols_str = ', '.join(CRYPTO_LIST)
        help_text = f"""📋 *Comenzi disponibile:*

/start - Pornește botul
/help - Afișează acest mesaj

*Predicții & Analiză:*
`/predict SIMBOL ZILE` - Predicție preț AI (Ex: `/predict BTC 7`)
`/predict_chart SIMBOL ZILE` - Grafic predicție AI (Ex: `/predict_chart ETH 5`)
`/summary SIMBOL` - Sumar zilnic (preț, trend {TREND_DAYS}z, pred 1z) (Ex: `/summary SOL`)
`/trend SIMBOL` - Analiză trend ({TREND_DAYS} zile) (Ex: `/trend ADA`)
`/grafic SIMBOL` - Grafic preț ({CHART_DAYS} zile) (Ex: `/grafic BNB`)
`/compare SIMBOL1 SIMBOL2` - Compară prețurile ({CHART_DAYS} zile) (Ex: `/compare BTC ETH`)

*Informații & Știri:*
`/crypto SIMBOL` - Preț live și date 24h (Ex: `/crypto XRP`)
`/news` - Ultimele {NEWS_COUNT} știri crypto din mai multe surse (RSS)

*Portofoliu:*
`/portfolio SIMBOL CANTITATE` - Adaugă/Actualizează (Ex: `/portfolio DOT 15.5`, `/portfolio BTC 0` pt ștergere)
`/myportfolio` - Afișează portofoliul tău

*Simboluri suportate:* {symbols_str}
*Nr. zile predicție:* 1-{PREDICT_MAX_DAYS}
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")
        logger.info(f"Help message sent successfully to user {user_id}")
    except Exception as e:
        logger.error(f"Error sending /help message to user {user_id}: {e}", exc_info=True)
        await update.message.reply_text("❌ Oops! A apărut o eroare la afișarea comenzilor.")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generează predicții textuale AI."""
    if len(context.args) != 2:
        await update.message.reply_text(f"❌ Format incorect. Exemplu: `/predict BTC 7`")
        return

    symbol = context.args[0].upper()
    days_str = context.args[1]

    if not is_valid_symbol(symbol):
        await update.message.reply_text(f"❌ Simbolul '{symbol}' nu este suportat. Folosește /help pentru listă.")
        return
    if not is_valid_days(days_str):
        await update.message.reply_text(f"❌ Număr de zile invalid. Trebuie să fie între 1 și {PREDICT_MAX_DAYS}.")
        return

    days = int(days_str)
    await update.message.reply_text(f"🔄 Generare predicție AI pentru {symbol} ({days} zile)...")

    try:
        predictions = predict_sequence(symbol, days)
        if not predictions:
            # predict_sequence deja loghează eroarea internă
            await update.message.reply_text(f"❌ Nu s-a putut genera predicția pentru {symbol}. Verifică log-urile sau încearcă mai târziu.")
            return

        # Creează datele pentru afișare
        start_date = datetime.now().date() + timedelta(days=1)
        prediction_dates = pd.date_range(start_date, periods=days)

        # Formatează textul predicțiilor folosind Markdown
        prediction_text_lines = []
        for date, price in zip(prediction_dates, predictions):
            prediction_text_lines.append(f"📅 {date.strftime('%Y-%m-%d')}: *{price:,.2f}* USD")

        response_message = f"📊 *Predicții AI {symbol}:*\n" + "\n".join(prediction_text_lines)
        await update.message.reply_text(response_message, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Eroare neașteptată în handler-ul /predict pentru {symbol}: {e}", exc_info=True)
        await update.message.reply_text("❌ A apărut o eroare neașteptată la generarea predicției.")


async def predict_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generează graficul predicțiilor AI."""
    if len(context.args) != 2:
        await update.message.reply_text(f"❌ Format incorect. Exemplu: `/predict_chart ETH 5`")
        return

    symbol = context.args[0].upper()
    days_str = context.args[1]

    if not is_valid_symbol(symbol):
        await update.message.reply_text(f"❌ Simbolul '{symbol}' nu este suportat.")
        return
    if not is_valid_days(days_str):
        await update.message.reply_text(f"❌ Număr de zile invalid (1-{PREDICT_MAX_DAYS}).")
        return

    days = int(days_str)
    await update.message.reply_text(f"🔄 Generare grafic AI pentru {symbol} ({days} zile)...")

    try:
        predictions = predict_sequence(symbol, days)
        if not predictions:
            await update.message.reply_text(f"❌ Eroare la obținerea datelor de predicție pentru graficul {symbol}.")
            return

        start_date = datetime.now().date() + timedelta(days=1)
        prediction_dates = pd.date_range(start_date, periods=days)

        # --- Plotare ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(prediction_dates, predictions, marker='o', linestyle='-', color='blue', label='Predicție AI')

        ax.set_title(f"Predicție AI Preț {symbol} - Următoarele {days} zile")
        ax.set_xlabel("Dată")
        ax.set_ylabel("Preț (USD)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Formatare axa X pentru lizibilitate
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout() # Ajustează layout-ul pentru a preveni suprapunerile

        # Salvare grafic în buffer de memorie
        buf = BytesIO()
        plt.savefig(buf, format='PNG')
        plt.close(fig) # Închide figura pentru a elibera memoria
        buf.seek(0) # Resetează poziția buffer-ului la început

        # Trimite fotografia
        await update.message.reply_photo(
            photo=InputFile(buf, filename=f"{symbol}_predict_chart_{days}d.png"),
            caption=f"Grafic predicție AI {symbol} ({days} zile)"
        )

    except Exception as e:
        logger.error(f"Eroare la generarea /predict_chart pentru {symbol}: {e}", exc_info=True)
        # Închide figura în caz de eroare, dacă a fost creată
        try: plt.close(fig)
        except NameError: pass # fig nu a fost definit dacă eroarea a apărut înainte
        await update.message.reply_text("❌ A apărut o eroare la generarea graficului de predicție.")


# --- Funcții RSS (preluate din codul anterior, par ok) ---
async def parse_rss_feed(feed_info: Dict[str, str]) -> Optional[feedparser.FeedParserDict]:
    """Descarcă și parsează asincron un flux RSS."""
    url = feed_info["url"]
    name = feed_info["name"]
    logger.info(f"Preluare RSS: {name} ({url})")
    try:
        # Folosim User-Agent pentru a evita blocarea
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Python CryptoBot'}
        # Rulează funcția blocantă feedparser.parse într-un thread separat
        feed_data = await asyncio.to_thread(feedparser.parse, url, request_headers=headers)

        # feedparser returnează feed_data.bozo = 1 dacă sunt erori de parsare
        if not feed_data or feed_data.bozo:
            bozo_exception = getattr(feed_data, 'bozo_exception', 'Necunoscută')
            logger.warning(f"Eroare parsare RSS {name}: {bozo_exception}")
            return None

        # Adaugă numele sursei la fiecare intrare pentru referință ulterioară
        for entry in feed_data.entries:
            entry['source_name'] = name
        logger.info(f"Preluare RSS OK: {name} ({len(feed_data.entries)} intrări)")
        return feed_data

    except Exception as e:
        logger.error(f"Excepție la preluarea RSS pentru {name} ({url}): {e}", exc_info=True)
        return None

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obține, combină, sortează și afișează știri din multiple fluxuri RSS."""
    await update.message.reply_text(f"🔄 Căutare ultimelor {NEWS_COUNT} știri crypto din multiple surse (RSS)...")

    # Preluare fluxuri în paralel folosind asyncio.gather
    tasks = [parse_rss_feed(feed_info) for feed_info in RSS_FEEDS]
    results = await asyncio.gather(*tasks)

    all_entries = []
    successful_feeds = 0
    for feed_data in results:
        if feed_data and feed_data.entries:
            all_entries.extend(feed_data.entries)
            successful_feeds += 1

    if not all_entries:
        logger.warning("Nicio intrare RSS nu a putut fi colectată.")
        await update.message.reply_text("❌ Nu s-au putut obține știri din sursele RSS momentan.")
        return

    logger.info(f"Intrări brute colectate: {len(all_entries)} din {successful_feeds}/{len(RSS_FEEDS)} surse RSS.")

    # Deduplicare pe baza titlului (ignorând majuscule/minuscule și spații extra) și sortare
    unique_entries_dict: Dict[str, Dict] = {}
    for entry in all_entries:
        title = entry.get('title', '').strip().lower()
        if title: # Ignoră intrările fără titlu
            # Folosește data publicării parsată sau un fallback la Epoch pentru sortare
            published_time = entry.get('published_parsed', None)
            # Convertim struct_time în timestamp float pentru sortare simplă, fallback la 0
            entry_timestamp = time.mktime(published_time) if published_time else 0
            entry['published_timestamp'] = entry_timestamp # Stocăm pt sortare

            # Dacă titlul nu e văzut sau intrarea curentă e mai nouă decât cea existentă cu același titlu
            if title not in unique_entries_dict or entry_timestamp > unique_entries_dict[title].get('published_timestamp', 0):
                 unique_entries_dict[title] = entry

    unique_entries = list(unique_entries_dict.values())
    # Sortare descrescătoare după timestamp (cele mai noi primele)
    unique_entries.sort(key=lambda x: x.get('published_timestamp', 0), reverse=True)

    logger.info(f"Intrări unice după deduplicare și sortare: {len(unique_entries)}")

    # Formatare top N știri pentru afișare (folosind HTML pentru linkuri)
    texts_to_send = []
    for i, entry in enumerate(unique_entries[:NEWS_COUNT]):
        title = entry.get('title', 'Fără titlu')
        link = entry.get('link', '#') # Link-ul articolului
        source_name = entry.get('source_name', 'Sursă Necunoscută')
        published_timestamp = entry.get('published_timestamp', 0)

        published_time_str = "dată necunoscută"
        if published_timestamp > 0:
            try:
                # Formatează data și ora locală
                published_dt = datetime.fromtimestamp(published_timestamp)
                published_time_str = published_dt.strftime('%d %b %H:%M') # Ex: 27 Mar 10:58
            except Exception as date_e:
                logger.warning(f"Eroare formatare dată pentru știre '{title}': {date_e}")

        # Escape HTML special characters in title and source name to prevent injection
        safe_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        safe_source = source_name.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Formatare linie știre
        texts_to_send.append(f"{i+1}. <a href='{link}'>{safe_title}</a>\n    <i>({safe_source} - {published_time_str})</i>")

    if not texts_to_send:
        await update.message.reply_text("⚠️ Nu s-au găsit știri relevante după filtrare.")
        return

    # Construiește mesajul final și trimite-l
    final_message = f"📰 <b>Top {len(texts_to_send)} Știri Crypto Recente (RSS):</b>\n\n" + "\n\n".join(texts_to_send)
    await update.message.reply_text(
        final_message,
        parse_mode="HTML",
        disable_web_page_preview=True # Dezactivează preview-ul link-urilor pentru un mesaj mai curat
    )

# --- Sfârșit funcții RSS ---


async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generează sumar zilnic: preț live, trend 7z, predicție 1z."""
    if len(context.args) != 1:
        await update.message.reply_text("❌ Format incorect. Exemplu: `/summary BTC`")
        return

    symbol = context.args[0].upper()
    if not is_valid_symbol(symbol):
        await update.message.reply_text(f"❌ Simbolul '{symbol}' nu este suportat.")
        return

    await update.message.reply_text(f"🔄 Generare sumar pentru {symbol}...")

    live_price_str = "N/A"
    trend_msg = "N/A"
    pred_msg = "N/A"

    async with aiohttp.ClientSession() as session:
        # 1. Preț live (Binance)
        try:
            url = f"{BINANCE_API_BASE}/ticker/price?symbol={symbol}USDT"
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    live_price = float(data.get('price', 0))
                    if live_price > 0:
                        live_price_str = f"*{live_price:,.2f}* $"
                    else:
                        live_price_str = "Preț invalid (0)"
                elif response.status == 400: # Simbol invalid pentru Binance
                     live_price_str = f"Indisponibil ({symbol}USDT negăsit)"
                else:
                    logger.warning(f"Binance API /ticker/price {symbol} status {response.status}")
                    live_price_str = f"Eroare API ({response.status})"
        except asyncio.TimeoutError:
             logger.warning(f"Timeout la obținere preț live Binance pentru {symbol}")
             live_price_str = "Eroare (Timeout)"
        except aiohttp.ClientError as e:
            logger.error(f"Eroare rețea preț live {symbol}: {e}")
            live_price_str = "Eroare (Rețea)"
        except Exception as e:
            logger.error(f"Eroare necunoscută preț live {symbol}: {e}", exc_info=True)
            live_price_str = "Eroare (Necunoscută)"

    # 2. Trend (TREND_DAYS zile) (yfinance - sincron, dar rapid)
    try:
        logger.info(f"[Sumar Trend {symbol}] Calcul trend...")
        # Perioada trebuie să fie TREND_DAYS + 1 pentru a avea capetele intervalului
        df_trend = yf.download(f"{symbol}-USD", period=f"{TREND_DAYS + 1}d", interval="1d", progress=False, auto_adjust=True)
        if not df_trend.empty and len(df_trend) >= 2:
            # Folosim .iloc pentru a accesa prima și ultima valoare 'Close'
            # *** MODIFICARE *** Adăugat .item()
            start_price = df_trend["Close"].iloc[0].item()
            end_price = df_trend["Close"].iloc[-1].item()
            logger.info(f"[Sumar Trend {symbol}] Prețuri OK: Start={start_price:.4f}, End={end_price:.4f}") # Logarea ar trebui să funcționeze acum

            change = end_price - start_price
            start_price_safe = start_price if abs(start_price) > 1e-9 else 1e-9 # Evită împărțirea la zero
            change_pct = (change / start_price_safe) * 100
            trend_symbol = "📈" if change >= 0 else "📉"
            trend_msg = f"{trend_symbol} Trend {TREND_DAYS}z: *{change:+.2f}$* ({change_pct:+.2f}%)"
        else:
            trend_msg = f"ℹ️ Date insuficiente trend ({TREND_DAYS}z)."
            logger.warning(f"[Sumar Trend {symbol}] Date insuficiente yfinance ({len(df_trend)} rânduri).")
    except Exception as e:
        logger.error(f"[Sumar Trend {symbol}] Eroare calcul: {e}", exc_info=True) # Loghează eroarea completă
        trend_msg = f"⚠️ Eroare calcul trend ({TREND_DAYS}z)." # Mesaj generic pentru user

    # 3. Predicție AI (1 zi) - Folosește modelul preîncărcat
    try:
        prediction = predict_sequence(symbol, 1) # Cere predicție pentru 1 zi
        if prediction:
            pred_msg = f"🔮 Predicție AI (1 zi): *{prediction[0]:,.2f}* $"
        else:
            # predict_sequence deja loghează eroarea internă
            pred_msg = "ℹ️ Predicție AI indisponibilă."
    except Exception as e:
        logger.error(f"Eroare necunoscută predicție sumar {symbol}: {e}", exc_info=True)
        pred_msg = "⚠️ Eroare generală predicție AI."


    # 4. Construiește mesajul final
    full_name = CRYPTO_SYMBOLS_CONFIG.get(symbol, symbol) # Numele complet al monedei
    message = f"""*🧠 Sumar zilnic: {full_name} ({symbol})*

💰 Preț live: {live_price_str}
{trend_msg}
{pred_msg}
"""
    await update.message.reply_text(message, parse_mode="Markdown")


async def crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Afișează date live de pe Binance (ticker 24hr)."""
    if len(context.args) != 1:
        await update.message.reply_text("❌ Format incorect. Exemplu: `/crypto BTC`")
        return

    symbol = context.args[0].upper()
    # Nu verificăm is_valid_symbol aici, lăsăm Binance să ne spună dacă există perechea USDT

    await update.message.reply_text(f"🔄 Căutare date live 24h pentru {symbol}USDT pe Binance...")

    try:
        url = f"{BINANCE_API_BASE}/ticker/24hr?symbol={symbol}USDT"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    full_name = CRYPTO_SYMBOLS_CONFIG.get(symbol, symbol) # Folosim numele cunoscut dacă există

                    # Extragem și formatăm datele cu valori default 0
                    price = float(data.get('lastPrice', 0))
                    high = float(data.get('highPrice', 0))
                    low = float(data.get('lowPrice', 0))
                    volume = float(data.get('volume', 0))
                    quote_volume = float(data.get('quoteVolume', 0))
                    change_pct = float(data.get('priceChangePercent', 0))

                    message = f"""📊 *Info Live 24h: {full_name} ({symbol})*
💵 Preț: *{price:,.2f} $*
📉 Low 24h: {low:,.2f} $
📈 High 24h: {high:,.2f} $
📊 Var 24h: {change_pct:+.2f}%
🔄 Vol 24h ({symbol}): {volume:,.2f}
💰 Vol 24h (USDT): {quote_volume:,.2f} $
_(Sursa: Binance)_"""
                    await update.message.reply_text(message, parse_mode="Markdown")

                elif response.status == 400:
                    # Eroare comună dacă simbolul nu există pe Binance cu perechea USDT
                    logger.warning(f"Binance API /ticker/24hr: Simbol {symbol}USDT probabil invalid (status 400).")
                    await update.message.reply_text(f"❌ Simbolul '{symbol}USDT' nu a fost găsit pe Binance.")
                else:
                    # Alte erori HTTP
                    logger.error(f"Eroare Binance API /ticker/24hr pentru {symbol}USDT. Status: {response.status}")
                    await update.message.reply_text(f"❌ Eroare la comunicarea cu Binance (Status: {response.status}). Încearcă mai târziu.")

    except asyncio.TimeoutError:
        logger.error(f"Timeout la requestul Binance /ticker/24hr pentru {symbol}USDT.")
        await update.message.reply_text("❌ Serviciul Binance nu răspunde (timeout). Încearcă mai târziu.")
    except aiohttp.ClientError as e:
        logger.error(f"Eroare de rețea la /crypto pentru {symbol}USDT: {e}")
        await update.message.reply_text("❌ Eroare de rețea la obținerea datelor live. Verifică conexiunea.")
    except Exception as e:
        logger.error(f"Eroare necunoscută în handler-ul /crypto pentru {symbol}USDT: {e}", exc_info=True)
        await update.message.reply_text("❌ A apărut o eroare neașteptată la obținerea datelor live.")


async def grafic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generează grafic preț pentru ultimele CHART_DAYS zile."""
    if len(context.args) != 1:
        await update.message.reply_text(f"❌ Format incorect. Exemplu: `/grafic BTC`")
        return
    symbol = context.args[0].upper()
    if not is_valid_symbol(symbol):
        await update.message.reply_text(f"❌ Simbolul '{symbol}' nu este suportat.")
        return

    await update.message.reply_text(f"🔄 Generare grafic preț {symbol} ({CHART_DAYS} zile)...")

    try:
        # Descarcă datele folosind yfinance
        df = yf.download(f"{symbol}-USD", period=f"{CHART_DAYS}d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            logger.warning(f"Nu s-au putut descărca date yfinance pentru /grafic {symbol}.")
            await update.message.reply_text(f"❌ Nu s-au putut obține datele istorice pentru {symbol}.")
            return

        # --- Plotare ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df["Close"], label=f"{symbol} Preț", color='purple')

        ax.set_title(f"Evoluție Preț {symbol} - Ultimele {CHART_DAYS} zile")
        ax.set_xlabel("Dată")
        ax.set_ylabel("Preț (USD)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Formatare axa X
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Salvare și trimitere
        buf = BytesIO()
        plt.savefig(buf, format='PNG')
        plt.close(fig) # Închide figura
        buf.seek(0)

        await update.message.reply_photo(
            photo=InputFile(buf, filename=f"{symbol}_{CHART_DAYS}d_chart.png"),
            caption=f"Grafic preț {symbol} - {CHART_DAYS} zile"
        )

    except Exception as e:
        logger.error(f"Eroare la generarea /grafic pentru {symbol}: {e}", exc_info=True)
        try: plt.close(fig)
        except NameError: pass
        await update.message.reply_text("❌ A apărut o eroare la generarea graficului.")


async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Calculează și afișează trendul prețului pentru ultimele TREND_DAYS zile."""
    if len(context.args) != 1:
        await update.message.reply_text(f"❌ Format incorect. Exemplu: `/trend BTC`")
        return
    symbol = context.args[0].upper()
    if not is_valid_symbol(symbol):
        await update.message.reply_text(f"❌ Simbolul '{symbol}' nu este suportat.")
        return

    await update.message.reply_text(f"🔄 Analiză trend {symbol} ({TREND_DAYS} zile)...")
    logger.info(f"Pornire calcul trend pentru {symbol}")

    try:
        # Descărcăm TREND_DAYS + 1 zile pentru a avea capetele intervalului
        df = yf.download(f"{symbol}-USD", period=f"{TREND_DAYS + 1}d", interval="1d", progress=False, auto_adjust=True)
        logger.info(f"Trend {symbol}: Date yfinance descărcate. Nr. rânduri: {len(df)}")

        if df.empty or len(df) < 2:
            logger.warning(f"Trend {symbol}: Date insuficiente ({len(df)} rânduri) returnate de yfinance.")
            await update.message.reply_text(f"❌ Date istorice insuficiente pentru a calcula trendul {symbol}.")
            return

        try:
            # Extrage prețul de start și de final
            # *** MODIFICARE *** Adăugat .item()
            start_price = df["Close"].iloc[0].item()
            end_price = df["Close"].iloc[-1].item()
            logger.info(f"Trend {symbol}: Prețuri extrase OK: Start={start_price:.4f}, End={end_price:.4f}") # Logarea ar trebui să funcționeze acum
        except IndexError as ie:
             logger.error(f"Trend {symbol}: Eroare la extragerea prețurilor din DataFrame (IndexError): {ie}", exc_info=True)
             await update.message.reply_text(f"❌ Eroare la procesarea datelor de preț pentru {symbol}.")
             return
        except Exception as e_item: # Captură eroare dacă .item() eșuează (ex: pe NaN)
            logger.error(f"Trend {symbol}: Eroare la extragerea valorii scalare cu .item(): {e_item}", exc_info=True)
            await update.message.reply_text(f"❌ Eroare la extragerea valorilor de preț pentru {symbol}.")
            return


        # Calcul variație și procentaj
        change = end_price - start_price
        # Protecție împărțire la zero dacă prețul de start e foarte mic sau zero
        start_price_safe = start_price if abs(start_price) > 1e-9 else 1e-9
        change_pct = (change / start_price_safe) * 100
        logger.info(f"Trend {symbol}: Calcul OK: Variație={change:.4f}, Procent={change_pct:.2f}%")

        # Determinare simbol și descriere trend
        trend_symbol = "📈" if change >= 0 else "📉"
        trend_desc = "ascendent" if change >= 0 else "descendent"

        # Construire mesaj final
        message = f"{trend_symbol} *Trend {symbol} ({TREND_DAYS} zile): {trend_desc}*\nVariație: *{change:+.2f}$* ({change_pct:+.2f}%)"
        logger.info(f"Trend {symbol}: Mesaj final pregătit.")

        await update.message.reply_text(message, parse_mode="Markdown")

    except Exception as e:
        # Captură generală pentru erori neașteptate (ex: probleme rețea yfinance)
        logger.error(f"Eroare majoră în handler-ul /trend pentru {symbol}: {e}", exc_info=True)
        await update.message.reply_text("❌ A apărut o eroare neașteptată la analiza trendului.")


async def compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generează grafic comparativ preț pentru două simboluri (CHART_DAYS zile)."""
    if len(context.args) != 2:
        await update.message.reply_text("❌ Format incorect. Exemplu: `/compare BTC ETH`")
        return

    sym1 = context.args[0].upper()
    sym2 = context.args[1].upper()

    if not is_valid_symbol(sym1):
        await update.message.reply_text(f"❌ Simbolul '{sym1}' nu este suportat.")
        return
    if not is_valid_symbol(sym2):
        await update.message.reply_text(f"❌ Simbolul '{sym2}' nu este suportat.")
        return
    if sym1 == sym2:
        await update.message.reply_text("❌ Te rog alege două simboluri diferite pentru comparație.")
        return

    await update.message.reply_text(f"🔄 Generare grafic comparativ {sym1} vs {sym2} ({CHART_DAYS} zile)...")

    try:
        # Descarcă datele pentru ambele simboluri
        df1 = yf.download(f"{sym1}-USD", period=f"{CHART_DAYS}d", interval="1d", progress=False, auto_adjust=True)["Close"]
        df2 = yf.download(f"{sym2}-USD", period=f"{CHART_DAYS}d", interval="1d", progress=False, auto_adjust=True)["Close"]

        missing_symbols = []
        if df1.empty: missing_symbols.append(sym1)
        if df2.empty: missing_symbols.append(sym2)

        if missing_symbols:
            logger.warning(f"Date yfinance lipsă pentru /compare: {', '.join(missing_symbols)}")
            await update.message.reply_text(f"❌ Nu s-au putut obține datele istorice pentru: {', '.join(missing_symbols)}.")
            return

        # --- Plotare cu 2 axe Y ---
        fig, ax1 = plt.subplots(figsize=(12, 7))
        color1 = 'tab:blue'
        ax1.set_xlabel('Dată')
        ax1.set_ylabel(f'Preț {sym1} (USD)', color=color1)
        ax1.plot(df1.index, df1.values, color=color1, label=sym1)
        ax1.tick_params(axis='y', labelcolor=color1)
        # Adaugă grid doar pentru axa Y primară pentru claritate
        ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

        # Creează a doua axă Y care împarte aceeași axă X
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(f'Preț {sym2} (USD)', color=color2)
        ax2.plot(df2.index, df2.values, color=color2, label=sym2)
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title(f"Comparație Preț: {sym1} vs {sym2} ({CHART_DAYS} zile)")

        # Adaugă legenda combinată
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        # Formatare axa X
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        fig.tight_layout() # Ajustează pentru a evita suprapunerea label-urilor

        # Salvare și trimitere
        buf = BytesIO()
        plt.savefig(buf, format='PNG')
        plt.close(fig) # Închide figura
        buf.seek(0)

        await update.message.reply_photo(
            photo=InputFile(buf, filename=f"{sym1}_vs_{sym2}_{CHART_DAYS}d_compare.png"),
            caption=f"Comparație preț {sym1} vs {sym2} - {CHART_DAYS} zile"
        )

    except Exception as e:
        logger.error(f"Eroare la generarea /compare pentru {sym1} vs {sym2}: {e}", exc_info=True)
        try: plt.close(fig)
        except NameError: pass
        await update.message.reply_text("❌ A apărut o eroare la generarea graficului comparativ.")


# === GESTIUNE PORTOFOLIU ===
portfolio_lock = asyncio.Lock() # Lock pentru acces concurent la fișierul CSV

async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Adaugă/Actualizează (sau șterge dacă amount=0) o monedă în portofoliul utilizatorului (CSV)."""
    if len(context.args) != 2:
        await update.message.reply_text("❌ Format incorect. Exemplu: `/portfolio BTC 0.5` sau `/portfolio BTC 0` pentru ștergere.")
        return

    user_id = str(update.effective_user.id) # ID-ul utilizatorului ca string
    symbol = context.args[0].upper()
    amount_str = context.args[1].replace(',', '.') # Înlocuiește virgula cu punct pentru float

    if not is_valid_symbol(symbol):
        await update.message.reply_text(f"❌ Simbolul '{symbol}' nu este suportat.")
        return

    try:
        amount = float(amount_str)
        if amount < 0:
            await update.message.reply_text("❌ Cantitatea nu poate fi negativă.")
            return
    except ValueError:
        await update.message.reply_text("❌ Cantitate invalidă. Folosește numere (ex: `0.5` sau `10`).")
        return

    # Folosim lock pentru a asigura operațiuni atomice pe fișierul CSV
    async with portfolio_lock:
        logger.info(f"Început operațiune portfolio pentru user {user_id}, {symbol}, {amount}")
        current_data: List[Dict[str, Any]] = []
        file_exists = os.path.exists(PORTFOLIO_FILE)
        header = ["user_id", "symbol", "amount", "timestamp"]

        # 1. Citim datele existente (dacă există fișierul)
        if file_exists:
            try:
                with open(PORTFOLIO_FILE, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    # Verificăm header-ul înainte de a citi
                    if reader.fieldnames != header:
                        logger.warning(f"Header incorect detectat în {PORTFOLIO_FILE}. Fișierul va fi rescris.")
                        # Tratăm ca și cum fișierul nu ar exista corect formatat
                        file_exists = False
                    else:
                        # Citim toate rândurile existente
                        current_data = list(reader)
            except Exception as e:
                logger.error(f"Eroare la citirea fișierului {PORTFOLIO_FILE} în /portfolio: {e}", exc_info=True)
                await update.message.reply_text("❌ Eroare internă la accesarea portofoliului. Încearcă din nou.")
                return # Ieșim dacă nu putem citi fișierul

        # 2. Pregătim noile date
        new_data: List[Dict[str, Any]] = []
        entry_updated = False
        # Iterăm prin datele vechi și le păstrăm pe cele care nu corespund user/symbol curent
        for row in current_data:
            if row["user_id"] == user_id and row["symbol"] == symbol:
                # Am găsit intrarea existentă. O vom înlocui/șterge.
                entry_updated = True
                if amount > 1e-9: # Dacă noua cantitate e pozitivă, o adăugăm
                    new_entry = {
                        "user_id": user_id,
                        "symbol": symbol,
                        "amount": f"{amount:.8f}", # Formatăm cu 8 zecimale
                        "timestamp": datetime.now().isoformat()
                    }
                    new_data.append(new_entry)
                    logger.info(f"Actualizat {symbol} la {amount} pentru user {user_id}")
                else:
                    logger.info(f"Șters {symbol} pentru user {user_id} (amount=0)")
                    # Nu adăugăm nimic în new_data, efectiv ștergând intrarea
            else:
                # Păstrăm rândurile care nu sunt ale userului/simbolului curent
                new_data.append(row)

        # Dacă intrarea nu exista și cantitatea e pozitivă, o adăugăm
        if not entry_updated and amount > 1e-9:
            new_entry = {
                "user_id": user_id,
                "symbol": symbol,
                "amount": f"{amount:.8f}",
                "timestamp": datetime.now().isoformat()
            }
            new_data.append(new_entry)
            logger.info(f"Adăugat {symbol} cu {amount} pentru user {user_id}")

        # 3. Rescriem fișierul complet cu datele actualizate
        try:
            with open(PORTFOLIO_FILE, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writeheader()
                writer.writerows(new_data)

            # Trimitem mesaj de confirmare utilizatorului
            if amount > 1e-9:
                await update.message.reply_text(f"✅ Portofoliu actualizat: {amount:.8f} {symbol}")
            else:
                await update.message.reply_text(f"✅ {symbol} eliminat din portofoliu.")
            logger.info(f"Operațiune portfolio finalizată cu succes pentru user {user_id}")

        except Exception as e:
            logger.error(f"Eroare la scrierea fișierului {PORTFOLIO_FILE} în /portfolio: {e}", exc_info=True)
            await update.message.reply_text("❌ Eroare internă la salvarea portofoliului. Modificările pot fi pierdute.")


async def fetch_price_for_portfolio(session: aiohttp.ClientSession, symbol: str, amount: float) -> Tuple[str, float]:
    """Funcție helper pentru /myportfolio: Obține preț și calculează valoare."""
    try:
        url = f"{BINANCE_API_BASE}/ticker/price?symbol={symbol}USDT"
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                price = float(data.get("price", 0))
                if price > 0:
                    total_value = price * amount
                    # Formatare linie pentru afișare (Markdown)
                    line = f"• {symbol}: {amount:.8f} × ${price:,.2f} = *${total_value:,.2f}*"
                    return line, total_value
                else:
                    logger.warning(f"Preț invalid (0) de la Binance pentru {symbol} în portfolio.")
                    return f"• {symbol}: {amount:.8f} × $? = Preț invalid (0)", 0.0
            else:
                # Eșec la obținerea prețului (simbol invalid, eroare API etc.)
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


async def myportfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Afișează portofoliul curent al utilizatorului și valoarea totală estimată."""
    user_id = str(update.effective_user.id)
    await update.message.reply_text("🔄 Încărcare și evaluare portofoliu...")

    user_holdings: Dict[str, float] = {} # Dicționar simbol -> cantitate totală

    # Folosim lock și la citire pentru a evita citirea parțială în timpul unei scrieri
    async with portfolio_lock:
        logger.info(f"Început citire portofoliu pentru user {user_id}")
        if not os.path.exists(PORTFOLIO_FILE):
            await update.message.reply_text("📭 Portofoliul tău este gol. Folosește `/portfolio SIMBOL CANTITATE` pentru a adăuga monede.")
            logger.info(f"Portofoliu gol pentru user {user_id} (fișier inexistent).")
            return

        try:
            with open(PORTFOLIO_FILE, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                header = ["user_id", "symbol", "amount", "timestamp"]
                if reader.fieldnames != header:
                    logger.error(f"Header incorect în {PORTFOLIO_FILE} la citirea pentru /myportfolio.")
                    await update.message.reply_text("❌ Eroare internă la citirea datelor de portofoliu (format invalid).")
                    return

                # Agregăm cantitățile pentru fiecare simbol deținut de utilizator
                for row in reader:
                    if row["user_id"] == user_id:
                        try:
                            symbol = row["symbol"]
                            amount = float(row["amount"])
                            if amount > 1e-9: # Adăugăm doar dacă avem cantitate pozitivă
                                user_holdings[symbol] = user_holdings.get(symbol, 0) + amount
                        except ValueError:
                            logger.warning(f"Cantitate invalidă în {PORTFOLIO_FILE} pentru user {user_id}, simbol {row.get('symbol', '?')}. Rând ignorat: {row}")
                        except KeyError:
                             logger.warning(f"Cheie lipsă în {PORTFOLIO_FILE} pentru user {user_id}. Rând ignorat: {row}")

            logger.info(f"Citire portofoliu finalizată pentru user {user_id}. Intrări agregate: {len(user_holdings)}")

        except Exception as e:
            logger.error(f"Eroare majoră la citirea {PORTFOLIO_FILE} în /myportfolio pentru user {user_id}: {e}", exc_info=True)
            await update.message.reply_text("❌ Eroare internă la accesarea datelor de portofoliu.")
            return

    # Verificăm dacă utilizatorul are ceva în portofoliu după agregare
    if not user_holdings:
        await update.message.reply_text("📭 Portofoliul tău este gol sau conține doar cantități zero.")
        logger.info(f"Portofoliu gol sau zero pentru user {user_id} după agregare.")
        return

    # Obținem prețurile și calculăm valorile în paralel
    message_lines = ["📊 *Portofoliul tău:*"]
    total_portfolio_value = 0.0
    has_price_errors = False

    logger.info(f"Preluare prețuri pentru {len(user_holdings)} active din portofoliul user {user_id}...")
    try:
        async with aiohttp.ClientSession() as session:
            # Creăm task-uri pentru a obține prețul fiecărei monede
            tasks = [fetch_price_for_portfolio(session, sym, amt) for sym, amt in user_holdings.items()]
            results = await asyncio.gather(*tasks) # Executăm task-urile în paralel

            # Procesăm rezultatele
            for line, value in results:
                message_lines.append(line)
                total_portfolio_value += value
                if "Eroare" in line: # Marcăm dacă a apărut vreo eroare de preț
                    has_price_errors = True

        # Adăugăm valoarea totală la sfârșit
        message_lines.append(f"\n💰 *Valoare totală estimată:* ${total_portfolio_value:,.2f}")

        final_message = "\n".join(message_lines)

        # Adăugăm o notă dacă au existat erori la prețuri
        if has_price_errors:
            final_message += "\n\n⚠️ _Notă: Valoarea totală poate fi imprecisă din cauza erorilor la obținerea unor prețuri._"

        await update.message.reply_text(final_message, parse_mode="Markdown")
        logger.info(f"Portofoliu afișat cu succes pentru user {user_id}. Valoare totală: ${total_portfolio_value:,.2f}")

    except Exception as e:
        # Eroare generală în timpul obținerii prețurilor sau formatării mesajului
        logger.error(f"Eroare la afișarea portofoliului pentru user {user_id}: {e}", exc_info=True)
        # Trimitem ce am putut formata plus un mesaj de eroare generală
        error_note = "\n\n⚠️ A apărut o eroare majoră la calcularea valorii portofoliului."
        await update.message.reply_text("\n".join(message_lines) + error_note, parse_mode="Markdown")


# === PORNIRE BOT ===
if __name__ == "__main__":
    if not TOKEN:
        logger.critical("FATAL: Variabila de mediu TELEGRAM_BOT_TOKEN nu este setată!")
        exit(1) # Oprește execuția dacă token-ul lipsește

    # --- Preîncărcare modele AI ---
    # Rulează sincron la început
    preload_models()
    # -----------------------------

    # Construiește aplicația Telegram
    app = ApplicationBuilder().token(TOKEN).build()

    # Adaugă handler-ele pentru comenzi
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("predict_chart", predict_chart))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("news", news)) # Folosește noua funcție RSS
    app.add_handler(CommandHandler("crypto", crypto))
    app.add_handler(CommandHandler("grafic", grafic))
    app.add_handler(CommandHandler("trend", trend))
    app.add_handler(CommandHandler("compare", compare))
    app.add_handler(CommandHandler("portfolio", portfolio))
    app.add_handler(CommandHandler("myportfolio", myportfolio))

    # Pornire bot (rulează până la oprire manuală - Ctrl+C)
    logger.info("🚀 Botul Crypto AI pornește...")
    print("🚀 Botul Crypto AI pornește...")
    app.run_polling()

    # Aceste linii vor rula doar dacă botul este oprit elegant
    logger.info("🛑 Botul Crypto AI s-a oprit.")
    print("🛑 Botul Crypto AI s-a oprit.")