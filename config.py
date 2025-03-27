# config.py
import os
from dotenv import load_dotenv

# --- Încărcare variabile de mediu ---
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
COINSTATS_API_KEY = os.getenv("COINSTATS_API_KEY")
SENTRY_DSN = os.getenv("SENTRY_DSN") # <-- LINIA ADĂUGATĂ

# --- Directoare ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
DATABASE_FILE = os.path.join(BASE_DIR, "portfolio.db") # Numele fișierului bazei de date SQLite

# --- Setări Generale ---
CACHE_EXPIRY_SECONDS = 86400 # 1 zi
SEQUENCE_LENGTH = 60
TREND_DAYS = 7
CHART_DAYS = 30
PREDICT_MAX_DAYS = 30
NEWS_COUNT = 7
LOG_LEVEL = "INFO"
ZERO_THRESHOLD = 1e-9

# --- API Endpoints ---
BINANCE_API_BASE = "https://api.binance.com/api/v3"

# --- Configurații Criptomonede ---
CRYPTO_SYMBOLS_CONFIG = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance Coin",
    "XRP": "XRP", "ADA": "Cardano", "SOL": "Solana",
    "DOGE": "Dogecoin", "DOT": "Polkadot", "MATIC": "Polygon",
    "LTC": "Litecoin"
}
CRYPTO_LIST = list(CRYPTO_SYMBOLS_CONFIG.keys())

# --- Surse RSS ---
RSS_FEEDS = [
    {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"name": "CryptoSlate", "url": "https://cryptoslate.com/feed/"},
    {"name": "Decrypt", "url": "https://decrypt.co/feed"},
]

# --- Creare directoare necesare (dacă nu există) ---
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
    print(f"Directorul cache creat: {CACHE_DIR}")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Directorul models creat: {MODEL_DIR}")