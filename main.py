# main.py
import logging
import os
import sys
import matplotlib
import sentry_sdk                             # Import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration # Import integrare logging
# Import sqlite3 integration nu mai este necesar explicit

# Setare backend matplotlib
matplotlib.use('Agg')

# Importuri DUPĂ setarea backend-ului
import config # Încarcă config (acum include și SENTRY_DSN)
from ai_utils import preload_models
from portfolio_utils import init_db
# --- Importurile sunt acum CURATE, fără 'test_error' ---
from bot_commands import (
    start, help_command, predict, predict_chart, summary, news,
    crypto, grafic, trend, compare, portfolio, myportfolio
)
from telegram.ext import ApplicationBuilder, CommandHandler

# --- Configurare Logging ---
log_level_str = config.LOG_LEVEL.upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__) # Obține logger-ul principal

# --- Inițializare Sentry --- # (Rămâne la fel)
if config.SENTRY_DSN:
    try:
        sentry_logging = LoggingIntegration(level=logging.INFO, event_level=logging.WARNING)
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            integrations=[
                sentry_logging,
                # Sqlite3Integration(), # Nu mai este necesar explicit
            ],
            environment="development",
            # release="crypto-ai-bot@1.0.1", # Opțional
            traces_sample_rate=0.0,
            profiles_sample_rate=0.0,
            send_default_pii=True
        )
        logger.info("Sentry SDK inițializat cu succes (fără Sqlite3Integration explicit).")
    except Exception as e:
        logger.error(f"Eroare la inițializarea Sentry SDK: {e}", exc_info=True)
else:
    logger.warning("SENTRY_DSN nu este setat în .env. Monitorizarea Sentry este dezactivată.")
# --- Sfârșit Bloc Sentry ---


if __name__ == "__main__":
    logger.info("============================================")
    logger.info("=== Inițializare Bot Crypto AI ===")
    logger.info("============================================")

    # Verificare Token
    if not config.TOKEN: logger.critical("FATAL: TELEGRAM_BOT_TOKEN nu este setat! Oprește execuția."); exit(1)
    logger.info("Token Telegram încărcat.")

    # --- Inițializare Bază de Date ---
    init_db()
    # --- Preîncărcare Modele AI ---
    preload_models()
    # -----------------------------

    # Construire aplicație Telegram
    logger.info("Construire aplicație Telegram...")
    app = ApplicationBuilder().token(config.TOKEN).build()
    logger.info("Aplicație construită.")

    # Adăugare handlere
    logger.info("Adăugare handlere comenzi...")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("predict_chart", predict_chart))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("news", news))
    app.add_handler(CommandHandler("crypto", crypto))
    app.add_handler(CommandHandler("grafic", grafic))
    app.add_handler(CommandHandler("trend", trend))
    app.add_handler(CommandHandler("compare", compare))
    app.add_handler(CommandHandler("portfolio", portfolio))
    app.add_handler(CommandHandler("myportfolio", myportfolio))
    # --- Handler-ul temporar ESTE ȘTERS ---
    logger.info("Handlere adăugate.") # <-- Logul este acum curat
    # --------------------------------------

    # Pornire bot
    logger.info("🚀 Botul Crypto AI pornește polling-ul...")
    print("---")
    print("🚀 Botul Crypto AI pornește...")
    print("---")
    app.run_polling()

    # Mesaj de oprire
    logger.info("🛑 Botul Crypto AI s-a oprit.")
    print("---")
    print("🛑 Botul Crypto AI s-a oprit.")
    print("---")