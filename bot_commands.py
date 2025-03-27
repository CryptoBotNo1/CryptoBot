# bot_commands.py - Corectat Final (v5 - Verificare finală 'compare')
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from io import BytesIO
from telegram import Update, InputFile
from telegram.ext import ContextTypes
from typing import List, Dict, Optional
import time
import sentry_sdk # Import Sentry SDK

# Importă configurările și utilitarele necesare
import config
import ai_utils
import data_utils
import portfolio_utils
import plot_utils # Importăm plot_utils

logger = logging.getLogger(__name__)

# --- Funcții Utilitare Specifice Comenzilor ---
def is_valid_symbol(symbol: str) -> bool:
    """Verifică dacă simbolul este în lista de criptomonede suportate."""
    return symbol.upper() in config.CRYPTO_LIST

def is_valid_days(days_str: str) -> bool:
    """Verifică dacă nr. de zile e valid (1-PREDICT_MAX_DAYS)."""
    try:
        days = int(days_str)
        return 1 <= days <= config.PREDICT_MAX_DAYS
    except ValueError:
        return False

# --- Funcție Helper Sentry Context ---
def _set_sentry_user_context(update: Optional[Update]):
    """Setează contextul utilizatorului în Sentry dacă update conține user."""
    # Verificam daca sentry_sdk a fost initializat cu succes inainte de a-l folosi
    # (presupunem ca hub/client exista daca init a rulat fara erori critice)
    if config.SENTRY_DSN and hasattr(sentry_sdk, 'Hub') and sentry_sdk.Hub.current.client:
        if update and update.effective_user:
            sentry_sdk.set_user({"id": str(update.effective_user.id)})
        else:
            sentry_sdk.set_user(None) # Resetează contextul user dacă nu există

# --- Handlere Comenzi Telegram ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /start."""
    _set_sentry_user_context(update)
    await update.message.reply_text("👋 Salut! Sunt botul tău AI pentru crypto.\nFolosește /help pentru lista completă de comenzi.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /help."""
    _set_sentry_user_context(update)
    user_id = update.effective_user.id if update.effective_user else "Unknown"
    logger.info(f"/help command requested by user {user_id}")
    try:
        symbols_str = ', '.join(config.CRYPTO_LIST)
        help_text = f"""📋 *Comenzi disponibile:*

/start - Pornește botul
/help - Afișează acest mesaj

*Predicții & Analiză:*
`/predict SIMBOL ZILE` - Predicție preț AI (Ex: `/predict BTC 7`)
`/predict_chart SIMBOL ZILE` - Grafic predicție AI (Ex: `/predict_chart ETH 5`)
`/summary SIMBOL` - Sumar zilnic (preț, trend {config.TREND_DAYS}z, pred 1z) (Ex: `/summary SOL`)
`/trend SIMBOL` - Analiză trend ({config.TREND_DAYS} zile) (Ex: `/trend ADA`)
`/grafic SIMBOL` - Grafic preț ({config.CHART_DAYS} zile) (Ex: `/grafic BNB`)
`/compare SIMBOL1 SIMBOL2` - Compară prețurile ({config.CHART_DAYS} zile) (Ex: `/compare BTC ETH`)

*Informații & Știri:*
`/crypto SIMBOL` - Preț live și date 24h (Ex: `/crypto XRP`)
`/news` - Ultimele {config.NEWS_COUNT} știri crypto din mai multe surse (RSS)

*Portofoliu:*
`/portfolio SIMBOL CANTITATE` - Adaugă/Actualizează (Ex: `/portfolio DOT 15.5`, `/portfolio BTC 0` pt ștergere)
`/myportfolio` - Afișează portofoliul tău

*Simboluri suportate:* {symbols_str}
*Nr. zile predicție:* 1-{config.PREDICT_MAX_DAYS}
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")
        logger.info(f"Help message sent successfully to user {user_id}")
    except Exception as e:
        logger.error(f"Error sending /help message to user {user_id}: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        await update.message.reply_text("❌ Oops! A apărut o eroare la afișarea comenzilor.")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /predict."""
    _set_sentry_user_context(update)
    if len(context.args) != 2: await update.message.reply_text(f"❌ Format incorect. Exemplu: `/predict BTC 7`"); return
    symbol = context.args[0].upper(); days_str = context.args[1]
    if not is_valid_symbol(symbol): await update.message.reply_text(f"❌ Simbol '{symbol}' nesuportat. Folosește /help."); return
    if not is_valid_days(days_str): await update.message.reply_text(f"❌ Nr. zile invalid (1-{config.PREDICT_MAX_DAYS})."); return
    days = int(days_str)
    await update.message.reply_text(f"🔄 Generare predicție AI pentru {symbol} ({days} zile)...")
    try:
        predictions = ai_utils.predict_sequence(symbol, days)
        if not predictions: await update.message.reply_text(f"❌ Nu s-a putut genera predicția pentru {symbol}."); return
        start_date = datetime.now().date() + timedelta(days=1)
        prediction_dates = pd.date_range(start_date, periods=days)
        pred_text = "\n".join([f"📅 {d.strftime('%Y-%m-%d')}: *{p:,.2f}* USD" for d, p in zip(prediction_dates, predictions)])
        await update.message.reply_text(f"📊 *Predicții AI {symbol}:*\n{pred_text}", parse_mode="Markdown")
    except Exception as e: logger.error(f"Eroare neașteptată /predict {symbol}: {e}", exc_info=True); await update.message.reply_text("❌ Eroare generare predicție.")

async def predict_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /predict_chart."""
    _set_sentry_user_context(update)
    if len(context.args) != 2: await update.message.reply_text(f"❌ Format: `/predict_chart ETH 5`"); return
    symbol = context.args[0].upper(); days_str = context.args[1]
    if not is_valid_symbol(symbol): await update.message.reply_text(f"❌ Simbolul '{symbol}' nesuportat."); return
    if not is_valid_days(days_str): await update.message.reply_text(f"❌ Nr. zile invalid (1-{config.PREDICT_MAX_DAYS})."); return
    days = int(days_str)
    await update.message.reply_text(f"🔄 Generare grafic AI pentru {symbol} ({days} zile)...")
    try:
        predictions = ai_utils.predict_sequence(symbol, days)
        if not predictions: await update.message.reply_text(f"❌ Eroare date predicție grafic {symbol}."); return
        start_date = datetime.now().date() + timedelta(days=1)
        prediction_dates = pd.date_range(start_date, periods=days)
        buf = plot_utils.generate_prediction_plot(symbol, prediction_dates, predictions)
        if buf.getbuffer().nbytes == 0: await update.message.reply_text("❌ Eroare internă generare imagine grafic."); return
        await update.message.reply_photo(photo=InputFile(buf, filename=f"{symbol}_predict_chart_{days}d.png"), caption=f"Grafic predicție AI {symbol} ({days} zile)")
    except Exception as e: logger.error(f"Eroare /predict_chart {symbol}: {e}", exc_info=True); await update.message.reply_text("❌ Eroare generare grafic AI.")

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /news."""
    _set_sentry_user_context(update)
    await update.message.reply_text(f"🔄 Căutare ultimelor {config.NEWS_COUNT} știri crypto (RSS)...")
    try:
        tasks = [data_utils.parse_rss_feed(feed_info) for feed_info in config.RSS_FEEDS]
        results = await asyncio.gather(*tasks)
        all_entries = []; successful_feeds = 0
        for feed_data in results:
            if feed_data and feed_data.entries: all_entries.extend(feed_data.entries); successful_feeds += 1
        if not all_entries: await update.message.reply_text("❌ Nu s-au putut obține știri RSS."); return
        logger.info(f"Intrări RSS brute: {len(all_entries)} din {successful_feeds}/{len(config.RSS_FEEDS)} surse.")
        unique_entries_dict: Dict[str, Dict] = {}
        for entry in all_entries:
            title = entry.get('title', '').strip().lower()
            if title:
                published_time = entry.get('published_parsed', None)
                entry_timestamp = time.mktime(published_time) if published_time else 0
                entry['published_timestamp'] = entry_timestamp
                if title not in unique_entries_dict or entry_timestamp > unique_entries_dict[title].get('published_timestamp', 0):
                     unique_entries_dict[title] = entry
        unique_entries = sorted(list(unique_entries_dict.values()), key=lambda x: x.get('published_timestamp', 0), reverse=True)
        logger.info(f"Intrări RSS unice: {len(unique_entries)}")
        texts_to_send = []
        for i, entry in enumerate(unique_entries[:config.NEWS_COUNT]):
            title = entry.get('title', 'Fără titlu'); link = entry.get('link', '#'); source = entry.get('source_name', '?')
            ts = entry.get('published_timestamp', 0)
            time_str = datetime.fromtimestamp(ts).strftime('%d %b %H:%M') if ts > 0 else "dată nec."
            safe_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            safe_source = source.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            texts_to_send.append(f"{i+1}. <a href='{link}'>{safe_title}</a>\n    <i>({safe_source} - {time_str})</i>")
        if not texts_to_send: await update.message.reply_text("⚠️ Nu s-au găsit știri relevante."); return
        final_message = f"📰 <b>Top {len(texts_to_send)} Știri Crypto Recente (RSS):</b>\n\n" + "\n\n".join(texts_to_send)
        await update.message.reply_text(final_message, parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Eroare majoră în handler-ul /news: {e}", exc_info=True)
        await update.message.reply_text("❌ A apărut o eroare la preluarea știrilor.")

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /summary."""
    _set_sentry_user_context(update)
    if len(context.args) != 1: await update.message.reply_text("❌ Format: `/summary BTC`"); return
    symbol = context.args[0].upper()
    if not is_valid_symbol(symbol): await update.message.reply_text(f"❌ Simbolul '{symbol}' nesuportat."); return
    await update.message.reply_text(f"🔄 Generare sumar pentru {symbol}...")
    live_price_str, trend_msg, pred_msg = "N/A", "N/A", "N/A"
    async with aiohttp.ClientSession() as session:
        try:
            url = f"{config.BINANCE_API_BASE}/ticker/price?symbol={symbol}USDT"
            async with session.get(url, timeout=5) as r:
                if r.status == 200: live_price = float((await r.json()).get('price', 0)); live_price_str = f"*{live_price:,.2f}* $" if live_price > 0 else "Preț invalid"
                elif r.status == 400: live_price_str = f"Indisponibil ({symbol}USDT)"
                else: logger.warning(f"Binance API {symbol} status {r.status}"); live_price_str = f"Eroare API ({r.status})"
        except Exception as e: logger.error(f"Eroare preț live sumar {symbol}: {e}"); live_price_str = "Eroare conex."
    try:
        df_trend = yf.download(f"{symbol}-USD", period=f"{config.TREND_DAYS + 1}d", interval="1d", progress=False, auto_adjust=True)
        if not df_trend.empty and len(df_trend) >= 2:
            start_price = df_trend["Close"].iloc[0].item()
            end_price = df_trend["Close"].iloc[-1].item()
            change = end_price - start_price; start_price_safe = start_price if abs(start_price) > 1e-9 else 1e-9
            change_pct = (change / start_price_safe) * 100; trend_symbol = "📈" if change >= 0 else "📉"
            trend_msg = f"{trend_symbol} Trend {config.TREND_DAYS}z: *{change:+.2f}$* ({change_pct:+.2f}%)"
        else: trend_msg = f"ℹ️ Date trend ({config.TREND_DAYS}z) insuf."; logger.warning(f"Date trend {symbol} insuf.")
    except Exception as e: logger.error(f"Eroare calcul trend sumar {symbol}: {e}", exc_info=True); trend_msg = f"⚠️ Eroare trend ({config.TREND_DAYS}z)."
    try:
        prediction = ai_utils.predict_sequence(symbol, 1)
        if prediction: pred_msg = f"🔮 Predicție AI (1 zi): *{prediction[0]:,.2f}* $"
        else: pred_msg = "ℹ️ Predicție AI indisponibilă."
    except Exception as e: logger.error(f"Eroare predicție sumar {symbol}: {e}"); pred_msg = "⚠️ Eroare predicție AI."
    full_name = config.CRYPTO_SYMBOLS_CONFIG.get(symbol, symbol)
    message = f"""*🧠 Sumar zilnic: {full_name} ({symbol})*

💰 Preț live: {live_price_str}
{trend_msg}
{pred_msg}
"""
    await update.message.reply_text(message, parse_mode="Markdown")

async def crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /crypto."""
    _set_sentry_user_context(update)
    if len(context.args) != 1: await update.message.reply_text("❌ Format: `/crypto BTC`"); return
    symbol = context.args[0].upper()
    await update.message.reply_text(f"🔄 Căutare date live 24h {symbol}USDT pe Binance...")
    try:
        url = f"{config.BINANCE_API_BASE}/ticker/24hr?symbol={symbol}USDT"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as r:
                if r.status == 200:
                    data = await r.json(); full_name = config.CRYPTO_SYMBOLS_CONFIG.get(symbol, symbol)
                    price = float(data.get('lastPrice', 0)); high = float(data.get('highPrice', 0)); low = float(data.get('lowPrice', 0))
                    volume = float(data.get('volume', 0)); quote_volume = float(data.get('quoteVolume', 0)); change_pct = float(data.get('priceChangePercent', 0))
                    msg = f"""📊 *Info Live 24h: {full_name} ({symbol})*
💵 Preț: *{price:,.2f} $*
📉 Low 24h: {low:,.2f} $
📈 High 24h: {high:,.2f} $
📊 Var 24h: {change_pct:+.2f}%
🔄 Vol 24h ({symbol}): {volume:,.2f}
💰 Vol 24h (USDT): {quote_volume:,.2f} $
_(Sursa: Binance)_"""
                    await update.message.reply_text(msg, parse_mode="Markdown")
                elif r.status == 400: await update.message.reply_text(f"❌ '{symbol}USDT' negăsit pe Binance.")
                else: logger.error(f"Eroare Binance {symbol} status {r.status}"); await update.message.reply_text(f"❌ Eroare Binance (Status: {r.status}).")
    except asyncio.TimeoutError: logger.error(f"Timeout Binance {symbol}"); await update.message.reply_text("❌ Binance timeout.")
    except aiohttp.ClientError as e: logger.error(f"Eroare rețea /crypto {symbol}: {e}"); await update.message.reply_text("❌ Eroare rețea date live.")
    except Exception as e: logger.error(f"Eroare /crypto {symbol}: {e}", exc_info=True); await update.message.reply_text("❌ Eroare date live.")

async def grafic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /grafic."""
    _set_sentry_user_context(update)
    if len(context.args) != 1: await update.message.reply_text(f"❌ Format: `/grafic BTC`"); return
    symbol = context.args[0].upper()
    if not is_valid_symbol(symbol): await update.message.reply_text(f"❌ Simbolul '{symbol}' nesuportat."); return
    await update.message.reply_text(f"🔄 Generare grafic preț {symbol} ({config.CHART_DAYS} zile)...")
    try:
        df = yf.download(f"{symbol}-USD", period=f"{config.CHART_DAYS}d", interval="1d", progress=False, auto_adjust=True)
        if df.empty: await update.message.reply_text(f"❌ Eroare date yfinance {symbol}."); return
        buf = plot_utils.generate_price_history_plot(symbol, df, config.CHART_DAYS)
        if buf.getbuffer().nbytes == 0: await update.message.reply_text("❌ Eroare internă generare imagine grafic."); return
        await update.message.reply_photo(photo=InputFile(buf, filename=f"{symbol}_{config.CHART_DAYS}d_chart.png"), caption=f"Grafic preț {symbol} - {config.CHART_DAYS} zile")
    except Exception as e: logger.error(f"Eroare /grafic {symbol}: {e}", exc_info=True); await update.message.reply_text("❌ Eroare generare grafic.")

async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /trend."""
    _set_sentry_user_context(update)
    if len(context.args) != 1: await update.message.reply_text(f"❌ Format: `/trend BTC`"); return
    symbol = context.args[0].upper()
    if not is_valid_symbol(symbol): await update.message.reply_text(f"❌ Simbolul '{symbol}' nesuportat."); return
    await update.message.reply_text(f"🔄 Analiză trend {symbol} ({config.TREND_DAYS} zile)...")
    logger.info(f"Pornire calcul trend pentru {symbol}")
    try:
        df = yf.download(f"{symbol}-USD", period=f"{config.TREND_DAYS + 1}d", interval="1d", progress=False, auto_adjust=True)
        logger.info(f"Trend {symbol}: yf OK. Len={len(df)}")
        if df.empty or len(df) < 2: logger.warning(f"Trend {symbol}: date insuf."); await update.message.reply_text(f"❌ Date insuf. trend {symbol}."); return
        try:
            start_price = df["Close"].iloc[0].item()
            end_price = df["Close"].iloc[-1].item()
            logger.info(f"Trend {symbol}: Prețuri OK: Start={start_price:.4f}, End={end_price:.4f}")
        except Exception as e_item: logger.error(f"Trend {symbol}: Eroare extragere preț: {e_item}", exc_info=True); await update.message.reply_text(f"❌ Eroare procesare preț {symbol}."); return
        change = end_price - start_price; start_price_safe = start_price if abs(start_price) > 1e-9 else 1e-9
        change_pct = (change / start_price_safe) * 100; logger.info(f"Trend {symbol}: Calcul OK: Chg={change:.4f}, Pct={change_pct:.2f}")
        trend_symbol = "📈" if change >= 0 else "📉"; trend_desc = "ascendent" if change >= 0 else "descendent"
        msg = f"{trend_symbol} *Trend {symbol} ({config.TREND_DAYS} zile): {trend_desc}*\nVariație: *{change:+.2f}$* ({change_pct:+.2f}%)"
        logger.info(f"Trend {symbol}: Mesaj OK.")
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e: logger.error(f"Eroare majoră /trend {symbol}: {e}", exc_info=True); await update.message.reply_text("❌ Eroare analiză trend.")

# --- Funcția compare CORECTATĂ DEFINITIV ---
async def compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /compare (VERSIUNE CORECTATĂ)."""
    _set_sentry_user_context(update)
    if len(context.args) != 2: await update.message.reply_text("❌ Format: `/compare BTC ETH`"); return
    sym1, sym2 = context.args[0].upper(), context.args[1].upper()
    if not is_valid_symbol(sym1): await update.message.reply_text(f"❌ Simbolul '{sym1}' nesuportat."); return
    if not is_valid_symbol(sym2): await update.message.reply_text(f"❌ Simbolul '{sym2}' nesuportat."); return
    if sym1 == sym2: await update.message.reply_text("❌ Alege simboluri diferite."); return

    await update.message.reply_text(f"🔄 Generare grafic comparativ {sym1} vs {sym2} ({config.CHART_DAYS} zile)...")
    try:
        # Descărcăm datele complete mai întâi
        df1_full = yf.download(f"{sym1}-USD", period=f"{config.CHART_DAYS}d", interval="1d", progress=False, auto_adjust=True)
        df2_full = yf.download(f"{sym2}-USD", period=f"{config.CHART_DAYS}d", interval="1d", progress=False, auto_adjust=True)

        # Verificăm dacă DF-urile sunt goale ÎNAINTE de a accesa ["Close"]
        missing = []
        if df1_full.empty: missing.append(sym1)
        if df2_full.empty: missing.append(sym2)
        if missing:
            logger.warning(f"Date yfinance lipsă pentru /compare: {', '.join(missing)}")
            await update.message.reply_text(f"❌ Eroare date yfinance: {', '.join(missing)}.")
            return

        # Extragem coloana 'Close' doar dacă DF-urile sunt valide
        df1_close = df1_full["Close"]
        df2_close = df2_full["Close"]

        # Generăm și trimitem graficul
        buf = plot_utils.generate_comparison_plot(sym1, sym2, df1_close, df2_close, config.CHART_DAYS)
        if buf.getbuffer().nbytes == 0:
            await update.message.reply_text("❌ Eroare internă generare imagine grafic.")
            return

        await update.message.reply_photo(
            photo=InputFile(buf, filename=f"{sym1}_vs_{sym2}_{config.CHART_DAYS}d_compare.png"),
            caption=f"Comparație preț {sym1} vs {sym2} - {config.CHART_DAYS} zile"
        )
    except Exception as e:
        logger.error(f"Eroare majoră /compare {sym1} vs {sym2}: {e}", exc_info=True)
        await update.message.reply_text("❌ Eroare grafic comparativ.")

async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /portfolio. Folosește portfolio_utils."""
    _set_sentry_user_context(update)
    if len(context.args) != 2: await update.message.reply_text("❌ Format: `/portfolio BTC 0.5` sau `/portfolio BTC 0`"); return
    user_id = str(update.effective_user.id); symbol = context.args[0].upper(); amount_str = context.args[1].replace(',', '.')
    if not is_valid_symbol(symbol): await update.message.reply_text(f"❌ Simbolul '{symbol}' nesuportat."); return
    try: amount = float(amount_str); assert amount >= 0
    except (ValueError, AssertionError): await update.message.reply_text("❌ Cantitate invalidă (nr. pozitiv sau 0)."); return
    success = await portfolio_utils.add_or_update_holding(user_id, symbol, amount)
    if success:
        if amount > config.ZERO_THRESHOLD: await update.message.reply_text(f"✅ Portofoliu actualizat: {amount:.8f} {symbol}")
        else: await update.message.reply_text(f"✅ {symbol} eliminat din portofoliu.")
    else: await update.message.reply_text("❌ Eroare internă la salvarea în baza de date a portofoliului.")

async def myportfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pentru comanda /myportfolio. Folosește portfolio_utils."""
    _set_sentry_user_context(update)
    user_id = str(update.effective_user.id)
    await update.message.reply_text("🔄 Încărcare și evaluare portofoliu din baza de date...")
    user_holdings = await portfolio_utils.get_user_holdings(user_id)
    if user_holdings is None: await update.message.reply_text("❌ Eroare internă la accesarea bazei de date a portofoliului."); return
    if not user_holdings: await update.message.reply_text("📭 Portofoliu gol. Folosește `/portfolio SIMBOL CANTITATE`."); return
    msg_lines = ["📊 *Portofoliul tău:*"]; total_portfolio_value = 0.0; has_errors = False
    logger.info(f"Evaluare portofoliu pentru {len(user_holdings)} active user {user_id}...")
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [data_utils.fetch_price_for_portfolio(session, sym, amt) for sym, amt in user_holdings.items()]
            results = await asyncio.gather(*tasks)
            for line, value in results: msg_lines.append(line); total_portfolio_value += value; has_errors = has_errors or "Eroare" in line
        msg_lines.append(f"\n💰 *Valoare totală estimată:* ${total_portfolio_value:,.2f}")
        final_message = "\n".join(msg_lines)
        if has_errors: final_message += "\n\n⚠️ _Notă: Valoare imprecisă (erori preț)._"
        await update.message.reply_text(final_message, parse_mode="Markdown")
        logger.info(f"Portofoliu afișat user {user_id}. Valoare: ${total_portfolio_value:,.2f}")
    except Exception as e:
        logger.error(f"Eroare evaluare /myportfolio user {user_id}: {e}", exc_info=True)
        await update.message.reply_text("❌ Eroare la evaluarea valorii portofoliului.")