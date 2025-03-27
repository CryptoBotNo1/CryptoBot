# tests/test_bot_commands.py
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch # Importăm și 'patch'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from io import BytesIO
import time

# Importăm obiectele Telegram necesare pentru simulare
from telegram import Update, Message, Chat, User, MessageEntity

# Importăm handler-ele și funcțiile testate
from bot_commands import (
    start, help_command, predict, predict_chart, summary, news,
    crypto, grafic, trend, compare, portfolio, myportfolio,
    _set_sentry_user_context, is_valid_symbol, is_valid_days
)
# Importăm și constante/utilitare folosite de handlere
import config
import ai_utils
import data_utils # Necesar pentru a patch-ui parse_rss_feed
import portfolio_utils
# import plot_utils

# Importăm aioresponses pentru simulare aiohttp
from aioresponses import aioresponses

# --- Mock Global pentru Sentry Context ---
@pytest.fixture(autouse=True)
def mock_sentry_context(mocker):
    mocker.patch("bot_commands._set_sentry_user_context")

# --- Mock Simplu pentru Context ---
@pytest.fixture
def mock_context():
    return MagicMock()

# --- Mock Simplu pentru Update și Mesaj ---
@pytest.fixture
def mock_update_message():
    mock_update = MagicMock(spec=Update)
    mock_update.effective_user = User(id=111, first_name="TestUser", is_bot=False)
    mock_update.message = AsyncMock(spec=Message)
    return mock_update, mock_update.message

# --- Test pentru /start (neschimbat) ---
@pytest.mark.asyncio
async def test_start_command(mock_update_message):
    mock_update, mock_message = mock_update_message
    mock_context = MagicMock()
    await start(mock_update, mock_context)
    mock_message.reply_text.assert_called_once()
    args, kwargs = mock_message.reply_text.call_args
    assert "👋 Salut! Sunt botul tău AI pentru crypto." in args[0]

# --- Teste pentru /predict (verificate, ar trebui să fie OK) ---
@pytest.mark.asyncio
async def test_predict_command_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message
    symbol = "MOCKBTC"; days = 3; mock_user_id = 456
    mock_update.effective_user = User(id=mock_user_id, first_name="PredTester", is_bot=False)
    mock_context.args = [symbol, str(days)]
    mocker.patch("bot_commands.is_valid_symbol", return_value=True) # Mock valid symbol
    mock_prediction_result = [100.0, 110.5, 108.0]
    mock_predict_seq = mocker.patch("bot_commands.ai_utils.predict_sequence", return_value=mock_prediction_result)
    await predict(mock_update, mock_context)
    mock_predict_seq.assert_called_once_with(symbol, days)
    assert mock_message.reply_text.call_count == 2
    call_list = mock_message.reply_text.call_args_list
    assert "🔄 Generare predicție AI" in call_list[0].args[0]
    assert "📊 *Predicții AI" in call_list[1].args[0]
    assert "*100.00* USD" in call_list[1].args[0]
    assert "*110.50* USD" in call_list[1].args[0]
    assert "*108.00* USD" in call_list[1].args[0]
    assert call_list[1].kwargs.get('parse_mode') == 'Markdown'

@pytest.mark.asyncio
async def test_predict_command_invalid_symbol(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message
    symbol = "INVALID"; days = 5
    mock_update.effective_user = User(id=789, first_name="InvUser", is_bot=False)
    mock_context.args = [symbol, str(days)]
    mocker.patch("bot_commands.is_valid_symbol", return_value=False)
    await predict(mock_update, mock_context)
    mock_message.reply_text.assert_called_once()
    # --- FIX 1 (Din Nou!): Corectăm textul așteptat în test ---
    expected_error_text = f"❌ Simbol '{symbol}' nesuportat. Folosește /help."
    actual_text = mock_message.reply_text.call_args.args[0]
    assert actual_text == expected_error_text
    # ------------------------------------------------------

@pytest.mark.asyncio
async def test_predict_command_invalid_days(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message
    symbol = "BTC"; days_str = "abc"
    mock_update.effective_user = User(id=101, first_name="DayUser", is_bot=False)
    mock_context.args = [symbol, days_str]
    await predict(mock_update, mock_context)
    mock_message.reply_text.assert_called_once()
    expected_error_text = f"❌ Nr. zile invalid (1-{config.PREDICT_MAX_DAYS})."
    assert expected_error_text in mock_message.reply_text.call_args.args[0]

# --- Teste pentru /summary (neschimbate, ar trebui să fie OK) ---
@pytest.mark.asyncio
async def test_summary_command_success(mocker, mock_update_message, mock_context):
    symbol = "MOCKSOL"; mock_update, mock_message = mock_update_message; mock_context.args = [symbol]
    mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    trend_days_count = config.TREND_DAYS + 1
    mock_yf_data = pd.DataFrame({'Close': np.linspace(100, 110, trend_days_count)}, index=pd.date_range(end=datetime.now(), periods=trend_days_count, freq='D'))
    mock_yf = mocker.patch("bot_commands.yf.download", return_value=mock_yf_data)
    mock_prediction = [115.5]; mock_predict = mocker.patch("bot_commands.ai_utils.predict_sequence", return_value=mock_prediction)
    binance_url = f"{config.BINANCE_API_BASE}/ticker/price?symbol={symbol}USDT"; mock_live_price = "112.34"
    with aioresponses() as m:
        m.get(binance_url, payload={"price": mock_live_price}, status=200)
        await summary(mock_update, mock_context)
    mock_yf.assert_called_once(); mock_predict.assert_called_once_with(symbol, 1)
    assert mock_message.reply_text.call_count == 2; calls = mock_message.reply_text.call_args_list
    assert f"🔄 Generare sumar pentru {symbol}" in calls[0].args[0]
    final_reply = calls[1].args[0]
    assert f"*{float(mock_live_price):,.2f}* $" in final_reply
    assert f"*{10.00:+.2f}$*" in final_reply; assert f"*{mock_prediction[0]:,.2f}* $" in final_reply
    assert calls[1].kwargs.get('parse_mode') == 'Markdown'

# --- Teste pentru /portfolio (neschimbate, ar trebui să fie OK) ---
@pytest.mark.asyncio
async def test_portfolio_command_add_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id); symbol = "MOCKDOT"; amount = 25.5
    mock_context.args = [symbol, str(amount)]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_db_write = mocker.patch("bot_commands.portfolio_utils.add_or_update_holding", return_value=True)
    await portfolio(mock_update, mock_context)
    mock_db_write.assert_called_once_with(user_id, symbol, amount); mock_message.reply_text.assert_called_once()
    assert f"✅ Portofoliu actualizat: {amount:.8f} {symbol}" in mock_message.reply_text.call_args.args[0]

@pytest.mark.asyncio
async def test_portfolio_command_delete_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id); symbol = "MOCKBTC"; amount = 0.0
    mock_context.args = [symbol, str(amount)]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_db_write = mocker.patch("bot_commands.portfolio_utils.add_or_update_holding", return_value=True)
    await portfolio(mock_update, mock_context)
    mock_db_write.assert_called_once_with(user_id, symbol, amount); mock_message.reply_text.assert_called_once()
    assert f"✅ {symbol} eliminat din portofoliu" in mock_message.reply_text.call_args.args[0]

@pytest.mark.asyncio
async def test_portfolio_command_db_error(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id); symbol = "MOCKETH"; amount = 1.0
    mock_context.args = [symbol, str(amount)]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_db_write = mocker.patch("bot_commands.portfolio_utils.add_or_update_holding", return_value=False) # Simulare eșec DB
    await portfolio(mock_update, mock_context)
    mock_db_write.assert_called_once_with(user_id, symbol, amount); mock_message.reply_text.assert_called_once()
    assert "❌ Eroare internă la salvarea în baza de date" in mock_message.reply_text.call_args.args[0]

# --- Teste pentru /myportfolio (neschimbate, ar trebui să fie OK) ---
@pytest.mark.asyncio
async def test_myportfolio_command_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id)
    mock_holdings = {"BTC": 0.5, "ETH": 2.0}; mock_db_read = mocker.patch("bot_commands.portfolio_utils.get_user_holdings", return_value=mock_holdings)
    btc_price = "80000.00"; eth_price = "4000.00"; btc_url = f"{config.BINANCE_API_BASE}/ticker/price?symbol=BTCUSDT"; eth_url = f"{config.BINANCE_API_BASE}/ticker/price?symbol=ETHUSDT"
    with aioresponses() as m:
        m.get(btc_url, payload={"price": btc_price}); m.get(eth_url, payload={"price": eth_price})
        await myportfolio(mock_update, mock_context)
    mock_db_read.assert_called_once_with(user_id); assert mock_message.reply_text.call_count == 2; calls = mock_message.reply_text.call_args_list
    assert "🔄 Încărcare și evaluare portofoliu" in calls[0].args[0]
    final_reply = calls[1].args[0]; assert "📊 *Portofoliul tău:*" in final_reply
    assert f"• BTC: {mock_holdings['BTC']:.8f} × ${float(btc_price):,.2f} = *${mock_holdings['BTC'] * float(btc_price):,.2f}*" in final_reply
    assert f"• ETH: {mock_holdings['ETH']:.8f} × ${float(eth_price):,.2f} = *${mock_holdings['ETH'] * float(eth_price):,.2f}*" in final_reply
    total_expected = (mock_holdings['BTC'] * float(btc_price)) + (mock_holdings['ETH'] * float(eth_price))
    assert f"💰 *Valoare totală estimată:* ${total_expected:,.2f}" in final_reply; assert calls[1].kwargs.get('parse_mode') == 'Markdown'

@pytest.mark.asyncio
async def test_myportfolio_command_empty(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id)
    mock_db_read = mocker.patch("bot_commands.portfolio_utils.get_user_holdings", return_value={}) # Portofoliu gol
    await myportfolio(mock_update, mock_context)
    mock_db_read.assert_called_once_with(user_id); assert mock_message.reply_text.call_count == 2
    assert "📭 Portofoliu gol." in mock_message.reply_text.call_args_list[1].args[0]

# --- Test Nou Corectat pentru /news ---
@pytest.mark.asyncio
async def test_news_command_success(mocker, mock_update_message, mock_context):
    """Testează /news simulând parse_rss_feed cu side_effect."""
    mock_update, mock_message = mock_update_message

    # Creăm intrări RSS simulate
    entry1_ts = time.time() - 3600; entry2_ts = time.time() - 7200
    mock_entry1 = {"title": "Știre Nouă 1", "link": "link1", "published_parsed": time.gmtime(entry1_ts), 'source_name': 'Sursa A'}
    mock_entry2 = {"title": "Știre Veche 2", "link": "link2", "published_parsed": time.gmtime(entry2_ts), 'source_name': 'Sursa B'}

    # Simulăm rezultatele pe care le-ar returna apelurile succesive către parse_rss_feed
    mock_feed_result1 = MagicMock(); mock_feed_result1.entries = [mock_entry1]
    mock_feed_result2 = MagicMock(); mock_feed_result2.entries = [mock_entry2]
    # Presupunem că avem 4 feed-uri în config, două reușesc, două eșuează (returnează None)
    side_effects = [mock_feed_result1, mock_feed_result2, None, None]
    # --- FIX 2: Mock parse_rss_feed în loc de asyncio.gather ---
    mock_parse_func = mocker.patch("bot_commands.data_utils.parse_rss_feed", side_effect=side_effects)
    # ---------------------------------------------------------

    await news(mock_update, mock_context)

    # Verificăm că parse_rss_feed a fost apelat pentru fiecare feed din config
    assert mock_parse_func.call_count == len(config.RSS_FEEDS)

    # Verificăm răspunsurile bot-ului
    assert mock_message.reply_text.call_count == 2
    calls = mock_message.reply_text.call_args_list
    assert f"🔄 Căutare ultimelor {config.NEWS_COUNT} știri crypto (RSS)..." in calls[0].args[0] # Folosim constanta

    final_reply = calls[1].args[0]
    assert "📰 <b>Top 2 Știri Crypto" in final_reply # Am simulat 2 intrări valide
    assert "Știre Nouă 1" in final_reply; assert "Știre Veche 2" in final_reply
    assert final_reply.find("Știre Nouă 1") < final_reply.find("Știre Veche 2") # Verifică ordinea
    assert "link1" in final_reply; assert "link2" in final_reply
    assert "<i>(Sursa A" in final_reply; assert "<i>(Sursa B" in final_reply
    assert calls[1].kwargs.get('parse_mode') == 'HTML'
    assert calls[1].kwargs.get('disable_web_page_preview') == True

# --- Teste Noi Corectate pentru /crypto ---
@pytest.mark.asyncio
async def test_crypto_command_success(mock_update_message, mock_context):
    """Testează /crypto cu succes folosind aioresponses."""
    symbol = "MOCKXRP"; mock_update, mock_message = mock_update_message; mock_context.args = [symbol]
    url = f"{config.BINANCE_API_BASE}/ticker/24hr?symbol={symbol}USDT"
    mock_payload = {"lastPrice": "2.50", "highPrice": "2.60", "lowPrice": "2.40", "volume": "1000000", "quoteVolume": "2500000", "priceChangePercent": "4.17"}

    with aioresponses() as m:
        m.get(url, payload=mock_payload, status=200)
        await crypto(mock_update, mock_context)

    assert mock_message.reply_text.call_count == 2; calls = mock_message.reply_text.call_args_list
    assert "🔄 Căutare date live 24h" in calls[0].args[0]
    final_reply = calls[1].args[0]
    assert f"Info Live 24h: {config.CRYPTO_SYMBOLS_CONFIG.get(symbol, symbol)} ({symbol})" in final_reply
    # --- FIX 3: Adăugăm spațiu înainte de $ ---
    assert f"*{float(mock_payload['lastPrice']):,.2f} $*" in final_reply
    # ----------------------------------------
    assert f"{float(mock_payload['lowPrice']):,.2f} $" in final_reply
    assert f"{float(mock_payload['highPrice']):,.2f} $" in final_reply
    assert f"{float(mock_payload['priceChangePercent']):+.2f}%" in final_reply
    assert f"{float(mock_payload['volume']):,.2f}" in final_reply
    assert f"{float(mock_payload['quoteVolume']):,.2f} $" in final_reply
    assert calls[1].kwargs.get('parse_mode') == 'Markdown'

@pytest.mark.asyncio
async def test_crypto_command_not_found(mock_update_message, mock_context):
    """Testează /crypto pentru un simbol negăsit pe Binance."""
    symbol = "NOTFOUND"; mock_update, mock_message = mock_update_message; mock_context.args = [symbol]
    url = f"{config.BINANCE_API_BASE}/ticker/24hr?symbol={symbol}USDT"

    with aioresponses() as m:
        m.get(url, status=400)
        await crypto(mock_update, mock_context)

    assert mock_message.reply_text.call_count == 2
    # --- FIX 4: Corectăm textul așteptat ---
    expected_error_text = f"❌ '{symbol}USDT' negăsit pe Binance."
    actual_text = mock_message.reply_text.call_args_list[1].args[0]
    assert actual_text == expected_error_text
    # -----------------------------------