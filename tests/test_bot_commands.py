# tests/test_bot_commands.py - Corectat Final (v4 - Sper!)
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from io import BytesIO
import time

# Importăm obiectele Telegram necesare pentru simulare
from telegram import Update, Message, Chat, User, MessageEntity, InputFile

# Importăm handler-ele și funcțiile testate
from bot_commands import (
    start, help_command, predict, predict_chart, summary, news,
    crypto, grafic, trend, compare, portfolio, myportfolio,
    _set_sentry_user_context, is_valid_symbol, is_valid_days
)
# Importăm și constante/utilitare folosite de handlere
import config
import ai_utils
import data_utils
import portfolio_utils
import plot_utils # Importăm plot_utils

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
    mock_update.message.reply_photo = AsyncMock()
    return mock_update, mock_update.message

# --- Test pentru /start ---
@pytest.mark.asyncio
async def test_start_command(mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message
    await start(mock_update, mock_context); mock_message.reply_text.assert_called_once()
    args, kwargs = mock_message.reply_text.call_args
    assert "👋 Salut! Sunt botul tău AI pentru crypto." in args[0]

# --- Test pentru /help ---
@pytest.mark.asyncio
async def test_help_command(mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message
    await help_command(mock_update, mock_context); mock_message.reply_text.assert_called_once()
    args, kwargs = mock_message.reply_text.call_args; reply_text = args[0]
    assert "📋 *Comenzi disponibile:*" in reply_text; assert "/start - Pornește botul" in reply_text
    assert "/predict SIMBOL ZILE" in reply_text; assert "/portfolio SIMBOL CANTITATE" in reply_text
    assert f"*Nr. zile predicție:* 1-{config.PREDICT_MAX_DAYS}" in reply_text
    assert "BTC" in reply_text; assert kwargs.get('parse_mode') == 'Markdown'

# --- Teste pentru /predict ---
@pytest.mark.asyncio
async def test_predict_command_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; symbol = "MOCKBTC"; days = 3
    mock_update.effective_user = User(id=456, first_name="PredTester", is_bot=False)
    mock_context.args = [symbol, str(days)]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_prediction_result = [100.0, 110.5, 108.0]; mock_predict_seq = mocker.patch("bot_commands.ai_utils.predict_sequence", return_value=mock_prediction_result)
    await predict(mock_update, mock_context); mock_predict_seq.assert_called_once_with(symbol, days); assert mock_message.reply_text.call_count == 2
    call_list = mock_message.reply_text.call_args_list; assert "🔄 Generare predicție AI" in call_list[0].args[0]
    assert "📊 *Predicții AI" in call_list[1].args[0]; assert "*100.00* USD" in call_list[1].args[0]
    assert "*110.50* USD" in call_list[1].args[0]; assert "*108.00* USD" in call_list[1].args[0]
    assert call_list[1].kwargs.get('parse_mode') == 'Markdown'

@pytest.mark.asyncio
async def test_predict_command_invalid_symbol(mocker, mock_update_message, mock_context):
    """Testează /predict cu un simbol invalid."""
    mock_update, mock_message = mock_update_message
    symbol = "INVALID"; days = 5
    mock_update.effective_user = User(id=789, first_name="InvUser", is_bot=False)
    mock_context.args = [symbol, str(days)]
    mocker.patch("bot_commands.is_valid_symbol", return_value=False) # Mock is_valid_symbol să returneze False
    await predict(mock_update, mock_context)
    mock_message.reply_text.assert_called_once()
    # --- FIX Definitiv: Textul așteptat corect, FĂRĂ 'ul' ---
    expected_error_text = f"❌ Simbol '{symbol}' nesuportat. Folosește /help."
    # ------------------------------------------------------
    actual_text = mock_message.reply_text.call_args.args[0]
    assert actual_text == expected_error_text

@pytest.mark.asyncio
async def test_predict_command_invalid_days(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; symbol = "BTC"; days_str = "abc"
    mock_update.effective_user = User(id=101, first_name="DayUser", is_bot=False)
    mock_context.args = [symbol, days_str]; await predict(mock_update, mock_context)
    mock_message.reply_text.assert_called_once(); expected_error_text = f"❌ Nr. zile invalid (1-{config.PREDICT_MAX_DAYS})."
    assert expected_error_text in mock_message.reply_text.call_args.args[0]

# --- Teste pentru /predict_chart ---
@pytest.mark.asyncio
async def test_predict_chart_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; symbol = "MOCKET"; days = 5
    mock_context.args = [symbol, str(days)]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_predictions = [200.0, 210.0, 205.0, 215.0, 220.0]; mock_predict_seq = mocker.patch("bot_commands.ai_utils.predict_sequence", return_value=mock_predictions)
    dummy_bytes = BytesIO(b"dummy png data"); mock_plot = mocker.patch("bot_commands.plot_utils.generate_prediction_plot", return_value=dummy_bytes)
    mock_dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=days); mocker.patch("bot_commands.pd.date_range", return_value=mock_dates)
    await predict_chart(mock_update, mock_context); mock_predict_seq.assert_called_once_with(symbol, days); mock_plot.assert_called_once()
    plot_args, _ = mock_plot.call_args; assert plot_args[0] == symbol; assert len(plot_args[1]) == days; assert plot_args[2] == mock_predictions
    assert mock_message.reply_text.call_count == 1; assert f"🔄 Generare grafic AI pentru {symbol}" in mock_message.reply_text.call_args.args[0]
    mock_message.reply_photo.assert_called_once(); photo_kwargs = mock_message.reply_photo.call_args.kwargs
    assert f"Grafic predicție AI {symbol} ({days} zile)" in photo_kwargs.get('caption', ''); assert isinstance(photo_kwargs.get('photo'), InputFile)

@pytest.mark.asyncio
async def test_predict_chart_predict_fails(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; symbol = "MOCKFAIL"; days = 3
    mock_context.args = [symbol, str(days)]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_predict_seq = mocker.patch("bot_commands.ai_utils.predict_sequence", return_value=[]); mock_plot = mocker.patch("bot_commands.plot_utils.generate_prediction_plot")
    await predict_chart(mock_update, mock_context); mock_predict_seq.assert_called_once_with(symbol, days)
    assert mock_message.reply_text.call_count == 2; assert f"❌ Eroare date predicție grafic {symbol}" in mock_message.reply_text.call_args_list[1].args[0]
    mock_plot.assert_not_called(); mock_message.reply_photo.assert_not_called()

# --- Teste pentru /summary ---
@pytest.mark.asyncio
async def test_summary_command_success(mocker, mock_update_message, mock_context):
    symbol = "MOCKSOL"; mock_update, mock_message = mock_update_message; mock_context.args = [symbol]
    mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    trend_days_count = config.TREND_DAYS + 1; mock_yf_data = pd.DataFrame({'Close': np.linspace(100, 110, trend_days_count)}, index=pd.date_range(end=datetime.now(), periods=trend_days_count, freq='D'))
    mock_yf = mocker.patch("bot_commands.yf.download", return_value=mock_yf_data)
    mock_prediction = [115.5]; mock_predict = mocker.patch("bot_commands.ai_utils.predict_sequence", return_value=mock_prediction)
    binance_url = f"{config.BINANCE_API_BASE}/ticker/price?symbol={symbol}USDT"; mock_live_price = "112.34"
    with aioresponses() as m:
        m.get(binance_url, payload={"price": mock_live_price}, status=200)
        await summary(mock_update, mock_context)
    mock_yf.assert_called_once(); mock_predict.assert_called_once_with(symbol, 1)
    assert mock_message.reply_text.call_count == 2; calls = mock_message.reply_text.call_args_list
    assert f"🔄 Generare sumar pentru {symbol}" in calls[0].args[0]; final_reply = calls[1].args[0]
    assert f"*{float(mock_live_price):,.2f}* $" in final_reply; assert f"*{10.00:+.2f}$*" in final_reply
    assert f"*{mock_prediction[0]:,.2f}* $" in final_reply; assert calls[1].kwargs.get('parse_mode') == 'Markdown'

# --- Teste pentru /portfolio ---
@pytest.mark.asyncio
async def test_portfolio_command_add_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id); symbol = "MOCKDOT"; amount = 25.5
    mock_context.args = [symbol, str(amount)]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_db_write = mocker.patch("bot_commands.portfolio_utils.add_or_update_holding", return_value=True)
    await portfolio(mock_update, mock_context); mock_db_write.assert_called_once_with(user_id, symbol, amount); mock_message.reply_text.assert_called_once()
    assert f"✅ Portofoliu actualizat: {amount:.8f} {symbol}" in mock_message.reply_text.call_args.args[0]

@pytest.mark.asyncio
async def test_portfolio_command_delete_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id); symbol = "MOCKBTC"; amount = 0.0
    mock_context.args = [symbol, str(amount)]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_db_write = mocker.patch("bot_commands.portfolio_utils.add_or_update_holding", return_value=True)
    await portfolio(mock_update, mock_context); mock_db_write.assert_called_once_with(user_id, symbol, amount); mock_message.reply_text.assert_called_once()
    assert f"✅ {symbol} eliminat din portofoliu" in mock_message.reply_text.call_args.args[0]

@pytest.mark.asyncio
async def test_portfolio_command_db_error(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id); symbol = "MOCKETH"; amount = 1.0
    mock_context.args = [symbol, str(amount)]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_db_write = mocker.patch("bot_commands.portfolio_utils.add_or_update_holding", return_value=False) # Simulare eșec DB
    await portfolio(mock_update, mock_context); mock_db_write.assert_called_once_with(user_id, symbol, amount); mock_message.reply_text.assert_called_once()
    assert "❌ Eroare internă la salvarea în baza de date" in mock_message.reply_text.call_args.args[0]

# --- Teste pentru /myportfolio ---
@pytest.mark.asyncio
async def test_myportfolio_command_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id)
    mock_holdings = {"BTC": 0.5, "ETH": 2.0}; mock_db_read = mocker.patch("bot_commands.portfolio_utils.get_user_holdings", return_value=mock_holdings)
    btc_price = "80000.00"; eth_price = "4000.00"; btc_url = f"{config.BINANCE_API_BASE}/ticker/price?symbol=BTCUSDT"; eth_url = f"{config.BINANCE_API_BASE}/ticker/price?symbol=ETHUSDT"
    with aioresponses() as m:
        m.get(btc_url, payload={"price": btc_price}); m.get(eth_url, payload={"price": eth_price})
        await myportfolio(mock_update, mock_context)
    mock_db_read.assert_called_once_with(user_id); assert mock_message.reply_text.call_count == 2; calls = mock_message.reply_text.call_args_list
    assert "🔄 Încărcare și evaluare portofoliu" in calls[0].args[0]; final_reply = calls[1].args[0]; assert "📊 *Portofoliul tău:*" in final_reply
    assert f"• BTC: {mock_holdings['BTC']:.8f} × ${float(btc_price):,.2f} = *${mock_holdings['BTC'] * float(btc_price):,.2f}*" in final_reply
    assert f"• ETH: {mock_holdings['ETH']:.8f} × ${float(eth_price):,.2f} = *${mock_holdings['ETH'] * float(eth_price):,.2f}*" in final_reply
    total_expected = (mock_holdings['BTC'] * float(btc_price)) + (mock_holdings['ETH'] * float(eth_price))
    assert f"💰 *Valoare totală estimată:* ${total_expected:,.2f}" in final_reply; assert calls[1].kwargs.get('parse_mode') == 'Markdown'

@pytest.mark.asyncio
async def test_myportfolio_command_empty(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; user_id = str(mock_update.effective_user.id)
    mock_db_read = mocker.patch("bot_commands.portfolio_utils.get_user_holdings", return_value={}) # Portofoliu gol
    await myportfolio(mock_update, mock_context); mock_db_read.assert_called_once_with(user_id); assert mock_message.reply_text.call_count == 2
    assert "📭 Portofoliu gol." in mock_message.reply_text.call_args_list[1].args[0]

# --- Test pentru /news ---
@pytest.mark.asyncio
async def test_news_command_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; entry1_ts = time.time() - 3600; entry2_ts = time.time() - 7200
    mock_entry1 = {"title": "Știre Nouă 1", "link": "link1", "published_parsed": time.gmtime(entry1_ts), 'source_name': 'Sursa A'}
    mock_entry2 = {"title": "Știre Veche 2", "link": "link2", "published_parsed": time.gmtime(entry2_ts), 'source_name': 'Sursa B'}
    mock_feed_result1 = MagicMock(); mock_feed_result1.entries = [mock_entry1]; mock_feed_result2 = MagicMock(); mock_feed_result2.entries = [mock_entry2]
    side_effects = [mock_feed_result1, mock_feed_result2, None, None] # Simulăm 4 feed-uri
    mock_parse_func = mocker.patch("bot_commands.data_utils.parse_rss_feed", side_effect=side_effects)
    await news(mock_update, mock_context); assert mock_parse_func.call_count == len(config.RSS_FEEDS)
    assert mock_message.reply_text.call_count == 2; calls = mock_message.reply_text.call_args_list
    assert f"🔄 Căutare ultimelor {config.NEWS_COUNT} știri crypto (RSS)..." in calls[0].args[0]; final_reply = calls[1].args[0]
    assert "📰 <b>Top 2 Știri Crypto" in final_reply; assert "Știre Nouă 1" in final_reply; assert "Știre Veche 2" in final_reply
    assert final_reply.find("Știre Nouă 1") < final_reply.find("Știre Veche 2"); assert "link1" in final_reply; assert "link2" in final_reply
    assert "<i>(Sursa A" in final_reply; assert "<i>(Sursa B" in final_reply
    assert calls[1].kwargs.get('parse_mode') == 'HTML'; assert calls[1].kwargs.get('disable_web_page_preview') == True

# --- Teste pentru /crypto ---
@pytest.mark.asyncio
async def test_crypto_command_success(mock_update_message, mock_context):
    symbol = "MOCKXRP"; mock_update, mock_message = mock_update_message; mock_context.args = [symbol]
    url = f"{config.BINANCE_API_BASE}/ticker/24hr?symbol={symbol}USDT"
    mock_payload = {"lastPrice": "2.50", "highPrice": "2.60", "lowPrice": "2.40", "volume": "1000000", "quoteVolume": "2500000", "priceChangePercent": "4.17"}
    with aioresponses() as m:
        m.get(url, payload=mock_payload, status=200)
        await crypto(mock_update, mock_context)
    assert mock_message.reply_text.call_count == 2; calls = mock_message.reply_text.call_args_list; assert "🔄 Căutare date live 24h" in calls[0].args[0]
    final_reply = calls[1].args[0]; assert f"Info Live 24h: {config.CRYPTO_SYMBOLS_CONFIG.get(symbol, symbol)} ({symbol})" in final_reply
    assert f"*{float(mock_payload['lastPrice']):,.2f} $*" in final_reply; assert f"{float(mock_payload['lowPrice']):,.2f} $" in final_reply
    assert f"{float(mock_payload['highPrice']):,.2f} $" in final_reply; assert f"{float(mock_payload['priceChangePercent']):+.2f}%" in final_reply
    assert f"{float(mock_payload['volume']):,.2f}" in final_reply; assert f"{float(mock_payload['quoteVolume']):,.2f} $" in final_reply
    assert calls[1].kwargs.get('parse_mode') == 'Markdown'

@pytest.mark.asyncio
async def test_crypto_command_not_found(mock_update_message, mock_context):
    symbol = "NOTFOUND"; mock_update, mock_message = mock_update_message; mock_context.args = [symbol]
    url = f"{config.BINANCE_API_BASE}/ticker/24hr?symbol={symbol}USDT"
    with aioresponses() as m: m.get(url, status=400)
    await crypto(mock_update, mock_context); assert mock_message.reply_text.call_count == 2
    expected_error_text = f"❌ '{symbol}USDT' negăsit pe Binance."
    actual_text = mock_message.reply_text.call_args_list[1].args[0]
    assert actual_text == expected_error_text

# --- Teste pentru /trend ---
@pytest.mark.asyncio
async def test_trend_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; symbol = "MOCKADA"
    mock_context.args = [symbol]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    trend_days_count = config.TREND_DAYS + 1; mock_yf_data = pd.DataFrame({'Close': np.linspace(50, 40, trend_days_count)}, index=pd.date_range(end=datetime.now(), periods=trend_days_count, freq='D'))
    mock_yf = mocker.patch("bot_commands.yf.download", return_value=mock_yf_data)
    await trend(mock_update, mock_context); mock_yf.assert_called_once()
    assert f"{config.TREND_DAYS + 1}d" in mock_yf.call_args.kwargs.get("period", "")
    assert mock_message.reply_text.call_count == 2; calls = mock_message.reply_text.call_args_list; assert f"🔄 Analiză trend {symbol}" in calls[0].args[0]
    final_reply = calls[1].args[0]; assert "📉 *Trend" in final_reply; assert "descendent*" in final_reply
    assert f"*{-10.00:+.2f}$*" in final_reply; assert f"({-20.00:+.2f}%)" in final_reply
    assert calls[1].kwargs.get('parse_mode') == 'Markdown'

@pytest.mark.asyncio
async def test_trend_yf_fails(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; symbol = "YFAIL"
    mock_context.args = [symbol]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_yf = mocker.patch("bot_commands.yf.download", return_value=pd.DataFrame()) # DataFrame gol
    await trend(mock_update, mock_context); mock_yf.assert_called_once()
    assert mock_message.reply_text.call_count == 2; assert f"❌ Date insuf. trend {symbol}" in mock_message.reply_text.call_args_list[1].args[0]

# --- Teste pentru /grafic ---
@pytest.mark.asyncio
async def test_grafic_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; symbol = "MOCKBNB"
    mock_context.args = [symbol]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_yf_data = pd.DataFrame({'Close': np.random.rand(config.CHART_DAYS)}, index=pd.date_range(end=datetime.now(), periods=config.CHART_DAYS, freq='D'))
    mock_yf = mocker.patch("bot_commands.yf.download", return_value=mock_yf_data)
    dummy_bytes = BytesIO(b"dummy history png"); mock_plot = mocker.patch("bot_commands.plot_utils.generate_price_history_plot", return_value=dummy_bytes)
    await grafic(mock_update, mock_context); mock_yf.assert_called_once(); assert f"{config.CHART_DAYS}d" in mock_yf.call_args.kwargs.get("period", "")
    mock_plot.assert_called_once(); plot_args, _ = mock_plot.call_args
    assert plot_args[0] == symbol; assert plot_args[1].equals(mock_yf_data); assert plot_args[2] == config.CHART_DAYS
    assert mock_message.reply_text.call_count == 1; mock_message.reply_photo.assert_called_once()
    photo_kwargs = mock_message.reply_photo.call_args.kwargs; assert f"Grafic preț {symbol} - {config.CHART_DAYS} zile" in photo_kwargs.get('caption', '')
    assert isinstance(photo_kwargs.get('photo'), InputFile)

@pytest.mark.asyncio
async def test_grafic_yf_fails(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; symbol = "YFAIL2"
    mock_context.args = [symbol]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_yf = mocker.patch("bot_commands.yf.download", return_value=pd.DataFrame()); mock_plot = mocker.patch("bot_commands.plot_utils.generate_price_history_plot")
    await grafic(mock_update, mock_context); mock_yf.assert_called_once()
    assert mock_message.reply_text.call_count == 2; assert f"❌ Eroare date yfinance {symbol}" in mock_message.reply_text.call_args_list[1].args[0]
    mock_plot.assert_not_called(); mock_message.reply_photo.assert_not_called()

# --- Teste pentru /compare ---
@pytest.mark.asyncio
async def test_compare_success(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; sym1 = "MCMP1"; sym2 = "MCMP2"
    mock_context.args = [sym1, sym2]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    df1_close_data = pd.Series(np.random.rand(config.CHART_DAYS), index=pd.date_range(end=datetime.now(), periods=config.CHART_DAYS, freq='D'), name='Close')
    df2_close_data = pd.Series(np.random.rand(config.CHART_DAYS), index=pd.date_range(end=datetime.now(), periods=config.CHART_DAYS, freq='D'), name='Close')
    mock_df1 = pd.DataFrame(df1_close_data); mock_df2 = pd.DataFrame(df2_close_data)
    mock_yf = mocker.patch("bot_commands.yf.download", side_effect=[mock_df1, mock_df2])
    dummy_bytes = BytesIO(b"dummy compare png"); mock_plot = mocker.patch("bot_commands.plot_utils.generate_comparison_plot", return_value=dummy_bytes)
    await compare(mock_update, mock_context); assert mock_yf.call_count == 2
    assert f"{sym1}-USD" in mock_yf.call_args_list[0].args[0]; assert f"{sym2}-USD" in mock_yf.call_args_list[1].args[0]
    assert f"{config.CHART_DAYS}d" == mock_yf.call_args_list[0].kwargs.get("period"); assert f"{config.CHART_DAYS}d" == mock_yf.call_args_list[1].kwargs.get("period")
    mock_plot.assert_called_once(); plot_args, _ = mock_plot.call_args; assert plot_args[0] == sym1; assert plot_args[1] == sym2
    assert plot_args[2].equals(mock_df1["Close"]); assert plot_args[3].equals(mock_df2["Close"]); assert plot_args[4] == config.CHART_DAYS
    assert mock_message.reply_text.call_count == 1; mock_message.reply_photo.assert_called_once()
    photo_kwargs = mock_message.reply_photo.call_args.kwargs; assert f"Comparație preț {sym1} vs {sym2}" in photo_kwargs.get('caption', '')
    assert isinstance(photo_kwargs.get('photo'), InputFile)

@pytest.mark.asyncio
async def test_compare_same_symbol(mocker, mock_update_message, mock_context):
    mock_update, mock_message = mock_update_message; sym = "MSAME"
    mock_context.args = [sym, sym]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    mock_yf = mocker.patch("bot_commands.yf.download"); mock_plot = mocker.patch("bot_commands.plot_utils.generate_comparison_plot")
    await compare(mock_update, mock_context); assert mock_message.reply_text.call_count == 1
    assert "❌ Alege simboluri diferite" in mock_message.reply_text.call_args.args[0]; mock_yf.assert_not_called(); mock_plot.assert_not_called(); mock_message.reply_photo.assert_not_called()

@pytest.mark.asyncio
async def test_compare_yf_fails(mocker, mock_update_message, mock_context):
    """Testează /compare când unul din apelurile yf.download eșuează."""
    mock_update, mock_message = mock_update_message; sym1 = "MOK"; sym2 = "MFAIL"
    mock_context.args = [sym1, sym2]; mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    # FIX: Primul returnează DataFrame, al doilea DataFrame gol
    df1_close_data = pd.Series(np.random.rand(config.CHART_DAYS), name='Close')
    mock_df1 = pd.DataFrame(df1_close_data)
    mock_yf = mocker.patch("bot_commands.yf.download", side_effect=[mock_df1, pd.DataFrame()]) # Al doilea e DF gol
    # ---------------------------------------------
    mock_plot = mocker.patch("bot_commands.plot_utils.generate_comparison_plot")
    await compare(mock_update, mock_context); assert mock_yf.call_count == 2
    assert mock_message.reply_text.call_count == 2; assert f"❌ Eroare date yfinance: {sym2}" in mock_message.reply_text.call_args_list[1].args[0]
    mock_plot.assert_not_called(); mock_message.reply_photo.assert_not_called()

# --- Aici se termină codul fișierului tests/test_bot_commands.py ---