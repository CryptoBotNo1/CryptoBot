# tests/test_bot_commands.py
import pytest
from unittest.mock import AsyncMock, MagicMock, call # Importăm și 'call'
import pandas as pd
from datetime import datetime, timedelta

# Importăm obiectele Telegram necesare pentru simulare
from telegram import Update, Message, Chat, User, MessageEntity

# Importăm handler-ele și funcțiile testate
from bot_commands import start, predict, _set_sentry_user_context, is_valid_symbol, is_valid_days
# Importăm și constante/utilitare folosite de handlere, dacă e necesar pt setup
import config
import ai_utils # Importăm pentru a putea mock-ui funcția de predicție

# --- Mock Global pentru Sentry Context ---
@pytest.fixture(autouse=True)
def mock_sentry_context(mocker):
    mocker.patch("bot_commands._set_sentry_user_context")

# --- Test pentru /start ---
@pytest.mark.asyncio
async def test_start_command():
    """Testează dacă handler-ul /start răspunde cu mesajul corect."""
    # Arrange
    mock_update = MagicMock(spec=Update)
    mock_update.message = AsyncMock(spec=Message)
    mock_update.effective_user = User(id=123, first_name="Tester", is_bot=False)
    mock_context = AsyncMock()

    # Act
    await start(mock_update, mock_context)

    # Assert
    mock_update.message.reply_text.assert_called_once()
    args, kwargs = mock_update.message.reply_text.call_args
    assert "👋 Salut! Sunt botul tău AI pentru crypto." in args[0]

# --- Teste pentru /predict ---
@pytest.mark.asyncio
async def test_predict_command_success(mocker):
    """Testează /predict pentru un input valid, simulând un răspuns AI."""
    # Arrange
    symbol = "MOCKBTC" # Folosim un simbol mock
    days = 3
    mock_user_id = 456
    mock_update = MagicMock(spec=Update)
    mock_update.effective_user = User(id=mock_user_id, first_name="PredTester", is_bot=False)
    mock_update.message = AsyncMock(spec=Message)
    mock_context = MagicMock()
    mock_context.args = [symbol, str(days)]

    # --- FIX 1: Mock is_valid_symbol să returneze True pentru simbolul mock ---
    mocker.patch("bot_commands.is_valid_symbol", return_value=True)
    # ---------------------------------------------------------------------

    # Mock funcția predict_sequence din ai_utils
    mock_prediction_result = [100.0, 110.5, 108.0]
    mock_predict_seq = mocker.patch("bot_commands.ai_utils.predict_sequence", return_value=mock_prediction_result)

    # Act
    await predict(mock_update, mock_context)

    # Assert
    # 0. Verificăm că is_valid_symbol a fost apelat (opțional, dar bun pt debug)
    #    Nu putem face assert direct pe mock-ul de mai sus DACA e folosit și în alte teste
    #    fără a-l reseta. Mai simplu e să verificăm pașii următori.

    # 1. Verificăm dacă predict_sequence a fost apelat corect ACUM
    mock_predict_seq.assert_called_once_with(symbol, days)

    # 2. Verificăm dacă reply_text a fost apelat de două ori
    assert mock_update.message.reply_text.call_count == 2

    # 3. Verificăm conținutul apelurilor reply_text
    call_list = mock_update.message.reply_text.call_args_list
    assert "🔄 Generare predicție AI" in call_list[0].args[0]
    assert "📊 *Predicții AI" in call_list[1].args[0]
    assert "*100.00* USD" in call_list[1].args[0]
    assert "*110.50* USD" in call_list[1].args[0]
    assert "*108.00* USD" in call_list[1].args[0]
    assert call_list[1].kwargs.get('parse_mode') == 'Markdown'

@pytest.mark.asyncio
async def test_predict_command_invalid_symbol(mocker):
    """Testează /predict cu un simbol invalid."""
    # Arrange
    symbol = "INVALID"
    days = 5
    mock_update = MagicMock(spec=Update)
    mock_update.effective_user = User(id=789, first_name="InvUser", is_bot=False)
    mock_update.message = AsyncMock(spec=Message)
    mock_context = MagicMock()
    mock_context.args = [symbol, str(days)]

    mocker.patch("bot_commands.is_valid_symbol", return_value=False)

    # Act
    await predict(mock_update, mock_context)

    # Assert
    mock_update.message.reply_text.assert_called_once()
    # --- FIX 2: Folosim egalitate exactă '==' în loc de 'in' ---
    expected_error_text = f"❌ Simbol '{symbol}' nesuportat. Folosește /help."
    actual_text = mock_update.message.reply_text.call_args.args[0]
    # print(f"\nDEBUG: Expected: {expected_error_text!r}") # Debug (opțional)
    # print(f"DEBUG: Actual:   {actual_text!r}") # Debug (opțional)
    assert actual_text == expected_error_text
    # ------------------------------------------------------

@pytest.mark.asyncio
async def test_predict_command_invalid_days(mocker):
    """Testează /predict cu un număr de zile invalid."""
    # Arrange
    symbol = "BTC" # Folosim un simbol valid aici
    days_str = "abc" # Invalid
    mock_update = MagicMock(spec=Update)
    mock_update.effective_user = User(id=101, first_name="DayUser", is_bot=False)
    mock_update.message = AsyncMock(spec=Message)
    mock_context = MagicMock()
    mock_context.args = [symbol, days_str]

    # Nu mai e nevoie să mock-uim is_valid_symbol dacă folosim "BTC"
    # mocker.patch("bot_commands.is_valid_symbol", return_value=True)

    # Act
    await predict(mock_update, mock_context)

    # Assert
    mock_update.message.reply_text.assert_called_once()
    # Verificăm cu 'in' sau '=='
    expected_error_text = f"❌ Nr. zile invalid (1-{config.PREDICT_MAX_DAYS})."
    assert expected_error_text in mock_update.message.reply_text.call_args.args[0]