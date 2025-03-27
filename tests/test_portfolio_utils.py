# tests/test_portfolio_utils.py
import pytest
import pytest_asyncio # Necesar pentru fixtures async
import os
import sqlite3
from datetime import datetime

# Importăm modulele/funcțiile pe care le testăm și config
# Ajustează calea dacă e necesar, dar pytest ar trebui să găsească modulele
import config
import portfolio_utils

# --- Fixture Pytest pentru Baza de Date Temporară ---
@pytest_asyncio.fixture(scope="function", autouse=True)
async def temp_db(tmp_path, monkeypatch):
    """
    Fixture care creează o bază de date SQLite temporară pentru fiecare test,
    configurează portfolio_utils să o folosească, și o șterge după test.
    'autouse=True' face ca fixture-ul să fie rulat automat pentru fiecare test din fișier.
    'scope="function"' înseamnă că se re-creează pentru fiecare funcție de test.
    'tmp_path' este un fixture pytest care oferă un director temporar unic.
    'monkeypatch' este un fixture pytest pentru a modifica temporar variabile/atribute.
    """
    # Creează un nume de fișier unic în directorul temporar
    test_db_path = tmp_path / "test_portfolio.db"
    print(f"Using temporary DB: {test_db_path}") # Debug: Afișează calea DB de test

    # Modifică temporar config.DATABASE_FILE pentru ca portfolio_utils să folosească DB-ul de test
    monkeypatch.setattr(config, 'DATABASE_FILE', str(test_db_path))

    # Inițializează schema în baza de date de test (rulează sincron, e ok aici)
    portfolio_utils.init_db()

    # --- Rulează testul ---
    yield test_db_path # Ofera calea către DB testului, dacă e necesar

    # --- Curățenie după test ---
    # (tmp_path este șters automat de pytest, deci nu trebuie să ștergem fișierul manual)
    # Dar putem verifica dacă există dacă vrem
    # if os.path.exists(test_db_path):
    #     os.remove(test_db_path)
    #     print(f"Removed temporary DB: {test_db_path}")


# --- Teste ---
# Folosim 'pytest.mark.asyncio' pentru funcțiile de test asincrone

@pytest.mark.asyncio
async def test_add_single_holding():
    """Testează adăugarea unei singure dețineri."""
    user_id = "user1"
    symbol = "BTC"
    amount = 0.5

    success = await portfolio_utils.add_or_update_holding(user_id, symbol, amount)
    assert success == True # Verifică dacă funcția a raportat succes

    # Verifică direct în baza de date (folosind sqlite3 sincron în test e ok)
    conn = sqlite3.connect(config.DATABASE_FILE) # Se va conecta la test_db_path datorită monkeypatch
    cursor = conn.cursor()
    cursor.execute("SELECT amount FROM portfolio WHERE user_id = ? AND symbol = ?", (user_id, symbol))
    result = cursor.fetchone()
    conn.close()

    assert result is not None # Verifică dacă înregistrarea există
    assert result[0] == pytest.approx(amount) # Verifică dacă amount este corect (folosim approx pt float)

@pytest.mark.asyncio
async def test_update_holding():
    """Testează actualizarea unei dețineri existente."""
    user_id = "user2"
    symbol = "ETH"
    initial_amount = 1.0
    updated_amount = 1.2

    await portfolio_utils.add_or_update_holding(user_id, symbol, initial_amount) # Adaugă inițial
    success = await portfolio_utils.add_or_update_holding(user_id, symbol, updated_amount) # Actualizează
    assert success == True

    # Verifică dacă amount a fost actualizat
    holdings = await portfolio_utils.get_user_holdings(user_id)
    assert holdings is not None
    assert len(holdings) == 1
    assert symbol in holdings
    assert holdings[symbol] == pytest.approx(updated_amount)

@pytest.mark.asyncio
async def test_delete_holding_by_zero_amount():
    """Testează ștergerea unei dețineri prin setarea cantității la zero."""
    user_id = "user3"
    symbol = "ADA"
    initial_amount = 10.0

    await portfolio_utils.add_or_update_holding(user_id, symbol, initial_amount) # Adaugă
    success = await portfolio_utils.add_or_update_holding(user_id, symbol, 0) # Setează la 0 pt ștergere
    assert success == True

    # Verifică dacă deținerea nu mai există
    holdings = await portfolio_utils.get_user_holdings(user_id)
    assert holdings is not None
    assert len(holdings) == 0 # Dicționarul ar trebui să fie gol

@pytest.mark.asyncio
async def test_get_user_holdings_multiple():
    """Testează obținerea mai multor dețineri pentru același utilizator."""
    user_id = "user4"
    holdings_to_add = {"BTC": 0.1, "ETH": 2.5, "DOT": 100.0}

    for symbol, amount in holdings_to_add.items():
        await portfolio_utils.add_or_update_holding(user_id, symbol, amount)

    retrieved_holdings = await portfolio_utils.get_user_holdings(user_id)
    assert retrieved_holdings is not None
    assert len(retrieved_holdings) == len(holdings_to_add)
    # Verifică dacă toate cantitățile sunt corecte
    for symbol, amount in holdings_to_add.items():
        assert symbol in retrieved_holdings
        assert retrieved_holdings[symbol] == pytest.approx(amount)

@pytest.mark.asyncio
async def test_get_user_holdings_isolation():
    """Testează dacă get_user_holdings returnează doar datele utilizatorului specificat."""
    user_id_1 = "user5"
    user_id_2 = "user6"

    await portfolio_utils.add_or_update_holding(user_id_1, "SOL", 5.0)
    await portfolio_utils.add_or_update_holding(user_id_2, "LTC", 2.0)

    holdings1 = await portfolio_utils.get_user_holdings(user_id_1)
    holdings2 = await portfolio_utils.get_user_holdings(user_id_2)

    assert holdings1 is not None and "SOL" in holdings1 and len(holdings1) == 1
    assert holdings2 is not None and "LTC" in holdings2 and len(holdings2) == 1
    assert "LTC" not in holdings1
    assert "SOL" not in holdings2

@pytest.mark.asyncio
async def test_get_nonexistent_user_holdings():
    """Testează obținerea deținerilor pentru un utilizator care nu are date."""
    holdings = await portfolio_utils.get_user_holdings("nonexistent_user")
    assert holdings is not None # Funcția ar trebui să ruleze fără eroare
    assert len(holdings) == 0 # și să returneze un dicționar gol