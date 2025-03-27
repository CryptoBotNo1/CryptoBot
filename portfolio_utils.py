# portfolio_utils.py
import logging
import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any

# Importă configurările necesare
import config

logger = logging.getLogger(__name__)

# --- Inițializare Bază de Date ---

def init_db():
    """
    Creează tabela portfolio în baza de date SQLite dacă nu există.
    Rulează sincron la pornirea aplicației.
    """
    try:
        with sqlite3.connect(config.DATABASE_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    amount REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    PRIMARY KEY (user_id, symbol)
                )
            """)
            conn.commit()
            logger.info(f"Baza de date SQLite inițializată/verificată: {config.DATABASE_FILE}")
    except sqlite3.Error as e:
        logger.critical(f"FATAL: Nu s-a putut inițializa baza de date SQLite! {e}", exc_info=True)
        # Oprim aplicația dacă nu putem inițializa DB-ul, deoarece portofoliul nu va funcționa
        raise SystemExit(f"Database initialization failed: {e}")
    except Exception as e:
        logger.critical(f"FATAL: Eroare necunoscută la inițializarea bazei de date: {e}", exc_info=True)
        raise SystemExit(f"Unknown database initialization error: {e}")

# --- Helper pentru Rularea Query-urilor DB în Thread Separat ---

async def _run_db_query(sql: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False, is_write: bool = False):
    """
    Rulează sincron o operație SQLite într-un thread separat pentru a nu bloca asyncio.
    Gestionează conexiunea, cursorul, commit-ul și erorile.
    """
    def db_operation():
        # Funcția sincronă care va rula în thread
        conn = None # Inițializăm conn cu None
        try:
            conn = sqlite3.connect(config.DATABASE_FILE, timeout=10) # Timeout de 10 secunde
            cursor = conn.cursor()
            cursor.execute(sql, params)

            if is_write:
                conn.commit()
                logger.debug(f"Query WRITE executat cu succes: {sql[:50]}...")
                return True # Succes pentru operații de scriere

            result = None
            if fetch_one:
                result = cursor.fetchone()
            elif fetch_all:
                result = cursor.fetchall()

            logger.debug(f"Query READ executat cu succes: {sql[:50]}... -> Rows: {'1' if fetch_one and result else len(result) if fetch_all else 'N/A'}")
            return result

        except sqlite3.Error as e:
            logger.error(f"Eroare SQLite la executarea query: {sql[:50]}... Params: {params}. Eroare: {e}", exc_info=True)
            if is_write and conn:
                 try:
                     conn.rollback() # Încearcă rollback la eroare de scriere
                 except sqlite3.Error as rb_e:
                     logger.error(f"Eroare la rollback SQLite: {rb_e}")
            return False if is_write else None # Returnează False la eroare de scriere, None la eroare de citire

        except Exception as e:
             logger.error(f"Eroare generală în db_operation: {e}", exc_info=True)
             return False if is_write else None
        finally:
            if conn:
                try:
                    conn.close()
                except sqlite3.Error as close_e:
                     logger.error(f"Eroare la închiderea conexiunii SQLite: {close_e}")

    # Rulează funcția sincronă db_operation în thread pool-ul default asyncio
    try:
        return await asyncio.to_thread(db_operation)
    except Exception as e:
        logger.error(f"Eroare la rularea asyncio.to_thread pentru DB: {e}", exc_info=True)
        return False if is_write else None


# --- Funcții Principale de Gestionare Portofoliu ---

async def add_or_update_holding(user_id: str, symbol: str, amount: float) -> bool:
    """
    Adaugă, actualizează sau șterge o deținere în baza de date.
    Folosește INSERT OR REPLACE pentru simplitate.
    Șterge intrarea dacă amount este sub pragul ZERO_THRESHOLD.
    Returnează True la succes, False la eșec.
    """
    current_timestamp = datetime.now().isoformat()

    if amount > config.ZERO_THRESHOLD:
        # Adaugă sau înlocuiește intrarea existentă
        sql = """
            INSERT OR REPLACE INTO portfolio (user_id, symbol, amount, timestamp)
            VALUES (?, ?, ?, ?)
        """
        params = (user_id, symbol, amount, current_timestamp)
        logger.info(f"DB: Pregătire INSERT/REPLACE pentru user {user_id}, {symbol}, {amount:.8f}")
        return await _run_db_query(sql, params, is_write=True)
    else:
        # Șterge intrarea dacă amount este considerat zero
        sql = "DELETE FROM portfolio WHERE user_id = ? AND symbol = ?"
        params = (user_id, symbol)
        logger.info(f"DB: Pregătire DELETE pentru user {user_id}, {symbol} (amount <= {config.ZERO_THRESHOLD})")
        return await _run_db_query(sql, params, is_write=True)

async def get_user_holdings(user_id: str) -> Optional[Dict[str, float]]:
    """
    Obține toate deținerile (cu amount > ZERO_THRESHOLD) pentru un utilizator specific.
    Returnează un dicționar {simbol: cantitate} sau None în caz de eroare.
    """
    sql = "SELECT symbol, amount FROM portfolio WHERE user_id = ? AND amount > ?"
    params = (user_id, config.ZERO_THRESHOLD)
    logger.debug(f"DB: Pregătire SELECT pentru user {user_id}")

    results = await _run_db_query(sql, params, fetch_all=True)

    if results is None: # Verificăm dacă a fost eroare la query
         logger.error(f"DB: Query SELECT pentru user {user_id} a eșuat returnând None.")
         return None

    # Procesează rezultatele într-un dicționar
    holdings_dict = {symbol: amount for symbol, amount in results}
    logger.info(f"DB: Găsit {len(holdings_dict)} dețineri pentru user {user_id}")
    return holdings_dict