# migrate_csv_to_db.py
import csv
import sqlite3
import logging
from datetime import datetime
import os # Necesităm os pentru a găsi fișierele

# Configurare minimă necesară pentru scriptul de migrare
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_CSV_FILE = os.path.join(BASE_DIR, "portfolio.csv") # Fișierul CSV vechi
DATABASE_FILE = os.path.join(BASE_DIR, "portfolio.db") # Fișierul DB nou
PORTFOLIO_HEADER = ["user_id", "symbol", "amount", "timestamp"] # Header-ul CSV așteptat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MigrationScript")

def create_table_if_not_exists(conn):
    """Creează tabela în DB dacă nu există."""
    try:
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
        logger.info("Tabela 'portfolio' verificată/creată în baza de date.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Eroare la crearea tabelei: {e}")
        return False

def migrate_data():
    """Citește din CSV și inserează/înlocuiește în SQLite."""
    if not os.path.exists(OLD_CSV_FILE):
        logger.warning(f"Fișierul CSV '{OLD_CSV_FILE}' nu a fost găsit. Nu se poate migra.")
        return

    conn = None
    inserted_count = 0
    skipped_count = 0
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        if not create_table_if_not_exists(conn):
            return # Oprește dacă nu se poate crea tabela

        cursor = conn.cursor()

        with open(OLD_CSV_FILE, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames != PORTFOLIO_HEADER:
                 logger.error(f"Header incorect în {OLD_CSV_FILE}. Așteptat: {PORTFOLIO_HEADER}. Migrare anulată.")
                 return

            logger.info(f"Început migrarea datelor din {OLD_CSV_FILE} în {DATABASE_FILE}...")
            for row in reader:
                try:
                    user_id = row['user_id']
                    symbol = row['symbol'].upper() # Asigurăm uppercase la simbol
                    amount = float(row['amount'])
                    # Folosim timestamp-ul din CSV dacă există și e valid, altfel folosim data curentă
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp']).isoformat()
                    except (ValueError, KeyError):
                        timestamp = datetime.now().isoformat()

                    # Folosim INSERT OR REPLACE pentru a gestiona duplicatele (înlocuiește cu rândul curent)
                    # sau INSERT OR IGNORE dacă vrei să păstrezi prima intrare întâlnită.
                    # Alegem REPLACE pentru a avea cele mai noi date dacă CSV avea duplicate.
                    sql = """
                        INSERT OR REPLACE INTO portfolio (user_id, symbol, amount, timestamp)
                        VALUES (?, ?, ?, ?)
                    """
                    cursor.execute(sql, (user_id, symbol, amount, timestamp))
                    inserted_count += 1
                    if inserted_count % 100 == 0: # Loghează progresul la fiecare 100 de rânduri
                        logger.info(f"Migrat {inserted_count} rânduri...")

                except (ValueError, KeyError) as e:
                    logger.warning(f"Rând invalid în CSV ignorat: {row}. Eroare: {e}")
                    skipped_count += 1
                    continue

        conn.commit() # Salvează toate modificările la sfârșit
        logger.info(f"Migrare finalizată! {inserted_count} rânduri procesate și inserate/înlocuite.")
        if skipped_count > 0:
             logger.warning(f"{skipped_count} rânduri au fost ignorate din cauza erorilor.")

    except FileNotFoundError:
        logger.error(f"Fișierul CSV '{OLD_CSV_FILE}' nu a fost găsit.")
    except sqlite3.Error as e:
        logger.error(f"Eroare SQLite în timpul migrării: {e}", exc_info=True)
        if conn:
             try: conn.rollback()
             except: pass # Ignoră erori la rollback
    except Exception as e:
        logger.error(f"Eroare necunoscută în timpul migrării: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    migrate_data()