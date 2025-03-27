# plot_utils.py
import logging
import matplotlib
# matplotlib.use('Agg') # Se setează în main.py
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from io import BytesIO
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

def generate_prediction_plot(symbol: str, dates: pd.DatetimeIndex, predictions: List[float]) -> BytesIO:
    """Generează graficul predicțiilor AI și returnează bufferul PNG."""
    logger.debug(f"Generare grafic predicție pentru {symbol} ({len(predictions)} zile)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, predictions, marker='o', linestyle='-', color='blue', label='Predicție AI')
    ax.set_title(f"Predicție AI Preț {symbol} - Următoarele {len(predictions)} zile")
    ax.set_xlabel("Dată")
    ax.set_ylabel("Preț (USD)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buf = BytesIO()
    try:
        plt.savefig(buf, format='PNG')
        logger.debug(f"Grafic predicție {symbol} salvat în buffer.")
    except Exception as e:
         logger.error(f"Eroare la salvarea graficului de predicție {symbol} în buffer: {e}", exc_info=True)
         # buf va fi gol sau parțial, dar îl returnăm oricum
    finally:
        plt.close(fig) # Esențial pentru a elibera memoria
    buf.seek(0)
    return buf

def generate_price_history_plot(symbol: str, df: pd.DataFrame, days: int) -> BytesIO:
    """Generează graficul istoricului de preț și returnează bufferul PNG."""
    logger.debug(f"Generare grafic istoric preț pentru {symbol} ({days} zile)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["Close"], label=f"{symbol} Preț", color='purple')
    ax.set_title(f"Evoluție Preț {symbol} - Ultimele {days} zile")
    ax.set_xlabel("Dată")
    ax.set_ylabel("Preț (USD)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buf = BytesIO()
    try:
        plt.savefig(buf, format='PNG')
        logger.debug(f"Grafic istoric {symbol} salvat în buffer.")
    except Exception as e:
         logger.error(f"Eroare la salvarea graficului istoric {symbol} în buffer: {e}", exc_info=True)
    finally:
        plt.close(fig)
    buf.seek(0)
    return buf

def generate_comparison_plot(sym1: str, sym2: str, df1: pd.Series, df2: pd.Series, days: int) -> BytesIO:
    """Generează graficul comparativ (2 axe Y) și returnează bufferul PNG."""
    logger.debug(f"Generare grafic comparativ {sym1} vs {sym2} ({days} zile)")
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color1 = 'tab:blue'
    ax1.set_xlabel('Dată')
    ax1.set_ylabel(f'Preț {sym1} (USD)', color=color1)
    ax1.plot(df1.index, df1.values, color=color1, label=sym1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(f'Preț {sym2} (USD)', color=color2)
    ax2.plot(df2.index, df2.values, color=color2, label=sym2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f"Comparație Preț: {sym1} vs {sym2} ({days} zile)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    buf = BytesIO()
    try:
        plt.savefig(buf, format='PNG')
        logger.debug(f"Grafic comparativ {sym1} vs {sym2} salvat în buffer.")
    except Exception as e:
         logger.error(f"Eroare la salvarea graficului comparativ {sym1} vs {sym2} în buffer: {e}", exc_info=True)
    finally:
        plt.close(fig)
    buf.seek(0)
    return buf