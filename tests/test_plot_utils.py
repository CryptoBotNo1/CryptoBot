# tests/test_plot_utils.py
import matplotlib
matplotlib.use('Agg') # <-- FORȚEAZĂ BACKEND NON-INTERACTIV ÎNAINTE DE ALTE IMPORTURI MATPLOTLIB

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO

# Importăm funcțiile testate (acum matplotlib știe ce backend să folosească)
from plot_utils import (
    generate_prediction_plot,
    generate_price_history_plot,
    generate_comparison_plot
)

# Date de test simple
DATES = pd.date_range(start="2025-01-01", periods=10)
VALUES1 = np.random.rand(10) * 100
VALUES2 = np.random.rand(10) * 50
DF1 = pd.DataFrame({'Close': VALUES1}, index=DATES)
DF2 = pd.DataFrame({'Close': VALUES2}, index=DATES)

def test_generate_prediction_plot():
    """Testează generarea graficului de predicție."""
    symbol = "PREDICT"
    days = 5
    dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=days)
    predictions = list(np.random.rand(days) * 1000)
    buf = generate_prediction_plot(symbol, dates, predictions)
    assert isinstance(buf, BytesIO)
    assert buf.getbuffer().nbytes > 0

def test_generate_price_history_plot():
    """Testează generarea graficului de istoric."""
    symbol = "HISTORY"
    days = 10
    buf = generate_price_history_plot(symbol, DF1, days)
    assert isinstance(buf, BytesIO)
    assert buf.getbuffer().nbytes > 0

def test_generate_comparison_plot():
    """Testează generarea graficului comparativ."""
    sym1 = "COMP1"
    sym2 = "COMP2"
    days = 10
    buf = generate_comparison_plot(sym1, sym2, DF1["Close"], DF2["Close"], days)
    assert isinstance(buf, BytesIO)
    assert buf.getbuffer().nbytes > 0