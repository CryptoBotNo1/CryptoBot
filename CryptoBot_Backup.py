import os
import logging
import json
import aiohttp
import requests
import pickle
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from sklearn.preprocessing import MinMaxScaler
from matplotlib.dates import DateFormatter

# === ÎNCĂRCARE .env ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
COINSTATS_API_KEY = os.getenv("COINSTATS_API_KEY")

# === CONFIG ===
MODEL_DIR = "models"
CRYPTO_LIST = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "MATIC", "LTC"]

# === MODEL ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# === FUNCȚII UTILITARE ===
def is_valid_symbol(symbol):
    return symbol in CRYPTO_LIST

def is_valid_days(days_str):
    try:
        days = int(days_str)
        return days > 0
    except ValueError:
        return False

# === FUNCȚII AI ===
def load_model_and_scaler(symbol):
    model_path = os.path.join(MODEL_DIR, f"lstm_{symbol}.pt")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{symbol}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"meta_{symbol}.json")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = LSTMModel(input_size=len(meta["features"]))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    # === TorchScript + Cuantizare dinamică (opțional) ===
    script_path = os.path.join(MODEL_DIR, f"lstm_{symbol}_scripted.pt")
    quant_path = os.path.join(MODEL_DIR, f"lstm_{symbol}_quantized.pt")

    if not os.path.exists(script_path):
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(script_path)
        logging.info(f"Model TorchScript salvat: {script_path}")

    if not os.path.exists(quant_path):
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
            )
            scripted_quant = torch.jit.script(quantized_model)
            scripted_quant.save(quant_path)
            logging.info(f"Model cuantizat salvat: {quant_path}")
        except Exception as e:
            logging.warning(f"Cuantizarea a eșuat pentru {symbol}: {e}")

    return model, scaler, meta

def fetch_latest_data(symbol, features):
    cache_path = f"cache_{symbol}.pkl"
    try:
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() < 86400:
                with open(cache_path, "rb") as f:
                    df = pickle.load(f)
                    logging.info(f"Date încărcate din cache pentru {symbol}")
                    return df[features].tail(60)

        df = yf.download(f"{symbol}-USD", period="90d").dropna()
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["EMA_10"] = df["Close"].ewm(span=10).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        RS = gain / loss
        df["RSI"] = 100 - (100 / (1 + RS))
        df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
        df = df.dropna()

        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
            logging.info(f"Date salvate în cache pentru {symbol}")

        return df[features].tail(60)
    except Exception as e:
        logging.error(f"Eroare la fetch_latest_data pentru {symbol}: {e}")
        return pd.DataFrame()
    

def predict_sequence(symbol, days):
    model, scaler, meta = load_model_and_scaler(symbol)
    data = fetch_latest_data(symbol, meta["features"])
    if data.empty:
        return []
    scaled_input = scaler.transform(data.values)
    sequence = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)

    preds = []
    for _ in range(days):
        with torch.no_grad():
            out = model(sequence)
        next_val = out.item()
        preds.append(next_val)
        next_row = torch.tensor(scaler.transform([sequence[0, -1, :].numpy()]), dtype=torch.float32)
        sequence = torch.cat((sequence[:, 1:, :], next_row.unsqueeze(0)), dim=1)

    result = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return result.tolist()

# === COMENZI BOT ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Salut! Sunt botul tău AI pentru crypto.\nFolosește /help pentru listă completă de comenzi.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """📋 Comenzi disponibile:
/start - Pornește botul
/help - Afișează acest mesaj
/predict BTC 7 - Predicție AI
/news - Știri crypto
/crypto BTC - Preț live
/grafic BTC - Grafic 30 zile
/trend BTC - Analiză trend
/predict_chart BTC 7 - Grafic AI
/compare BTC ETH - Compară 2 monede
/summary BTC - Sumar zilnic
/portfolio BTC 0.5 - Adaugă în portofoliu
/myportfolio - Afișează portofoliul
"""
    await update.message.reply_text(text)

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2 or not is_valid_symbol(context.args[0].upper()) or not is_valid_days(context.args[1]):
        await update.message.reply_text("❌ Format: /predict BTC 7")
        return
    symbol = context.args[0].upper()
    days = int(context.args[1])
    await update.message.reply_text("🔄 Generare predicție...")
    try:
        preds = predict_sequence(symbol, days)
        if not preds:
            await update.message.reply_text("❌ Nu s-au putut genera predicțiile.")
            return
        dates = pd.date_range(datetime.now(), periods=days+1)[1:]
        pred_text = "\n".join([f"📅 {d.date()}: {p:,.2f} USD" for d, p in zip(dates, preds)])
        await update.message.reply_text(f"📊 *Predicții pentru {symbol}:*\n" + pred_text, parse_mode="Markdown")
    except Exception as e:
        logging.error(f"Eroare la predict: {e}")
        await update.message.reply_text("❌ Eroare la generarea predicției.")

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        url = f"https://openapi.coinstats.app/public/v1/news?limit=5"
        headers = {"X-API-KEY": COINSTATS_API_KEY}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as r:
                if r.status != 200:
                    raise Exception("Eroare CoinStats")
                data = await r.json()
                news_list = data.get("news", [])
        if not news_list:
            await update.message.reply_text("⚠️ Nu sunt știri disponibile momentan.")
        else:
            texts = [f"📰 {n['title']}" for n in news_list[:5]]
            await update.message.reply_text("\n".join(texts))
    except Exception as e:
        logging.error(f"Eroare la news: {e}")
        await update.message.reply_text("❌ Nu am putut obține știrile.")

async def crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Format: /crypto BTC")
        return
    symbol = context.args[0].upper()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT") as r:
                if r.status != 200:
                    raise Exception("Eroare Binance")
                data = await r.json()
        msg = f"""📊 {symbol} Info:
💵 Preț: {float(data['lastPrice']):,.2f} $
📈 High: {float(data['highPrice']):,.2f} $
📉 Low: {float(data['lowPrice']):,.2f} $
🔁 Volum: {float(data['volume']):,.2f} {symbol}"""
        await update.message.reply_text(msg)
    except Exception as e:
        logging.error(f"Eroare la crypto: {e}")
        await update.message.reply_text("❌ Nu s-au putut obține datele live.")

async def grafic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Format: /grafic BTC")
        return
    symbol = context.args[0].upper()
    df = yf.download(f"{symbol}-USD", period="30d")
    if df.empty:
        await update.message.reply_text("Nu am putut prelua datele pentru grafic.")
        return
    plt.figure()
    df["Close"].plot(title=f"Grafic {symbol} - Ultimele 30 zile")
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    await update.message.reply_photo(photo=InputFile(buf, filename="grafic.png"))

async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Format: /trend BTC")
        return
    symbol = context.args[0].upper()
    df = yf.download(f"{symbol}-USD", period="7d")
    if df.empty:
        await update.message.reply_text("Nu am putut determina trendul.")
        return
    change = df["Close"].iloc[-1] - df["Close"].iloc[0]
    msg = f"📈 {symbol} este în trend ascendent (+{change:.2f}$)" if change > 0 else f"📉 {symbol} este în trend descendent ({change:.2f}$)"
    await update.message.reply_text(msg)
async def myportfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not os.path.exists(PORTFOLIO_FILE):
        await update.message.reply_text("📭 Nu ai niciun portofoliu salvat.")
        return

    user_holdings = {}
    try:
        with open(PORTFOLIO_FILE, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["user_id"] == user_id:
                    symbol = row["symbol"]
                    amount = float(row["amount"])
                    user_holdings[symbol] = user_holdings.get(symbol, 0) + amount

        if not user_holdings:
            await update.message.reply_text("📭 Nu ai nimic în portofoliu.")
            return

        msg = "📊 Portofoliul tău:\n"  # corectare: string închis corect pe un singur rând
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol, amount in user_holdings.items():
                tasks.append(fetch_price(session, symbol, amount))
            results = await asyncio.gather(*tasks)
            for line in results:
                msg += line

        await update.message.reply_text(msg)
    except Exception as e:
        logging.error(f"Eroare la myportfolio: {e}")
        await update.message.reply_text("❌ Eroare la afișarea portofoliului.")
import csv
PORTFOLIO_FILE = "portfolio.csv"

async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("❌ Format: /portfolio BTC 0.5")
        return

    user_id = str(update.effective_user.id)
    symbol = context.args[0].upper()
    if not is_valid_symbol(symbol):
        await update.message.reply_text("❌ Simbol invalid.")
        return

    try:
        amount = float(context.args[1])
        if amount <= 0:
            await update.message.reply_text("❌ Cantitatea trebuie să fie pozitivă.")
            return
    except ValueError:
        await update.message.reply_text("❌ Cantitate invalidă.")
        return

    # Salvează în fișier CSV
    new_entry = [user_id, symbol, amount, str(datetime.now())]
    file_exists = os.path.exists(PORTFOLIO_FILE)
    with open(PORTFOLIO_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["user_id", "symbol", "amount", "timestamp"])
        writer.writerow(new_entry)

    await update.message.reply_text(f"✅ Portofoliu actualizat: {amount} {symbol}")
async def predict_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2 or not is_valid_symbol(context.args[0].upper()) or not is_valid_days(context.args[1]):
        await update.message.reply_text("❌ Format: /predict_chart BTC 7")
        return

    symbol = context.args[0].upper()
    days = int(context.args[1])
    await update.message.reply_text("🔄 Generare grafic AI...")

    try:
        preds = predict_sequence(symbol, days)
        if not preds:
            await update.message.reply_text("❌ Nu s-au putut genera predicțiile.")
            return
        dates = pd.date_range(datetime.now(), periods=days+1)[1:]

        plt.figure(figsize=(10, 5))
        plt.plot(dates, preds, marker='o', linestyle='-', label='Predicție AI')
        plt.title(f"Predicție {symbol} - Următoarele {days} zile")
        plt.xlabel("Dată")
        plt.ylabel("Preț (USD)")
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        await update.message.reply_photo(photo=InputFile(buf, filename="predict_chart.png"))
    except Exception as e:
        logging.error(f"Eroare la predict_chart: {e}")
        await update.message.reply_text("❌ Eroare la generarea graficului.")

async def compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("❌ Format: /compare BTC ETH")
        return

    sym1, sym2 = context.args[0].upper(), context.args[1].upper()
    if not is_valid_symbol(sym1) or not is_valid_symbol(sym2):
        await update.message.reply_text("❌ Simbol invalid.")
        return

    df1 = yf.download(f"{sym1}-USD", period="30d")["Close"]
    df2 = yf.download(f"{sym2}-USD", period="30d")["Close"]

    if df1.empty or df2.empty:
        await update.message.reply_text("❌ Nu s-au putut prelua datele.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df1.index, df1.values, label=sym1)
    plt.plot(df2.index, df2.values, label=sym2)
    plt.title(f"Comparatie: {sym1} vs {sym2} (30 zile)")
    plt.xlabel("Dată")
    plt.ylabel("Preț (USD)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    await update.message.reply_photo(photo=InputFile(buf, filename="compare.png"))

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("❌ Format: /summary BTC")
        return

    symbol = context.args[0].upper()
    if not is_valid_symbol(symbol):
        await update.message.reply_text("❌ Simbol invalid.")
        return

    try:
        df = yf.download(f"{symbol}-USD", period="7d")
        if df.empty:
            await update.message.reply_text("❌ Date indisponibile pentru analiză.")
            return
        change = df["Close"].iloc[-1] - df["Close"].iloc[0]
        trend_msg = f"📈 Trend: Ascendent (+{change:.2f}$)" if change > 0 else f"📉 Trend: Descendent ({change:.2f}$)"

        pred = predict_sequence(symbol, 1)
        if not pred:
            await update.message.reply_text("❌ Nu s-a putut calcula predicția.")
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT") as r:
                    if r.status != 200:
                        raise Exception(f"Binance API status {r.status}")
                data = await r.json()
            live_price = float(data['lastPrice'])
        except Exception as e:
            logging.error(f"Eroare Binance API: {e}")
            live_price = "N/A"

        try:
            news_url = "https://openapi.coinstats.app/public/v1/news?limit=1"
            headers = {"X-API-KEY": COINSTATS_API_KEY}
            async with aiohttp.ClientSession() as session:
                async with session.get(news_url, headers=headers) as resp:
                    if resp.status != 200:
                        raise Exception(f"CoinStats API status {resp.status}")
                    news_data = await resp.json()
            news_title = news_data["news"][0]["title"] if "news" in news_data and news_data["news"] else "Fără știri."
        except Exception as e:
            logging.error(f"Eroare CoinStats API: {e}")
            news_title = "Știrile nu sunt disponibile."

        msg = f"""🧠 *Sumar zilnic {symbol}*
💰 Preț live: {live_price} $
🔮 Predicție AI (1 zi): {pred[0]:.2f} $
{trend_msg}
📰 Știre: {news_title}""": {pred[0]:.2f} $\n"
    f"{trend_msg}\n"
    f"📰 Știre: {news_title}"
)
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logging.error(f"Eroare la summary: {e}")
        await update.message.reply_text("❌ Eroare la generarea sumarului.")

# === RUN BOT ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TOKEN).build()

    # === HANDLERE CLIASICE ===
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("news", news))
    app.add_handler(CommandHandler("crypto", crypto))
    app.add_handler(CommandHandler("grafic", grafic))
    app.add_handler(CommandHandler("trend", trend))

    app.add_handler(CommandHandler("predict_chart", predict_chart))
    app.add_handler(CommandHandler("compare", compare))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("portfolio", portfolio))
    app.add_handler(CommandHandler("myportfolio", myportfolio))

    logging.info("🤖 Botul este online!")
    app.run_polling()
