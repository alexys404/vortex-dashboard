# -*- coding: utf-8 -*-
"""
VORTEX_IA - Bot Multi-Agents (14 Stratégies + Filtres Multi-TF & Risk)
Version complète avec API Mini App Telegram, XGBoost, Clôture Partielle et Trailing
"""

import time
import datetime
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import requests
import joblib

# --- AJOUT API POUR TELEGRAM MINI APP ---
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale contenant les données lues par la Mini App
bot_status = {
    "equity": 0.0,
    "consensus_signal": "NEUTRAL",
    "consensus_score": 0,
    "prob_buy": 0,
    "prob_sell": 0,
    "trend_h1": "NEUTRAL",
    "votes": [0]*14,
    "open_positions": 0
}

@app.get("/api/status")
def get_status():
    return bot_status

def start_api():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

# Lancement du serveur API sur le port 8000 en arrière-plan
threading.Thread(target=start_api, daemon=True).start()
# ----------------------------------------

# ==============================================================================
# 0. CONFIGURATION TELEGRAM
# ==============================================================================
TELEGRAM_TOKEN = "8897464855:AAFzPpSWIygNtUtsZ60MJMMSqT4AvBkcG80"
TELEGRAM_CHAT_ID = "8377154744"

def send_telegram_message(message):
    """Envoie une notification sur Telegram"""
    if TELEGRAM_TOKEN == "TON_TOKEN_BOT_ICI" or TELEGRAM_CHAT_ID == "TON_CHAT_ID_ICI":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Erreur Telegram : {e}")

# ==============================================================================
# 1. IDENTIFIANTS DU COMPTE MT5
# ==============================================================================
ACCOUNT_ID = 700183397
ACCOUNT_PASSWORD = "5UjcdC&p"
ACCOUNT_SERVER = "PUPrime-Demo"

# ==============================================================================
# 2. PARAMÈTRES ET CONFIGURATION DE VORTEX_IA MULTI-AGENTS
# ==============================================================================
SYMBOL = "XAUUSD.s"
TIMEFRAME = mt5.TIMEFRAME_M5
RISK_PERCENT = 1.0
MAGIC_NUMBER = 888111
CHECK_INTERVAL = 5

# SEUIL DE CONSENSUS DE L'ENSEMBLE (75% = au moins 11/14 stratégies)
CONSENSUS_THRESHOLD = 75.0

# SEUILS DE CONFORT POUR L'AGENT XGBOOST (Stratégie 1)
AI_BUY_MIN_PROBA = 0.38
AI_SELL_MIN_PROBA = 0.45

# MULTIPLICATEURS SL ET TP BASÉS SUR L'ATR
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 3.0

# PARAMÈTRES CLÔTURE PARTIELLE & BREAK-EVEN
PARTIAL_CLOSE_ENABLED = True
PARTIAL_CLOSE_RATIO = 0.5         # Clôturer 50% du volume
PARTIAL_TRIGGER_ATR_MULT = 1.2    # Seuil de déclenchement (1.2 * ATR)

# PARAMÈTRES TRAILING STOP
TRAILING_STOP_ENABLED = True
TRAILING_ATR_MULTIPLIER = 1.8

# LIMITATION DES SPREADS DANGEREUX SUR L'OR
MAX_ALLOWED_SPREAD = 0.50

last_executed_trade_type = None
partially_closed_tickets = set()

# ==============================================================================
# 3. CHARGEMENT DU MODÈLE D'IA (XGBOOST)
# ==============================================================================
print("🧠 Chargement du modèle IA 'vortex_ia_model.pkl'...")
try:
    ai_model = joblib.load('vortex_ia_model.pkl')
    print("✅ Modèle IA XGBoost chargé avec succès !")
except Exception as e:
    print(f"❌ IMPOSSIBLE DE CHARGER LE MODÈLE IA : {e}")
    exit()

# ==============================================================================
# 4. CALCULS INDICATEURS ET XGBOOST
# ==============================================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def prepare_features(df):
    df["time_dt"] = pd.to_datetime(df["time"], unit="s")
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["dist_EMA_50"] = (df["close"] - df["EMA_50"]) / df["close"]
    df["dist_EMA_200"] = (df["close"] - df["EMA_200"]) / df["close"]
    
    df["RSI"] = calculate_rsi(df["close"], 14)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    df["ATR"] = calculate_atr(df, 14)
    df["body_size"] = np.abs(df["close"] - df["open"]) / df["close"]
    df["returns_1"] = df["close"].pct_change(1)
    df["returns_3"] = df["close"].pct_change(3)

    df["hour"] = df["time_dt"].dt.hour
    df["day_of_week"] = df["time_dt"].dt.dayofweek
    return df

def get_ai_prediction(df):
    feature_cols = [
        'dist_EMA_50', 'dist_EMA_200', 'RSI', 'MACD', 'MACD_hist',
        'ATR', 'body_size', 'returns_1', 'returns_3', 'hour', 'day_of_week'
    ]
    last_row = df[feature_cols].iloc[[-1]]
    probs = ai_model.predict_proba(last_row)[0]
    return probs[1], probs[2], probs[0]  # prob_buy, prob_sell, prob_neutre

# ==============================================================================
# 5. MATRICE DES 14 STRATÉGIES & AGENTS DE SÉCURITÉ
# ==============================================================================

def run_14_strategies(df):
    votes = []
    close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    
    # 1. XGBoost IA
    p_buy, p_sell, _ = get_ai_prediction(df)
    if p_buy >= AI_BUY_MIN_PROBA: votes.append(1)
    elif p_sell >= AI_SELL_MIN_PROBA: votes.append(-1)
    else: votes.append(0)

    # 2. Cross EMA 50 / 200
    ema50 = df['EMA_50'].iloc[-1]
    ema200 = df['EMA_200'].iloc[-1]
    votes.append(1 if ema50 > ema200 else -1)

    # 3. Position du Prix vs EMA 50
    votes.append(1 if close > ema50 else -1)

    # 4. Signal MACD
    macd = df['MACD'].iloc[-1]
    macd_sig = df['MACD_signal'].iloc[-1]
    votes.append(1 if macd > macd_sig else -1)

    # 5. Momentum / Bougie
    votes.append(1 if close > df['open'].iloc[-1] else -1)

    # 6. RSI Dynamic
    rsi = df['RSI'].iloc[-1]
    votes.append(1 if rsi < 35 else (-1 if rsi > 65 else 0))

    # 7. Bandes de Bollinger
    sma20 = df['close'].rolling(20).mean().iloc[-1]
    std20 = df['close'].rolling(20).std().iloc[-1]
    lower_b = sma20 - (2 * std20)
    upper_b = sma20 + (2 * std20)
    votes.append(1 if close <= lower_b else (-1 if close >= upper_b else 0))

    # 8. Stochastic Reversal
    votes.append(1 if rsi < 30 else (-1 if rsi > 70 else 0))

    # 9. Approximation VWAP
    vwap = (df['tick_volume'] * (df['high'] + df['low'] + df['close']) / 3).sum() / df['tick_volume'].sum()
    votes.append(1 if close < vwap else -1)

    # 10. Support / Résistance M15
    res = df['high'].rolling(30).max().iloc[-1]
    sup = df['low'].rolling(30).min().iloc[-1]
    votes.append(1 if close <= sup * 1.0005 else (-1 if close >= res * 0.9995 else 0))

    # 11. Impulsion Volatilité ATR
    atr = df['ATR'].iloc[-1]
    votes.append(1 if (df['high'].iloc[-1] - df['low'].iloc[-1]) > (1.3 * atr) and close > prev_close else 0)

    # 12. Breakout Session
    votes.append(1 if close > res else (-1 if close < sup else 0))

    # 13. Direction de Histogramme MACD
    macd_hist = df['MACD_hist'].iloc[-1]
    prev_hist = df['MACD_hist'].iloc[-2]
    votes.append(1 if macd_hist > prev_hist else -1)

    # 14. Keltner Channel Breakout
    votes.append(1 if close > (ema50 + 1.5 * atr) else (-1 if close < (ema50 - 1.5 * atr) else 0))

    return votes, p_buy, p_sell

def agent_consensus(votes):
    tot_buy = sum(1 for v in votes if v == 1)
    tot_sell = sum(1 for v in votes if v == -1)
    
    score_buy = (tot_buy / 14.0) * 100.0
    score_sell = (tot_sell / 14.0) * 100.0
    
    if score_buy >= CONSENSUS_THRESHOLD:
        return "BUY", score_buy
    if score_sell >= CONSENSUS_THRESHOLD:
        return "SELL", score_sell
    return "NEUTRAL", max(score_buy, score_sell)

def agent_tendance_h1():
    rates_h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 200)
    if rates_h1 is None or len(rates_h1) == 0:
        return "NEUTRAL"
    df_h1 = pd.DataFrame(rates_h1)
    ema200_h1 = df_h1['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    current_price = df_h1['close'].iloc[-1]
    return "BUY" if current_price > ema200_h1 else "SELL"

def agent_risk_manager():
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        return False
    spread = tick.ask - tick.bid
    return spread <= MAX_ALLOWED_SPREAD

# ==============================================================================
# 6. GESTION DYNAMIQUE DU RISQUE ET ORDRES MT5
# ==============================================================================

def calculate_dynamic_lot(sl_distance_price):
    account_info = mt5.account_info()
    symbol_info = mt5.symbol_info(SYMBOL)

    if account_info is None or symbol_info is None or sl_distance_price == 0:
        return 0.01

    equity = account_info.equity
    risk_amount = equity * (RISK_PERCENT / 100.0)

    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size

    if tick_value == 0 or tick_size == 0:
        return 0.01

    points_sl = sl_distance_price / symbol_info.point
    risk_per_lot = (points_sl * symbol_info.point / tick_size) * tick_value

    if risk_per_lot == 0:
        return 0.01

    raw_lot = risk_amount / risk_per_lot
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step

    lot = round(raw_lot / lot_step) * lot_step
    return round(max(min_lot, min(max_lot, lot)), 2)

def get_open_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []
    return [p for p in positions if p.magic == MAGIC_NUMBER]

def open_trade(order_type, current_atr, consensus_score):
    global last_executed_trade_type
    
    symbol_info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)

    if symbol_info is None or tick is None:
        return False

    sl_distance = current_atr * SL_ATR_MULTIPLIER
    tp_distance = current_atr * TP_ATR_MULTIPLIER

    if order_type == "BUY":
        price = tick.ask
        sl = price - sl_distance
        tp = price + tp_distance
        trade_type = mt5.ORDER_TYPE_BUY
    elif order_type == "SELL":
        price = tick.bid
        sl = price + sl_distance
        tp = price - tp_distance
        trade_type = mt5.ORDER_TYPE_SELL
    else:
        return False

    sl = round(sl, symbol_info.digits)
    tp = round(tp, symbol_info.digits)
    lot_to_use = calculate_dynamic_lot(sl_distance)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_to_use,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "Vortex_IA Multi-Agents",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Échec ordre {order_type} : {result.comment}")
        return False

    last_executed_trade_type = order_type

    print(f"\n🤖✅ ORDRE UNIQUE {order_type} EXÉCUTÉ | Consensus: {consensus_score:.0f}% | Lot: {lot_to_use} | Prix: {price:.2f}")
    
    send_telegram_message(
        f"🤖 *VORTEX_IA MULTI-AGENTS - TRADE EXÉCUTÉ*\n🔹 *Type* : {order_type}\n🎯 *Consensus 14 Stratégies* : {consensus_score:.0f}%\n💰 *Lot* : {lot_to_use}\n📍 *Prix* : {price:.2f}\n🛑 *SL* : {sl:.2f}\n🎯 *TP* : {tp:.2f}"
    )
    return True

def manage_partial_close_and_trailing(df):
    positions = get_open_positions()
    if not positions:
        return

    symbol_info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    if not symbol_info or not tick:
        return

    current_atr = df['ATR'].iloc[-1]
    partial_trigger = current_atr * PARTIAL_TRIGGER_ATR_MULT
    trailing_distance = current_atr * TRAILING_ATR_MULTIPLIER
    min_stop_distance = symbol_info.trade_stops_level * symbol_info.point

    for pos in positions:
        ticket = pos.ticket
        
        if PARTIAL_CLOSE_ENABLED and ticket not in partially_closed_tickets:
            profit_points = 0
            close_price = 0
            order_type_close = None

            if pos.type == mt5.ORDER_TYPE_BUY:
                profit_points = tick.bid - pos.price_open
                close_price = tick.bid
                order_type_close = mt5.ORDER_TYPE_SELL
            elif pos.type == mt5.ORDER_TYPE_SELL:
                profit_points = pos.price_open - tick.ask
                close_price = tick.ask
                order_type_close = mt5.ORDER_TYPE_BUY

            if profit_points >= partial_trigger:
                lot_step = symbol_info.volume_step
                partial_volume = round((pos.volume * PARTIAL_CLOSE_RATIO) / lot_step) * lot_step
                partial_volume = max(symbol_info.volume_min, partial_volume)

                if partial_volume < pos.volume:
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": ticket,
                        "symbol": SYMBOL,
                        "volume": partial_volume,
                        "type": order_type_close,
                        "price": close_price,
                        "deviation": 20,
                        "magic": MAGIC_NUMBER,
                        "comment": "Clôture partielle Vortex_IA",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    res = mt5.order_send(close_request)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        partially_closed_tickets.add(ticket)
                        print(f"\n💰 Clôture partielle de {partial_volume} lot(s) sur le ticket #{ticket}")
                        send_telegram_message(f"💰 *CLÔTURE PARTIELLE* sur #{ticket}\nVolume fermé : {partial_volume} lot(s)")

                        be_sl = round(pos.price_open, symbol_info.digits)
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "symbol": SYMBOL,
                            "sl": be_sl,
                            "tp": pos.tp,
                        }
                        mt5.order_send(modify_request)

        if TRAILING_STOP_ENABLED:
            if pos.type == mt5.ORDER_TYPE_BUY:
                current_price = tick.bid
                if current_price > pos.price_open:
                    new_sl = round(current_price - trailing_distance, symbol_info.digits)
                    if new_sl > pos.sl and (current_price - new_sl) >= min_stop_distance:
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "symbol": SYMBOL,
                            "sl": new_sl,
                            "tp": pos.tp,
                        }
                        mt5.order_send(modify_request)

            elif pos.type == mt5.ORDER_TYPE_SELL:
                current_price = tick.ask
                if current_price < pos.price_open:
                    new_sl = round(current_price + trailing_distance, symbol_info.digits)
                    if new_sl < pos.sl and (new_sl - current_price) >= min_stop_distance:
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "symbol": SYMBOL,
                            "sl": new_sl,
                            "tp": pos.tp,
                        }
                        mt5.order_send(modify_request)

# ==============================================================================
# 7. BOUCLE PRINCIPALE MULTI-AGENTS
# ==============================================================================

def main():
    global last_executed_trade_type
    
    print("==================================================")
    print("🚀 DÉMARRAGE DE VORTEX_IA (VERSION MULTI-AGENTS + API)")
    print("==================================================")

    if not mt5.initialize():
        print("❌ Impossible de lancer l'API MetaTrader 5.")
        return

    if not mt5.login(login=ACCOUNT_ID, password=ACCOUNT_PASSWORD, server=ACCOUNT_SERVER):
        print(f"❌ Échec de la connexion au compte {ACCOUNT_ID}.")
        mt5.shutdown()
        return

    account_info = mt5.account_info()
    print(f"✅ Connecté au compte : {account_info.login} ({account_info.server})")
    print(f"📌 Capital: ${account_info.equity:.2f} | Risk/Trade: {RISK_PERCENT}% | Symbol: {SYMBOL}")
    print("--------------------------------------------------")

    send_telegram_message(f"🟢 *Vortex_IA Multi-Agents Démarré*\nCapital: ${account_info.equity:.2f}")

    mt5.symbol_select(SYMBOL, True)

    try:
        while True:
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 200)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df = prepare_features(df)

                current_atr = df["ATR"].iloc[-1]
                open_positions = get_open_positions()

                if not np.isnan(current_atr):
                    votes, prob_buy, prob_sell = run_14_strategies(df)
                    consensus_signal, consensus_score = agent_consensus(votes)
                    trend_h1 = agent_tendance_h1()
                    risk_ok = agent_risk_manager()

                    # --- ENVOI DES DONNÉES TEMPS RÉEL À LA MINI APP ---
                    bot_status.update({
                        "equity": account_info.equity,
                        "consensus_signal": consensus_signal,
                        "consensus_score": round(consensus_score),
                        "prob_buy": round(prob_buy * 100),
                        "prob_sell": round(prob_sell * 100),
                        "trend_h1": trend_h1,
                        "votes": votes,
                        "open_positions": len(open_positions)
                    })
                    # --------------------------------------------------

                    timestamp = time.strftime('%H:%M:%S')
                    print(f"[{timestamp}] Consensus: {consensus_signal} ({consensus_score:.0f}%) | IA Buy: {prob_buy*100:.0f}% Sell: {prob_sell*100:.0f}% | H1: {trend_h1} | Pos: {len(open_positions)}", end="\r", flush=True)

                    if len(open_positions) > 0:
                        manage_partial_close_and_trailing(df)
                        time.sleep(CHECK_INTERVAL)
                        continue

                    if last_executed_trade_type is not None and consensus_score < 50:
                        last_executed_trade_type = None

                    if consensus_signal == trend_h1 and consensus_signal in ["BUY", "SELL"] and risk_ok:
                        if last_executed_trade_type != consensus_signal:
                            print(f"\n🔥 SIGNAL UNIQUE VALIDÉ PAR LES AGENTS : Direction {consensus_signal} ({consensus_score:.0f}%)")
                            open_trade(consensus_signal, current_atr, consensus_score)

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Arrêt du bot par l'utilisateur.")
        send_telegram_message("🔴 *Vortex_IA Arrêté*")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()