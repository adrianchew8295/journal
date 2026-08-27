import calendar
import datetime
from datetime import timedelta
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st
import yfinance as yf

# =====================================================================
# 1. 基础配置与凭证锁定
# =====================================================================
st.set_page_config(page_title="QQQ 2B与战区同频座舱", page_icon="🎯", layout="wide")

TIINGO_TOKEN = "bcffe3a5cf7eeef085e405cfa4a3e5691b976217"
TICKER = "QQQ"
CSV_FILE = "monthly_trade_records.csv"

tz_myt = pytz.timezone("Asia/Kuala_Lumpur")
tz_ny = pytz.timezone("America/New_York")

now_myt = datetime.datetime.now(tz_myt)
now_ny = datetime.datetime.now(tz_ny)

# =====================================================================
# 2. QQQ 数据抓取引擎
# =====================================================================
def fetch_raw_data_with_retry(period_5m="1mo", max_retries=3):
    df_1h, df_5m = None, None
    err_log = []
    start_str = (now_myt - timedelta(days=60)).strftime("%Y-%m-%d")

    for attempt in range(max_retries):
        url = f"https://api.tiingo.com/iex/{TICKER}/prices?startDate={start_str}&resampleFreq=1hour&token={TIINGO_TOKEN}&columns=open,high,low,close,volume"
        try:
            resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and len(data) >= 30:
                    df_t = pd.DataFrame(data)
                    df_t["date"] = pd.to_datetime(df_t["date"])
                    df_t.set_index("date", inplace=True)
                    df_t.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
                    df_1h = df_t[["Open", "High", "Low", "Close", "Volume"]].sort_index()
                    df_1h.index = df_1h.index.tz_localize("UTC").tz_convert(tz_ny) if df_1h.index.tz is None else df_1h.index.tz_convert(tz_ny)
                    break
        except Exception:
            time.sleep(1)

    if df_1h is None:
        try:
            df_yf = yf.download(TICKER, period="2mo", interval="1h", prepost=True, progress=False)
            if df_yf is not None and not df_yf.empty:
                if isinstance(df_yf.columns, pd.MultiIndex):
                    df_yf.columns = df_yf.columns.get_level_values(0)
                df_1h = df_yf[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
                df_1h.index = df_1h.index.tz_localize("UTC").tz_convert(tz_ny) if df_1h.index.tz is None else df_1h.index.tz_convert(tz_ny)
        except Exception as e:
            err_log.append("YahooFinance 1H 失败: " + str(e))

    for attempt in range(max_retries):
        try:
            df_5m_raw = yf.download(TICKER, period=period_5m, interval="5m", prepost=True, progress=False)
            if df_5m_raw is not None and not df_5m_raw.empty:
                if isinstance(df_5m_raw.columns, pd.MultiIndex):
                    df_5m_raw.columns = df_5m_raw.columns.get_level_values(0)
                df_5m = df_5m_raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
                df_5m.index = df_5m.index.tz_localize("UTC").tz_convert(tz_ny) if df_5m.index.tz is None else df_5m.index.tz_convert(tz_ny)
                break
        except Exception as e:
            err_log.append("YahooFinance 5M 失败: " + str(e))
            time.sleep(1)

    return df_1h, df_5m, err_log

# =====================================================================
# 3. 核心运算：富途 13 行参数抽取
# =====================================================================
def compute_futu_13_params(df_1h, df_5m, as_of_ny_time):
    if df_1h is None: return None
    sub_1h = df_1h[df_1h.index <= as_of_ny_time].copy()
    if len(sub_1h) < 25: return None

    today_ny = as_of_ny_time.date()
    df_rth = sub_1h[(sub_1h.index.hour > 9) | ((sub_1h.index.hour == 9) & (sub_1h.index.minute >= 30))]
    df_rth = df_rth[df_rth.index.hour < 16]
    past_dates = sorted(list(set(df_rth.index.date)))
    past_dates = [d for d in past_dates if d < today_ny]

    if past_dates:
        prev_df = df_rth[df_rth.index.date == past_dates[-1]]
        pdh_idx, pdl_idx = prev_df["High"].idxmax(), prev_df["Low"].idxmin()
        pdh_val, pdl_val = float(prev_df.loc[pdh_idx, "High"]), float(prev_df.loc[pdl_idx, "Low"])
        pdh_time, pdl_time = pdh_idx.strftime("%Y-%m-%d %H:%M ET"), pdl_idx.strftime("%Y-%m-%d %H:%M ET")
    else:
        pdh_val, pdl_val = float(sub_1h["High"].iloc[-10:].max()), float(sub_1h["Low"].iloc[-10:].min())
        pdh_time, pdl_time = "Prior Session", "Prior Session"

    sub_5m_pm = df_5m[(df_5m.index.date == today_ny) & (df_5m.index.hour >= 4) & (df_5m.index < as_of_ny_time)] if df_5m is not None else None
    if sub_5m_pm is not None and not sub_5m_pm.empty:
        pmh_idx, pml_idx = sub_5m_pm["High"].idxmax(), sub_5m_pm["Low"].idxmin()
        pmh_val, pml_val = float(sub_5m_pm.loc[pmh_idx, "High"]), float(sub_5m_pm.loc[pml_idx, "Low"])
        pmh_time, pml_time = pmh_idx.strftime("%Y-%m-%d %H:%M ET"), pml_idx.strftime("%Y-%m-%d %H:%M ET")
        live_price = float(sub_5m_pm["Close"].iloc[-1])
    else:
        pmh_val, pml_val = float(sub_1h["High"].iloc[-4:].max()), float(sub_1h["Low"].iloc[-4:].min())
        pmh_time, pml_time = "Recent 1H", "Recent 1H"
        live_price = float(sub_1h["Close"].iloc[-1])

    sub_1h["EMA20"] = sub_1h["Close"].ewm(span=20, adjust=False).mean()
    sub_1h["SMA50"] = sub_1h["Close"].rolling(window=50).mean()

    tr = np.maximum(sub_1h["High"] - sub_1h["Low"], np.maximum((sub_1h["High"] - sub_1h["Close"].shift(1)).abs(), (sub_1h["Low"] - sub_1h["Close"].shift(1)).abs()))
    atr = float(tr.rolling(14).mean().iloc[-1]) if not np.isnan(tr.rolling(14).mean().iloc[-1]) else (live_price * 0.008)

    subset = sub_1h.iloc[-60:].copy()
    highs, lows, opens, closes, times = subset["High"].values, subset["Low"].values, subset["Open"].values, subset["Close"].values, subset.index

    pivots_high, pivots_low = [], []
    for i in range(2, len(subset) - 2):
        if highs[i] == max(highs[i-2:i+3]): pivots_high.append((float(highs[i]), float(max(opens[i], closes[i])), times[i].strftime("%m-%d %H:%M ET")))
        if lows[i] == min(lows[i-2:i+3]): pivots_low.append((float(min(opens[i], closes[i])), float(lows[i]), times[i].strftime("%m-%d %H:%M ET")))

    valid_highs = [p for p in pivots_high if p[0] > live_price]
    valid_highs.sort(key=lambda x: x[0])
    sbr_top, sbr_bot, sbr_time = valid_highs[0] if len(valid_highs) >= 1 else (live_price + 1.2 * atr, live_price + 0.6 * atr, "Range High")
    sbr2_top, sbr2_bot, sbr2_time = valid_highs[1] if len(valid_highs) >= 2 else (sbr_top + 1.2 * atr, sbr_top + 0.5 * atr, "Tier-2 High")

    valid_lows = [p for p in pivots_low if p[1] < live_price]
    valid_lows.sort(key=lambda x: x[1], reverse=True)
    rbs_top, rbs_bot, rbs_time = valid_lows[0] if len(valid_lows) >= 1 else (live_price - 0.6 * atr, live_price - 1.2 * atr, "Range Low")
    rbs2_top, rbs2_bot, rbs2_time = valid_lows[1] if len(valid_lows) >= 2 else (rbs_bot - 0.5 * atr, rbs_bot - 1.2 * atr, "Tier-2 Low")

    ema20_now = float(sub_1h["EMA20"].iloc[-1])
    sma50_now = float(sub_1h["SMA50"].iloc[-1]) if not np.isnan(sub_1h["SMA50"].iloc[-1]) else ema20_now
    score_ma = 1 if (live_price > ema20_now and ema20_now >= sma50_now) else (-1 if (live_price < ema20_now and ema20_now <= sma50_now) else 0)

    score_hhll = 0
    if len(pivots_high) >= 2 and len(pivots_low) >= 2:
        last_2_h, last_2_l = [p[0] for p in pivots_high[-2:]], [p[1] for p in pivots_low[-2:]]
        if last_2_h[1] > last_2_h[0] and last_2_l[1] > last_2_l[0]: score_hhll = 1
        elif last_2_h[1] < last_2_h[0] and last_2_l[1] < last_2_l[0]: score_hhll = -1

    ema20_prev = float(sub_1h["EMA20"].iloc[-5])
    ema_slope = (ema20_now - ema20_prev) / ema20_prev * 100
    score_slope = 1 if ema_slope > 0.15 else (-1 if ema_slope < -0.15 else 0)

    total_score = score_ma + score_hhll + score_slope
    trend_bias = 1 if total_score >= 2 else (-1 if total_score <= -2 else 0)
    bias_desc = "🟢 绿灯 (做多为主)" if trend_bias == 1 else ("🔴 红灯 (做空为主)" if trend_bias == -1 else "🟡 黄灯 (震荡防守)")

    return {
        "live_price": live_price, "TREND_BIAS": trend_bias, "BIAS_DESC": bias_desc,
        "EMA20_1H": round(ema20_now, 2), "ATR_1H": round(atr, 2),
        "SBR_TOP": sbr_top, "SBR_BOT": sbr_bot, "SBR_TIME": sbr_time,
        "RBS_TOP": rbs_top, "RBS_BOT": rbs_bot, "RBS_TIME": rbs_time,
        "SBR2_TOP": sbr2_top, "SBR2_BOT": sbr2_bot, "SBR2_TIME": sbr2_time,
        "RBS2_TOP": rbs2_top, "RBS2_BOT": rbs2_bot, "RBS2_TIME": rbs2_time,
        "PDH": pdh_val, "PDH_TIME": pdh_time, "PDL": pdl_val, "PDL_TIME": pdl_time,
        "PMH": pmh_val, "PMH_TIME": pmh_time, "PML": pml_val, "PML_TIME": pml_time
    }

# =====================================================================
# 4. 100% 完整对齐富途指标逻辑的 5M 回测引擎 (0.5 ATR 止损 / 1:2 止盈)
# =====================================================================
def simulate_trades_with_2b(df_5m, p, start_cutoff_ny, window_end_ny):
    trades = []
    if p is None or df_5m is None: return trades

    day_5m = df_5m[(df_5m.index >= start_cutoff_ny - timedelta(hours=3)) & (df_5m.index <= window_end_ny)].copy()
    if len(day_5m) < 25: return trades

    weights = np.arange(1, 21)
    day_5m["LWMA20"] = day_5m["Close"].rolling(20).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    
    tr = np.maximum(day_5m["High"] - day_5m["Low"], np.maximum((day_5m["High"] - day_5m["Close"].shift(1)).abs(), (day_5m["Low"] - day_5m["Close"].shift(1)).abs()))
    day_5m["ATR14"] = tr.rolling(14).mean()
    day_5m["VOL_MA"] = day_5m["Volume"].rolling(20).mean()
    day_5m["VOL_HEAVY"] = day_5m["Volume"] >= 1.25 * day_5m["VOL_MA"]

    rbs_top, rbs_bot = p["RBS_TOP"], p["RBS_BOT"]
    rbs2_top, rbs2_bot = p["RBS2_TOP"], p["RBS2_BOT"]
    sbr_top, sbr_bot = p["SBR_TOP"], p["SBR_BOT"]
    sbr2_top, sbr2_bot = p["SBR2_TOP"], p["SBR2_BOT"]
    pdl_line, pdh_line = p["PDL"], p["PDH"]
    pml_line, pmh_line = p["PML"], p["PMH"]
    bias = p["TREND_BIAS"]

    in_rbs1 = (day_5m["Low"] <= rbs_top) & (day_5m["Close"] >= rbs_bot)
    in_rbs2 = (rbs2_top > 0) & (day_5m["Low"] <= rbs2_top) & (day_5m["Close"] >= rbs2_bot)
    in_sbr1 = (day_5m["High"] >= sbr_bot) & (day_5m["Close"] <= sbr_top)
    in_sbr2 = (sbr2_top > 0) & (day_5m["High"] >= sbr2_bot) & (day_5m["Close"] <= sbr2_top)

    buy_zone = in_rbs1 | in_rbs2 | ((day_5m["Low"] <= pdl_line) & (day_5m["Close"] > pdl_line)) | ((day_5m["Low"] <= pml_line) & (day_5m["Close"] > pml_line))
    sell_zone = in_sbr1 | in_sbr2 | ((day_5m["High"] >= pdh_line) & (day_5m["Close"] < pdh_line)) | ((day_5m["High"] >= pmh_line) & (day_5m["Close"] < pmh_line))

    llv5_ref1 = day_5m["Low"].rolling(5).min().shift(1)
    hhv5_ref1 = day_5m["High"].rolling(5).max().shift(1)

    bull_2b_raw = ((day_5m["Low"] < llv5_ref1) | (day_5m["Low"] < pdl_line) | (day_5m["Low"] < pml_line)) & (day_5m["Close"] > llv5_ref1) & (day_5m["Close"] > day_5m["Open"])
    bear_2b_raw = ((day_5m["High"] > hhv5_ref1) | (day_5m["High"] > pdh_line) | (day_5m["High"] > pmh_line)) & (day_5m["Close"] < hhv5_ref1) & (day_5m["Close"] < day_5m["Open"])

    bull_engulf_raw = buy_zone & (day_5m["Close"] > day_5m["Open"]) & (day_5m["Close"].shift(1) < day_5m["Open"].shift(1)) & (day_5m["Close"] >= day_5m["Open"].shift(1)) & (day_5m["Open"] <= day_5m["Close"].shift(1))
    bear_engulf_raw = sell_zone & (day_5m["Close"] < day_5m["Open"]) & (day_5m["Close"].shift(1) > day_5m["Open"].shift(1)) & (day_5m["Close"] <= day_5m["Open"].shift(1)) & (day_5m["Open"] >= day_5m["Close"].shift(1))

    bull_star_raw = buy_zone & (day_5m["Close"].shift(2) < day_5m["Open"].shift(2)) & ((day_5m["Close"].shift(1) - day_5m["Open"].shift(1)).abs() <= 0.35 * (day_5m["High"].shift(1) - day_5m["Low"].shift(1))) & (day_5m["Close"] > day_5m["Open"]) & (day_5m["Close"] >= (day_5m["Open"].shift(2) + day_5m["Close"].shift(2)) / 2)
    bear_star_raw = sell_zone & (day_5m["Close"].shift(2) > day_5m["Open"].shift(2)) & ((day_5m["Close"].shift(1) - day_5m["Open"].shift(1)).abs() <= 0.35 * (day_5m["High"].shift(1) - day_5m["Low"].shift(1))) & (day_5m["Close"] < day_5m["Open"]) & (day_5m["Close"] <= (day_5m["Open"].shift(2) + day_5m["Close"].shift(2)) / 2)

    bull_123_raw = buy_zone & (day_5m["Close"] > day_5m["LWMA20"]) & (day_5m["Close"].shift(1) <= day_5m["LWMA20"].shift(1)) & (day_5m["Low"] > llv5_ref1) & (day_5m["Close"] > day_5m["Open"])
    bear_123_raw = sell_zone & (day_5m["Close"] < day_5m["LWMA20"]) & (day_5m["Close"].shift(1) >= day_5m["LWMA20"].shift(1)) & (day_5m["High"] < hhv5_ref1) & (day_5m["Close"] < day_5m["Open"])

    std_buy_setup = bull_engulf_raw | bull_star_raw | bull_123_raw
    std_sell_setup = bear_engulf_raw | bear_star_raw | bear_123_raw

    vol_heavy_or_ref1 = day_5m["VOL_HEAVY"] | day_5m["VOL_HEAVY"].shift(1)
    
    buy_2b_confirmed = bull_2b_raw.shift(1) & (day_5m["High"] > day_5m["High"].shift(1)) & (day_5m["Close"] > day_5m["Open"]) & vol_heavy_or_ref1
    sell_2b_confirmed = bear_2b_raw.shift(1) & (day_5m["Low"] < day_5m["Low"].shift(1)) & (day_5m["Close"] < day_5m["Open"]) & vol_heavy_or_ref1

    buy_std_confirmed = std_buy_setup.shift(1) & (day_5m["High"] > day_5m["High"].shift(1)) & (day_5m["Close"] > day_5m["Open"]) & (day_5m["Close"] > day_5m["LWMA20"]) & vol_heavy_or_ref1
    sell_std_confirmed = std_sell_setup.shift(1) & (day_5m["Low"] < day_5m["Low"].shift(1)) & (day_5m["Close"] < day_5m["Open"]) & (day_5m["Close"] < day_5m["LWMA20"]) & vol_heavy_or_ref1

    buy_2b_sig = (bias >= 0) & buy_2b_confirmed & (buy_2b_confirmed.rolling(5).sum() == 1)
    sell_2b_sig = (bias <= 0) & sell_2b_confirmed & (sell_2b_confirmed.rolling(5).sum() == 1)
    buy_std_sig = (bias >= 0) & buy_std_confirmed & (buy_std_confirmed.rolling(5).sum() == 1)
    sell_std_sig = (bias <= 0) & sell_std_confirmed & (sell_std_confirmed.rolling(5).sum() == 1)

    in_pos, pos_type = False, 0
    entry_p, sl_p, tp_p = 0.0, 0.0, 0.0
    entry_time_ny = None
    daily_trade_count = 0
    futu_signal_tag = ""

    start_idx = 0
    for idx_i, t_idx in enumerate(day_5m.index):
        if t_idx >= start_cutoff_ny:
            start_idx = idx_i
            break

    for i in range(start_idx, len(day_5m)):
        cur_t_ny = day_5m.index[i]
        c, h, l = day_5m["Close"].iloc[i], day_5m["High"].iloc[i], day_5m["Low"].iloc[i]
        atr_v = day_5m["ATR14"].iloc[i] if not np.isnan(day_5m["ATR14"].iloc[i]) else 0.8
        is_window_close = (cur_t_ny >= window_end_ny - timedelta(minutes=5))

        if in_pos:
            exit_flag, reason, exit_p = False, "", 0.0
            exit_time_ny = cur_t_ny + timedelta(minutes=5)
            
            if pos_type == 1:
                if is_window_close: exit_flag, reason, exit_p = True, "24:00 纪律清仓", c
                elif l <= sl_p: exit_flag, reason, exit_p = True, "SL (0.5 ATR)", sl_p
                elif h >= tp_p: exit_flag, reason, exit_p = True, "TP (1:2 止盈)", tp_p
            elif pos_type == -1:
                if is_window_close: exit_flag, reason, exit_p = True, "24:00 纪律清仓", c
                elif h >= sl_p: exit_flag, reason, exit_p = True, "SL (0.5 ATR)", sl_p
                elif l <= tp_p: exit_flag, reason, exit_p = True, "TP (1:2 止盈)", tp_p

            if exit_flag:
                pnl = (exit_p - entry_p) if pos_type == 1 else (entry_p - exit_p)
                trades.append({
                    "Signal": futu_signal_tag,
                    "Entry_MYT": entry_time_ny.astimezone(tz_myt).strftime("%H:%M"), "Entry_ET": entry_time_ny.strftime("%H:%M"),
                    "Exit_MYT": exit_time_ny.astimezone(tz_myt).strftime("%H:%M"), "Exit_ET": exit_time_ny.strftime("%H:%M"),
                    "Entry_Price": round(entry_p, 2), "Exit_Price": round(exit_p, 2),
                    "SL": round(sl_p, 2), "TP": round(tp_p, 2), "PnL_Points": round(pnl, 2),
                    "Reason": reason, "Result": "盈利" if pnl > 0 else ("保本" if pnl == 0 else "亏损"),
                    "Entry_DT_NY": entry_time_ny, "Exit_DT_NY": exit_time_ny
                })
                in_pos = False
                daily_trade_count += 1
                break

        if not in_pos and daily_trade_count == 0 and cur_t_ny < (window_end_ny - timedelta(minutes=15)):
            is_b2b = bool(buy_2b_sig.iloc[i])
            is_s2b = bool(sell_2b_sig.iloc[i])
            is_bstd = bool(buy_std_sig.iloc[i]) and not is_b2b
            is_sstd = bool(sell_std_sig.iloc[i]) and not is_s2b

            sl_dist = 0.5 * atr_v
            tp_dist = 1.0 * atr_v

            if is_b2b or is_bstd:
                in_pos, pos_type = True, 1
                entry_p = c
                sl_p = c - sl_dist
                tp_p = c + tp_dist
                entry_time_ny = cur_t_ny + timedelta(minutes=5)
                futu_signal_tag = "▲▲ 2B" if is_b2b else "▲ CALL"
            elif is_s2b or is_sstd:
                in_pos, pos_type = True, -1
                entry_p = c
                sl_p = c + sl_dist
                tp_p = c - tp_dist
                entry_time_ny = cur_t_ny + timedelta(minutes=5)
                futu_signal_tag = "▼▼ 2B" if is_s2b else "▼ PUT"

    return trades

# =====================================================================
# 5. 账本存储
# =====================================================================
RECORD_COLUMNS = [
    "Date_MYT", "TREND_BIAS", "EMA20_1H", "ATR_1H", "SBR_TOP", "SBR_BOT", "RBS_TOP", "RBS_BOT",
    "SBR2_TOP", "SBR2_BOT", "RBS2_TOP", "RBS2_BOT", "PDH", "PDL", "PMH", "PML",
    "Signal", "Entry_MYT", "Entry_ET", "Exit_MYT", "Exit_ET", "Entry_Price", "Exit_Price", "SL", "TP", "PnL_Points", "Reason", "Result"
]

def load_journal():
    if not os.path.exists(CSV_FILE):
        df_init = pd.DataFrame(columns=RECORD_COLUMNS)
        df_init.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
        return df_init
    df_read = pd.read_csv(CSV_FILE)
    for col in RECORD_COLUMNS:
        if col not in df_read.columns: df_read[col] = np.nan
    return df_read

def append_to_journal(date_str, params, trades):
    df_cur = load_journal()
    if not df_cur.empty and date_str in df_cur["Date_MYT"].astype(str).values: return False, "当天记录已存在"

    rows = []
    base_info = {
        "Date_MYT": date_str, "TREND_BIAS": params["TREND_BIAS"], "EMA20_1H": params.get("EMA20_1H", 0.0), "ATR_1H": params.get("ATR_1H", 0.0),
        "SBR_TOP": params["SBR_TOP"], "SBR_BOT": params["SBR_BOT"], "RBS_TOP": params["RBS_TOP"], "RBS_BOT": params["RBS_BOT"],
        "SBR2_TOP": params["SBR2_TOP"], "SBR2_BOT": params["SBR2_BOT"], "RBS2_TOP": params["RBS2_TOP"], "RBS2_BOT": params["RBS2_BOT"],
        "PDH": params["PDH"], "PDL": params["PDL"], "PMH": params["PMH"], "PML": params["PML"]
    }

    if trades:
        for t in trades:
            r = dict(base_info); r.update(t); rows.append(r)
    else:
        empty_t = {
            "Signal": "NO_TRADE", "Entry_MYT": "-", "Entry_ET": "-", "Exit_MYT": "-", "Exit_ET": "-",
            "Entry_Price": 0.0, "Exit_Price": 0.0, "SL": 0.0, "TP": 0.0, "PnL_Points": 0.0, "Reason": "窗口期无2B/战区信号", "Result": "无"
        }
        r = dict(base_info); r.update(empty_t); rows.append(r)

    df_new = pd.DataFrame(rows)[[c for c in RECORD_COLUMNS if c in rows[0]]]
    df_new.to_csv(CSV_FILE, index=False, encoding="utf-8-sig", mode="a" if os.path.exists(CSV_FILE) else "w", header=not os.path.exists(CSV_FILE))
    return True, f"成功记录 {len(rows)} 条明细"

# =====================================================================
# 6. UI 渲染与多模块看板
# =====================================================================
df_j = load_journal()
yesterday_d = now_myt.date() - timedelta(days=1)
yesterday_myt_str = yesterday_d.strftime("%Y-%m-%d")
has_10pm_p = (now_myt.hour >= 22 or now_myt.hour < 5)
has_8am_report = yesterday_myt_str in df_j["Date_MYT"].astype(str).values if not df_j.empty else False

s1, s2, s3, s4 = st.columns(4)
s1.success("✅ 10:00 PM 战区引擎已就绪" if has_10pm_p else "⏳ 10:00 PM 战区引擎等待中")
s2.success(f"✅ 战报已交付 ({yesterday_myt_str})" if has_8am_report else f"⏳ 战报待更新 ({yesterday_myt_str})")
s3.info("🎯 纪律窗口：22:00 - 24:00 (MYT) | 0.5 ATR 止损 / 1:2 TP")

with s4:
    if st.button("🧪 执行系统全链路测试"):
        with st.spinner("正在自检..."):
            d1, d5, errs = fetch_raw_data_with_retry(period_5m="5d")
            if errs: st.error("异常: " + "; ".join(errs))
            else: st.success("自检通过：接口正常。")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🎯 QQQ 战区座舱 (13行富途参数复制)", "📅 QQQ 2B同频月历与复盘图", "⚡ 当天 5M 实时执行座舱 (0.5 ATR)"])

with tab1:
    st.subheader("🎯 QQQ 5M 战区座舱 (含 SBR/SBR2/RBS/RBS2 & 2B)")
    c_t1, c_t2 = st.columns(2)
    c_t1.info("🕒 大马时间 (MYT): " + now_myt.strftime("%Y-%m-%d %H:%M:%S"))
    c_t2.info("🇺🇸 美东时间 (ET): " + now_ny.strftime("%Y-%m-%d %H:%M:%S"))

    if not has_10pm_p:
        st.warning("🔒 处于日间准备期。大马时间 22:00 准时解锁并生成今晚 13 行战区代码。")
    else:
        if st.button("🔄 刷新最新点位"): st.cache_data.clear(); st.rerun()
        d1h, d5m, _ = fetch_raw_data_with_retry(period_5m="5d")
        if d1h is not None:
            p = compute_futu_13_params(d1h, d5m, now_ny)
            if p:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🎯 QQQ 现价", f"${p['live_price']:.2f}")
                m2.metric("🚦 三灯信号定调", p["BIAS_DESC"])
                m3.metric("📈 1H EMA20 均线", f"${p['EMA20_1H']:.2f}")
                m4.metric("📊 1H ATR 波动", f"${p['ATR_1H']:.2f}")

                out_lines = [
                    f"TREND_BIAS := {p['TREND_BIAS']};       {{ 1. QQQ三灯判定: 1=绿灯做多, -1=红灯做空, 0=黄灯防守 }}",
                    "",
                    "{ --- 第一梯队主战区 (PRIMARY ZONES) --- }",
                    f"SBR_TOP := {round(p['SBR_TOP'], 2)}; {{ 2. PRIMARY 1H 阻力顶沿 [{p['SBR_TIME']}] }}",
                    f"SBR_BOT := {round(p['SBR_BOT'], 2)}; {{ 3. PRIMARY 1H 阻力底沿 [{p['SBR_TIME']}] }}",
                    f"RBS_TOP := {round(p['RBS_TOP'], 2)}; {{ 4. PRIMARY 1H 支撑顶沿 [{p['RBS_TIME']}] }}",
                    f"RBS_BOT := {round(p['RBS_BOT'], 2)}; {{ 5. PRIMARY 1H 支撑底沿 [{p['RBS_TIME']}] }}",
                    "",
                    "{ --- 第二梯队拓展战区 (SECONDARY ZONES) --- }",
                    f"SBR2_TOP := {round(p['SBR2_TOP'], 2)}; {{ 6. SECONDARY 1H 更高阻力顶沿 [{p['SBR2_TIME']}] }}",
                    f"SBR2_BOT := {round(p['SBR2_BOT'], 2)}; {{ 7. SECONDARY 1H 更高阻力底沿 [{p['SBR2_TIME']}] }}",
                    f"RBS2_TOP := {round(p['RBS2_TOP'], 2)}; {{ 8. SECONDARY 1H 更低支撑顶沿 [{p['RBS2_TIME']}] }}",
                    f"RBS2_BOT := {round(p['RBS2_BOT'], 2)}; {{ 9. SECONDARY 1H 更低支撑底沿 [{p['RBS2_TIME']}] }}",
                    "",
                    "{ --- 全市场客观极值 (SWEEP ANCHORS) --- }",
                    f"PDH_LINE := {round(p['PDH'], 2)}; {{ 10. 昨日最高价 PDH [{p['PDH_TIME']}] }}",
                    f"PDL_LINE := {round(p['PDL'], 2)}; {{ 11. 昨日最低价 PDL [{p['PDL_TIME']}] }}",
                    f"PMH_LINE := {round(p['PMH'], 2)}; {{ 12. 盘前最高价 PMH [{p['PMH_TIME']}] }}",
                    f"PML_LINE := {round(p['PML'], 2)}; {{ 13. 盘前最低价 PML [{p['PML_TIME']}] }}"
                ]
                st.markdown("#### 📋 复制到富途指标顶部 13 行代码 (点右上角复制):")
                st.code("\n".join(out_lines), language="pascal")

with tab2:
    st.subheader("📅 QQQ 2B 同频月历与执行细节")
    col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1.5])
    with col_btn1:
        if st.button("🛠️ 结算昨夜 22:00-24:00 账本"):
            with st.spinner("正在结算..."):
                d1h, d5m, _ = fetch_raw_data_with_retry(period_5m="5d")
                target_d = now_myt.date() - timedelta(days=1)
                dt_10pm_myt = tz_myt.localize(datetime.datetime.combine(target_d, datetime.time(22, 0, 0)))
                cutoff_ny = dt_10pm_myt.astimezone(tz_ny)
                window_end_ny = cutoff_ny + timedelta(hours=2)
                p = compute_futu_13_params(d1h, d5m, cutoff_ny)
                if p:
                    trades = simulate_trades_with_2b(d5m, p, cutoff_ny, window_end_ny)
                    ok, msg = append_to_journal(target_d.strftime("%Y-%m-%d"), p, trades)
                    if ok: st.success(msg); st.rerun()
                    else: st.warning(msg)
    with col_btn2:
        if st.button("⚡ 一键回溯补录当月所有历史交易日 (Backfill)"):
            with st.spinner("正在回溯运算..."):
                d1h, d5m, _ = fetch_raw_data_with_retry(period_5m="1mo")
                if d1h is not None and d5m is not None:
                    dates_in_5m = sorted(list(set(d5m.index.date)))
                    added_cnt = 0
                    for d in dates_in_5m:
                        if d >= now_ny.date(): continue
                        dt_10pm_myt = tz_myt.localize(datetime.datetime.combine(d, datetime.time(22, 0, 0)))
                        cutoff_ny = dt_10pm_myt.astimezone(tz_ny)
                        window_end_ny = cutoff_ny + timedelta(hours=2)
                        p_day = compute_futu_13_params(d1h, d5m, cutoff_ny)
                        if p_day:
                            trades_day = simulate_trades_with_2b(d5m, p_day, cutoff_ny, window_end_ny)
                            ok, _ = append_to_journal(d.strftime("%Y-%m-%d"), p_day, trades_day)
                            if ok: added_cnt += 1
                    st.success(f"🎉 回溯完成，新增 {added_cnt} 个交易日记录！")
                    st.rerun()
    with col_btn3:
        if st.button("🗑️ 清空历史账本重新生成"):
            if os.path.exists(CSV_FILE): os.remove(CSV_FILE); st.success("账本已重置！"); st.rerun()

    df_journal = load_journal()
    if not df_journal.empty and "Date_MYT" in df_journal.columns:
        df_journal["Date_MYT_dt"] = pd.to_datetime(df_journal["Date_MYT"]).dt.date
        df_journal["Year"] = pd.to_datetime(df_journal["Date_MYT"]).dt.year
        df_journal["Month"] = pd.to_datetime(df_journal["Date_MYT"]).dt.month
    else:
        df_journal["Year"], df_journal["Month"], df_journal["Date_MYT_dt"] = [], [], []

    cy, cm, cdl = st.columns([1.5, 1.5, 2])
    with cy: sel_y = st.selectbox("年份", options=[2025, 2026, 2027], index=1)
    with cm: sel_m = st.selectbox("月份", options=list(range(1, 13)), index=now_myt.month - 1)

    df_m = df_journal[(df_journal["Year"] == sel_y) & (df_journal["Month"] == sel_m)] if not df_journal.empty else pd.DataFrame()
    valid_t = df_m[df_m["Signal"] != "NO_TRADE"] if not df_m.empty else pd.DataFrame()
    tot_pts = valid_t["PnL_Points"].sum() if not valid_t.empty else 0.0
    tot_cnt = len(valid_t)
    w_cnt = len(valid_t[valid_t["PnL_Points"] > 0]) if not valid_t.empty else 0
    w_rate = (w_cnt / tot_cnt * 100) if tot_cnt > 0 else 0.0

    with cdl:
        csv_bytes = df_m.to_csv(index=False).encode("utf-8-sig") if not df_m.empty else "".encode("utf-8-sig")
        st.download_button(
            label=f"📥 导出 {sel_y}年{sel_m}月 完整账本 (.csv)",
            data=csv_bytes, file_name=f"Futu_Full_Journal_{sel_y}_{str(sel_m).zfill(2)}.csv", mime="text/csv", disabled=df_m.empty
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🗓️ 选定月份", f"{sel_y} 年 {sel_m} 月")
    k2.metric("💰 窗口盈亏", f"{tot_pts:+.2f} pt")
    k3.metric("🎯 窗口胜率", f"{w_rate:.1f}%", f"{w_cnt}/{tot_cnt} 胜")
    k4.metric("📊 开仓总笔数", f"{tot_cnt} 笔")

    st.markdown("---")
    weekdays = ["周一 (Mon)", "周二 (Tue)", "周三 (Wed)", "周四 (Thu)", "周五 (Fri)", "周六 (Sat)", "周日 (Sun)"]
    h_cols = st.columns(7)
    for idx, hc in enumerate(h_cols): hc.markdown(f"<div style='text-align:center; font-weight:bold; color:#4a5568;'>{weekdays[idx]}</div>", unsafe_allow_html=True)

    cal = calendar.monthcalendar(sel_y, sel_m)
    for week in cal:
        w_cols = st.columns(7)
        for d_idx, day in enumerate(week):
            with w_cols[d_idx]:
                if day == 0: st.markdown("<div style='height:120px;'></div>", unsafe_allow_html=True); continue
                cur_d = datetime.date(sel_y, sel_m, day)
                is_weekend = (d_idx >= 5)
                if is_weekend:
                    st.markdown(f"<div style='background-color:#edf2f7; border-radius:8px; padding:8px; height:120px; border:1px dashed #cbd5e0; text-align:center;'><div style='font-size:13px; color:#a0aec0; text-align:left;'><b>{day}</b></div><div style='font-size:18px; margin-top:10px;'>❌</div><div style='font-size:11px; color:#a0aec0;'>周末休市</div></div>", unsafe_allow_html=True)
                else:
                    d_recs = df_m[df_m["Date_MYT_dt"] == cur_d] if not df_m.empty else pd.DataFrame()
                    if not d_recs.empty:
                        r_t = d_recs[d_recs["Signal"] != "NO_TRADE"]
                        cnt = len(r_t); pts = r_t["PnL_Points"].sum() if not r_t.empty else 0.0
                        b_val = d_recs.iloc[0].get("TREND_BIAS", 0)
                        b_str = "多" if b_val == 1 else ("空" if b_val == -1 else "黄灯")
                        if cnt == 0:
                            st.markdown(f"<div style='background-color:#f7fafc; border-radius:8px; padding:8px; height:120px; border:1px solid #e2e8f0; text-align:center;'><div style='font-size:13px; color:#718096; text-align:left;'><b>{day}</b> <span style='font-size:10px; color:#a0aec0;'>({b_str})</span></div><div style='font-size:12px; color:#718096; margin-top:15px;'>⚪ 未触及战区</div><div style='font-size:10px; color:#a0aec0;'>空仓休战</div></div>", unsafe_allow_html=True)
                        else:
                            bg = "#e6fffa" if pts >= 0 else "#fff5f5"
                            bd = "#38b2ac" if pts >= 0 else "#e53e3e"
                            tc = "#234e52" if pts >= 0 else "#742a2a"
                            sgn = "+" if pts > 0 else ""
                            st.markdown(f"<div style='background-color:{bg}; border-radius:8px; padding:8px; height:120px; border:2px solid {bd}; text-align:center;'><div style='font-size:13px; color:{tc}; text-align:left;'><b>{day}</b> <span style='font-size:10px; color:{bd};'>({b_str})</span></div><div style='font-size:15px; font-weight:bold; color:{bd}; margin-top:2px;'>{sgn}{pts:.2f} pt</div><div style='font-size:11px; color:{tc};'>{cnt} 笔交易 (22-24点)</div></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color:#ffffff; border-radius:8px; padding:8px; height:120px; border:1px solid #edf2f7; text-align:center;'><div style='font-size:13px; color:#cbd5e0; text-align:left;'><b>{day}</b></div><div style='font-size:11px; color:#cbd5e0; margin-top:25px;'>-</div></div>", unsafe_allow_html=True)

    # ---------------- 历史交易日 5M K线逐笔复盘视窗 ----------------
    st.markdown("---")
    st.subheader("🔍 历史交易日 5M K线与信号复盘图 (22:00 - 24:00 MYT)")

    recorded_dates = sorted(list(set(df_journal["Date_MYT"].dropna().astype(str).tolist())), reverse=True) if not df_journal.empty else []
    
    if recorded_dates:
        selected_date_str = st.selectbox("选择复盘交易日 (MYT)", options=recorded_dates)
        
        # 提取当天的记录
        day_record = df_journal[df_journal["Date_MYT"] == selected_date_str].iloc[0]
        
        # 获取 5M 数据
        _, df_5m_all, _ = fetch_raw_data_with_retry(period_5m="1mo")
        
        if df_5m_all is not None and not df_5m_all.empty:
            sel_d = pd.to_datetime(selected_date_str).date()
            dt_start_myt = tz_myt.localize(datetime.datetime.combine(sel_d, datetime.time(21, 30, 0)))
            dt_end_myt = tz_myt.localize(datetime.datetime.combine(sel_d, datetime.time(24, 0, 0))) + timedelta(minutes=15)
            
            start_ny_view = dt_start_myt.astimezone(tz_ny)
            end_ny_view = dt_end_myt.astimezone(tz_ny)
            
            sub_chart = df_5m_all[(df_5m_all.index >= start_ny_view) & (df_5m_all.index <= end_ny_view)].copy()
            
            if not sub_chart.empty:
                # 转换索引时间为 MYT 方便对照
                sub_chart["MYT_Time"] = sub_chart.index.tz_convert(tz_myt)
                
                fig_replay = go.Figure()
                fig_replay.add_trace(go.Candlestick(
                    x=sub_chart["MYT_Time"],
                    open=sub_chart['Open'], high=sub_chart['High'],
                    low=sub_chart['Low'], close=sub_chart['Close'],
                    name="5M K线 (MYT)"
                ))
                
                # 标记战区阻力支撑线
                rbs_top_val = float(day_record.get("RBS_TOP", 0))
                sbr_bot_val = float(day_record.get("SBR_BOT", 0))
                if rbs_top_val > 0:
                    fig_replay.add_hline(y=rbs_top_val, line_dash="dash", line_color="cyan", annotation_text="RBS 支撑顶")
                if sbr_bot_val > 0:
                    fig_replay.add_hline(y=sbr_bot_val, line_dash="dash", line_color="magenta", annotation_text="SBR 阻力底")

                # 标记交易信号及入场/平仓线
                sig_type = str(day_record.get("Signal", "NO_TRADE"))
                if sig_type != "NO_TRADE" and not pd.isna(day_record.get("Entry_Price")):
                    ep = float(day_record["Entry_Price"])
                    xp = float(day_record["Exit_Price"])
                    sl = float(day_record["SL"])
                    tp = float(day_record["TP"])
                    
                    fig_replay.add_hline(y=ep, line_color="gold", line_width=1.5, annotation_text=f"入场价: {ep}")
                    fig_replay.add_hline(y=sl, line_dash="dot", line_color="red", annotation_text=f"止损 (0.5 ATR): {sl}")
                    fig_replay.add_hline(y=tp, line_dash="dot", line_color="green", annotation_text=f"止盈 (1:2): {tp}")

                fig_replay.update_layout(
                    title=f"{selected_date_str} 交易执行结构 | 信号: {sig_type} | 结果: {day_record.get('Result', '-')}",
                    xaxis_rangeslider_visible=False,
                    height=500,
                    margin=dict(l=10, r=10, t=40, b=10),
                    template="plotly_dark"
                )
                st.plotly_chart(fig_replay, use_container_width=True)
                
                # 当天关键数据卡片
                rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                rc1.metric("信号类型", sig_type)
                rc2.metric("入场时间 (MYT)", str(day_record.get("Entry_MYT", "-")))
                rc3.metric("离场时间 (MYT)", str(day_record.get("Exit_MYT", "-")))
                rc4.metric("离场原因", str(day_record.get("Reason", "-")))
                rc5.metric("盈亏点数", f"{float(day_record.get('PnL_Points', 0)):+.2f} pt")
            else:
                st.warning("所选日期的 5M K线数据在当前缓冲池中未找到。")
    else:
        st.info("暂无历史记录，点击上方【一键回溯补录】或【结算】生成数据后即可在此复盘。")

    with st.expander("🔍 展开查看完整明细表 (Full Data Table)"):
        if not df_m.empty: st.dataframe(df_m.drop(columns=["Date_MYT_dt", "Year", "Month"], errors="ignore"), use_container_width=True)
        else: st.info("当月暂无交易明细。")

with tab3:
    st.subheader("⚡ 当天 5M 实时执行座舱 (0.5 ATR 止损 / 1:2 TP)")
    
    col_c1, col_c2 = st.columns([1, 3])
    with col_c1:
        risk_per_trade = st.number_input("单笔固定风险金额 (USD)", min_value=10.0, value=200.0, step=50.0)
        if st.button("🔄 刷新 5M 实时数据"):
            st.cache_data.clear()
            st.rerun()

    _, df_5m_live, _ = fetch_raw_data_with_retry(period_5m="5d")
    if df_5m_live is not None and not df_5m_live.empty:
        df_live = df_5m_live.copy()
        df_live["EMA_9"] = df_live["Close"].ewm(span=9, adjust=False).mean()
        df_live["EMA_21"] = df_live["Close"].ewm(span=21, adjust=False).mean()
        
        tr_live = np.maximum(df_live["High"] - df_live["Low"], np.maximum((df_live["High"] - df_live["Close"].shift(1)).abs(), (df_live["Low"] - df_live["Close"].shift(1)).abs()))
        df_live["ATR"] = tr_live.rolling(14).mean()
        
        typical_p = (df_live["High"] + df_live["Low"] + df_live["Close"]) / 3
        df_live["VWAP"] = (typical_p * df_live["Volume"]).cumsum() / df_live["Volume"].cumsum()

        cur_bar = df_live.iloc[-1]
        prev_bar = df_live.iloc[-2]
        c_price = float(cur_bar["Close"])
        c_atr = float(cur_bar["ATR"]) if not np.isnan(cur_bar["ATR"]) else 0.8
        
        sl_dist = 0.5 * c_atr
        tp_dist = 1.0 * c_atr

        long_sl = c_price - sl_dist
        long_tp = c_price + tp_dist
        long_shares = int(risk_per_trade / sl_dist) if sl_dist > 0 else 0

        short_sl = c_price + sl_dist
        short_tp = c_price - tp_dist
        short_shares = int(risk_per_trade / sl_dist) if sl_dist > 0 else 0

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("QQQ 实时价 (5M)", f"${c_price:.2f}", f"{(c_price - prev_bar['Close']):+.2f}")
        mc2.metric("5M ATR (14)", f"${c_atr:.2f}", f"0.5 ATR = ${sl_dist:.2f}")
        mc3.metric("多头建议仓位", f"{long_shares} 股", f"总敞口: ${long_shares * c_price:,.0f}")
        mc4.metric("空头建议仓位", f"{short_shares} 股", f"总敞口: ${short_shares * c_price:,.0f}")

        t_long, t_short = st.columns(2)
        with t_long:
            st.markdown("**做多方案 (Long Setup)**")
            st.table(pd.DataFrame({
                "指标/参数": ["进场参考价", "止损位 (0.5 ATR)", "止盈位 (1:2 TP)", "单笔止损幅度", "建议买入手数"],
                "数值": [f"${c_price:.2f}", f"${long_sl:.2f}", f"${long_tp:.2f}", f"-${sl_dist:.2f} ({(sl_dist/c_price)*100:.2f}%)", f"{long_shares} 股"]
            }))
        with t_short:
            st.markdown("**做空方案 (Short Setup)**")
            st.table(pd.DataFrame({
                "指标/参数": ["进场参考价", "止损位 (0.5 ATR)", "止盈位 (1:2 TP)", "单笔止损幅度", "建议卖空手数"],
                "数值": [f"${c_price:.2f}", f"${short_sl:.2f}", f"${short_tp:.2f}", f"+${sl_dist:.2f} ({(sl_dist/c_price)*100:.2f}%)", f"{short_shares} 股"]
            }))

        plot_df = df_live.tail(60)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open'], high=plot_df['High'],
            low=plot_df['Low'], close=plot_df['Close'],
            name="5M K线"
        ))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_9'], line=dict(color='orange', width=1), name="EMA 9"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_21'], line=dict(color='blue', width=1), name="EMA 21"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['VWAP'], line=dict(color='purple', width=1.5, dash='dot'), name="VWAP"))
        
        fig.add_hline(y=long_tp, line_dash="dash", line_color="green", annotation_text="做多目标 TP (1:2)")
        fig.add_hline(y=long_sl, line_dash="dash", line_color="red", annotation_text="做多止损 SL (0.5 ATR)")

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("暂未获取到 5M 实时数据，请点击刷新重试。")
