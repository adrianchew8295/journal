import calendar
import datetime
from datetime import timedelta
import os
import time
import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import yfinance as yf

# =====================================================================
# 1. 基础配置与凭证锁定
# =====================================================================
st.set_page_config(
    page_title="QQQ 5M 战区座舱",
    page_icon="🌊",
    layout="wide"
)

TIINGO_TOKEN = "bcffe3a5cf7eeef085e405cfa4a3e5691b976217"
TICKER = "QQQ"
CSV_FILE = "monthly_trade_records.csv"

tz_myt = pytz.timezone("Asia/Kuala_Lumpur")
tz_ny = pytz.timezone("America/New_York")

now_myt = datetime.datetime.now(tz_myt)
now_ny = datetime.datetime.now(tz_ny)

# =====================================================================
# 2. 数据抓取引擎 (带重试保护)
# =====================================================================
def fetch_raw_data_with_retry(max_retries=3):
    df_1h, source_1h = None, "None"
    df_5m, source_5m = None, "None"
    err_log = []
    start_d = now_myt - timedelta(days=45)
    start_str = start_d.strftime("%Y-%m-%d")

    for attempt in range(max_retries):
        url = (
            "https://api.tiingo.com/iex/"
            + TICKER
            + "/prices?startDate="
            + start_str
            + "&resampleFreq=1hour&token="
            + TIINGO_TOKEN
            + "&columns=open,high,low,close,volume"
        )
        try:
            resp = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                timeout=6
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and len(data) >= 30:
                    df_t = pd.DataFrame(data)
                    df_t["date"] = pd.to_datetime(df_t["date"])
                    df_t.set_index("date", inplace=True)
                    df_t.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume"
                        },
                        inplace=True
                    )
                    df_1h = df_t[["Open", "High", "Low", "Close", "Volume"]].sort_index()
                    if df_1h.index.tz is None:
                        df_1h.index = df_1h.index.tz_localize("UTC").tz_convert(tz_ny)
                    else:
                        df_1h.index = df_1h.index.tz_convert(tz_ny)
                    source_1h = "Tiingo IEX API"
                    break
        except Exception as e:
            err_log.append("Tiingo 1H 失败: " + str(e))
            time.sleep(1)

    if df_1h is None:
        try:
            df_yf = yf.download(
                TICKER,
                period="1mo",
                interval="1h",
                prepost=True,
                progress=False
            )
            if df_yf is not None and not df_yf.empty:
                if isinstance(df_yf.columns, pd.MultiIndex):
                    df_yf.columns = df_yf.columns.get_level_values(0)
                df_1h = df_yf[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
                if df_1h.index.tz is None:
                    df_1h.index = df_1h.index.tz_localize("UTC").tz_convert(tz_ny)
                else:
                    df_1h.index = df_1h.index.tz_convert(tz_ny)
                source_1h = "YahooFinance (1H)"
        except Exception as e:
            err_log.append("YahooFinance 1H 失败: " + str(e))

    for attempt in range(max_retries):
        try:
            df_5m_raw = yf.download(
                TICKER,
                period="5d",
                interval="5m",
                prepost=True,
                progress=False
            )
            if df_5m_raw is not None and not df_5m_raw.empty:
                if isinstance(df_5m_raw.columns, pd.MultiIndex):
                    df_5m_raw.columns = df_5m_raw.columns.get_level_values(0)
                df_5m = df_5m_raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
                if df_5m.index.tz is None:
                    df_5m.index = df_5m.index.tz_localize("UTC").tz_convert(tz_ny)
                else:
                    df_5m.index = df_5m.index.tz_convert(tz_ny)
                source_5m = "YahooFinance (5M)"
                break
        except Exception as e:
            err_log.append("YahooFinance 5M 失败: " + str(e))
            time.sleep(1)

    return df_1h, source_1h, df_5m, source_5m, err_log

# =====================================================================
# 3. 核心运算：10 PM 参数生成与 5M 信号复盘
# =====================================================================
def compute_futu_13_params(df_1h, df_5m, as_of_ny_time):
    if df_1h is None:
        return None
    sub_1h = df_1h[df_1h.index <= as_of_ny_time].copy()
    if len(sub_1h) < 25:
        return None

    today_ny = as_of_ny_time.date()
    df_rth = sub_1h[
        (sub_1h.index.hour > 9)
        | ((sub_1h.index.hour == 9) & (sub_1h.index.minute >= 30))
    ]
    df_rth = df_rth[df_rth.index.hour < 16]
    past_dates = sorted(list(set(df_rth.index.date)))
    past_dates = [d for d in past_dates if d < today_ny]

    if past_dates:
        prev_df = df_rth[df_rth.index.date == past_dates[-1]]
        pdh_idx = prev_df["High"].idxmax()
        pdl_idx = prev_df["Low"].idxmin()
        pdh_val = float(prev_df.loc[pdh_idx, "High"])
        pdl_val = float(prev_df.loc[pdl_idx, "Low"])
        pdh_time = pdh_idx.strftime("%Y-%m-%d %H:%M ET")
        pdl_time = pdl_idx.strftime("%Y-%m-%d %H:%M ET")
    else:
        pdh_val = float(sub_1h["High"].iloc[-10:].max())
        pdl_val = float(sub_1h["Low"].iloc[-10:].min())
        pdh_time = "Prior Session"
        pdl_time = "Prior Session"

    if df_5m is not None:
        sub_5m_pm = df_5m[
            (df_5m.index.date == today_ny)
            & (df_5m.index.hour >= 4)
            & (df_5m.index < as_of_ny_time)
        ]
    else:
        sub_5m_pm = None

    if sub_5m_pm is not None and not sub_5m_pm.empty:
        pmh_idx = sub_5m_pm["High"].idxmax()
        pml_idx = sub_5m_pm["Low"].idxmin()
        pmh_val = float(sub_5m_pm.loc[pmh_idx, "High"])
        pml_val = float(sub_5m_pm.loc[pml_idx, "Low"])
        pmh_time = pmh_idx.strftime("%Y-%m-%d %H:%M ET")
        pml_time = pml_idx.strftime("%Y-%m-%d %H:%M ET")
        live_price = float(sub_5m_pm["Close"].iloc[-1])
    else:
        pmh_val = float(sub_1h["High"].iloc[-4:].max())
        pml_val = float(sub_1h["Low"].iloc[-4:].min())
        pmh_time = "Recent 1H"
        pml_time = "Recent 1H"
        live_price = float(sub_1h["Close"].iloc[-1])

    sub_1h["EMA20"] = sub_1h["Close"].ewm(span=20, adjust=False).mean()
    sub_1h["SMA50"] = sub_1h["Close"].rolling(window=50).mean()

    tr = np.maximum(
        sub_1h["High"] - sub_1h["Low"],
        np.maximum(
            (sub_1h["High"] - sub_1h["Close"].shift(1)).abs(),
            (sub_1h["Low"] - sub_1h["Close"].shift(1)).abs()
        )
    )
    atr_series = tr.rolling(14).mean()
    if not np.isnan(atr_series.iloc[-1]):
        atr = float(atr_series.iloc[-1])
    else:
        atr = live_price * 0.008

    subset = sub_1h.iloc[-60:].copy()
    highs = subset["High"].values
    lows = subset["Low"].values
    opens = subset["Open"].values
    closes = subset["Close"].values
    times = subset.index

    pivots_high = []
    pivots_low = []
    for i in range(2, len(subset) - 2):
        if highs[i] == max(highs[i-2:i+3]):
            pivots_high.append((
                float(highs[i]),
                float(max(opens[i], closes[i])),
                times[i].strftime("%m-%d %H:%M ET")
            ))
        if lows[i] == min(lows[i-2:i+3]):
            pivots_low.append((
                float(min(opens[i], closes[i])),
                float(lows[i]),
                times[i].strftime("%m-%d %H:%M ET")
            ))

    valid_highs = [p for p in pivots_high if p[0] > live_price]
    valid_highs.sort(key=lambda x: x[0])
    if len(valid_highs) >= 1:
        sbr_top, sbr_bot, sbr_time = valid_highs[0]
    else:
        sbr_top, sbr_bot, sbr_time = live_price + 1.2 * atr, live_price + 0.6 * atr, "Range High"

    if len(valid_highs) >= 2:
        sbr2_top, sbr2_bot, sbr2_time = valid_highs[1]
    else:
        sbr2_top, sbr2_bot, sbr2_time = sbr_top + 1.2 * atr, sbr_top + 0.5 * atr, "Tier-2 High"

    valid_lows = [p for p in pivots_low if p[1] < live_price]
    valid_lows.sort(key=lambda x: x[1], reverse=True)
    if len(valid_lows) >= 1:
        rbs_top, rbs_bot, rbs_time = valid_lows[0]
    else:
        rbs_top, rbs_bot, rbs_time = live_price - 0.6 * atr, live_price - 1.2 * atr, "Range Low"

    if len(valid_lows) >= 2:
        rbs2_top, rbs2_bot, rbs2_time = valid_lows[1]
    else:
        rbs2_top, rbs2_bot, rbs2_time = rbs_bot - 0.5 * atr, rbs_bot - 1.2 * atr, "Tier-2 Low"

    ema20_now = float(sub_1h["EMA20"].iloc[-1])
    if not np.isnan(sub_1h["SMA50"].iloc[-1]):
        sma50_now = float(sub_1h["SMA50"].iloc[-1])
    else:
        sma50_now = ema20_now

    if live_price > ema20_now and ema20_now >= sma50_now:
        score_ma = 1
    elif live_price < ema20_now and ema20_now <= sma50_now:
        score_ma = -1
    else:
        score_ma = 0

    score_hhll = 0
    if len(pivots_high) >= 2 and len(pivots_low) >= 2:
        last_2_h = [p[0] for p in pivots_high[-2:]]
        last_2_l = [p[1] for p in pivots_low[-2:]]
        if last_2_h[1] > last_2_h[0] and last_2_l[1] > last_2_l[0]:
            score_hhll = 1
        elif last_2_h[1] < last_2_h[0] and last_2_l[1] < last_2_l[0]:
            score_hhll = -1

    ema20_prev = float(sub_1h["EMA20"].iloc[-5])
    ema_slope = (ema20_now - ema20_prev) / ema20_prev * 100
    if ema_slope > 0.15:
        score_slope = 1
    elif ema_slope < -0.15:
        score_slope = -1
    else:
        score_slope = 0

    total_score = score_ma + score_hhll + score_slope
    if total_score >= 2:
        trend_bias = 1
    elif total_score <= -2:
        trend_bias = -1
    else:
        trend_bias = 0

    return {
        "live_price": live_price,
        "TREND_BIAS": trend_bias,
        "TOTAL_SCORE": total_score,
        "SBR_TOP": sbr_top,
        "SBR_BOT": sbr_bot,
        "SBR_TIME": sbr_time,
        "RBS_TOP": rbs_top,
        "RBS_BOT": rbs_bot,
        "RBS_TIME": rbs_time,
        "SBR2_TOP": sbr2_top,
        "SBR2_BOT": sbr2_bot,
        "SBR2_TIME": sbr2_time,
        "RBS2_TOP": rbs2_top,
        "RBS2_BOT": rbs2_bot,
        "RBS2_TIME": rbs2_time,
        "PDH": pdh_val,
        "PDH_TIME": pdh_time,
        "PDL": pdl_val,
        "PDL_TIME": pdl_time,
        "PMH": pmh_val,
        "PMH_TIME": pmh_time,
        "PML": pml_val,
        "PML_TIME": pml_time
    }

def simulate_night_trades(df_5m, p, start_cutoff_ny, close_ny):
    trades = []
    if p is None or p["TREND_BIAS"] == 0 or df_5m is None:
        return trades

    day_5m = df_5m[
        (df_5m.index >= start_cutoff_ny)
        & (df_5m.index <= close_ny)
    ].copy()
    if len(day_5m) < 20:
        return trades

    weights = np.arange(1, 21)
    day_5m["LWMA20"] = day_5m["Close"].rolling(20).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(),
        raw=True
    )
    tr_5m = np.maximum(
        day_5m["High"] - day_5m["Low"],
        np.maximum(
            (day_5m["High"] - day_5m["Close"].shift(1)).abs(),
            (day_5m["Low"] - day_5m["Close"].shift(1)).abs()
        )
    )
    day_5m["ATR14"] = tr_5m.rolling(14).mean()
    day_5m["VOL_MA"] = day_5m["Volume"].rolling(20).mean()
    day_5m["VOL_HEAVY"] = day_5m["Volume"] >= (1.25 * day_5m["VOL_MA"])

    in_pos = False
    pos_type = 0
    entry_p = 0.0
    sl_p = 0.0
    tp_p = 0.0
    be_trigger_p = 0.0
    entry_idx = 0
    entry_time_ny = None
    futu_signal_tag = ""

    for i in range(20, len(day_5m)):
        cur_t_ny = day_5m.index[i]
        c = day_5m["Close"].iloc[i]
        o = day_5m["Open"].iloc[i]
        h = day_5m["High"].iloc[i]
        l = day_5m["Low"].iloc[i]
        if not np.isnan(day_5m["ATR14"].iloc[i]):
            atr_v = day_5m["ATR14"].iloc[i]
        else:
            atr_v = 0.5
        vol_h = day_5m["VOL_HEAVY"].iloc[i]
        lwma = day_5m["LWMA20"].iloc[i]

        if in_pos:
            bars_held = i - entry_idx
            exit_flag = False
            reason = ""
            exit_p = 0.0
            is_eod = (cur_t_ny >= close_ny - timedelta(minutes=10))

            if pos_type == 1:
                if h >= entry_p + (entry_p - sl_p):
                    be_trigger_p = entry_p
                if is_eod:
                    exit_flag, reason, exit_p = True, "EOD (收盘清仓)", c
                elif l <= sl_p:
                    exit_flag, reason, exit_p = True, "SL (极限止损)", sl_p
                elif h >= tp_p:
                    exit_flag, reason, exit_p = True, "TP (2R止盈)", tp_p
                elif be_trigger_p > 0 and l <= be_trigger_p:
                    exit_flag, reason, exit_p = True, "BE (1R保本损)", be_trigger_p
                elif c < lwma and bars_held >= 5:
                    exit_flag, reason, exit_p = True, "MA Cut (均线破位)", c
                elif bars_held >= 15:
                    exit_flag, reason, exit_p = True, "Time (15根K超时)", c
            elif pos_type == -1:
                if l <= entry_p - (sl_p - entry_p):
                    be_trigger_p = entry_p
                if is_eod:
                    exit_flag, reason, exit_p = True, "EOD (收盘清仓)", c
                elif h >= sl_p:
                    exit_flag, reason, exit_p = True, "SL (极限止损)", sl_p
                elif l <= tp_p:
                    exit_flag, reason, exit_p = True, "TP (2R止盈)", tp_p
                elif be_trigger_p > 0 and h >= be_trigger_p:
                    exit_flag, reason, exit_p = True, "BE (1R保本损)", be_trigger_p
                elif c > lwma and bars_held >= 5:
                    exit_flag, reason, exit_p = True, "MA Cut (均线破位)", c
                elif bars_held >= 15:
                    exit_flag, reason, exit_p = True, "Time (15根K超时)", c

            if exit_flag:
                if pos_type == 1:
                    pnl = exit_p - entry_p
                else:
                    pnl = entry_p - exit_p
                trades.append({
                    "Signal": futu_signal_tag,
                    "Entry_MYT": entry_time_ny.astimezone(tz_myt).strftime("%H:%M"),
                    "Entry_ET": entry_time_ny.strftime("%H:%M"),
                    "Exit_MYT": cur_t_ny.astimezone(tz_myt).strftime("%H:%M"),
                    "Exit_ET": cur_t_ny.strftime("%H:%M"),
                    "Entry_Price": round(entry_p, 2),
                    "Exit_Price": round(exit_p, 2),
                    "SL": round(sl_p, 2),
                    "TP": round(tp_p, 2),
                    "PnL_Points": round(pnl, 2),
                    "Reason": reason,
                    "Result": "盈利" if pnl > 0 else "亏损"
                })
                in_pos = False
                continue

        if not in_pos and i >= 2:
            prev_c = day_5m["Close"].iloc[i-1]
            prev_o = day_5m["Open"].iloc[i-1]
            prev_h = day_5m["High"].iloc[i-1]
            prev_l = day_5m["Low"].iloc[i-1]
            vol_ok = vol_h or day_5m["VOL_HEAVY"].iloc[i-1]

            buy_zone = (
                (prev_l <= p["RBS_TOP"] and prev_c >= p["RBS_BOT"])
                or (prev_l <= p["PDL"] and prev_c > p["PDL"])
                or (prev_l <= p["PML"] and prev_c > p["PML"])
            )
            sell_zone = (
                (prev_h >= p["SBR_BOT"] and prev_c <= p["SBR_TOP"])
                or (prev_h >= p["PDH"] and prev_c < p["PDH"])
                or (prev_h >= p["PMH"] and prev_c < p["PMH"])
            )

            llv5 = day_5m["Low"].iloc[i-6:i-1].min()
            hhv5 = day_5m["High"].iloc[i-6:i-1].max()

            b_2b = (
                (prev_l < llv5 or prev_l < p["PDL"] or prev_l < p["PML"])
                and (prev_c > llv5)
                and (prev_c > prev_o)
            )
            s_2b = (
                (prev_h > hhv5 or prev_h > p["PDH"] or prev_h > p["PMH"])
                and (prev_c < hhv5)
                and (prev_c < prev_o)
            )

            b_engulf = (
                buy_zone
                and (prev_c > prev_o)
                and (day_5m["Close"].iloc[i-2] < day_5m["Open"].iloc[i-2])
                and (prev_c >= day_5m["Open"].iloc[i-2])
            )
            s_engulf = (
                sell_zone
                and (prev_c < prev_o)
                and (day_5m["Close"].iloc[i-2] > day_5m["Open"].iloc[i-2])
                and (prev_c <= day_5m["Open"].iloc[i-2])
            )

            buy_ok = (
                p["TREND_BIAS"] >= 0
                and (h > prev_h)
                and (c > o)
                and (c > lwma)
                and vol_ok
                and (b_2b or b_engulf)
            )
            sell_ok = (
                p["TREND_BIAS"] <= 0
                and (l < prev_l)
                and (c < o)
                and (c < lwma)
                and vol_ok
                and (s_2b or s_engulf)
            )

            if buy_ok:
                in_pos = True
                pos_type = 1
                entry_p = c
                sl_p = l - 0.15 * atr_v
                tp_p = c + 2.0 * (c - sl_p)
                entry_idx = i
                entry_time_ny = cur_t_ny
                futu_signal_tag = "▲▲ 2B" if b_2b else "▲ CALL"
                be_trigger_p = 0.0
            elif sell_ok:
                in_pos = True
                pos_type = -1
                entry_p = c
                sl_p = h + 0.15 * atr_v
                tp_p = c - 2.0 * (sl_p - c)
                entry_idx = i
                entry_time_ny = cur_t_ny
                futu_signal_tag = "▼▼ 2B" if s_2b else "▼ PUT"
                be_trigger_p = 0.0

    return trades

# =====================================================================
# 4. 持久化账本存储
# =====================================================================
ALL_COLS = [
    "Date_MYT", "Signal", "Entry_MYT", "Entry_ET", "Exit_MYT", "Exit_ET",
    "Entry_Price", "Exit_Price", "SL", "TP", "PnL_Points", "Reason", "Result",
    "TREND_BIAS", "TOTAL_SCORE", "SBR_TOP", "SBR_BOT", "RBS_TOP", "RBS_BOT",
    "SBR2_TOP", "SBR2_BOT", "RBS2_TOP", "RBS2_BOT", "PDH", "PDL", "PMH", "PML"
]

def load_journal():
    if not os.path.exists(CSV_FILE):
        df_init = pd.DataFrame(columns=ALL_COLS)
        df_init.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
        return df_init
    df_read = pd.read_csv(CSV_FILE)
    for col in ALL_COLS:
        if col not in df_read.columns:
            df_read[col] = np.nan
    return df_read

def append_to_journal(date_str, params, trades):
    df_cur = load_journal()
    if not df_cur.empty and date_str in df_cur["Date_MYT"].astype(str).values:
        return False, "当天记录已存在"

    rows = []
    base_info = {
        "Date_MYT": date_str,
        "TREND_BIAS": params["TREND_BIAS"],
        "TOTAL_SCORE": params["TOTAL_SCORE"],
        "SBR_TOP": params["SBR_TOP"],
        "SBR_BOT": params["SBR_BOT"],
        "RBS_TOP": params["RBS_TOP"],
        "RBS_BOT": params["RBS_BOT"],
        "SBR2_TOP": params["SBR2_TOP"],
        "SBR2_BOT": params["SBR2_BOT"],
        "RBS2_TOP": params["RBS2_TOP"],
        "RBS2_BOT": params["RBS2_BOT"],
        "PDH": params["PDH"],
        "PDL": params["PDL"],
        "PMH": params["PMH"],
        "PML": params["PML"]
    }

    if trades:
        for t in trades:
            row_dict = dict(base_info)
            row_dict.update(t)
            rows.append(row_dict)
    else:
        empty_t = {
            "Signal": "NO_TRADE",
            "Entry_MYT": "-",
            "Entry_ET": "-",
            "Exit_MYT": "-
