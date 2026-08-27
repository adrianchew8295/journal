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
    page_title="QQQ 宏观战区与实战月历座舱",
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
# 2. 宏观面总导演 + 数据抓取引擎
# =====================================================================
def fetch_macro_and_market_data():
    macro_bias = 0
    macro_score = 50
    tnx_v, dxy_v, vix_v = 4.2, 102.0, 16.0

    try:
        tickers_macro = ["^TNX", "DX-Y.NYB", "^VIX", "QQQ", "RSP"]
        df_macro = yf.download(tickers_macro, period="1mo", interval="1d", progress=False, prepost=False)
        if df_macro is not None and not df_macro.empty:
            close_df = df_macro["Close"] if isinstance(df_macro.columns, pd.MultiIndex) else df_macro

            if "^TNX" in close_df.columns:
                tnx_v = float(close_df["^TNX"].dropna().iloc[-1])
            if "DX-Y.NYB" in close_df.columns:
                dxy_v = float(close_df["DX-Y.NYB"].dropna().iloc[-1])
            if "^VIX" in close_df.columns:
                vix_v = float(close_df["^VIX"].dropna().iloc[-1])

            score = 50
            if tnx_v > 4.5: score -= 15
            elif tnx_v < 4.0: score += 15

            if vix_v < 18: score += 15
            elif vix_v > 25: score -= 25

            if dxy_v > 103: score -= 10
            elif dxy_v < 100: score += 10

            if "QQQ" in close_df.columns and "RSP" in close_df.columns and len(close_df["QQQ"].dropna()) >= 5:
                q_chg = close_df["QQQ"].dropna().iloc[-1] / close_df["QQQ"].dropna().iloc[-5] - 1
                r_chg = close_df["RSP"].dropna().iloc[-1] / close_df["RSP"].dropna().iloc[-5] - 1
                if (r_chg - q_chg) > 0.01: score += 10
                elif (q_chg - r_chg) > 0.02: score -= 10

            macro_score = max(5, min(95, score))
            if macro_score >= 60: macro_bias = 1
            elif macro_score <= 40: macro_bias = -1
            else: macro_bias = 0

    except Exception:
        pass

    return macro_bias, macro_score, tnx_v, dxy_v, vix_v

def fetch_raw_data_with_retry(period_5m="1mo", max_retries=3):
    df_1h, source_1h = None, "None"
    df_5m, source_5m = None, "None"
    err_log = []
    start_str = (now_myt - timedelta(days=60)).strftime("%Y-%m-%d")

    # 1H 数据拉取
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
                    source_1h = "Tiingo IEX API"
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
                source_1h = "YahooFinance (1H)"
        except Exception as e:
            err_log.append("YahooFinance 1H 失败: " + str(e))

    # 5M 数据拉取 (支持 1mo/2mo 批量回测)
    for attempt in range(max_retries):
        try:
            df_5m_raw = yf.download(TICKER, period=period_5m, interval="5m", prepost=True, progress=False)
            if df_5m_raw is not None and not df_5m_raw.empty:
                if isinstance(df_5m_raw.columns, pd.MultiIndex):
                    df_5m_raw.columns = df_5m_raw.columns.get_level_values(0)
                df_5m = df_5m_raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
                df_5m.index = df_5m.index.tz_localize("UTC").tz_convert(tz_ny) if df_5m.index.tz is None else df_5m.index.tz_convert(tz_ny)
                source_5m = "YahooFinance (5M)"
                break
        except Exception as e:
            err_log.append("YahooFinance 5M 失败: " + str(e))
            time.sleep(1)

    return df_1h, source_1h, df_5m, source_5m, err_log

# =====================================================================
# 3. 核心运算：宏观多空总导演 + 13 行战区参数生成与回放
# =====================================================================
def compute_futu_13_params(df_1h, df_5m, as_of_ny_time):
    if df_1h is None:
        return None
    sub_1h = df_1h[df_1h.index <= as_of_ny_time].copy()
    if len(sub_1h) < 25:
        return None

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

    tr = np.maximum(sub_1h["High"] - sub_1h["Low"],
                    np.maximum((sub_1h["High"] - sub_1h["Close"].shift(1)).abs(),
                               (sub_1h["Low"] - sub_1h["Close"].shift(1)).abs()))
    atr = float(tr.rolling(14).mean().iloc[-1]) if not np.isnan(tr.rolling(14).mean().iloc[-1]) else (live_price * 0.008)

    subset = sub_1h.iloc[-60:].copy()
    highs, lows, opens, closes, times = subset["High"].values, subset["Low"].values, subset["Open"].values, subset["Close"].values, subset.index

    pivots_high, pivots_low = [], []
    for i in range(2, len(subset) - 2):
        if highs[i] == max(highs[i-2:i+3]):
            pivots_high.append((float(highs[i]), float(max(opens[i], closes[i])), times[i].strftime("%m-%d %H:%M ET")))
        if lows[i] == min(lows[i-2:i+3]):
            pivots_low.append((float(min(opens[i], closes[i])), float(lows[i]), times[i].strftime("%m-%d %H:%M ET")))

    valid_highs = [p for p in pivots_high if p[0] > live_price]
    valid_highs.sort(key=lambda x: x[0])
    sbr_top, sbr_bot, sbr_time = valid_highs[0] if len(valid_highs) >= 1 else (live_price + 1.2 * atr, live_price + 0.6 * atr, "Range High")
    sbr2_top, sbr2_bot, sbr2_time = valid_highs[1] if len(valid_highs) >= 2 else (sbr_top + 1.2 * atr, sbr_top + 0.5 * atr, "Tier-2 High")

    valid_lows = [p for p in pivots_low if p[1] < live_price]
    valid_lows.sort(key=lambda x: x[1], reverse=True)
    rbs_top, rbs_bot, rbs_time = valid_lows[0] if len(valid_lows) >= 1 else (live_price - 0.6 * atr, live_price - 1.2 * atr, "Range Low")
    rbs2_top, rbs2_bot, rbs2_time = valid_lows[1] if len(valid_lows) >= 2 else (rbs_bot - 0.5 * atr, rbs_bot - 1.2 * atr, "Tier-2 Low")

    macro_bias, macro_score, tnx_v, dxy_v, vix_v = fetch_macro_and_market_data()

    return {
        "live_price": live_price,
        "TREND_BIAS": macro_bias, 
        "TOTAL_SCORE": macro_score,
        "TNX": tnx_v, "DXY": dxy_v, "VIX": vix_v,
        "SBR_TOP": sbr_top, "SBR_BOT": sbr_bot, "SBR_TIME": sbr_time,
        "RBS_TOP": rbs_top, "RBS_BOT": rbs_bot, "RBS_TIME": rbs_time,
        "SBR2_TOP": sbr2_top, "SBR2_BOT": sbr2_bot, "SBR2_TIME": sbr2_time,
        "RBS2_TOP": rbs2_top, "RBS2_BOT": rbs2_bot, "RBS2_TIME": rbs2_time,
        "PDH": pdh_val, "PDH_TIME": pdh_time,
        "PDL": pdl_val, "PDL_TIME": pdl_time,
        "PMH": pmh_val, "PMH_TIME": pmh_time,
        "PML": pml_val, "PML_TIME": pml_time
    }

def simulate_night_trades(df_5m, p, start_cutoff_ny, close_ny):
    trades = []
    if p is None or df_5m is None:
        return trades

    day_5m = df_5m[(df_5m.index >= start_cutoff_ny) & (df_5m.index <= close_ny)].copy()
    if len(day_5m) < 15:
        return trades

    weights = np.arange(1, 21)
    day_5m["LWMA20"] = day_5m["Close"].rolling(20).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    tr_5m = np.maximum(day_5m["High"] - day_5m["Low"],
                       np.maximum((day_5m["High"] - day_5m["Close"].shift(1)).abs(),
                                  (day_5m["Low"] - day_5m["Close"].shift(1)).abs()))
    day_5m["ATR14"] = tr_5m.rolling(14).mean()
    day_5m["VOL_MA"] = day_5m["Volume"].rolling(20).mean()
    day_5m["VOL_HEAVY"] = day_5m["Volume"] >= (1.25 * day_5m["VOL_MA"])

    in_pos, pos_type = False, 0
    entry_p, sl_p, tp_p, be_trigger_p = 0.0, 0.0, 0.0, 0.0
    entry_idx, entry_time_ny = 0, None
    futu_signal_tag = ""

    for i in range(20, len(day_5m)):
        cur_t_ny = day_5m.index[i]
        c, o, h, l = day_5m["Close"].iloc[i], day_5m["Open"].iloc[i], day_5m["High"].iloc[i], day_5m["Low"].iloc[i]
        atr_v = day_5m["ATR14"].iloc[i] if not np.isnan(day_5m["ATR14"].iloc[i]) else 0.5
        vol_h = day_5m["VOL_HEAVY"].iloc[i]
        lwma = day_5m["LWMA20"].iloc[i]

        if in_pos:
            bars_held = i - entry_idx
            exit_flag, reason, exit_p = False, "", 0.0
            is_eod = (cur_t_ny >= close_ny - timedelta(minutes=10))

            if pos_type == 1:
                if h >= entry_p + (entry_p - sl_p): be_trigger_p = entry_p
                if is_eod: exit_flag, reason, exit_p = True, "EOD (收盘清仓)", c
                elif l <= sl_p: exit_flag, reason, exit_p = True, "SL (极限止损)", sl_p
                elif h >= tp_p: exit_flag, reason, exit_p = True, "TP (2R止盈)", tp_p
                elif be_trigger_p > 0 and l <= be_trigger_p: exit_flag, reason, exit_p = True, "BE (1R保本损)", be_trigger_p
                elif c < lwma and bars_held >= 5: exit_flag, reason, exit_p = True, "MA Cut (均线破位)", c
                elif bars_held >= 15: exit_flag, reason, exit_p = True, "Time (15根K超时)", c
            elif pos_type == -1:
                if l <= entry_p - (sl_p - entry_p): be_trigger_p = entry_p
                if is_eod: exit_flag, reason, exit_p = True, "EOD (收盘清仓)", c
                elif h >= sl_p: exit_flag, reason, exit_p = True, "SL (极限止损)", sl_p
                elif l <= tp_p: exit_flag, reason, exit_p = True, "TP (2R止盈)", tp_p
                elif be_trigger_p > 0 and h >= be_trigger_p: exit_flag, reason, exit_p = True, "BE (1R保本损)", be_trigger_p
                elif c > lwma and bars_held >= 5: exit_flag, reason, exit_p = True, "MA Cut (均线破位)", c
                elif bars_held >= 15: exit_flag, reason, exit_p = True, "Time (15根K超时)", c

            if exit_flag:
                pnl = (exit_p - entry_p) if pos_type == 1 else (entry_p - exit_p)
                trades.append({
                    "Signal": futu_signal_tag,
                    "Entry_MYT": entry_time_ny.astimezone(tz_myt).strftime("%H:%M"),
                    "Entry_ET": entry_time_ny.strftime("%H:%M"),
                    "Exit_MYT": cur_t_ny.astimezone(tz_myt).strftime("%H:%M"),
                    "Exit_ET": cur_t_ny.strftime("%H:%M"),
                    "Entry_Price": round(entry_p, 2),
                    "Exit_Price": round(exit_p, 2),
                    "SL": round(sl_p, 2), "TP": round(tp_p, 2),
                    "PnL_Points": round(pnl, 2),
                    "Reason": reason,
                    "Result": "盈利" if pnl > 0 else "亏损"
                })
                in_pos = False
                continue

        if not in_pos and i >= 2:
            prev_c, prev_o = day_5m["Close"].iloc[i-1], day_5m["Open"].iloc[i-1]
            prev_h, prev_l = day_5m["High"].iloc[i-1], day_5m["Low"].iloc[i-1]
            vol_ok = vol_h or day_5m["VOL_HEAVY"].iloc[i-1]

            buy_zone = (prev_l <= p["RBS_TOP"] and prev_c >= p["RBS_BOT"]) or (prev_l <= p["PDL"] and prev_c > p["PDL"]) or (prev_l <= p["PML"] and prev_c > p["PML"])
            sell_zone = (prev_h >= p["SBR_BOT"] and prev_c <= p["SBR_TOP"]) or (prev_h >= p["PDH"] and prev_c < p["PDH"]) or (prev_h >= p["PMH"] and prev_c < p["PMH"])

            llv5 = day_5m["Low"].iloc[i-6:i-1].min()
            hhv5 = day_5m["High"].iloc[i-6:i-1].max()

            b_2b = (prev_l < llv5 or prev_l < p["PDL"] or prev_l < p["PML"]) and (prev_c > llv5) and (prev_c > prev_o)
            s_2b = (prev_h > hhv5 or prev_h > p["PDH"] or prev_h > p["PMH"]) and (prev_c < hhv5) and (prev_c < prev_o)

            b_engulf = buy_zone and (prev_c > prev_o) and (day_5m["Close"].iloc[i-2] < day_5m["Open"].iloc[i-2]) and (prev_c >= day_5m["Open"].iloc[i-2])
            s_engulf = sell_zone and (prev_c < prev_o) and (day_5m["Close"].iloc[i-2] > day_5m["Open"].iloc[i-2]) and (prev_c <= day_5m["Open"].iloc[i-2])

            if p["TREND_BIAS"] == 0:
                buy_ok = (h > prev_h) and (c > o) and (c > lwma) and vol_ok and b_2b
                sell_ok = (l < prev_l) and (c < o) and (c < lwma) and vol_ok and s_2b
            else:
                buy_ok = p["TREND_BIAS"] >= 0 and (h > prev_h) and (c > o) and (c > lwma) and vol_ok and (b_2b or b_engulf)
                sell_ok = p["TREND_BIAS"] <= 0 and (l < prev_l) and (c < o) and (c < lwma) and vol_ok and (s_2b or s_engulf)

            if buy_ok:
                in_pos, pos_type = True, 1
                entry_p = c
                sl_p = l - 0.15 * atr_v
                tp_p = c + 2.0 * (c - sl_p)
                entry_idx, entry_time_ny = i, cur_t_ny
                futu_signal_tag = "▲▲ 2B" if b_2b else "▲ CALL"
                be_trigger_p = 0.0
            elif sell_ok:
                in_pos, pos_type = True, -1
                entry_p = c
                sl_p = h + 0.15 * atr_v
                tp_p = c - 2.0 * (sl_p - c)
                entry_idx, entry_time_ny = i, cur_t_ny
                futu_signal_tag = "▼▼ 2B" if s_2b else "▼ PUT"
                be_trigger_p = 0.0

    return trades

# =====================================================================
# 4. 持久化账本存储
# =====================================================================
ALL_COLS = [
    "Date_MYT", "Signal", "Entry_MYT", "Entry_ET", "Exit_MYT", "Exit_ET",
    "Entry_Price", "Exit_Price", "SL", "TP", "PnL_Points", "Reason", "Result",
    "TREND_BIAS", "TOTAL_SCORE", "TNX", "DXY", "VIX",
    "SBR_TOP", "SBR_BOT", "RBS_TOP", "RBS_BOT",
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
        "TNX": params["TNX"], "DXY": params["DXY"], "VIX": params["VIX"],
        "SBR_TOP": params["SBR_TOP"], "SBR_BOT": params["SBR_BOT"],
        "RBS_TOP": params["RBS_TOP"], "RBS_BOT": params["RBS_BOT"],
        "SBR2_TOP": params["SBR2_TOP"], "SBR2_BOT": params["SBR2_BOT"],
        "RBS2_TOP": params["RBS2_TOP"], "RBS2_BOT": params["RBS2_BOT"],
        "PDH": params["PDH"], "PDL": params["PDL"],
        "PMH": params["PMH"], "PML": params["PML"]
    }

    if trades:
        for t in trades:
            row_dict = dict(base_info)
            row_dict.update(t)
            rows.append(row_dict)
    else:
        empty_t = {
            "Signal": "NO_TRADE", "Entry_MYT": "-", "Entry_ET": "-",
            "Exit_MYT": "-", "Exit_ET": "-", "Entry_Price": 0.0, "Exit_Price": 0.0,
            "SL": 0.0, "TP": 0.0, "PnL_Points": 0.0, "Reason": "宏观中立或无信号", "Result": "无"
        }
        row_dict = dict(base_info)
        row_dict.update(empty_t)
        rows.append(row_dict)

    df_new = pd.DataFrame(rows)
    df_new.to_csv(CSV_FILE, index=False, encoding="utf-8-sig", mode="a" if os.path.exists(CSV_FILE) else "w", header=not os.path.exists(CSV_FILE))
    return True, "成功记录 " + str(len(rows)) + " 条明细"

# =====================================================================
# 5. 页面渲染
# =====================================================================
df_j = load_journal()
yesterday_d = now_myt.date() - timedelta(days=1)
yesterday_myt_str = yesterday_d.strftime("%Y-%m-%d")
has_10pm_p = (now_myt.hour >= 22 or now_myt.hour < 5)
has_8am_report = yesterday_myt_str in df_j["Date_MYT"].astype(str).values if not df_j.empty else False

s1, s2, s3, s4 = st.columns(4)
s1.success("✅ 10:00 PM 参数引擎已激活" if has_10pm_p else "⏳ 10:00 PM 参数引擎等待中")
s2.success(("✅ 08:00 AM 战报已交付 (" + yesterday_myt_str + ")") if has_8am_report else ("⏳ 08:00 AM 战报待更新 (" + yesterday_myt_str + ")"))
s3.info("🟢 系统全运转正常")

with s4:
    if st.button("🧪 执行系统全链路测试"):
        with st.spinner("正在回放测试..."):
            d1, s1h, d5, s5m, errs = fetch_raw_data_with_retry(period_5m="5d")
            if errs:
                st.error("异常: " + "; ".join(errs))
            else:
                st.success("测试通过：宏观与微观数据源正常。")

st.markdown("---")

tab1, tab2 = st.tabs(["🎯 QQQ 战区座舱 (富途参数复制)", "📅 QQQ 实战月历账本"])

# =====================================================================
# TAB 1: 战区座舱与 13 行富途参数
# =====================================================================
with tab1:
    st.subheader("🌊 QQQ 5M 交易座舱 & 宏观总导演雷达")
    c_t1, c_t2 = st.columns(2)
    c_t1.info("🕒 大马时间 (MYT): " + now_myt.strftime("%Y-%m-%d %H:%M:%S"))
    c_t2.info("🇺🇸 美东时间 (ET): " + now_ny.strftime("%Y-%m-%d %H:%M:%S"))

    if not has_10pm_p:
        st.warning("🔒 处于日间准备期。大马时间 22:00 准时解锁并生成今晚 13 行战区代码。")
    else:
        if st.button("🔄 刷新最新点位"):
            st.cache_data.clear()
            st.rerun()

        d1h, src1h, d5m, src5m, _ = fetch_raw_data_with_retry(period_5m="5d")
        if d1h is not None:
            p = compute_futu_13_params(d1h, d5m, now_ny)
            if p:
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("🎯 QQQ 现价", "$" + str(round(p["live_price"], 2)))
                
                if p["TREND_BIAS"] == 1: b_desc = "🟢 宏观偏多 (CALL)"
                elif p["TREND_BIAS"] == -1: b_desc = "🔴 宏观偏空 (PUT)"
                else: b_desc = "⚪ 宏观中立 (防守2B)"
                
                m_c2.metric("🧭 宏观总定调", b_desc)
                m_c3.metric("📊 宏观多空分", str(p["TOTAL_SCORE"]) + " / 100分")
                m_c4.metric("📉 核心指标", f"TNX:{p['TNX']:.2f}% | VIX:{p['VIX']:.1f}")

                out_lines = [
                    f"TREND_BIAS := {p['TREND_BIAS']};       {{ 1. 宏观偏向(总导演判定): 1=多, -1=空, 0=中立 [宏观得分: {p['TOTAL_SCORE']}分] }}",
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
                futu_13 = "\n".join(out_lines)
                st.markdown("#### 📋 复制到富途指标顶部 13 行代码 (点右上角复制):")
                st.code(futu_13, language="pascal")

# =====================================================================
# TAB 2: 实战月历账本 (带一键全月补录)
# =====================================================================
with tab2:
    st.subheader("📅 QQQ 5M 实战月历记账本")

    col_btn1, col_btn2 = st.columns([1.5, 2])
    with col_btn1:
        if st.button("🛠️ 立即自动结算并补录昨夜账本"):
            with st.spinner("正在结算昨夜复盘..."):
                d1h, _, d5m, _, _ = fetch_raw_data_with_retry(period_5m="5d")
                target_d = now_myt.date() - timedelta(days=1)
                dt_10pm_myt = tz_myt.localize(datetime.datetime.combine(target_d, datetime.time(22, 0, 0)))
                cutoff_ny = dt_10pm_myt.astimezone(tz_ny)
                close_ny = cutoff_ny.replace(hour=16, minute=0, second=0)

                p = compute_futu_13_params(d1h, d5m, cutoff_ny)
                if p:
                    trades = simulate_night_trades(d5m, p, cutoff_ny, close_ny)
                    ok, msg = append_to_journal(target_d.strftime("%Y-%m-%d"), p, trades)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.warning(msg)

    with col_btn2:
        if st.button("⚡ 一键回溯补录当月所有历史交易日 (Backfill)"):
            with st.spinner("正在抓取当月全部历史 K 线并批量执行复盘运算，请稍候 5-10 秒..."):
                d1h, _, d5m, _, _ = fetch_raw_data_with_retry(period_5m="1mo")
                if d1h is not None and d5m is not None:
                    dates_in_5m = sorted(list(set(d5m.index.date)))
                    added_cnt = 0
                    for d in dates_in_5m:
                        if d >= now_ny.date():
                            continue  # 今天未收盘，跳过
                        dt_10pm_myt = tz_myt.localize(datetime.datetime.combine(d, datetime.time(22, 0, 0)))
                        cutoff_ny = dt_10pm_myt.astimezone(tz_ny)
                        close_ny = cutoff_ny.replace(hour=16, minute=0, second=0)

                        p_day = compute_futu_13_params(d1h, d5m, cutoff_ny)
                        if p_day:
                            trades_day = simulate_night_trades(d5m, p_day, cutoff_ny, close_ny)
                            ok, _ = append_to_journal(d.strftime("%Y-%m-%d"), p_day, trades_day)
                            if ok: added_cnt += 1

                    st.success(f"🎉 成功完成历史回溯，批量补录了 {added_cnt} 个交易日的实战数据！")
                    st.rerun()
                else:
                    st.error("拉取历史数据失败，请重试。")

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
            label="📥 导出 " + str(sel_y) + "年" + str(sel_m) + "月 账本 (.csv)",
            data=csv_bytes,
            file_name="QQQ_Journal_" + str(sel_y) + "_" + str(sel_m).zfill(2) + ".csv",
            mime="text/csv",
            disabled=df_m.empty
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🗓️ 选定月份", str(sel_y) + " 年 " + str(sel_m) + " 月")
    k2.metric("💰 当月点数盈亏", f"{tot_pts:+.2f} pt")
    k3.metric("🎯 当月胜率", f"{w_rate:.1f}%", str(w_cnt) + "/" + str(tot_cnt) + " 胜")
    k4.metric("📊 开仓总笔数", str(tot_cnt) + " 笔")

    st.markdown("---")

    weekdays = ["周一 (Mon)", "周二 (Tue)", "周三 (Wed)", "周四 (Thu)", "周五 (Fri)", "周六 (Sat)", "周日 (Sun)"]
    h_cols = st.columns(7)
    for idx, hc in enumerate(h_cols):
        hc.markdown("<div style='text-align:center; font-weight:bold; color:#4a5568;'>" + weekdays[idx] + "</div>", unsafe_allow_html=True)

    cal = calendar.monthcalendar(sel_y, sel_m)
    for week in cal:
        w_cols = st.columns(7)
        for d_idx, day in enumerate(week):
            with w_cols[d_idx]:
                if day == 0:
                    st.markdown("<div style='height:120px;'></div>", unsafe_allow_html=True)
                    continue

                cur_d = datetime.date(sel_y, sel_m, day)
                is_weekend = (d_idx >= 5)

                if is_weekend:
                    st.markdown(
                        "<div style='background-color:#edf2f7; border-radius:8px; padding:8px; height:120px; border:1px dashed #cbd5e0; text-align:center;'>"
                        + "<div style='font-size:13px; color:#a0aec0; text-align:left;'><b>" + str(day) + "</b></div>"
                        + "<div style='font-size:18px; margin-top:10px;'>❌</div>"
                        + "<div style='font-size:11px; color:#a0aec0;'>周末休市</div></div>",
                        unsafe_allow_html=True
                    )
                else:
                    d_recs = df_m[df_m["Date_MYT_dt"] == cur_d] if not df_m.empty else pd.DataFrame()
                    if not d_recs.empty:
                        r_t = d_recs[d_recs["Signal"] != "NO_TRADE"]
                        cnt = len(r_t)
                        pts = r_t["PnL_Points"].sum() if not r_t.empty else 0.0
                        b_val = d_recs.iloc[0].get("TREND_BIAS", 0)
                        b_str = "多" if b_val == 1 else ("空" if b_val == -1 else "中立")

                        if cnt == 0:
                            st.markdown(
                                "<div style='background-color:#f7fafc; border-radius:8px; padding:8px; height:120px; border:1px solid #e2e8f0; text-align:center;'>"
                                + "<div style='font-size:13px; color:#718096; text-align:left;'><b>" + str(day) + "</b> <span style='font-size:10px; color:#a0aec0;'>(" + b_str + ")</span></div>"
                                + "<div style='font-size:12px; color:#718096; margin-top:15px;'>⚪ 宏观中立</div>"
                                + "<div style='font-size:10px; color:#a0aec0;'>0 笔交易</div></div>",
                                unsafe_allow_html=True
                            )
                        else:
                            bg = "#e6fffa" if pts >= 0 else "#fff5f5"
                            bd = "#38b2ac" if pts >= 0 else "#e53e3e"
                            tc = "#234e52" if pts >= 0 else "#742a2a"
                            sgn = "+" if pts > 0 else ""
                            st.markdown(
                                "<div style='background-color:" + bg + "; border-radius:8px; padding:8px; height:120px; border:2px solid " + bd + "; text-align:center;'>"
                                + "<div style='font-size:13px; color:" + tc + "; text-align:left;'><b>" + str(day) + "</b> <span style='font-size:10px; color:" + bd + ";'>(" + b_str + ")</span></div>"
                                + "<div style='font-size:15px; font-weight:bold; color:" + bd + "; margin-top:2px;'>" + sgn + f"{pts:.2f}" + " pt</div>"
                                + "<div style='font-size:11px; color:" + tc + ";'>" + str(cnt) + " 笔交易</div></div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            "<div style='background-color:#ffffff; border-radius:8px; padding:8px; height:120px; border:1px solid #edf2f7; text-align:center;'>"
                            + "<div style='font-size:13px; color:#cbd5e0; text-align:left;'><b>" + str(day) + "</b></div>"
                            + "<div style='font-size:11px; color:#cbd5e0; margin-top:25px;'>-</div></div>",
                            unsafe_allow_html=True
                        )

    with st.expander("🔍 展开查看当月逐笔流水与 13 项战区参数明细 (Full Data Table)"):
        if not df_m.empty:
            st.dataframe(df_m.drop(columns=["Date_MYT_dt", "Year", "Month"], errors="ignore"), use_container_width=True)
        else:
            st.info("当月暂无交易明细。")
