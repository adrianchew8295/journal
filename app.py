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
    page_title="QQQ 黄金2小时极简战区座舱",
    page_icon="🎯",
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
# 2. QQQ 数据抓取引擎
# =====================================================================
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

    # 5M 数据拉取
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
# 3. 核心运算：QQQ 三灯信号 + 黄金 2 小时实战回测
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
    sub_1h["EMA20_slope"] = sub_1h["EMA20"].diff()

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

    ema_now = sub_1h["EMA20"].iloc[-1]
    slope_now = sub_1h["EMA20_slope"].iloc[-1]

    if live_price > ema_now and slope_now > 0:
        trend_bias = 1
        bias_desc = "🟢 绿灯 (做多为主)"
    elif live_price < ema_now and slope_now < 0:
        trend_bias = -1
        bias_desc = "🔴 红灯 (做空为主)"
    else:
        trend_bias = 0
        bias_desc = "🟡 黄灯 (震荡防守·仅做2B)"

    return {
        "live_price": live_price,
        "TREND_BIAS": trend_bias,
        "BIAS_DESC": bias_desc,
        "EMA20_1H": round(ema_now, 2),
        "ATR_1H": round(atr, 2),
        "SBR_TOP": sbr_top, "SBR_BOT": sbr_bot, "SBR_TIME": sbr_time,
        "RBS_TOP": rbs_top, "RBS_BOT": rbs_bot, "RBS_TIME": rbs_time,
        "SBR2_TOP": sbr2_top, "SBR2_BOT": sbr2_bot, "SBR2_TIME": sbr2_time,
        "RBS2_TOP": rbs2_top, "RBS2_BOT": rbs2_bot, "RBS2_TIME": rbs2_time,
        "PDH": pdh_val, "PDH_TIME": pdh_time,
        "PDL": pdl_val, "PDL_TIME": pdl_time,
        "PMH": pmh_val, "PMH_TIME": pmh_time,
        "PML": pml_val, "PML_TIME": pml_time
    }

def simulate_night_trades(df_5m, p, start_cutoff_ny, window_end_ny):
    """
    严格锁定在宿主纪律窗口：大马时间 22:00 至 24:00（美东 10:00 至 12:00）。
    单日最多开 1 笔，到 24:00 强制时间清仓，绝不熬夜。
    """
    trades = []
    if p is None or df_5m is None:
        return trades

    # 截取黄金 2 小时 K 线
    day_5m = df_5m[(df_5m.index >= start_cutoff_ny - timedelta(minutes=100)) & (df_5m.index <= window_end_ny)].copy()
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
    daily_trade_count = 0

    # 寻找入场点起始位置（22:00 MYT / 10:00 ET）
    start_idx = 0
    for idx_i, t_idx in enumerate(day_5m.index):
        if t_idx >= start_cutoff_ny:
            start_idx = max(20, idx_i)
            break

    for i in range(start_idx, len(day_5m)):
        cur_t_ny = day_5m.index[i]
        c, o, h, l = day_5m["Close"].iloc[i], day_5m["Open"].iloc[i], day_5m["High"].iloc[i], day_5m["Low"].iloc[i]
        atr_v = day_5m["ATR14"].iloc[i] if not np.isnan(day_5m["ATR14"].iloc[i]) else 0.8
        vol_h = bool(day_5m["VOL_HEAVY"].iloc[i])
        lwma = day_5m["LWMA20"].iloc[i]

        # 检查是否到达 24:00 MYT 纪律清仓点
        is_window_close = (cur_t_ny >= window_end_ny - timedelta(minutes=5))

        if in_pos:
            exit_flag, reason, exit_p = False, "", 0.0

            if pos_type == 1:
                if h >= entry_p + (entry_p - sl_p): be_trigger_p = entry_p
                if is_window_close: exit_flag, reason, exit_p = True, "24:00 纪律清仓", c
                elif l <= sl_p: exit_flag, reason, exit_p = True, "SL (止损)", sl_p
                elif h >= tp_p: exit_flag, reason, exit_p = True, "TP (2R止盈)", tp_p
                elif be_trigger_p > 0 and l <= be_trigger_p: exit_flag, reason, exit_p = True, "BE (1R保本损)", be_trigger_p
            elif pos_type == -1:
                if l <= entry_p - (sl_p - entry_p): be_trigger_p = entry_p
                if is_window_close: exit_flag, reason, exit_p = True, "24:00 纪律清仓", c
                elif h >= sl_p: exit_flag, reason, exit_p = True, "SL (止损)", sl_p
                elif l <= tp_p: exit_flag, reason, exit_p = True, "TP (2R止盈)", tp_p
                elif be_trigger_p > 0 and h >= be_trigger_p: exit_flag, reason, exit_p = True, "BE (1R保本损)", be_trigger_p

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
                    "Result": "盈利" if pnl > 0 else ("保本" if pnl == 0 else "亏损")
                })
                in_pos = False
                daily_trade_count += 1
                break  # 单日做满 1 笔立即锁定收工

        # 未持仓且未开过单，并在 23:45 前寻找入场
        if not in_pos and daily_trade_count == 0 and cur_t_ny < (window_end_ny - timedelta(minutes=15)):
            prev_c, prev_o = day_5m["Close"].iloc[i-1], day_5m["Open"].iloc[i-1]
            prev_h, prev_l = day_5m["High"].iloc[i-1], day_5m["Low"].iloc[i-1]
            vol_ok = vol_h or bool(day_5m["VOL_HEAVY"].iloc[i-1])

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
                buy_ok = (p["TREND_BIAS"] == 1) and (h > prev_h) and (c > o) and (c > lwma) and vol_ok and (b_2b or b_engulf)
                sell_ok = (p["TREND_BIAS"] == -1) and (l < prev_l) and (c < o) and (c < lwma) and vol_ok and (s_2b or s_engulf)

            stop_buffer = max(0.5 * atr_v, 1.2)

            if buy_ok:
                in_pos, pos_type = True, 1
                entry_p = c
                sl_p = min(prev_l, l) - stop_buffer
                tp_p = c + 2.0 * (c - sl_p)
                entry_idx, entry_time_ny = i, cur_t_ny
                futu_signal_tag = "▲▲ 2B" if b_2b else "▲ CALL"
                be_trigger_p = 0.0
            elif sell_ok:
                in_pos, pos_type = True, -1
                entry_p = c
                sl_p = max(prev_h, h) + stop_buffer
                tp_p = c - 2.0 * (sl_p - c)
                entry_idx, entry_time_ny = i, cur_t_ny
                futu_signal_tag = "▼▼ 2B" if s_2b else "▼ PUT"
                be_trigger_p = 0.0

    return trades

# =====================================================================
# 4. 账本存储
# =====================================================================
RECORD_COLUMNS = [
    "Date_MYT", "TREND_BIAS", "EMA20_1H", "ATR_1H",
    "SBR_TOP", "SBR_BOT", "RBS_TOP", "RBS_BOT",
    "SBR2_TOP", "SBR2_BOT", "RBS2_TOP", "RBS2_BOT",
    "PDH", "PDL", "PMH", "PML",
    "Signal", "Entry_MYT", "Entry_ET", "Exit_MYT", "Exit_ET",
    "Entry_Price", "Exit_Price", "SL", "TP", "PnL_Points", "Reason", "Result"
]

def load_journal():
    if not os.path.exists(CSV_FILE):
        df_init = pd.DataFrame(columns=RECORD_COLUMNS)
        df_init.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
        return df_init
    df_read = pd.read_csv(CSV_FILE)
    for col in RECORD_COLUMNS:
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
        "EMA20_1H": params.get("EMA20_1H", 0.0),
        "ATR_1H": params.get("ATR_1H", 0.0),
        "SBR_TOP": params["SBR_TOP"], "SBR_BOT": params["SBR_BOT"],
        "RBS_TOP": params["RBS_TOP"], "RBS_BOT": params["RBS_BOT"],
        "SBR2_TOP": params["SBR2_TOP"], "SBR2_BOT": params["SBR2_BOT"],
        "RBS2_TOP": params["RBS2_TOP"], "RBS2_BOT": params["RBS2_BOT"],
        "PDH": params["PDH"], "PDL": params["PDL"],
        "PMH": params["PMH"], "PML": params["PML"]
    }

    if trades:
        for t in trades:
            r = dict(base_info)
            r.update(t)
            rows.append(r)
    else:
        empty_t = {
            "Signal": "NO_TRADE", "Entry_MYT": "-", "Entry_ET": "-",
            "Exit_MYT": "-", "Exit_ET": "-", "Entry_Price": 0.0, "Exit_Price": 0.0,
            "SL": 0.0, "TP": 0.0, "PnL_Points": 0.0, "Reason": "窗口期无高胜率信号", "Result": "无"
        }
        r = dict(base_info)
        r.update(empty_t)
        rows.append(r)

    df_new = pd.DataFrame(rows)[RECORD_COLUMNS]
    df_new.to_csv(CSV_FILE, index=False, encoding="utf-8-sig", mode="a" if os.path.exists(CSV_FILE) else "w", header=not os.path.exists(CSV_FILE))
    return True, f"成功记录 {len(rows)} 条明细"

# =====================================================================
# 5. 页面渲染
# =====================================================================
df_j = load_journal()
yesterday_d = now_myt.date() - timedelta(days=1)
yesterday_myt_str = yesterday_d.strftime("%Y-%m-%d")
has_10pm_p = (now_myt.hour >= 22 or now_myt.hour < 5)
has_8am_report = yesterday_myt_str in df_j["Date_MYT"].astype(str).values if not df_j.empty else False

s1, s2, s3, s4 = st.columns(4)
s1.success("✅ 10:00 PM 参数引擎已就绪" if has_10pm_p else "⏳ 10:00 PM 参数引擎等待中")
s2.success(f"✅ 战报已交付 ({yesterday_myt_str})" if has_8am_report else f"⏳ 战报待更新 ({yesterday_myt_str})")
s3.info("🎯 纪律窗口：22:00 - 24:00 (MYT)")

with s4:
    if st.button("🧪 执行系统全链路测试"):
        with st.spinner("正在自检..."):
            d1, s1h, d5, s5m, errs = fetch_raw_data_with_retry(period_5m="5d")
            if errs:
                st.error("异常: " + "; ".join(errs))
            else:
                st.success("自检通过：QQQ 数据接口正常。")

st.markdown("---")

tab1, tab2 = st.tabs(["🎯 QQQ 战区座舱 (富途参数复制)", "📅 QQQ 黄金2小时实战月历"])

# =====================================================================
# TAB 1: 战区座舱与 13 行富途参数
# =====================================================================
with tab1:
    st.subheader("🎯 QQQ 5M 交易座舱 (黄金 2 小时窗口)")
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
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🎯 QQQ 现价", f"${p['live_price']:.2f}")
                m2.metric("🚦 三灯信号定调", p["BIAS_DESC"])
                m3.metric("📈 1H EMA20 均线", f"${p['EMA20_1H']:.2f}")
                m4.metric("📊 1H ATR 波动", f"${p['ATR_1H']:.2f}")

                out_lines = [
                    f"TREND_BIAS := {p['TREND_BIAS']};       {{ 1. QQQ三灯判定: 1=绿灯做多, -1=红灯做空, 0=黄灯震荡防守 }}",
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

# =====================================================================
# TAB 2: 实战月历账本
# =====================================================================
with tab2:
    st.subheader("📅 QQQ 黄金 2 小时 (22:00 - 24:00 MYT) 实战月历")

    col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1.5])
    with col_btn1:
        if st.button("🛠️ 结算昨夜 22:00-24:00 账本"):
            with st.spinner("正在结算昨夜 2 小时复盘..."):
                d1h, _, d5m, _, _ = fetch_raw_data_with_retry(period_5m="5d")
                target_d = now_myt.date() - timedelta(days=1)
                dt_10pm_myt = tz_myt.localize(datetime.datetime.combine(target_d, datetime.time(22, 0, 0)))
                cutoff_ny = dt_10pm_myt.astimezone(tz_ny)
                window_end_ny = cutoff_ny + timedelta(hours=2)

                p = compute_futu_13_params(d1h, d5m, cutoff_ny)
                if p:
                    trades = simulate_night_trades(d5m, p, cutoff_ny, window_end_ny)
                    ok, msg = append_to_journal(target_d.strftime("%Y-%m-%d"), p, trades)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.warning(msg)

    with col_btn2:
        if st.button("⚡ 一键回溯补录当月所有历史交易日 (Backfill)"):
            with st.spinner("正在严格按 22:00-24:00 黄金时段回溯运算，请稍候 5-10 秒..."):
                d1h, _, d5m, _, _ = fetch_raw_data_with_retry(period_5m="1mo")
                if d1h is not None and d5m is not None:
                    dates_in_5m = sorted(list(set(d5m.index.date)))
                    added_cnt = 0
                    for d in dates_in_5m:
                        if d >= now_ny.date():
                            continue
                        dt_10pm_myt = tz_myt.localize(datetime.datetime.combine(d, datetime.time(22, 0, 0)))
                        cutoff_ny = dt_10pm_myt.astimezone(tz_ny)
                        window_end_ny = cutoff_ny + timedelta(hours=2)

                        p_day = compute_futu_13_params(d1h, d5m, cutoff_ny)
                        if p_day:
                            trades_day = simulate_night_trades(d5m, p_day, cutoff_ny, window_end_ny)
                            ok, _ = append_to_journal(d.strftime("%Y-%m-%d"), p_day, trades_day)
                            if ok: added_cnt += 1

                    st.success(f"🎉 黄金 2 小时回溯完成，新增 {added_cnt} 个交易日记录！")
                    st.rerun()

    with col_btn3:
        if st.button("🗑️ 清空历史账本重新生成"):
            if os.path.exists(CSV_FILE):
                os.remove(CSV_FILE)
                st.success("账本已重置！")
                st.rerun()

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
            label=f"📥 导出 {sel_y}年{sel_m}月 黄金时段账本 (.csv)",
            data=csv_bytes,
            file_name=f"QQQ_Golden_Journal_{sel_y}_{str(sel_m).zfill(2)}.csv",
            mime="text/csv",
            disabled=df_m.empty
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🗓️ 选定月份", f"{sel_y} 年 {sel_m} 月")
    k2.metric("💰 黄金窗口盈亏", f"{tot_pts:+.2f} pt")
    k3.metric("🎯 黄金窗口胜率", f"{w_rate:.1f}%", f"{w_cnt}/{tot_cnt} 胜")
    k4.metric("📊 开仓总笔数", f"{tot_cnt} 笔 (每晚限1笔)")

    st.markdown("---")

    weekdays = ["周一 (Mon)", "周二 (Tue)", "周三 (Wed)", "周四 (Thu)", "周五 (Fri)", "周六 (Sat)", "周日 (Sun)"]
    h_cols = st.columns(7)
    for idx, hc in enumerate(h_cols):
        hc.markdown(f"<div style='text-align:center; font-weight:bold; color:#4a5568;'>{weekdays[idx]}</div>", unsafe_allow_html=True)

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
                        f"<div style='background-color:#edf2f7; border-radius:8px; padding:8px; height:120px; border:1px dashed #cbd5e0; text-align:center;'>"
                        f"<div style='font-size:13px; color:#a0aec0; text-align:left;'><b>{day}</b></div>"
                        f"<div style='font-size:18px; margin-top:10px;'>❌</div>"
                        f"<div style='font-size:11px; color:#a0aec0;'>周末休市</div></div>",
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
                                f"<div style='background-color:#f7fafc; border-radius:8px; padding:8px; height:120px; border:1px solid #e2e8f0; text-align:center;'>"
                                f"<div style='font-size:13px; color:#718096; text-align:left;'><b>{day}</b> <span style='font-size:10px; color:#a0aec0;'>({b_str})</span></div>"
                                f"<div style='font-size:12px; color:#718096; margin-top:15px;'>⚪ 无高胜率机会</div>"
                                f"<div style='font-size:10px; color:#a0aec0;'>空仓休战</div></div>",
                                unsafe_allow_html=True
                            )
                        else:
                            bg = "#e6fffa" if pts >= 0 else "#fff5f5"
                            bd = "#38b2ac" if pts >= 0 else "#e53e3e"
                            tc = "#234e52" if pts >= 0 else "#742a2a"
                            sgn = "+" if pts > 0 else ""
                            st.markdown(
                                f"<div style='background-color:{bg}; border-radius:8px; padding:8px; height:120px; border:2px solid {bd}; text-align:center;'>"
                                f"<div style='font-size:13px; color:{tc}; text-align:left;'><b>{day}</b> <span style='font-size:10px; color:{bd};'>({b_str})</span></div>"
                                f"<div style='font-size:15px; font-weight:bold; color:{bd}; margin-top:2px;'>{sgn}{pts:.2f} pt</div>"
                                f"<div style='font-size:11px; color:{tc};'>{cnt} 笔交易 (22:00-24:00)</div></div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            f"<div style='background-color:#ffffff; border-radius:8px; padding:8px; height:120px; border:1px solid #edf2f7; text-align:center;'>"
                            f"<div style='font-size:13px; color:#cbd5e0; text-align:left;'><b>{day}</b></div>"
                            f"<div style='font-size:11px; color:#cbd5e0; margin-top:25px;'>-</div></div>",
                            unsafe_allow_html=True
                        )

    with st.expander("🔍 展开查看黄金时段流水明细 (Full Data Table)"):
        if not df_m.empty:
            st.dataframe(df_m.drop(columns=["Date_MYT_dt", "Year", "Month"], errors="ignore"), use_container_width=True)
        else:
            st.info("当月暂无交易明细。")
