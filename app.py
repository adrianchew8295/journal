import io
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import yfinance as yf

# =====================================================================
# 1. 核心凭证与页面配置 (Title: 《《癸水(QQQ)》》)
# =====================================================================
TIINGO_TOKEN = "bcffe3a5cf7eeef085e405cfa4a3e5691b976217"

st.set_page_config(
    page_title="《《癸水(QQQ)》》",
    page_icon="🌊",
    layout="wide"
)

TICKER_QQQ = "QQQ"
BIG_SEVEN = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
LEADERS = ["MU", "TSM", "AMD", "AVGO", "QCOM", "ARM", "ASML", "PLTR", "NFLX", "INTC"]
ALL_TICKERS = [TICKER_QQQ] + BIG_SEVEN + LEADERS

# =====================================================================
# 2. 人性化时间引擎 (大马 MYT & 美东 ET)
# =====================================================================
tz_myt = pytz.timezone("Asia/Kuala_Lumpur")
tz_ny = pytz.timezone("America/New_York")

now_myt = datetime.datetime.now(tz_myt)
now_ny = datetime.datetime.now(tz_ny)

target_open_ny = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
if now_ny >= target_open_ny and now_ny.hour >= 16:
    target_open_ny += timedelta(days=1)
while target_open_ny.weekday() >= 5:
    target_open_ny += timedelta(days=1)

target_open_myt = target_open_ny.astimezone(tz_myt)
time_to_open = target_open_myt - now_myt

c_t1, c_t2, c_t3 = st.columns([1.5, 1.5, 2])
c_t1.info(f"🕒 **大马时间 (MYT):** {now_myt.strftime('%Y-%m-%d %H:%M:%S')}")
c_t2.info(f"🇺🇸 **美东时间 (ET):** {now_ny.strftime('%Y-%m-%d %H:%M:%S')}")

if 0 <= time_to_open.total_seconds() <= 86400:
    hours, remainder = divmod(int(time_to_open.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    c_t3.warning(f"⏳ **距离今晚美股开盘:** {hours}小时 {minutes}分 {seconds}秒")
else:
    c_t3.success("🟢 **美股交易中 / 盘后复盘阶段**")

# =====================================================================
# 3. 双模数据抓取引擎 (Tiingo 优先，429 自动无缝切换 yfinance)
# =====================================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_complete_data_audited(ticker, token):
    df_1h = None
    source_1h = "None"
    start_date = (datetime.datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
    
    # 3.1 优先请求 Tiingo 1H (IEX API)
    url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start_date}&resampleFreq=1hour&token={token}&columns=open,high,low,close,volume"
    headers = {'Content-Type': 'application/json'}
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and len(data) >= 30:
                df_t = pd.DataFrame(data)
                df_t['date'] = pd.to_datetime(df_t['date'])
                df_t.set_index('date', inplace=True)
                df_t.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                df_1h = df_t[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
                df_1h.index = df_1h.index.tz_localize("UTC").tz_convert("America/New_York") if df_1h.index.tz is None else df_1h.index.tz_convert("America/New_York")
                source_1h = "Tiingo (IEX 1H API)"
    except Exception:
        pass

    # 3.2 兜底请求 yfinance 1H
    if df_1h is None:
        try:
            df_yf = yf.download(ticker, period="1mo", interval="1h", prepost=True, progress=False)
            if df_yf is not None and not df_yf.empty:
                if isinstance(df_yf.columns, pd.MultiIndex):
                    df_yf.columns = df_yf.columns.get_level_values(0)
                df_1h = df_yf[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
                df_1h.index = df_1h.index.tz_localize("UTC").tz_convert("America/New_York") if df_1h.index.tz is None else df_1h.index.tz_convert("America/New_York")
                source_1h = "YahooFinance (1H prepost)"
        except Exception:
            pass

    # 3.3 请求 yfinance 5M 实时盘前
    df_5m = None
    source_5m = "None"
    try:
        df_5m_raw = yf.download(ticker, period="5d", interval="5m", prepost=True, progress=False)
        if df_5m_raw is not None and not df_5m_raw.empty:
            if isinstance(df_5m_raw.columns, pd.MultiIndex):
                df_5m_raw.columns = df_5m_raw.columns.get_level_values(0)
            df_5m = df_5m_raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
            df_5m.index = df_5m.index.tz_localize("UTC").tz_convert("America/New_York") if df_5m.index.tz is None else df_5m.index.tz_convert("America/New_York")
            source_5m = "YahooFinance (Live 5M Pre-market)"
    except Exception:
        pass

    return df_1h, source_1h, df_5m, source_5m

# =====================================================================
# 4. Primary + Secondary 双梯队战区与客观极值量化算法
# =====================================================================
def calculate_audited_levels(df_1h, source_1h, df_5m, source_5m, ticker):
    if df_1h is None or len(df_1h) < 25:
        return None
    
    today_ny = datetime.datetime.now(tz_ny).date()
    
    # 昨日 RTH 极值 (09:30 - 16:00 ET)
    df_rth = df_1h[(df_1h.index.hour > 9) | ((df_1h.index.hour == 9) & (df_1h.index.minute >= 30))]
    df_rth = df_rth[df_rth.index.hour < 16]
    past_dates = sorted(list(set(df_rth.index.date)))
    past_dates = [d for d in past_dates if d < today_ny]
    
    if past_dates:
        prev_df = df_rth[df_rth.index.date == past_dates[-1]]
        pdh_idx, pdl_idx = prev_df['High'].idxmax(), prev_df['Low'].idxmin()
        pdh_val, pdl_val = float(prev_df.loc[pdh_idx, 'High']), float(prev_df.loc[pdl_idx, 'Low'])
        pdh_time_str, pdl_time_str = pdh_idx.strftime("%Y-%m-%d %H:%M ET"), pdl_idx.strftime("%Y-%m-%d %H:%M ET")
    else:
        pdh_val, pdl_val = float(df_1h['High'].iloc[-10:].max()), float(df_1h['Low'].iloc[-10:].min())
        pdh_time_str, pdl_time_str = "Prior Session", "Prior Session"

    # 今日盘前极值 (04:00 - 09:30 ET)
    if df_5m is not None:
        today_pm = df_5m[(df_5m.index.date == today_ny) & (df_5m.index.hour >= 4) & ((df_5m.index.hour < 9) | ((df_5m.index.hour == 9) & (df_5m.index.minute < 30)))]
        if not today_pm.empty:
            pmh_idx, pml_idx = today_pm['High'].idxmax(), today_pm['Low'].idxmin()
            pmh_val, pml_val = float(today_pm.loc[pmh_idx, 'High']), float(today_pm.loc[pml_idx, 'Low'])
            pmh_time_str, pml_time_str = pmh_idx.strftime("%Y-%m-%d %H:%M ET"), pml_idx.strftime("%Y-%m-%d %H:%M ET")
            live_price = float(today_pm['Close'].iloc[-1])
        else:
            pmh_idx, pml_idx = df_5m['High'].iloc[-12:].idxmax(), df_5m['Low'].iloc[-12:].idxmin()
            pmh_val, pml_val = float(df_5m.loc[pmh_idx, 'High']), float(df_5m.loc[pml_idx, 'Low'])
            pmh_time_str, pml_time_str = pmh_idx.strftime("%Y-%m-%d %H:%M ET"), pml_idx.strftime("%Y-%m-%d %H:%M ET")
            live_price = float(df_5m['Close'].iloc[-1])
    else:
        pmh_val, pml_val = float(df_1h['High'].iloc[-4:].max()), float(df_1h['Low'].iloc[-4:].min())
        pmh_time_str, pml_time_str = "Recent 1H", "Recent 1H"
        live_price = float(df_1h['Close'].iloc[-1])

    # 1H 均线与 ATR
    df_1h_calc = df_1h.copy()
    df_1h_calc['EMA20'] = df_1h_calc['Close'].ewm(span=20, adjust=False).mean()
    df_1h_calc['SMA50'] = df_1h_calc['Close'].rolling(window=50).mean()
    
    tr = np.maximum(df_1h_calc['High'] - df_1h_calc['Low'], 
                    np.maximum((df_1h_calc['High'] - df_1h_calc['Close'].shift(1)).abs(), 
                               (df_1h_calc['Low'] - df_1h_calc['Close'].shift(1)).abs()))
    atr = float(tr.rolling(14).mean().iloc[-1]) if not np.isnan(tr.rolling(14).mean().iloc[-1]) else (live_price * 0.008)

    # 1H Grimes 拐点扫描 (Primary + Secondary)
    subset = df_1h_calc.iloc[-60:].copy()
    highs, lows, opens, closes, times = subset['High'].values, subset['Low'].values, subset['Open'].values, subset['Close'].values, subset.index
    
    pivots_high, pivots_low = [], []
    for i in range(2, len(subset) - 2):
        if highs[i] == max(highs[i-2:i+3]):
            pivots_high.append((float(highs[i]), float(max(opens[i], closes[i])), times[i].strftime("%m-%d %H:%M ET")))
        if lows[i] == min(lows[i-2:i+3]):
            pivots_low.append((float(min(opens[i], closes[i])), float(lows[i]), times[i].strftime("%m-%d %H:%M ET")))

    # SBR 排序
    valid_highs = [p for p in pivots_high if p[1] > live_price]
    valid_highs.sort(key=lambda x: x[1])
    sbr_top, sbr_bot, sbr_time = valid_highs[0] if len(valid_highs) >= 1 else (live_price + 1.2 * atr, live_price + 0.6 * atr, "Range High")
    sbr2_top, sbr2_bot, sbr2_time = valid_highs[1] if len(valid_highs) >= 2 else (sbr_top + 1.2 * atr, sbr_top + 0.5 * atr, "Tier-2 High")

    # RBS 排序
    valid_lows = [p for p in pivots_low if p[0] < live_price]
    valid_lows.sort(key=lambda x: x[0], reverse=True)
    rbs_top, rbs_bot, rbs_time = valid_lows[0] if len(valid_lows) >= 1 else (live_price - 0.6 * atr, live_price - 1.2 * atr, "Range Low")
    rbs2_top, rbs2_bot, rbs2_time = valid_lows[1] if len(valid_lows) >= 2 else (rbs_bot - 0.5 * atr, rbs_bot - 1.2 * atr, "Tier-2 Low")

    # =====================================================================
    # 三维共振判决 (已优化：解耦 SMA50 滞后死锁，引入极值突破与敏捷 EMA20)
    # =====================================================================
    ema20_now = float(df_1h_calc['EMA20'].iloc[-1])
    
    # 1. 均线站位分：纯粹基于 1H EMA20 站位
    score_ma = 1 if live_price > ema20_now else (-1 if live_price < ema20_now else 0)

    # 2. 结构与极值分：优先抓取昨高昨低或盘前极值突破，次级参考 Grimes 拐点
    score_hhll = 0
    if live_price > pdh_val or live_price > pmh_val:
        score_hhll = 1
    elif live_price < pdl_val or live_price < pml_val:
        score_hhll = -1
    elif len(pivots_high) >= 2 and len(pivots_low) >= 2:
        last_2_h, last_2_l = [p[0] for p in pivots_high[-2:]], [p[1] for p in pivots_low[-2:]]
        if last_2_h[1] > last_2_h[0] and last_2_l[1] > last_2_l[0]:
            score_hhll = 1
        elif last_2_h[1] < last_2_h[0] and last_2_l[1] < last_2_l[0]:
            score_hhll = -1

    # 3. 均线斜率分：回溯 3 根 1H K 线，提升对急跌急涨的捕捉速度
    ema20_prev = float(df_1h_calc['EMA20'].iloc[-3])
    ema_slope = (ema20_now - ema20_prev) / ema20_prev * 100
    score_slope = 1 if ema_slope > 0.10 else (-1 if ema_slope < -0.10 else 0)

    total_score = score_ma + score_hhll + score_slope
    final_bias = 1 if total_score >= 2 else (-1 if total_score <= -2 else 0)

    prev_close = float(df_1h['Close'].iloc[-2])
    chg_pct = (live_price - prev_close) / prev_close * 100
    
    # 轮动策略信号判定
    if live_price >= sbr_bot: action = "🔴 止盈高抛 (Take Profit)"
    elif live_price <= rbs_top: action = "🟢 支撑轮动 (Rotation In)"
    elif live_price > ema20_now: action = "📈 多头持仓 (Holding)"
    else: action = "📉 偏弱观望 (Weak)"

    return {
        "TICKER": ticker,
        "Group": "Mag 7" if ticker in BIG_SEVEN else ("Index" if ticker == "QQQ" else "Growth"),
        "Close": round(live_price, 2),
        "Change%": round(chg_pct, 2),
        "Action": action,
        "TREND_BIAS": final_bias,
        "TOTAL_SCORE": total_score,
        "EMA20": round(ema20_now, 2),
        "SBR_TOP": round(sbr_top, 2), "SBR_BOT": round(sbr_bot, 2), "SBR_TIME": sbr_time,
        "RBS_TOP": round(rbs_top, 2), "RBS_BOT": round(rbs_bot, 2), "RBS_TIME": rbs_time,
        "SBR2_TOP": round(sbr2_top, 2), "SBR2_BOT": round(sbr2_bot, 2), "SBR2_TIME": sbr2_time,
        "RBS2_TOP": round(rbs2_top, 2), "RBS2_BOT": round(rbs2_bot, 2), "RBS2_TIME": rbs2_time,
        "PDH": round(pdh_val, 2), "PDH_TIME": pdh_time_str,
        "PDL": round(pdl_val, 2), "PDL_TIME": pdl_time_str,
        "PMH": round(pmh_val, 2), "PMH_TIME": pmh_time_str,
        "PML": round(pml_val, 2), "PML_TIME": pml_time_str,
        "SOURCE_1H": source_1h, "SOURCE_5M": source_5m
    }

# =====================================================================
# 5. 主程序渲染 (《《癸水(QQQ)》》座舱)
# =====================================================================
st.title("🌊 《《癸水(QQQ)》》 0DTE 期权中枢 & 17 核心股轮动雷达")

results = []
all_hist_data = {}
with st.spinner("执行双梯队战区运算与 17 股轮动扫描中..."):
    for t in ALL_TICKERS:
        df_1h, src_1h, df_5m, src_5m = fetch_complete_data_audited(t, TIINGO_TOKEN)
        if df_1h is not None:
            all_hist_data[t] = df_1h
        res = calculate_audited_levels(df_1h, src_1h, df_5m, src_5m, t)
        if res:
            results.append(res)

if results:
    df_res = pd.DataFrame(results)
    df_stocks = df_res[df_res["TICKER"] != "QQQ"]
    qqq_row = df_res[df_res["TICKER"] == "QQQ"].iloc[0]

    # --- 5.1 顶部战况与市场宽度指标卡 ---
    up_cnt = sum(df_stocks["Change%"] > 0)
    down_cnt = sum(df_stocks["Change%"] <= 0)
    mag7_up = sum(df_stocks[df_stocks["Group"] == "Mag 7"]["Change%"] > 0)
    breadth_pct = int(sum(df_stocks["Close"] > df_stocks["EMA20"]) / len(df_stocks) * 100)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 QQQ 最新现价", f"${qqq_row['Close']}", f"{qqq_row['Change%']}%")
    m2.metric("📊 17 股多空分布", f"{up_cnt} 涨 / {down_cnt} 跌", f"宽度: {breadth_pct}% > 20EMA")
    m3.metric("👑 Big 7 巨头动能", f"{mag7_up} / 7 支上涨", "决定 QQQ 真实推力")
    bias_desc = "🟢 偏多 (CALL)" if qqq_row['TREND_BIAS'] == 1 else ("🔴 偏空 (PUT)" if qqq_row['TREND_BIAS'] == -1 else "⚪ 震荡 (NEUTRAL)")
    m4.metric("🧭 QQQ 宏观定调", bias_desc, f"共振得分: {qqq_row['TOTAL_SCORE']} / 3")

    st.markdown("---")

    # --- 5.2 核心左右双栏：左侧【买卖哪只股】 vs 右侧【QQQ 13 行参数复制】 ---
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.subheader("⚡ 【今日实战买卖雷达】(轮动加仓 & 止盈)")
        
        # 筛选买卖标的
        buy_list = df_stocks[df_stocks["Action"].str.contains("轮动")]
        profit_list = df_stocks[df_stocks["Action"].str.contains("止盈")]
        holding_list = df_stocks[df_stocks["Action"].str.contains("持仓")]
        
        st.markdown("#### 🟢 立即轮动买入区 (ROTATION IN)")
        if not buy_list.empty:
            for _, r in buy_list.iterrows():
                st.success(f"**{r['TICKER']}** (${r['Close']}, {r['Change%']}%) ➔ 踩入 1H RBS 支撑带 (`${r['RBS_BOT']} ~ ${r['RBS_TOP']}`)\n\n*📌 建议：QQQ 期权盈利资金，优先分批定投买入*")
        else:
            st.info("暂无个股踩入 1H RBS 支撑位（无错杀低吸点）")

        st.markdown("#### 🔴 立即止盈高抛区 (TAKE PROFIT)")
        if not profit_list.empty:
            for _, r in profit_list.iterrows():
                st.error(f"**{r['TICKER']}** (${r['Close']}, +{r['Change%']}%) ➔ 刺入 1H SBR 阻力带 (`${r['SBR_BOT']} ~ ${r['SBR_TOP']}`)\n\n*📌 建议：正股底仓分批止盈高抛，收回现金*")
        else:
            st.info("暂无个股触及 1H SBR 阻力位（持仓继续奔跑）")

        st.markdown("#### ⚪ 顺势持仓待命区 (HOLDING)")
        hold_str = ", ".join([f"**{r['TICKER']}**({r['Change%']}%)" for _, r in holding_list.iterrows()]) if not holding_list.empty else "无"
        st.caption(f"处于 20 EMA 上方顺势运行: {hold_str}")

    with col_right:
        st.subheader("📋 【QQQ 专属】富途 5M 复制座舱")
        st.markdown(f"* **现价通道:** `{qqq_row['SOURCE_5M']}` | **1H 通道:** `{qqq_row['SOURCE_1H']}`")
        st.markdown(f"* **⚡ 今日盘前极值:** `${qqq_row['PMH']}` ~ `${qqq_row['PML']}` *(时间: `{qqq_row['PMH_TIME']}`)*")
        st.markdown(f"* **📌 昨日常规极值:** `${qqq_row['PDH']}` ~ `${qqq_row['PDL']}` *(时间: `{qqq_row['PDH_TIME']}`)*")
        st.markdown(f"* **🔴 Primary SBR 阻力:** `${qqq_row['SBR_BOT']} ~ ${qqq_row['SBR_TOP']}` *(K线: `{qqq_row['SBR_TIME']}`)*")
        st.markdown(f"* **🟢 Primary RBS 支撑:** `${qqq_row['RBS_BOT']} ~ ${qqq_row['RBS_TOP']}` *(K线: `{qqq_row['RBS_TIME']}`)*")
        st.markdown(f"* **🔴 Secondary SBR2:** `${qqq_row['SBR2_BOT']} ~ ${qqq_row['SBR2_TOP']}` *(K线: `{qqq_row['SBR2_TIME']}`)*")
        st.markdown(f"* **🟢 Secondary RBS2:** `${qqq_row['RBS2_BOT']} ~ ${qqq_row['RBS2_TOP']}` *(K线: `{qqq_row['RBS2_TIME']}`)*")
        
        st.markdown("##### 复制到富途指标顶部 13 行代码 (点右上角直接复制):")
        futu_13_code = f"""TREND_BIAS := {int(qqq_row['TREND_BIAS'])};       {{ 1. 宏观偏向: 1=多, -1=空, 0=中立 [得分: {qqq_row['TOTAL_SCORE']}] }}

{{ --- 第一梯队主战区 (Primary Zones) --- }}
SBR_TOP    := {qqq_row['SBR_TOP']:.2f};  {{ 2. Primary 1H 阻力顶沿 [{qqq_row['SBR_TIME']}] }}
SBR_BOT    := {qqq_row['SBR_BOT']:.2f};  {{ 3. Primary 1H 阻力底沿 [{qqq_row['SBR_TIME']}] }}
RBS_TOP    := {qqq_row['RBS_TOP']:.2f};  {{ 4. Primary 1H 支撑顶沿 [{qqq_row['RBS_TIME']}] }}
RBS_BOT    := {qqq_row['RBS_BOT']:.2f};  {{ 5. Primary 1H 支撑底沿 [{qqq_row['RBS_TIME']}] }}

{{ --- 第二梯队拓展战区 (Secondary Zones - 突破后备用) --- }}
SBR2_TOP   := {qqq_row['SBR2_TOP']:.2f};  {{ 6. Secondary 1H 更高阻力顶沿 [{qqq_row['SBR2_TIME']}] }}
SBR2_BOT   := {qqq_row['SBR2_BOT']:.2f};  {{ 7. Secondary 1H 更高阻力底沿 [{qqq_row['SBR2_TIME']}] }}
RBS2_TOP   := {qqq_row['RBS2_TOP']:.2f};  {{ 8. Secondary 1H 更低支撑顶沿 [{qqq_row['RBS2_TIME']}] }}
RBS2_BOT   := {qqq_row['RBS2_BOT']:.2f};  {{ 9. Secondary 1H 更低支撑底沿 [{qqq_row['RBS2_TIME']}] }}

{{ --- 全市场客观极值 (Sweep Anchors) --- }}
PDH_LINE   := {qqq_row['PDH']:.2f};  {{ 10. 昨日最高价 PDH [{qqq_row['PDH_TIME']}] }}
PDL_LINE   := {qqq_row['PDL']:.2f};  {{ 11. 昨日最低价 PDL [{qqq_row['PDL_TIME']}] }}
PMH_LINE   := {qqq_row['PMH']:.2f};  {{ 12. 盘前最高价 PMH [{qqq_row['PMH_TIME']}] }}
PML_LINE   := {qqq_row['PML']:.2f};  {{ 13. 盘前最低价 PML [{qqq_row['PML_TIME']}] }}"""
        st.code(futu_13_code, language="pascal")

    st.markdown("---")

    # --- 5.3 17 股全景轮动雷达看板 ---
    st.subheader("🗺️ 17 支核心个股全景雷达 (轮动与止盈监控)")
    
    def highlight_action(val):
        if "止盈" in str(val):
            return "background-color: #49111c; color: #ffccd5; font-weight: bold;"
        elif "轮动" in str(val):
            return "background-color: #1b4332; color: #d8f3dc; font-weight: bold;"
        return ""

    display_cols = ["TICKER", "Group", "Close", "Change%", "Action", "EMA20", "RBS_TOP", "SBR_BOT", "PDL", "PDH", "SOURCE_5M"]
    styler = df_stocks[display_cols].sort_values(by="Change%", ascending=False).style
    if hasattr(styler, 'map'):
        styled_df = styler.map(highlight_action, subset=["Action"])
    else:
        styled_df = styler.applymap(highlight_action, subset=["Action"])
    st.dataframe(styled_df, use_container_width=True, height=480)

    # --- 5.4 历史时光机导出 (Pass Record) ---
    st.markdown("---")
    st.subheader("⏳ 历史复盘时光机与数据导出 (Pass Record)")
    
    col_p1, col_p2 = st.columns([2, 8])
    with col_p1:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_res.to_excel(writer, index=False, sheet_name="Market_Report")
        st.download_button(
            label="📥 导出今日全量复盘报表 (.xlsx)",
            data=output.getvalue(),
            file_name=f"QQQ_Portfolio_Report_{datetime.date.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col_p2:
        btn_clear = st.button("🧹 清除缓存并强制刷新看板", type="secondary")
        if btn_clear:
            st.cache_data.clear()
            st.rerun()
