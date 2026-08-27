import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="QQQ 5M Cockpit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- 数据与指标计算 -----------------
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

@st.cache_data(ttl=60)
def fetch_qqq_5m_data():
    ticker = yf.Ticker("QQQ")
    df = ticker.history(period="5d", interval="5m")
    if df.empty:
        return df
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['ATR'] = calculate_atr(df, period=14)
    df['VWAP'] = calculate_vwap(df)
    return df

# ----------------- 侧边栏设置 -----------------
st.sidebar.title("交易参数设置")
risk_per_trade = st.sidebar.number_input("单笔固定风险金额 (USD)", min_value=10.0, value=200.0, step=50.0)
atr_period = st.sidebar.number_input("ATR 周期", min_value=5, max_value=50, value=14)
auto_refresh = st.sidebar.button("刷新实时数据")

# ----------------- 主界面 -----------------
st.title("QQQ 5M 执行座舱 (0.5 ATR / 1:2 R:R)")

df = fetch_qqq_5m_data()

if df.empty:
    st.error("获取数据失败，请检查网络或稍后重试。")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2]

current_price = latest['Close']
current_atr = latest['ATR']
sl_distance = 0.5 * current_atr
tp_distance = 1.0 * current_atr  # 1:2 盈亏比 (0.5 * 2)

# 做多点位
long_sl = current_price - sl_distance
long_tp = current_price + tp_distance
long_shares = int(risk_per_trade / sl_distance) if sl_distance > 0 else 0

# 做空点位
short_sl = current_price + sl_distance
short_tp = current_price - tp_distance
short_shares = int(risk_per_trade / sl_distance) if sl_distance > 0 else 0

# ----------------- 核心指标看板 -----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="QQQ 当前价格 (5M)",
        value=f"${current_price:.2f}",
        delta=f"{(current_price - prev['Close']):.2f}"
    )

with col2:
    st.metric(
        label=f"ATR ({atr_period})",
        value=f"${current_atr:.2f}",
        delta=f"0.5 ATR = ${sl_distance:.2f}"
    )

with col3:
    st.metric(
        label="做多头寸建议",
        value=f"{long_shares} 股",
        delta=f"总暴露: ${long_shares * current_price:,.0f}"
    )

with col4:
    st.metric(
        label="做空头寸建议",
        value=f"{short_shares} 股",
        delta=f"总暴露: ${short_shares * current_price:,.0f}"
    )

st.divider()

# ----------------- 计划执行矩阵 -----------------
t1, t2 = st.columns(2)

with t1:
    st.subheader("做多计划 (Long Setup)")
    long_data = {
        "项目": ["进场参考价", "止损位 (0.5 ATR)", "目标止盈位 (1:2)", "单笔止损幅度", "建议买入数量"],
        "数值": [
            f"${current_price:.2f}",
            f"${long_sl:.2f}",
            f"${long_tp:.2f}",
            f"-${sl_distance:.2f} ({(sl_distance/current_price)*100:.2f}%)",
            f"{long_shares} 股"
        ]
    }
    st.table(pd.DataFrame(long_data))

with t2:
    st.subheader("做空计划 (Short Setup)")
    short_data = {
        "项目": ["进场参考价", "止损位 (0.5 ATR)", "目标止盈位 (1:2)", "单笔止损幅度", "建议卖空数量"],
        "数值": [
            f"${current_price:.2f}",
            f"${short_sl:.2f}",
            f"${short_tp:.2f}",
            f"+${sl_distance:.2f} ({(sl_distance/current_price)*100:.2f}%)",
            f"{short_shares} 股"
        ]
    }
    st.table(pd.DataFrame(short_data))

# ----------------- 5M 图表 -----------------
st.subheader("QQQ 5M 结构与执行区间")

plot_df = df.tail(60)

fig = go.Figure()

# K线
fig.add_trace(go.Candlestick(
    x=plot_df.index,
    open=plot_df['Open'],
    high=plot_df['High'],
    low=plot_df['Low'],
    close=plot_df['Close'],
    name="5M K线"
))

# 均线与指标
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_9'], line=dict(color='orange', width=1), name="EMA 9"))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_21'], line=dict(color='blue', width=1), name="EMA 21"))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['VWAP'], line=dict(color='purple', width=1.5, dash='dot'), name="VWAP"))

# 止损止盈参考水平线
fig.add_hline(y=long_tp, line_dash="dash", line_color="green", annotation_text="做多目标 TP (1:2)")
fig.add_hline(y=long_sl, line_dash="dash", line_color="red", annotation_text="做多止损 SL (0.5 ATR)")

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=550,
    margin=dict(l=10, r=10, t=30, b=10),
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
