import streamlit as st
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# 初始化 Gate.io API
api_key = 'YOUR_API_KEY'  # 替换为你的 API Key
api_secret = 'YOUR_API_SECRET'  # 替换为你的 API Secret
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 加载市场数据
exchange.load_markets()


def fetch_data(symbol, timeframe='4h', days=30, hours=0):
    """从交易所获取历史数据"""
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days, hours=hours)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    except Exception as e:
        st.error(f"抓取数据失败: {e}")
        return None


def calculate_vwap(df):
    """计算密集成交区 (VWAP)"""
    if df is None or df.empty:
        return None
    vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
    return round(vwap, 13)  # 保留 13 位小数


def plot_chart(df, vwap, symbol):
    """生成图表，包含成交量柱状图、MA7、MA34 和 VWAP"""
    # 计算 MA7 和 MA34 均线
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma34'] = df['close'].rolling(window=34).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # 绘制价格线、MA7 和 MA34
    ax1.plot(df['timestamp'], df['close'], label='收盘价', color='blue')
    ax1.plot(df['timestamp'], df['ma7'], label='MA7', color='purple', linestyle='--')
    ax1.plot(df['timestamp'], df['ma34'], label='MA34', color='orange', linestyle='--')
    ax1.axhline(vwap, color='red', linestyle='--', label=f'VWAP: {vwap:.13f}')
    ax1.set_ylabel('价格', fontsize=12)
    ax1.set_title(f'{symbol} 收盘价、VWAP、MA7 和 MA34', fontsize=16)
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='x', rotation=45)

    # 绘制成交量柱状图
    ax2.bar(df['timestamp'], df['volume'], color='green', alpha=0.7, label='成交量')
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('时间', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)


# Streamlit 界面
st.title("密集交易区计算工具")
symbol = st.text_input("输入货币对 (如 BTC/USDT):", value="BTC/USDT")
timeframe = st.selectbox("选择交易级别:", ["1m", "5m", "30m", "4h", "1d"], index=3)  # 添加 "1m" 选项，默认选择第4项 "4h"
days = st.number_input("输入时间跨度 (天):", min_value=0, value=30)  # 允许天为0
hours = st.number_input("输入时间跨度 (小时):", min_value=0, value=0, max_value=23)  # 限制在一天之内

if st.button("确定"):
    st.write(f"**输入的货币对为:** {symbol}")
    st.write(f"**选择的交易级别为:** {timeframe}")
    st.write(f"**时间跨度为:** {days} 天 {hours} 小时")

    # 获取历史数据
    df = fetch_data(symbol, timeframe, days, hours)

    if df is not None and not df.empty:
        # 计算 VWAP
        vwap = calculate_vwap(df)
        if vwap is not None:
            st.write(f"**密集成交区价格 (VWAP):** {vwap:.13f}")
            # 绘制图表
            plot_chart(df, vwap, symbol)
        else:
            st.error("无法计算 VWAP，可能数据不足。")
    else:
        st.error("未能获取历史数据，请检查输入的货币对是否正确。")
