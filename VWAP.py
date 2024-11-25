import streamlit as st
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# 初始化 Gate.io API
api_key = 'YOUR_API_KEY'  # 替换为你的 API Key
api_secret = 'YOUR_API_SECRET'  # 替换为你的 API Secret
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 20000
})

# 加载市场数据
exchange.load_markets()


def fetch_data(symbol, timeframe='1m', start_time=None, end_time=None):
    """
    从交易所获取历史数据
    :param symbol: 交易对 (如 BTC/USDT)
    :param timeframe: 时间级别 (如 '1m', '5m', '4h', '1d')
    :param start_time: 起始时间 (datetime 对象)
    :param end_time: 结束时间 (datetime 对象)
    :return: 包含历史数据的 DataFrame
    """
    try:
        since = int(start_time.timestamp() * 1000)
        until = int(end_time.timestamp() * 1000) if end_time else None

        ohlcvs = []
        while True:
            data = exchange.fetch_ohlcv(symbol, timeframe, since)
            ohlcvs += data
            if len(data) < 500 or (until and data[-1][0] >= until):
                break
            since = data[-1][0]

        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[(df['timestamp'] >= pd.Timestamp(start_time)) & (df['timestamp'] <= pd.Timestamp(end_time))]
        return df
    except Exception as e:
        st.error(f"抓取数据失败: {e}")
        return None


def calculate_vwap(df):
    """
    计算密集成交区 (VWAP)
    :param df: 历史数据 DataFrame
    :return: VWAP 值 (浮点数)
    """
    if df is None or df.empty:
        return None
    vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
    return round(vwap, 13)


def plot_chart(df, vwap, symbol):
    """
    生成图表，包含成交量柱状图、MA7、MA34 和 VWAP
    :param df: 历史数据 DataFrame
    :param vwap: VWAP 值
    :param symbol: 交易对
    """
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

# 用户输入货币对
symbol = st.text_input("输入货币对 (如 BTC/USDT):", value="BTC/USDT")

# 用户选择时间级别
timeframe = st.selectbox("选择交易级别:", ["1m", "5m", "30m", "4h", "1d"], index=3)

# 提示用户输入时间格式
st.write("请输入起始和结束时间（格式：YYYY/MM/DD HH:MM，例如 2024/11/25 08:00）：")

# 起始和结束时间通过文本框输入
start_datetime_str = st.text_input("输入起始时间:", value="2024/11/25 08:00")
end_datetime_str = st.text_input("输入结束时间:", value="2024/11/25 08:00")

# 转换字符串为 datetime 对象
try:
    start_datetime = datetime.strptime(start_datetime_str, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end_datetime_str, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
    valid_time = True
except ValueError:
    st.error("时间格式错误，请使用 YYYY/MM/DD HH:MM 格式输入。")
    valid_time = False

if st.button("确定") and valid_time:
    st.write(f"**输入的货币对为:** {symbol}")
    st.write(f"**选择的交易级别为:** {timeframe}")
    st.write(f"**时间跨度为:** 从 {start_datetime} 到 {end_datetime}")

    # 获取历史数据
    df = fetch_data(symbol, timeframe, start_datetime, end_datetime)

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
