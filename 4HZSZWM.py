import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time

# 使用API密钥初始化gate.io API
api_key = 'your_api_key'
api_secret = 'your_api_secret'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 20000
})

# 加载市场数据
exchange.load_markets()

# 获取日交易额超过1000万USDT的现货交易对
def get_high_volume_symbols(threshold=10_000_000):
    try:
        tickers = exchange.fetch_tickers()
        high_volume_symbols = [
            symbol for symbol, data in tickers.items()
            if 'spot' in data.get('type', '') and data['quoteVolume'] >= threshold
        ]
        return high_volume_symbols
    except Exception as e:
        st.error(f"获取高交易量交易对时出错: {e}")
        return []

# 获取OHLCV数据
def fetch_data(symbol, timeframe='4h', days=60):
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma34'] = df['close'].rolling(window=34).mean()
        df['ma170'] = df['close'].rolling(window=170).mean()
        return df
    except Exception as e:
        st.write(f"获取 {symbol} 数据时出错: {str(e)}")
        return None

# 找出MA34的波峰
def find_ma34_peaks(df):
    peaks = []
    for i in range(34, len(df) - 1):
        if df['ma34'].iloc[i] > df['ma34'].iloc[i - 1] and df['ma34'].iloc[i] > df['ma34'].iloc[i + 1]:
            peaks.append(df['ma34'].iloc[i])
    return peaks

# 获取最新价格
def get_latest_price(df):
    return df['close'].iloc[-1]

# 筛选满足条件的交易对
def display_results(symbols):
    total_symbols = len(symbols)
    detected = 0
    
    for idx, symbol in enumerate(symbols):
        st.write(f"正在检测交易对: {symbol} ({idx + 1}/{total_symbols})")
        df = fetch_data(symbol, days=130)
        if df is None or df.empty:
            continue

        peaks = find_ma34_peaks(df)
        ma170_min = df['ma170'].min()
        latest_price = get_latest_price(df)

        if not peaks or ma170_min is None:
            st.write(f"交易对 {symbol} 的数据不完整，跳过检测")
            continue

        min_peak_value = min(peaks)
        condition = ma170_min >= min_peak_value and latest_price > df['ma170'].iloc[-1]

        if condition:
            detected += 1
            st.write("---")
            st.write(f"交易对: {symbol}")
            st.write(f"最小MA34波峰值: {min_peak_value}")
            st.write(f"最新MA170值: {df['ma170'].iloc[-1]}")
            st.write(f"最新价格: {latest_price}")
            st.write(f"条件满足时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
            st.write("---")

        time.sleep(0.5)

    st.write(f"检测完成，共找到 {detected}/{total_symbols} 个符合条件的交易对！")

if __name__ == "__main__":
    st.title("实时交易对检测")

    symbols = get_high_volume_symbols()
    if not symbols:
        st.error("未找到符合条件的现货交易对！")
    else:
        st.success(f"成功加载 {len(symbols)} 个交易对！")
        while True:
            display_results(symbols)
            time.sleep(10)
