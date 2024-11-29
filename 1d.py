import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time

# 初始化 Gate.io API
api_key = 'your_api_key'  # 替换为你的 API Key
api_secret = 'your_api_secret'  # 替换为你的 API Secret
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 初始化交易市场数据
def load_markets_with_retry():
    """尝试加载交易所市场，最多重试 3 次"""
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.write(f"网络错误，正在重试 ({attempt + 1}/3)...")
        except Exception as e:
            st.write(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

def fetch_data(symbol, timeframe='1d', days=90):
    """
    获取交易对的历史 K 线数据
    """
    if symbol not in exchange.symbols:
        return None
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.write(f"抓取 {symbol} 的数据时出错: {str(e)}")
        return None

def calculate_vwap(df):
    """
    计算 VWAP
    """
    if df.empty:
        return None
    vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
    return round(vwap, 13)

def check_conditions(df):
    """
    检查条件
    """
    if df.empty or len(df) < 3:
        return False, None, None, None
    vwap = calculate_vwap(df)
    if vwap is None:
        return False, None, None, None
    latest_price = df['close'].iloc[-1]
    in_vwap_range = abs(latest_price - vwap) / vwap <= 0.15
    return in_vwap_range, df.index[-1], latest_price, vwap

def get_high_volume_symbols():
    """
    筛选交易量超过 100 万 USDT 的交易对
    """
    try:
        markets = exchange.fetch_tickers()
        symbols = [
            symbol for symbol, data in markets.items()
            if data['quoteVolume'] > 1_000_000 and '/USDT' in symbol
        ]
        return symbols
    except Exception as e:
        st.write(f"获取高交易量交易对时出错: {str(e)}")
        return []

def monitor_symbols(symbols, days=90):
    """
    监控交易对
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    num_symbols = len(symbols)

    for index, symbol in enumerate(symbols):
        df = fetch_data(symbol, days=days)
        if df is not None and not df.empty:
            condition_met, condition_time, latest_price, vwap = check_conditions(df)
            if condition_met:
                st.write(f"交易对: {symbol}")
                st.write(f"时间周期: 1天")
                st.write(f"最新价格: {latest_price:.13f}")
                st.write(f"VWAP: {vwap:.13f}")
                st.write(f"信号时间: {condition_time}")
                st.write("---")
        progress_bar.progress((index + 1) / num_symbols)
        status_text.text(f"正在检测交易对: {symbol}")
        time.sleep(1)  # 避免 API 速率限制

    progress_bar.empty()
    status_text.text("检测完成")

def main():
    st.title('长期交易信号检测 - VWAP')

    # 用户输入 VWAP 计算时间跨度
    days = st.sidebar.slider("选择 VWAP 时间跨度（天）", min_value=30, max_value=180, value=90, step=10)
    interval = st.sidebar.selectbox("检测间隔", options=["每日", "每小时"], index=0)

    while True:
        st.write("正在加载高交易量交易对，请稍候...")
        symbols = get_high_volume_symbols()

        if not symbols:
            st.warning("未找到满足条件的交易对")
        else:
            st.success("交易对加载成功！")
            monitor_symbols(symbols, days=days)

        # 根据用户选择的间隔休眠
        if interval == "每日":
            time.sleep(86400)  # 24 小时
        elif interval == "每小时":
            time.sleep(3600)  # 1 小时

if __name__ == "__main__":
    main()
