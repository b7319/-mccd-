import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time
import pytz

# 初始化Gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 20000
})

# 音频文件URL
alert_sound_url = "http://www.btc131419.cn/wp-content/uploads/2024/12/y1148.wav"

# 加载市场数据
def load_markets_with_retry():
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except Exception as e:
            st.error(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

# 获取高交易额交易对
def get_symbols_with_volume():
    try:
        tickers = exchange.fetch_tickers()
        symbols = [
            symbol for symbol, ticker in tickers.items()
            if 'USDT' in symbol and ticker.get('quoteVolume', 0) >= 5000000
        ]
        return symbols
    except Exception as e:
        st.error(f"获取高交易额交易对时出错: {str(e)}")
        return []

# 获取K线数据
def fetch_data(symbol, timeframe='5m', since=None):
    try:
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"获取 {symbol} 数据时出错: {str(e)}")
        return None

# 计算MA指标
def calculate_moving_averages(df):
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma34'] = df['close'].rolling(window=34).mean()
    df['ma170'] = df['close'].rolling(window=170).mean()

# 自动播放音频函数
def play_audio():
    audio_html = f"""
        <audio autoplay>
            <source src="{alert_sound_url}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# 滚动显示当前检测的交易对
def display_scrolling_symbol(symbol_placeholder, current_symbol):
    symbol_placeholder.write(f"**正在检测交易对：{current_symbol}**")

# 主逻辑
def main():
    st.title("实时交易检测 (5分钟级别)")
    st.markdown("---")

    # 符合条件的交易对区域
    st.header("符合条件的交易对")
    results_placeholder = st.container()  # 用于动态更新符合条件的交易对

    # 当前检测交易对滚动显示
    symbol_placeholder = st.empty()

    # 获取符合条件的交易对
    symbols = get_symbols_with_volume()
    if not symbols:
        st.warning("未找到符合条件的高交易额交易对")
        return

    st.success(f"已加载 {len(symbols)} 个交易对，开始检测...")
    st.markdown("---")

    # 实时检测交易对
    while True:
        for symbol in symbols:
            # 滚动显示当前检测的交易对
            display_scrolling_symbol(symbol_placeholder, symbol)

            # 获取最近15小时K线数据
            since_15h = exchange.parse8601((datetime.now(timezone.utc) - timedelta(hours=15)).isoformat())
            df = fetch_data(symbol, '5m', since=since_15h)

            if df is not None:
                calculate_moving_averages(df)
                ma7_valley = df['ma7'].iloc[-1]
                ma34_peak = df['ma34'].max()
                ma170_min = df['ma170'].min()

                # 判断符合条件的交易对
                if ma34_peak <= ma170_min <= ma7_valley:
                    with results_placeholder:
                        st.write(f"### 交易对: {symbol}")
                        st.write(f"- **MA34 波峰值**: {ma34_peak}")
                        st.write(f"- **MA170 最低值**: {ma170_min}")
                        st.write(f"- **MA7 波谷值**: {ma7_valley}")
                        st.write("---")
                    play_audio()  # 播放音频提示

            time.sleep(2)  # 每个交易对间隔2秒
        time.sleep(180)  # 所有交易对检测完毕后等待3分钟

if __name__ == "__main__":
    main()
