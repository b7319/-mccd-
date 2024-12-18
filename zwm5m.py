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

# 音频路径
alert_sound_url = 'http://www.btc131419.cn/wp-content/uploads/2024/12/y1148.wav'

# 加载市场数据
def load_markets_with_retry():
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError as e:
            st.write(f"网络错误，正在重试 ({attempt + 1}/3)...")
            time.sleep(5)
        except Exception as e:
            st.write(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

# 筛选出日交易额大于500万USDT的交易对
def get_symbols_with_volume():
    try:
        tickers = exchange.fetch_tickers()
        symbols = [
            symbol for symbol, ticker in tickers.items()
            if 'USDT' in symbol and ticker.get('quoteVolume', 0) >= 5000000
        ]
        return symbols
    except Exception as e:
        st.write(f"获取高交易额交易对时出错: {str(e)}")
        return []

# 获取指定时间段的K线数据（5分钟级别）
def fetch_data(symbol, timeframe='5m', since=None):
    if symbol not in exchange.symbols:
        return None
    try:
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.write(f"获取 {symbol} 数据时出错: {str(e)}")
        return None

# 获取明显的MA34波峰值（经过平滑处理）
def get_ma34_valid_peak(df):
    df['ma34'] = df['close'].rolling(window=34).mean()
    peaks = []

    for i in range(1, len(df) - 1):
        if df['ma34'].iloc[i] > df['ma34'].iloc[i - 1] and df['ma34'].iloc[i] > df['ma34'].iloc[i + 1]:
            peaks.append((df['ma34'].iloc[i], df.index[i]))

    if not peaks:
        return None, None

    valid_peak = min(peaks, key=lambda x: x[0])
    return valid_peak[0], valid_peak[1]

# 获取最新15小时内MA170的最低值
def get_min_ma170(df):
    df['ma170'] = df['close'].rolling(window=170).mean()
    if df['ma170'].isnull().all():
        return None, None
    min_ma170 = df['ma170'].min()
    min_ma170_time = df['ma170'].idxmin()
    return min_ma170, min_ma170_time

# 检测最新9根K线内是否存在MA7的波谷
def check_ma7_valley(df):
    df['ma7'] = df['close'].rolling(window=7).mean()
    recent_df = df.iloc[-9:]
    for i in range(1, len(recent_df) - 1):
        if recent_df['ma7'].iloc[i] < recent_df['ma7'].iloc[i - 1] and \
           recent_df['ma7'].iloc[i] < recent_df['ma7'].iloc[i + 1]:
            return recent_df['ma7'].iloc[i]
    return None

# 将UTC时间转换为北京时间（CST）
def convert_to_cst(utc_time):
    if utc_time is None:
        return "N/A"
    cst = pytz.timezone('Asia/Shanghai')
    utc_time = utc_time.replace(tzinfo=pytz.utc)
    return utc_time.astimezone(cst)

# 主逻辑
def main():
    st.title('实时交易检测 (5分钟级别)')

    st.markdown("---")
    progress_bar = st.progress(0)
    current_symbol_placeholder = st.empty()
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.header("符合条件的交易对")
    with col2:
        st.header("不符合条件的交易对")

    symbols = get_symbols_with_volume()
    if not symbols:
        st.warning("未找到符合条件的高交易额交易对")
        return

    st.success(f"加载 {len(symbols)} 个交易对进行检测")

    while True:
        for idx, symbol in enumerate(symbols):
            progress = (idx + 1) / len(symbols)
            progress_bar.progress(progress)
            current_symbol_placeholder.write(f"正在检测交易对: {symbol} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

            since_15h = exchange.parse8601((datetime.now(timezone.utc) - timedelta(hours=15)).isoformat())
            df = fetch_data(symbol, '5m', since=since_15h)

            if df is not None and not df.empty:
                ma34_peak, ma34_peak_time = get_ma34_valid_peak(df)
                min_ma170, min_ma170_time = get_min_ma170(df)
                ma7_valley = check_ma7_valley(df)

                if all(val is not None for val in [ma34_peak, min_ma170, ma7_valley]) and \
                        ma34_peak <= min_ma170 <= ma7_valley:
                    with col1:
                        st.write(f"交易对: {symbol}")
                        st.write(f"MA34 波峰值：{ma34_peak} (发生时间: {convert_to_cst(ma34_peak_time)})")
                        st.write(f"MA170 最低值：{min_ma170} (发生时间: {convert_to_cst(min_ma170_time)})")
                        st.write(f"MA7 波谷值：{ma7_valley}")
                        st.write(f"是否合格：是")
                        st.audio(alert_sound_url)
                        st.write("---")
                else:
                    with col2:
                        st.write(f"交易对: {symbol}")
                        st.write(f"MA34 波峰值：{ma34_peak} (发生时间: {convert_to_cst(ma34_peak_time)})")
                        st.write(f"MA170 最低值：{min_ma170} (发生时间: {convert_to_cst(min_ma170_time)})")
                        st.write(f"MA7 波谷值：{ma7_valley}")
                        st.write(f"是否合格：否")
                        st.write("---")

            time.sleep(3)
        time.sleep(180)

if __name__ == '__main__':
    main()
