import ccxt 
import pandas as pd
from datetime import datetime
import streamlit as st
import time
import pytz

# 初始化Gate.io API
api_key = 'c8e2fb89d031ca42a30ed7b674cb06dc'
api_secret = 'fab0bc8aeebeb31e46238eda033e2b6258e9c9185f262f74d4472489f9f03219'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 20000
})

# 加载市场数据
def load_markets_with_retry():
    """
    重试加载市场数据，最多尝试3次。
    """
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.write(f"网络错误，正在重试 ({attempt + 1}/3)...")
            time.sleep(5)
        except Exception as e:
            st.write(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

def get_market_volume(symbol):
    """
    获取指定交易对的24小时交易量（USDT计）。
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker.get('quoteVolume', None)  # 返回24小时交易量（USDT），若无数据则返回None
    except Exception as e:
        return None

def get_eligible_symbols(min_volume_usdt=3000000):
    """
    获取24小时交易量超过min_volume_usdt的现货USDT交易对。
    """
    eligible_symbols = []
    for symbol in exchange.symbols:
        if '/USDT' in symbol and 'USDT' not in symbol.split('/')[0] and ':' not in symbol:
            volume = get_market_volume(symbol)
            if volume is not None and volume >= min_volume_usdt:
                eligible_symbols.append(symbol)
    return eligible_symbols

def fetch_data(symbol, timeframe='30m', limit=386):
    """
    获取指定交易对的K线数据。
    """
    try:
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        return None

def calculate_ma(df, short_window=34, long_window=170):
    """
    计算MA34和MA170。
    """
    df['MA34'] = df['close'].rolling(window=short_window).mean()
    df['MA170'] = df['close'].rolling(window=long_window).mean()
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    计算 MACD 的 DIF 和 DEA。
    """
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()  # 快速EMA
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()  # 慢速EMA
    df['DIF'] = df['ema_fast'] - df['ema_slow']  # DIF
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()  # DEA
    return df

def detect_macd_dead_cross(df):
    """
    检测DIF死叉DEA的点。
    """
    dead_cross_points = []
    for i in range(1, len(df)):
        if df['DIF'].iloc[i] < df['DEA'].iloc[i] and df['DIF'].iloc[i - 1] > df['DEA'].iloc[i - 1]:
            dead_cross_points.append(i)
    return dead_cross_points

def get_ma34_valleys(df):
    """
    获取MA34波谷值，仅检测最新13根K线内。
    """
    valleys = []
    latest_13_df = df.iloc[-13:]
    for i in range(1, len(latest_13_df) - 1):
        if latest_13_df['MA34'].iloc[i] < latest_13_df['MA34'].iloc[i - 1] and latest_13_df['MA34'].iloc[i] < \
                latest_13_df['MA34'].iloc[i + 1]:
            valleys.append((latest_13_df.index[i], latest_13_df['MA34'].iloc[i]))
    return valleys

def get_effective_ma34_peaks(df, dead_cross_idx):
    """
    获取基于DIF死叉点之间区间内的有效MA34波峰值。
    """
    peaks = []
    for i in range(len(dead_cross_idx) - 1):
        start_idx = dead_cross_idx[i]
        end_idx = dead_cross_idx[i + 1]
        sub_df = df.iloc[start_idx:end_idx]
        if sub_df.empty:
            continue
        # 查找区间内的有效波峰
        for j in range(1, len(sub_df) - 1):
            # 局部高点
            if sub_df['MA34'].iloc[j] > sub_df['MA34'].iloc[j - 1] and sub_df['MA34'].iloc[j] > sub_df['MA34'].iloc[j + 1]:
                # 取区间内MA34最高点
                if sub_df['MA34'].iloc[j] == sub_df['MA34'].max():
                    peak_time = sub_df.index[j]
                    peak_value = sub_df['MA34'].iloc[j]
                    peaks.append((peak_time, peak_value))
    return peaks

def get_min_peak(df, peaks):
    """
    获取386根K线内的所有有效波峰中的最小波峰值。
    """
    if not peaks:
        return None, None
    min_peak = min(peaks, key=lambda x: x[1])
    return min_peak

def convert_to_cst(utc_time):
    """
    将UTC时间转换为北京时间，格式化为 YYYY/MM/DD HH:MM。
    """
    cst = pytz.timezone('Asia/Shanghai')
    if isinstance(utc_time, pd.Timestamp):
        utc_time = utc_time.replace(tzinfo=pytz.utc)
    return utc_time.astimezone(cst).strftime('%Y/%m/%d %H:%M')

def main():
    st.title('USDT交易对筛选器')

    st.write("正在加载符合条件的交易对...")

    if "displayed_results" not in st.session_state:
        st.session_state["displayed_results"] = []

    results_container = st.container()
    progress_container = st.empty()
    status_container = st.empty()

    # 获取符合条件的USDT交易对
    eligible_symbols = get_eligible_symbols()
    total_symbols = len(eligible_symbols)
    st.write(f"成功加载 {total_symbols} 个交易对！")
    
    if total_symbols == 0:
        st.write("未找到符合条件的交易对")
        return

    # 初始化进度条
    progress_bar = st.progress(0)
    
    # 无限循环检测
    while True:
        for idx, symbol in enumerate(eligible_symbols):
            progress_bar.progress((idx + 1) / total_symbols)

            status_container.write(f"正在检测交易对: {symbol}")

            df = fetch_data(symbol, timeframe='30m', limit=386)  # 更新为30分钟K线数据
            if df is not None and len(df) >= 13:
                df = calculate_ma(df)
                df = calculate_macd(df)
                valleys = get_ma34_valleys(df)
                if not valleys:
                    continue
                dead_cross_idx = detect_macd_dead_cross(df)
                if not dead_cross_idx:
                    continue
                peaks = get_effective_ma34_peaks(df, dead_cross_idx)
                if not peaks:
                    continue
                min_peak_time, min_peak_value = get_min_peak(df, peaks)
                if min_peak_time is None or min_peak_value is None:
                    continue
                for valley_time, valley_value in valleys:
                    ma170_valley = df.loc[valley_time]['MA170']
                    if valley_value >= ma170_valley:
                        ma170_peak = df.loc[min_peak_time]['MA170']
                        if min_peak_value <= ma170_peak:  # 确保最小 MA34 波峰值在 MA170 波峰值下方
                            detection_time = datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Shanghai')).strftime('%Y/%m/%d %H:%M')

                            with results_container:
                                st.session_state["displayed_results"].append({
                                    'symbol': symbol,
                                    'valley_value': valley_value,
                                    'valley_time': convert_to_cst(valley_time),
                                    'min_peak_value': min_peak_value,
                                    'min
                                    'min_peak_value': min_peak_value,
                                    'min_peak_time': convert_to_cst(min_peak_time),
                                    'detection_time': detection_time
                                })
                                # 每次显示新的结果，不会覆盖旧结果
                                for result in st.session_state["displayed_results"]:
                                    st.write(f"### 交易对: {result['symbol']}")
                                    st.write(f"波谷值：{result['valley_value']:.13f}, 时间：{result['valley_time']}")
                                    st.write(f"最小波峰值：{result['min_peak_value']:.13f}, 时间：{result['min_peak_time']}")
                                    st.write(f"条件满足时间：{result['valley_time']}")
                                    st.write(f"检测并输出时间: {result['detection_time']}")
                                    st.markdown("---")

            time.sleep(0)  # 每个交易对无停顿

        time.sleep(6)  # 每轮循环之间增加6秒延迟

if __name__ == "__main__":
    main()
