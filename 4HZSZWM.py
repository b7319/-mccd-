import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time

# 使用API密钥初始化gate.io API
api_key = '405876b4bb875f8c780de71e03bb2541'
api_secret = 'e0d65164f4f42867c49d55958242d3abd8f12e5df8704f7c03c27f177779bc9d'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 20000
})

# 尝试加载市场数据，最多重试3次
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

# 筛选日交易额大于1000万USDT的现货交易对
def get_high_volume_symbols():
    symbols = []
    for symbol in exchange.symbols:
        try:
            if '/' in symbol:  # 只保留现货交易对
                ticker = exchange.fetch_ticker(symbol)
                if 'USDT' in symbol and ticker['quoteVolume'] >= 10000000:  # 日交易额筛选条件
                    symbols.append(symbol)
        except ccxt.BaseError as e:
            st.warning(f"获取 {symbol} 数据时出错: {str(e)}")
        except Exception as e:
            st.warning(f"其他错误 - 跳过交易对 {symbol}: {str(e)}")
    return symbols

# 获取交易对数据
def fetch_data(symbol, timeframe='4h', days=60):
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df['ma5'] = df['close'].rolling(window=5).mean()
        if days == 60:
            df['ma34'] = df['close'].rolling(window=34).mean()
        elif days == 130:
            df['ma170'] = df['close'].rolling(window=170).mean()
        if df.isnull().any().any():  # 检查空值
            return None
        return df
    except ccxt.BaseError as e:
        st.warning(f"获取 {symbol} 数据时出错: {str(e)}")
        return None
    except Exception as e:
        st.warning(f"其他错误 - 跳过交易对 {symbol}: {str(e)}")
        return None

# 找出MA34的有效波峰值
def find_ma34_peaks(df):
    peaks = []
    valleys = []
    for i in range(34, len(df) - 1):
        if df['ma34'].iloc[i] > df['ma34'].iloc[i - 1] and df['ma34'].iloc[i] > df['ma34'].iloc[i + 1]:
            peaks.append(i)
        if df['ma34'].iloc[i] < df['ma34'].iloc[i - 1] and df['ma34'].iloc[i] < df['ma34'].iloc[i + 1]:
            valleys.append(i)

    valid_peaks = []
    for peak in peaks:
        left_valley = max([v for v in valleys if v < peak], default=None)
        right_valley = min([v for v in valleys if v > peak], default=None)
        if left_valley is not None and right_valley is not None:
            left_crossing = (df['ma5'][left_valley:peak] < df['ma34'][left_valley:peak]).any() and \
                            (df['ma5'][left_valley:peak] > df['ma34'][left_valley:peak]).any()
            right_crossing = (df['ma5'][peak:right_valley] < df['ma34'][peak:right_valley]).any() and \
                             (df['ma5'][peak:right_valley] > df['ma34'][peak:right_valley]).any()
            if left_crossing or right_crossing:
                valid_peaks.append(df['ma34'].iloc[peak])
    return valid_peaks

# 找出MA170的最低值
def find_ma170_min(df):
    return df['ma170'].min() if df is not None else None

# 获取最新价格
def get_latest_price(df):
    return df['close'].iloc[-1] if df is not None else None

# 实时展示筛选结果
def display_results():
    symbols = get_high_volume_symbols()
    total_symbols = len(symbols)
    
    if not symbols:
        st.warning("没有找到满足条件的交易对")
        return

    st.title('实时交易对检测')
    progress_placeholder = st.empty()
    detected_placeholder = st.empty()

    while True:
        for i, symbol in enumerate(symbols):
            progress_placeholder.write(f"检测进度: {i + 1}/{total_symbols} ({symbol})")
            
            df_60d = fetch_data(symbol, days=60)
            df_130d = fetch_data(symbol, days=130)

            if df_60d is None or df_130d is None:
                st.warning(f"交易对 {symbol} 数据缺失，跳过...")
                continue

            peaks = find_ma34_peaks(df_60d)
            ma170_min = find_ma170_min(df_130d)
            latest_price = get_latest_price(df_130d)

            if peaks and ma170_min is not None:
                min_peak_value = min(peaks)
                condition = ma170_min >= min_peak_value and latest_price > df_130d['ma170'].iloc[-1]
                if condition:
                    detected_placeholder.write(f"---\n交易对: {symbol}\n\n"
                                              f"最小MA34波峰值: {min_peak_value}\n\n"
                                              f"最新MA170值: {df_130d['ma170'].iloc[-1]}\n\n"
                                              f"最新价格: {latest_price}\n\n"
                                              f"条件满足时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n---")

        time.sleep(10)

if __name__ == "__main__":
    display_results()
