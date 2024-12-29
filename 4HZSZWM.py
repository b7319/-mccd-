import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time

# 初始化 gate.io API
api_key = 'your_api_key'
api_secret = 'your_api_secret'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 20000
})

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

# 筛选日交易额大于 1000 万 USDT 的现货交易对
def get_high_volume_symbols():
    markets = exchange.fetch_tickers()
    high_volume_symbols = []
    for symbol, data in markets.items():
        if symbol.endswith('/USDT') and 'spot' in data.get('info', {}).get('type', ''):
            if data['quoteVolume'] and data['quoteVolume'] >= 10_000_000:
                high_volume_symbols.append(symbol)
    return high_volume_symbols

# 获取交易数据
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

# 检查波峰后9条K线内是否有新的波峰或波谷
def is_valid_peak(df, peak):
    for i in range(peak + 1, min(peak + 10, len(df) - 1)):
        if df['ma34'].iloc[i] > df['ma34'].iloc[i - 1] and df['ma34'].iloc[i] > df['ma34'].iloc[i + 1]:
            return False
        if df['ma34'].iloc[i] < df['ma34'].iloc[i - 1] and df['ma34'].iloc[i] < df['ma34'].iloc[i + 1]:
            return False
    return True

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

            if (left_crossing or right_crossing) and is_valid_peak(df, peak):
                valid_peaks.append(df['ma34'].iloc[peak])

    return valid_peaks

# 找出MA170的最低值
def find_ma170_min(df):
    return df['ma170'].min() if not df['ma170'].isna().all() else None

# 获取最新价格
def get_latest_price(df):
    return df['close'].iloc[-1] if not df['close'].isna().all() else None

# 主检测逻辑
def detect_and_display():
    st.title('实时交易对检测')

    high_volume_symbols = get_high_volume_symbols()
    st.write(f"获取成功的现货交易对总数: {len(high_volume_symbols)}")
    progress = st.progress(0)

    result_container = st.container()

    while True:
        results = []
        for idx, symbol in enumerate(high_volume_symbols):
            progress.progress((idx + 1) / len(high_volume_symbols))

            df_60d = fetch_data(symbol, days=60)
            df_130d = fetch_data(symbol, days=130)

            if df_60d is not None and df_130d is not None:
                peaks = find_ma34_peaks(df_60d)
                ma170_min = find_ma170_min(df_130d)
                latest_price = get_latest_price(df_130d)

                if peaks and ma170_min is not None and latest_price is not None:
                    min_peak_value = min(peaks)
                    condition = ma170_min >= min_peak_value and latest_price > df_130d['ma170'].iloc[-1]

                    if condition:
                        results.append({
                            'symbol': symbol,
                            'min_ma34_peak': min_peak_value,
                            'ma170_latest': df_130d['ma170'].iloc[-1],
                            'latest_price': latest_price,
                            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                        })

        with result_container:
            if results:
                st.write(f"检测完成，共找到 {len(results)} 个符合条件的交易对：")
                for res in results:
                    st.write(f"交易对: {res['symbol']}")
                    st.write(f"最小MA34波峰值: {res['min_ma34_peak']}")
                    st.write(f"最新MA170值: {res['ma170_latest']}")
                    st.write(f"最新价格: {res['latest_price']}")
                    st.write(f"条件满足时间: {res['timestamp']}")
                    st.write("---")
            else:
                st.write("未找到符合条件的交易对！")

        time.sleep(60)  # 每分钟检测一次

if __name__ == "__main__":
    detect_and_display()
