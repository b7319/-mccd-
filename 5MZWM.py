import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time

# 使用API密钥初始化gate.io API
api_key = 'c8e2fb89d031ca42a30ed7b674cb06dc'
api_secret = 'fab0bc8aeebeb31e46238eda033e2b6258e9c9185f262f74d4472489f9f03219'
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

# 获取交易对数据
def fetch_data(symbol, timeframe='5m', days=14):
    if symbol not in exchange.symbols:
        return None
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df['ma5'] = df['close'].rolling(window=5).mean()
        if days == 3:
            df['ma34'] = df['close'].rolling(window=34).mean()
        elif days == 14:
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
    return df['ma170'].min()

# 获取最新价格
def get_latest_price(df):
    return df['close'].iloc[-1]

# 筛选满足条件的交易对并在Streamlit页面展示结果
def display_result(res):
    st.write(f"交易对: {res['symbol']}")
    st.write(f"最小MA34波峰值: {res['min_ma34_peak']}")
    st.write(f"最新MA170值: {res['ma170_latest']}")
    st.write(f"最新价格: {res['latest_price']}")
    st.write(f"条件满足时间: {res['time_detected']}")
    st.write("---")

# 主逻辑
def main():
    st.title('交易对MA34波峰和MA170最新值检测')

    # 筛选日交易额大于1000万USDT的现货交易对
    high_volume_symbols = []
    for symbol in exchange.symbols:
        if '/USDT' in symbol:
            try:
                ticker = exchange.fetch_ticker(symbol)
                if ticker['quoteVolume'] >= 10_000_000:
                    high_volume_symbols.append(symbol)
            except Exception as e:
                st.write(f"获取 {symbol} 日交易额时出错: {str(e)}")

    if len(high_volume_symbols) == 0:
        st.warning("未找到符合条件的交易对！")
        return

    st.success("符合条件的交易对加载成功!")

    while True:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for symbol in high_volume_symbols:
            df_3d = fetch_data(symbol, days=3)
            df_14d = fetch_data(symbol, days=14)
            if df_3d is not None and not df_3d.empty and df_14d is not None and not df_14d.empty:
                peaks = find_ma34_peaks(df_3d)
                ma170_min = find_ma170_min(df_14d)
                latest_price = get_latest_price(df_14d)
                if peaks:
                    min_peak_value = min(peaks)
                    condition = ma170_min >= min_peak_value and latest_price > df_14d['ma170'].iloc[-1]
                    if condition:
                        symbol_data = {
                            'symbol': symbol,
                            'min_ma34_peak': min_peak_value,
                            'ma170_latest': df_14d['ma170'].iloc[-1],
                            'latest_price': latest_price,
                            'condition': condition,
                            'time_detected': current_time
                        }
                        display_result(symbol_data)

if __name__ == "__main__":
    main()
