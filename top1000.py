import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time
import os
import pygame

# ------------------ 配置部分 ------------------
API_KEY = 'YOUR_REAL_API_KEY'
API_SECRET = 'YOUR_REAL_API_SECRET'
SYMBOLS_FILE_PATH = 'D:\\pycharm_study\\symbols.txt'
AUDIO_FILE_PATH = 'D:\\pycharm_study\\y2080.wav'
TIMEFRAMES = ['30m']
DETECTION_INTERVAL = 900

# ------------------ 初始化部分 ------------------
exchange = ccxt.gateio({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'timeout': 20000
})
pygame.mixer.init()

def load_markets_with_retry(max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.warning(f"网络错误，正在重试 ({attempt + 1}/{max_retries})...")
            time.sleep(delay)
        except Exception as e:
            st.error(f"加载市场数据时出错: {str(e)}")
            return None

load_markets_with_retry()

def fetch_data(symbol, timeframe='30m', days=14):
    if symbol not in exchange.symbols:
        return None
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)

        if df.empty:
            return None

        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma34'] = df['close'].rolling(window=34).mean()
        df['ma170'] = df['close'].rolling(window=170).mean()
        df['ma453'] = df['close'].rolling(window=453).mean()
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        return df
    except Exception as e:
        st.error(f"获取数据时出错: {str(e)}")
        return None

def ma_conditions(df):
    recent = df.iloc[-13:]
    ma7_diff = recent['ma7'].diff()
    ma7_turn_points = recent[(ma7_diff.shift(-1) > 0) & (ma7_diff <= 0)]
    if not ma7_turn_points.empty:
        ma7_turn_point_index = ma7_turn_points.index[0]
        condition_ma7_turn_up = True

        ma7_at_turn = recent.loc[ma7_turn_point_index, 'ma7']
        ma170_at_turn = recent.loc[ma7_turn_point_index, 'ma170']
        ma453_at_turn = recent.loc[ma7_turn_point_index, 'ma453']

        condition_above_ma170_or_ma453 = (ma7_at_turn >= ma170_at_turn) or (ma7_at_turn >= ma453_at_turn)
    else:
        condition_ma7_turn_up = False
        condition_above_ma170_or_ma453 = False

    condition_upward = ma7_diff.iloc[-1] > 0
    macd_diff_check = recent['macd'].iloc[-1] > recent['macd'].iloc[-2]
    macd_above_zero = recent['macd'].iloc[-1] >= 0

    if condition_ma7_turn_up and condition_above_ma170_or_ma453 and condition_upward and macd_diff_check and macd_above_zero:
        return True, ma7_turn_point_index
    else:
        return False, None

def display_result(res):
    st.write(f"交易对: {res['symbol']}")
    st.write(f"时间周期: {res['timeframe']}")
    st.write(f"条件满足时间: {res['condition_time']}")
    st.write(f"输出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("条件满足")
    st.write("---")
    if os.path.exists(AUDIO_FILE_PATH):
        pygame.mixer.music.load(AUDIO_FILE_PATH)
        pygame.mixer.music.play()
    else:
        st.error(f"音频文件 '{AUDIO_FILE_PATH}' 不存在！")

# ------------------ 主程序部分 ------------------
def main():
    st.title('30m周期条件检测')

    if os.path.exists(SYMBOLS_FILE_PATH):
        with open(SYMBOLS_FILE_PATH, 'r') as file:
            symbols = [line.strip() for line in file if line.strip() and line.strip() in exchange.symbols]
    else:
        st.error(f"文件 '{SYMBOLS_FILE_PATH}' 不存在！")
        symbols = []

    if len(symbols) == 0:
        st.warning("未在'symbols.txt'中找到有效的交易对")
    else:
        st.success("交易对加载成功!")

        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        displayed_signals = set()

        while True:
            for symbol in symbols:
                progress = 0
                step = 1 / len(TIMEFRAMES)

                status_placeholder.text(f"当前检测货币对: {symbol}")

                for timeframe in TIMEFRAMES:
                    df = fetch_data(symbol, timeframe)
                    if df is not None and not df.empty:
                        ma_result, turn_time = ma_conditions(df)
                        if ma_result:
                            ma7_latest = df['ma7'].iloc[-1]
                            ma7_previous = df['ma7'].iloc[-2]
                            macd_latest = df['macd'].iloc[-1]
                            macd_previous = df['macd'].iloc[-2]

                            # 确保展示的是最新检测结果
                            signal_key = (symbol, timeframe, turn_time.strftime('%Y-%m-%d %H:%M:00'))
                            if signal_key not in displayed_signals:
                                displayed_signals.add(signal_key)
                                symbol_data = {
                                    'symbol': symbol,
                                    'timeframe': timeframe,
                                    'condition_time': turn_time.strftime('%Y-%m-%d %H:%M:%S')
                                }
                                display_result(symbol_data)

                    progress += step
                    progress_placeholder.progress(min(progress, 1.0))

                time.sleep(0.1)

            st.write(f"等待{DETECTION_INTERVAL}秒后进行下一次检测...")
            time.sleep(DETECTION_INTERVAL)

if __name__ == "__main__":
    main()
