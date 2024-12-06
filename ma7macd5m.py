import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import asyncio
import os
import pygame
import time

# 初始化 gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 初始化 pygame 音频系统，用于播放声音
pygame.mixer.init()

# 初始化交易市场
def load_markets_with_retry():
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.write(f"网络错误，正在重试 ({attempt + 1}/3)...")
            asyncio.sleep(5)
        except Exception as e:
            st.write(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

# 获取数据并计算MA7、MA170和MACD
def fetch_data(symbol, timeframe='5m', max_bars=1000):
    if symbol not in exchange.symbols:
        return None

    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(minutes=max_bars * 5)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)

        if df.empty:
            st.write(f"{symbol} 在 {timeframe} 时间周期内没有数据。")
            return None

        # 计算MA7和MA170
        df['ma7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['ma170'] = df['close'].rolling(window=170, min_periods=1).mean()

        # 计算MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        return df
    except Exception as e:
        error_message = f"抓取 {symbol} 的数据时出错: {str(e)}"
        st.write(error_message)
        return None

# 检查条件是否满足
def check_conditions(df):
    if df is None or len(df) < 3:
        return False, None

    last_3 = df.iloc[-3:]
    ma7 = last_3['ma7']
    ma170 = last_3['ma170']
    macd = last_3['macd']
    macd_signal = last_3['macd_signal']

    # 条件1: MA7拐头向上且在MA170之上
    ma7_up = (ma7.iloc[0] < ma7.iloc[1] and ma7.iloc[1] >= ma7.iloc[2]) and (ma7.iloc[-1] > ma170.iloc[-1])

    # 条件2: MACD金叉发生在零轴之上
    macd_golden_cross = (macd.iloc[-2] < macd_signal.iloc[-2] and macd.iloc[-1] >= macd_signal.iloc[-1]) and (macd.iloc[-1] > 0)

    # 检查条件是否同时满足
    if ma7_up and macd_golden_cross:
        return True, last_3.index[-1]

    return False, None

valid_signals = set()

def display_result(res):
    st.write(f"交易对: {res['symbol']}")
    st.write(f"时间周期: {res['timeframe']}")
    st.write(f"满足条件的时间: {res['condition_time']}")
    st.write(f"输出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("满足条件")
    st.write("---")
    pygame.mixer.music.load('D:\\pycharm_study\\y2135.wav')
    pygame.mixer.music.play()

async def monitor_symbols(symbols, progress_bar, status_text):
    num_symbols = len(symbols)

    while True:
        for index, symbol in enumerate(symbols):
            df = fetch_data(symbol)
            if df is not None and not df.empty:
                condition_met, condition_time = check_conditions(df)
                if condition_met:
                    signal_key = (symbol, condition_time.strftime('%Y-%m-%d %H:%M:%S'))
                    if signal_key not in valid_signals:
                        valid_signals.add(signal_key)
                        symbol_data = {
                            'symbol': symbol,
                            'timeframe': '5分钟',
                            'condition_time': condition_time.strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        display_result(symbol_data)

            progress_bar.progress((index + 1) / num_symbols)
            status_text.text(f"正在检测交易对: {symbol}")

            # 每个货币对检测间隔3秒
            time.sleep(3)

        # 每轮检测完成后等待3分钟（后台处理，无输出到页面）
        progress_bar.progress(0)
        time.sleep(180)  # 每轮检测间隔3分钟

async def main():
    st.title('MA 和 MACD 筛选 (5分钟)')

    symbols_file_path = 'D:\\pycharm_study\\TOP100.txt'

    if os.path.exists(symbols_file_path):
        with open(symbols_file_path, 'r') as file:
            symbols = [line.strip() for line in file if line.strip() and line.strip() in exchange.symbols]
    else:
        st.error(f"文件 '{symbols_file_path}' 不存在！")
        symbols = []

    if not symbols:
        st.warning("未在'TOP100.txt'中找到有效的交易对")
    else:
        st.success("交易对加载成功！")
        progress_bar = st.progress(0)
        status_text = st.empty()
        await monitor_symbols(symbols, progress_bar, status_text)

if __name__ == "__main__":
    asyncio.run(main())
