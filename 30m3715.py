import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo  # Python 3.9+ 内置支持
import streamlit as st
import asyncio
import os
import pygame

# 初始化 gate.io API
api_key = 'YOUR_API_KEY'  # 替换为你的 API Key
api_secret = 'YOUR_API_SECRET'  # 替换为你的 API Secret
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 初始化 pygame 的音频系统，用于满足条件时播放声音
pygame.mixer.init()

# 加载交易市场数据，最多重试 3 次
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

# 调用初始化市场数据的函数
load_markets_with_retry()

# 异步获取交易对数据
async def fetch_data(symbol, timeframe='30m', days=7):
    if symbol not in exchange.symbols:
        return None
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = await asyncio.get_event_loop().run_in_executor(None, exchange.fetch_ohlcv, symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(ZoneInfo('Asia/Shanghai'))
        df.set_index('timestamp', inplace=True)

        if df.empty:
            st.write(f"{symbol} 在 {timeframe} 时间周期内没有数据。")
            return None

        # 计算移动平均线
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma34'] = df['close'].rolling(window=34).mean()

        return df
    except Exception as e:
        st.write(f"抓取 {symbol} 的数据时出错: {str(e)}")
        await asyncio.sleep(1)
        return None

# 找到有效波峰
def find_effective_peaks(series, window=15, min_threshold=0.002):
    peaks = []
    for i in range(window, len(series) - window):
        peak_value = series.iloc[i]
        surrounding_mean = (series.iloc[i - window:i].mean() + series.iloc[i + 1:i + window + 1].mean()) / 2
        if peak_value == series.iloc[i - window:i + window + 1].max() and peak_value > surrounding_mean + min_threshold * surrounding_mean:
            peaks.append(series.index[i])
    return peaks

# 找到最近的波谷
def find_recent_valley(series, window=5):
    valleys = []
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i - window:i + window + 1].min():
            valleys.append(series.index[i])
    return valleys[-1] if valleys else None

# 检查筛选条件
def check_conditions(df):
    recent_ma7 = df.iloc[-13:]
    ma7_valley_index = find_recent_valley(recent_ma7['ma7'])

    # 将当前时间转换为与 Pandas 时间戳一致的时区
    current_time = datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Shanghai'))

    # 波谷时间是否存在，且距离当前时间不超过 4 小时
    if ma7_valley_index is None or (current_time - ma7_valley_index).total_seconds() > 14400:
        return False, None, None

    ma7_valley_value = recent_ma7['ma7'].loc[ma7_valley_index]
    ma34_peaks_indices = find_effective_peaks(df['ma34'])
    ma34_peaks_values = df['ma34'].loc[ma34_peaks_indices]

    # 检查是否存在任意波峰符合条件
    for time, peak_value in ma34_peaks_values.items():
        if abs(ma7_valley_value - peak_value) / peak_value <= 0.03 and ma7_valley_value > peak_value:
            return True, ma7_valley_index, {time: peak_value}

    return False, None, None

# 展示结果
def display_result(symbol, ma7_value, ma7_time, ma34_values):
    st.write(f"交易对: {symbol}")
    st.write(f"ma7 波谷值: {ma7_value}, 时间: {ma7_time.strftime('%Y-%m-%d %H:%M:%S')}")
    for time, value in ma34_values.items():
        st.write(f"ma34 波峰值: {value}, 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"检测输出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("---")
    pygame.mixer.music.load('D:\\pycharm_study\\y1314.wav')
    pygame.mixer.music.play()

# 检测交易对
async def monitor_symbols(symbols, progress_bar, status_text):
    detected_signals = set()
    while True:
        for index, symbol in enumerate(symbols):
            df = await fetch_data(symbol)
            if df is not None:
                condition_met, ma7_time, ma34_peaks = check_conditions(df)
                if condition_met:
                    signal_id = (symbol, ma7_time)
                    if signal_id not in detected_signals:
                        detected_signals.add(signal_id)
                        display_result(symbol, df['ma7'].loc[ma7_time], ma7_time, ma34_peaks)
            progress_bar.progress((index + 1) / len(symbols))
            status_text.text(f"正在检测交易对: {symbol}")
            await asyncio.sleep(1)
        await asyncio.sleep(60)

# 主函数
async def main():
    st.title('ma7 回踩 ma34 (30分钟)')
    symbols_file = 'D:\\pycharm_study\\symbols.txt'

    if os.path.exists(symbols_file):
        with open(symbols_file, 'r') as file:
            symbols = [line.strip() for line in file if line.strip() in exchange.symbols]
    else:
        st.error(f"文件 '{symbols_file}' 不存在！")
        return

    if not symbols:
        st.warning("未找到有效交易对")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    await monitor_symbols(symbols, progress_bar, status_text)

if __name__ == "__main__":
    asyncio.run(main())
