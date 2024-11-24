import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import asyncio
import time
import requests
import pygame
import io

# 初始化 gate.io 的 API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 68686
})

# GitHub 音频文件 URL
audio_url = 'https://raw.githubusercontent.com/b7319/-mccd-/main/y1314.wav'

# 全局记录已显示的符号及其 MA7 波谷值
displayed_symbols = {}

def load_markets_with_retry():
    """加载市场数据，带重试机制"""
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.write(f"网络错误，重试中（{attempt + 1}/3）...") 
            time.sleep(5)
        except Exception as e:
            st.write(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

def fetch_and_filter_usdt_pairs():
    """筛选出交易量超过 1,000,000 的 USDT 交易对"""
    valid_pairs = []
    usdt_pairs = [symbol for symbol in exchange.symbols if symbol.endswith('/USDT')]

    for symbol in usdt_pairs:
        try:
            ticker = exchange.fetch_ticker(symbol)
            volume_quote = ticker['quoteVolume']
            if volume_quote > 1000000:
                valid_pairs.append(symbol)
        except ccxt.BaseError as e:
            if 'INVALID_CURRENCY' in str(e):
                st.write(f"{symbol} 已下架，跳过。")
            else:
                st.write(f"获取 {symbol} 数据时出错: {str(e)}")

    return valid_pairs

def calculate_ma(df, period):
    """计算移动平均值"""
    return df['close'].rolling(window=period, min_periods=1).mean()

def find_recent_valley(df, column='ma7'):
    """找出最近24小时内的MA7波谷"""
    recent_time = df.index[-1] - timedelta(hours=24)
    recent_data = df[df.index >= recent_time]
    if recent_data.empty:
        return None, None
    valley_value = recent_data[column].min()
    valley_index = recent_data[column].idxmin()
    return valley_index, valley_value

def find_valid_peaks(df, column='ma34', threshold=0.005):
    """找到显著的 MA34 波峰，排除无效波峰"""
    peaks = []
    for i in range(1, len(df) - 1):
        if df[column][i] > df[column][i - 1] and df[column][i] > df[column][i + 1]:
            peak_value = df[column][i]
            peak_index = df.index[i]
            # 阈值控制，避免波动造成的重复波峰
            if peaks and abs(peak_value - peaks[-1][1]) / peaks[-1][1] < threshold:
                continue
            peaks.append((peak_index, peak_value))
    return peaks

def fetch_data(symbol, timeframe='4h', days=68):
    """获取数据并计算MA7和MA34"""
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
    ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
    df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
    df.set_index('timestamp', inplace=True)
    df['ma7'] = calculate_ma(df, 7)
    df['ma34'] = calculate_ma(df, 34)
    return df

def play_sound():
    """播放提示音"""
    response = requests.get(audio_url)
    if response.status_code == 200:
        # 使用 pygame 播放音频
        pygame.mixer.init()
        pygame.mixer.music.load(io.BytesIO(response.content))  # 通过 BytesIO 加载音频数据
        pygame.mixer.music.play()
    else:
        st.warning("音频文件无法下载。")

async def monitor_symbols():
    global displayed_symbols
    symbols = fetch_and_filter_usdt_pairs()

    total_symbols = len(symbols)
    st.title('实时交易对检测')
    progress_bar = st.progress(0)  # 添加检测进度条
    status_text = st.empty()  # 显示当前检测进度

    while True:
        for index, symbol in enumerate(symbols):
            try:
                df = fetch_data(symbol)
                valley_time, valley_value = find_recent_valley(df, 'ma7')
                if valley_time and symbol not in displayed_symbols:
                    peaks = find_valid_peaks(df, 'ma34')
                    for peak_time, peak_value in peaks:
                        if abs((valley_value - peak_value) / peak_value) <= 0.03 and valley_value > peak_value:
                            displayed_symbols[symbol] = valley_value
                            st.write(f"交易对: {symbol}")
                            st.write(f"MA7波谷值: {valley_value} (时间: {valley_time})")
                            st.write(f"MA34波峰值: {peak_value} (时间: {peak_time})")
                            st.write(f"检测时间: {datetime.now()}")
                            st.write("---")
                            # 播放音频
                            play_sound()
                            break  # 找到一个符合条件的波峰后不再继续查找

                # 更新检测进度
                progress = (index + 1) / total_symbols
                progress_bar.progress(progress)
                status_text.text(f"检测进度: {index + 1}/{total_symbols} - 当前检测: {symbol}")

                # 每个货币对检测的间隔为 2 秒（减少请求频率）
                await asyncio.sleep(2)

            except Exception as e:
                st.write(f"{symbol} 错误: {str(e)}")

        # 每次检测完成后等待 60 秒再循环
        await asyncio.sleep(60)

# 主函数
async def main():
    await monitor_symbols()

# 运行程序
asyncio.run(main())
