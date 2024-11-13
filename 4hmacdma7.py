import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import asyncio
import os
import pygame

# 初始化 gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 初始化 pygame 音频系统，用于播放音频
pygame.mixer.init()

# 初始化 Exchange Markets
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

# 获取数据
def fetch_data(symbol, timeframe='4h', days=30):
    if symbol not in exchange.symbols:
        return None
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)

        if df.empty:
            st.write(f"{symbol} 在 {timeframe} 时间周期内没有数据。")
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
        df['dif'] = df['ema12'] - df['ema26']

        return df
    except Exception as e:
        st.write(f"抓取 {symbol} 的数据时出错: {str(e)}")
        return None

# 判断是否向上转折
def is_turning_up(series):
    diffs = series.diff().dropna()
    for i in range(1, len(diffs)):
        if diffs.iloc[i-1] < 0 and diffs.iloc[i] > 0:
            return series.index[i]
    return None

# 检查条件
def check_conditions(df):
    recent = df.iloc[-7:]  # 最新的七根K线
    ma7_turn_point = is_turning_up(recent['ma7'])  # 确定ma7具体向上转折的时间

    # 检查 MA7 和 MA170/MA453 条件
    ma7_condition = (
        ma7_turn_point is not None and
        (recent.loc[ma7_turn_point, 'ma7'] >= recent.loc[ma7_turn_point, 'ma170'] or recent.loc[ma7_turn_point, 'ma7'] >= recent.loc[ma7_turn_point, 'ma453']) and
        recent['ma7'].diff().iloc[-1] > 0  # 确保当前斜率向上
    )

    # 检查 DIF 条件
    dif_condition = (
        recent['dif'].iloc[-1] >= 0 and
        is_turning_up(recent['dif']) is not None and
        recent['dif'].diff().iloc[-1] > 0  # 确保 DIF 的斜率向上
    )

    combined_condition = ma7_condition and dif_condition

    if combined_condition:
        return True, ma7_turn_point
    return False, None

valid_symbols = set()

# 显示结果
def display_result(res):
    st.write(f"交易对: {res['symbol']}")
    st.write(f"时间周期: {res['timeframe']}")
    st.write(f"满足条件的时间: {res['condition_time']}")
    st.write(f"输出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("满足条件")
    st.write("---")
    pygame.mixer.music.load('D:\\pycharm_study\\y2080.wav')
    pygame.mixer.music.play()

# 监控交易对
async def monitor_symbols(symbols, progress_bar, status_text):
    num_symbols = len(symbols)

    while True:
        for index, symbol in enumerate(symbols):
            df = fetch_data(symbol)
            if df is not None and not df.empty:
                condition_met, condition_time = check_conditions(df)
                if condition_met:
                    symbol_data_tuple = (symbol, condition_time.strftime('%Y-%m-%d %H:%M:%S'))
                    if symbol_data_tuple not in valid_symbols:
                        valid_symbols.add(symbol_data_tuple)
                        symbol_data = {
                            'symbol': symbol,
                            'timeframe': '4小时',
                            'condition_time': condition_time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        display_result(symbol_data)

            # 更新进度条和状态文本
            progress_bar.progress((index + 1) / num_symbols)
            status_text.text(f"正在检测交易对: {symbol}")
            await asyncio.sleep(3)  # 每个交易对检测之间等待3秒

        await asyncio.sleep(600 - 3*num_symbols)  # 确保整体检测循环在约10分钟内完成

# 主函数
async def main():
    st.title('单时间周期条件检测 (4小时)')

    symbols_file_path = 'D:\\pycharm_study\\symbols.txt'

    if os.path.exists(symbols_file_path):
        with open(symbols_file_path, 'r') as file:
            symbols = [line.strip() for line in file if line.strip() and line.strip() in exchange.symbols]
    else:
        st.error(f"文件 '{symbols_file_path}' 不存在！")
        symbols = []

    if not symbols:
        st.warning("未在'symbols.txt'中找到有效的交易对")
    else:
        st.success("交易对加载成功！")
        progress_bar = st.progress(0)
        status_text = st.empty()  # Initialise a placeholder for status text
        await monitor_symbols(symbols, progress_bar, status_text)

if __name__ == "__main__":
    asyncio.run(main())
