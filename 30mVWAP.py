import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import asyncio
import os
import pygame

# 初始化 Gate.io API
api_key = 'YOUR_API_KEY'  # 替换为你的 API Key
api_secret = 'YOUR_API_SECRET'  # 替换为你的 API Secret
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 初始化 pygame 的音频系统，用于播放警报声音
pygame.mixer.init()

# 初始化交易市场数据，加载市场列表
def load_markets_with_retry():
    """尝试加载交易所市场，最多重试 3 次"""
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

# 获取指定交易对的历史数据
async def fetch_data(symbol, timeframe='30m', days=7):
    """从交易所获取交易对的历史 K 线数据"""
    if symbol not in exchange.symbols:
        return None
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = await asyncio.get_event_loop().run_in_executor(None, exchange.fetch_ohlcv, symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
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

# 计算密集成交区（VWAP）
def calculate_vwap(df, days=7):
    """计算成交量加权均价 (VWAP)"""
    recent_data = df[-days * 48:]  # 假设每小时2根K线，取7天的数据
    if recent_data.empty:
        return None
    vwap = (recent_data['close'] * recent_data['volume']).sum() / recent_data['volume'].sum()
    return vwap

# 检查 MA7 波谷信号
def check_conditions(df):
    """检查 MA7 是否形成波谷，并满足密集成交区条件"""
    recent_ma7 = df.iloc[-13:]
    ma7_valley_index = recent_ma7['ma7'].idxmin()
    ma7_valley_value = recent_ma7['ma7'].min()

    if len(recent_ma7) < 3:
        return False, None, None, None
    if ma7_valley_index not in recent_ma7.index[1:-1]:
        return False, None, None, None
    if recent_ma7.loc[ma7_valley_index, 'ma7'] >= recent_ma7.iloc[-2]['ma7']:
        return False, None, None, None

    vwap = calculate_vwap(df)
    if vwap is None:
        return False, None, None, None

    if ma7_valley_value > vwap and abs(ma7_valley_value - vwap) / vwap <= 0.03:
        return True, ma7_valley_index, ma7_valley_value, vwap

    return False, None, None, None

# 存储已触发的条件
triggered_conditions = set()

# 显示结果并播放警报
def display_result(res):
    """显示结果并播放警报"""
    st.write(f"交易对: {res['symbol']}")
    st.write(f"时间周期: {res['timeframe']}")
    st.write(f"MA7波谷对应时间点: {res['condition_time']}")
    st.write(f"密集成交区价格（VWAP）: {res['vwap']:.13f}")
    st.write(f"MA7波谷值: {res['ma7_valley']:.13f}")
    st.write(f"检测到信号时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("满足条件")
    st.write("---")
    pygame.mixer.music.load('D:\\\\pycharm_study\\\\y1314.wav')
    pygame.mixer.music.play()

# 监控交易对
async def monitor_symbols(symbols, progress_bar, status_text):
    """实时监控交易对，筛选满足条件的信号"""
    num_symbols = len(symbols)

    while True:
        current_time = datetime.now()
        for index, symbol in enumerate(symbols):
            df = await fetch_data(symbol)
            if df is not None and not df.empty:
                condition_met, condition_time, ma7_valley, vwap = check_conditions(df)
                if condition_met:
                    condition_key = (symbol, condition_time)  # 唯一标识
                    if condition_key not in triggered_conditions:
                        triggered_conditions.add(condition_key)
                        symbol_data = {
                            'symbol': symbol,
                            'timeframe': '30分钟',
                            'condition_time': condition_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'ma7_valley': ma7_valley,
                            'vwap': vwap
                        }
                        display_result(symbol_data)

            progress_bar.progress((index + 1) / num_symbols)
            status_text.text(f"正在检测交易对: {symbol}")

            await asyncio.sleep(1)

        elapsed_time = (datetime.now() - current_time).total_seconds()
        await asyncio.sleep(max(0, 60 - elapsed_time))

# 主函数
async def main():
    st.title('MA7波谷与密集成交区检测 (30分钟)')

    symbols_file_path = 'D:\\\\pycharm_study\\\\symbols.txt'

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
        status_text = st.empty()
        await monitor_symbols(symbols, progress_bar, status_text)

if __name__ == "__main__":
    asyncio.run(main())
