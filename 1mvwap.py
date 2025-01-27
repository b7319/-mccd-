import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import asyncio

# 初始化 Gate.io API
api_key = 'your_api_key'  # 替换为你的 API Key
api_secret = 'your_api_secret'  # 替换为你的 API Secret
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

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
async def fetch_data(symbol, timeframe='1m', minutes=180):
    """
    从交易所获取交易对的历史 K 线数据
    参数：
        symbol: 交易对名称，例如 'BTC/USDT'
        timeframe: 时间周期，例如 '1m'
        minutes: 获取最近多少分钟的数据
    返回：
        pandas.DataFrame 格式的 K 线数据，包含时间、开高低收和成交量
    """
    if symbol not in exchange.symbols:
        return None
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat())
        ohlcvs = await asyncio.get_event_loop().run_in_executor(None, exchange.fetch_ohlcv, symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)

        if df.empty:
            st.write(f"{symbol} 在 {timeframe} 时间周期内没有数据。")
            return None

        # 计算移动平均线
        df['ma7'] = df['close'].rolling(window=7).mean()
        return df
    except Exception as e:
        st.write(f"抓取 {symbol} 的数据时出错: {str(e)}")
        await asyncio.sleep(1)
        return None

# 计算密集成交区（VWAP）
def calculate_vwap(df):
    """
    计算 VWAP 值
    """
    if df.empty:
        return None
    vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
    return round(vwap, 13)

# 检查 MA7 波谷与 VWAP 的距离
def check_conditions(df):
    if len(df) < 13:
        return False, None, None, None

    # 计算 VWAP
    vwap = calculate_vwap(df)
    if vwap is None:
        return False, None, None, None

    # 检查 MA7 的波谷
    ma7_values = df['ma7'][-13:]
    ma7_valley = ma7_values.min()  # 波谷值
    valley_index = ma7_values.idxmin()  # 波谷时间点

    # 比较波谷与 VWAP 的距离
    valley_distance_condition = abs(ma7_valley - vwap) / vwap <= 0.03
    if valley_distance_condition:
        return True, valley_index, ma7_valley, vwap

    return False, None, None, None

# 存储已触发的条件和上次信号的时间
triggered_conditions = {}
last_trigger_times = {}

def display_result(res):
    st.write(f"交易对: {res['symbol']}")
    st.write(f"波谷时间点: {res['condition_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"VWAP 值: {res['vwap']:.13f}")
    st.write(f"MA7 波谷值: {res['ma7_valley']:.13f}")
    st.write(f"检测到信号时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("---")

async def monitor_symbols(symbols, progress_bar, status_text):
    num_symbols = len(symbols)
    while True:
        for index, symbol in enumerate(symbols):
            current_time = datetime.now()
            df = await fetch_data(symbol)
            if df is not None and not df.empty:
                condition_met, condition_time, ma7_valley, vwap = check_conditions(df)
                if condition_met:
                    last_trigger_time = last_trigger_times.get(symbol, None)
                    if last_trigger_time is None or (current_time - last_trigger_time).total_seconds() > 1800:
                        last_trigger_times[symbol] = current_time
                        condition_key = (symbol, condition_time)
                        if condition_key not in triggered_conditions:
                            triggered_conditions[condition_key] = current_time
                            symbol_data = {
                                'symbol': symbol,
                                'condition_time': condition_time,
                                'ma7_valley': ma7_valley,
                                'vwap': vwap
                            }
                            display_result(symbol_data)

            progress_bar.progress((index + 1) / num_symbols)
            status_text.text(f"正在检测交易对: {symbol}")
            await asyncio.sleep(3)

async def main():
    st.title('VWAP 和 MA7 波谷检测 (1分钟级别)')

    symbols = [
        'XRP5L/USDT', 'DOGE5L/USDT', 'SHIB5L/USDT', 'LINK5L/USDT',
        'GALA5L/USDT', 'UNI5L/USDT', 'ETH5L/USDT', 'DOT5L/USDT',
        'LTC5L/USDT', 'EOS5L/USDT', 'BTC5L/USDT', 'BCH5L/USDT',
        'BSV5L/USDT', 'AXS5L/USDT', 'XRP5S/USDT', 'DOGE5S/USDT',
        'SHIB5S/USDT', 'LINK5S/USDT', 'DOT5S/USDT', 'UNI5S/USDT',
        'ETH5S/USDT', 'EOS5S/USDT', 'GALA5S/USDT', 'LTC5S/USDT',
        'BTC5S/USDT', 'BCH5S/USDT', 'BSV5S/USDT', 'AXS5S/USDT'
    ]

    progress_bar = st.progress(0)
    status_text = st.empty()
    await monitor_symbols(symbols, progress_bar, status_text)

if __name__ == "__main__":
    asyncio.run(main())
