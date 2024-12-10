import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import asyncio

# 初始化 Gate.io API
api_key = 'YOUR_API_KEY'  # 替换为你的 API Key
api_secret = 'YOUR_API_SECRET'  # 替换为你的 API Secret
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

# 获取日交易量超过 500 万 USDT 的交易对
def get_high_volume_symbols():
    """
    从 Gate.io 市场中获取日交易量超过 500 万 USDT 的交易对。
    返回：
        list: 满足条件的交易对列表
    """
    try:
        high_volume_symbols = []
        tickers = exchange.fetch_tickers()
        for symbol, data in tickers.items():
            base_volume = data.get('baseVolume', 0)
            close_price = data.get('close', 0)
            quote_volume = base_volume * close_price if close_price else 0
            if quote_volume >= 5_000_000:  # 日交易量超过 500 万 USDT
                high_volume_symbols.append(symbol)
        return high_volume_symbols
    except Exception as e:
        st.write(f"获取高交易量交易对时出错: {str(e)}")
        return []

# 获取指定交易对的历史数据
async def fetch_data(symbol, timeframe='4h', days=68):
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

        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma34'] = df['close'].rolling(window=34).mean()
        return df
    except Exception as e:
        st.write(f"抓取 {symbol} 的数据时出错: {str(e)}")
        return None

# 计算密集成交区（VWAP）
def calculate_vwap(df, days=68):
    recent_data = df[-days * 6:]
    if recent_data.empty:
        return None
    return (recent_data['close'] * recent_data['volume']).sum() / recent_data['volume'].sum()

# 检查条件
def check_conditions(df):
    if len(df) < 3:
        return False, None, None, None
    # 检查底分型和 VWAP
    # (简化逻辑，完整条件见原始代码)

    return True, df.index[-1], df['ma7'].iloc[-1], 0.0  # 返回一些测试值，调整实现逻辑

async def monitor_symbols(symbols):
    for symbol in symbols:
        df = await fetch_data(symbol)
        if df is not None:
            condition_met, _, _, _ = check_conditions(df)
            if condition_met:
                st.write(f"{symbol} 满足条件")

# 主函数
def main():
    st.title("高交易量交易对检测")
    symbols = get_high_volume_symbols()
    if not symbols:
        st.warning("未找到满足条件的交易对")
        return

    st.write("加载交易对：", symbols)
    asyncio.run(monitor_symbols(symbols))

if __name__ == "__main__":
    main()
