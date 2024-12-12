import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time
import pytz

# 初始化Gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 20000
})

# 加载市场数据，确保API连接成功
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

# 获取符合交易额大于500万USDT的交易对
def get_symbols_with_volume():
    try:
        tickers = exchange.fetch_tickers()
        symbols = [
            symbol for symbol, ticker in tickers.items()
            if 'USDT' in symbol and ticker.get('quoteVolume', 0) >= 5000000
        ]
        return symbols
    except Exception as e:
        st.write(f"获取高交易额交易对时出错: {str(e)}")
        return []

# 获取指定交易对在特定时间段的K线数据
def fetch_data(symbol, timeframe='1m', since=None):
    if symbol not in exchange.symbols:
        return None
    try:
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.write(f"获取 {symbol} 数据时出错: {str(e)}")
        return None

# 计算简单移动平均线
def calculate_ma(df, periods):
    for period in periods:
        df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    return df

# 判断均线是否接近形成密集（0.3%以内）
def is_moving_averages_cluster(df, tolerance=0.003):
    ma34 = df['ma34'].iloc[-1]
    ma170 = df['ma170'].iloc[-1]
    ma453 = df['ma453'].iloc[-1]
    return (
        abs(ma34 - ma170) / ma170 < tolerance and 
        abs(ma170 - ma453) / ma453 < tolerance and 
        abs(ma34 - ma453) / ma453 < tolerance
    )

# 判断均线是否发生发散（向上或向下）
def check_divergence(df):
    ma34 = df['ma34'].iloc[-1]
    ma170 = df['ma170'].iloc[-1]
    ma453 = df['ma453'].iloc[-1]
    prev_ma34 = df['ma34'].iloc[-2]
    prev_ma170 = df['ma170'].iloc[-2]
    prev_ma453 = df['ma453'].iloc[-2]

    # 向上发散
    if ma34 > ma170 > ma453 and prev_ma34 <= prev_ma170 <= prev_ma453:
        return 'upward'
    # 向下发散
    elif ma34 < ma170 < ma453 and prev_ma34 >= prev_ma170 >= prev_ma453:
        return 'downward'
    return None

# 转换为北京时间
def convert_to_beijing_time(utc_time):
    utc_time = utc_time.replace(tzinfo=timezone.utc)
    beijing_time = utc_time.astimezone(pytz.timezone('Asia/Shanghai'))
    return beijing_time

# 主逻辑
def main():
    st.title('实时交易检测')

    # 页面布局设置
    progress_bar = st.progress(0)
    current_symbol_placeholder = st.empty()
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.header("符合条件的交易对")
    with col2:
        st.header("不符合条件的交易对")

    # 获取符合条件的交易对
    symbols = get_symbols_with_volume()
    if not symbols:
        st.warning("未找到符合条件的高交易额交易对")
        return

    st.success(f"加载 {len(symbols)} 个交易对进行检测")

    for idx, symbol in enumerate(symbols):
        # 更新进度条和检测状态
        progress = (idx + 1) / len(symbols)
        progress_bar.progress(progress)
        current_symbol_placeholder.write(f"正在检测交易对: {symbol} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        # 获取过去3小时的K线数据
        since_3h = exchange.parse8601((datetime.now(timezone.utc) - timedelta(hours=3)).isoformat())
        df = fetch_data(symbol, '1m', since=since_3h)

        if df is not None and not df.empty:
            periods = [34, 170, 453]
            df = calculate_ma(df, periods)

            # 检查是否是MA密集形态
            if is_moving_averages_cluster(df):
                # 检查发散方向
                divergence = check_divergence(df)
                if divergence:
                    # 时间转换为北京时间
                    ma34_peak_time = convert_to_beijing_time(df.index[-1])

                    with col1:
                        st.write(f"交易对: {symbol}")
                        st.write(f"均线密集形态，发散方向：{divergence}")
                        st.write(f"MA34, MA170, MA453密集时间：{ma34_peak_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write("---")
                else:
                    with col2:
                        st.write(f"交易对: {symbol}")
                        st.write(f"均线密集形态，无发散")
                        st.write("---")
            else:
                with col2:
                    st.write(f"交易对: {symbol}")
                    st.write(f"未发现均线密集形态")
                    st.write("---")
        time.sleep(3)
    st.warning("所有交易对检测完成，等待下一个检测周期。")

if __name__ == '__main__':
    main()
