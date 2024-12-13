import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time

# 初始化 gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 加载市场数据
def load_markets_with_retry():
    """
    加载市场数据，最多重试 3 次。
    """
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.write(f"网络错误，正在重试 ({attempt + 1}/3)...")
            time.sleep(5)
        except Exception as e:
            st.write(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

# 获取所有日交易量大于500万USDT的交易对
def get_high_volume_symbols():
    """
    筛选出所有日交易量大于 500 万 USDT 的交易对。
    """
    high_volume_symbols = []
    try:
        tickers = exchange.fetch_tickers()
        for symbol, data in tickers.items():
            if 'quoteVolume' in data and data['quoteVolume'] >= 5_000_000:
                high_volume_symbols.append(symbol)
    except Exception as e:
        st.write(f"获取高交易量交易对时出错: {str(e)}")
    return high_volume_symbols

# 获取数据并计算MA、MACD等指标
def fetch_data(symbol, timeframe='30m', max_bars=1000):
    """
    获取指定交易对的历史数据，并计算技术指标（MA 和 MACD）。
    
    :param symbol: 交易对符号
    :param timeframe: 时间周期
    :param max_bars: 最大数据条数
    :return: 包含技术指标的 DataFrame
    """
    if symbol not in exchange.symbols:
        return None

    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(minutes=max_bars * 30)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)

        if df.empty:
            st.write(f"{symbol} 在 {timeframe} 时间周期内没有数据。")
            return None

        # 计算MA
        df['ma7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['ma34'] = df['close'].rolling(window=34, min_periods=1).mean()
        df['ma170'] = df['close'].rolling(window=170, min_periods=1).mean()
        df['ma453'] = df['close'].rolling(window=453, min_periods=1).mean()

        # 计算MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_dif'] = ema12 - ema26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()

        return df
    except Exception as e:
        error_message = f"抓取 {symbol} 的数据时出错: {str(e)}"
        st.write(error_message)
        return None

# 筛选条件
def check_conditions(df):
    """
    检查交易对是否满足筛选条件。
    
    :param df: 包含技术指标的 DataFrame
    :return: 是否满足条件以及条件满足的时间
    """
    if df is None or len(df) < 13:
        return False, None

    last_13 = df.iloc[-13:]

    # 检查 MA7 波谷条件
    last_9 = df.iloc[-9:]
    ma7_changes = last_9['ma7'].diff()
    ma7_valley = (ma7_changes.iloc[-2] < 0 and ma7_changes.iloc[-1] > 0 and all(last_9['ma7'] > last_9['ma34']))

    if not ma7_valley:
        return False, None

    # 检查 MACD 条件
    macd_above_zero = all(last_9['macd_dif'] > 0) and all(last_9['macd_dea'] > 0)

    if not macd_above_zero:
        return False, None

    # 检查 MA34 上穿条件
    ma34_cross = (last_13['ma34'].iloc[-2] < last_13['ma170'].iloc[-2] and last_13['ma34'].iloc[-1] >= last_13['ma170'].iloc[-1]) or \
                 (last_13['ma34'].iloc[-2] < last_13['ma453'].iloc[-2] and last_13['ma34'].iloc[-1] >= last_13['ma453'].iloc[-1])

    if not ma34_cross:
        return False, None

    return True, df.index[-1]

valid_signals = set()

# 显示结果
def display_result(res):
    """
    显示满足条件的交易对信息。
    
    :param res: 包含交易对信息的字典
    """
    st.write(f"交易对: {res['symbol']}")
    st.write(f"时间周期: {res['timeframe']}")
    st.write(f"满足条件的时间: {res['condition_time']}")
    st.write(f"输出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("满足条件")
    st.write("---")

# 监控交易对
def monitor_symbols(symbols):
    """
    循环检测交易对是否满足筛选条件。
    
    :param symbols: 交易对列表
    """
    num_symbols = len(symbols)
    progress_bar = st.progress(0)  # 初始化进度条
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
                            'timeframe': '30分钟',
                            'condition_time': condition_time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        display_result(symbol_data)

            progress_bar.progress((index + 1) / num_symbols)
            time.sleep(3)  # 每个交易对间隔 3 秒
        progress_bar.progress(0)  # 完成一轮检测后重置进度条

# 主程序
def main():
    """
    主程序入口。
    """
    st.title('MA34金叉MA453 筛选系统')

    symbols = get_high_volume_symbols()

    if not symbols:
        st.warning("未找到满足条件的高交易量交易对")
    else:
        st.success(f"成功加载 {len(symbols)} 个交易对！")
        monitor_symbols(symbols)

if __name__ == "__main__":
    main()
