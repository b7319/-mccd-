import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import time
import os

# 使用API密钥初始化gate.io API
api_key = '405876b4bb875f8c780de71e03bb2541'
api_secret = 'e0d65164f4f42867c49d55958242d3abd8f12e5df8704f7c03c27f177779bc9d'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 20000
})

# 尝试加载市场数据，最多重试3次
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

# 获取交易对数据
def fetch_data(symbol, timeframe='4h', days=130):
    if symbol not in exchange.symbols:
        return None
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df['ma34'] = df['close'].rolling(window=34).mean()
        df['ma170'] = df['close'].rolling(window=170).mean()
        return df
    except Exception as e:
        st.write(f"获取 {symbol} 数据时出错: {str(e)}")
        return None

# 筛选符合条件的货币对
def find_golden_cross(df):
    golden_crosses = []

    for i in range(1, len(df)):
        if df['ma34'].iloc[i] > df['ma170'].iloc[i] and df['ma34'].iloc[i - 1] <= df['ma170'].iloc[i - 1] and df['close'].iloc[i] > df['ma170'].iloc[i]:
            golden_crosses.append({
                'timestamp': df.index[i],
                'price': df['close'].iloc[i],
                'kline_count': len(df) - i - 1  # 距离当前时间的K线数量
            })

    return golden_crosses

# 更新筛选结果
def update_results(symbols, results):
    new_results = []

    for symbol in symbols:
        df = fetch_data(symbol)
        if df is not None and not df.empty:
            golden_crosses = find_golden_cross(df)
            for cross in golden_crosses:
                if cross['kline_count'] <= 6 and df['close'].iloc[-1] > df['ma170'].iloc[-1]:  # 修改后的条件
                    if symbol not in results or cross['timestamp'] > results[symbol]['timestamp']:
                        results[symbol] = cross
                        new_results.append({
                            'symbol': symbol,
                            'timestamp': cross['timestamp'],
                            'price': cross['price'],
                            'kline_count': cross['kline_count']
                        })

    return new_results


# 在Streamlit页面展示结果
def display_results(results):
    if results:
        for res in results:
            st.write(f"交易对: {res['symbol']}\n")
            st.write(f"金叉发生时间: {res['timestamp']}\n")
            st.write("---")
    else:
        st.write("没有符合条件的数据")

if __name__ == "__main__":
    st.title('交易对MA34金叉MA170检测')

    symbols_file_path = 'D:\\pycharm_study\\symbols.txt'
    results_file_path = 'D:\\pycharm_study\\results.csv'

    if os.path.exists(symbols_file_path):
        with open(symbols_file_path, 'r') as file:
            symbols = [line.strip() for line in file if line.strip() and line.strip() in exchange.symbols]
    else:
        st.error(f"文件 '{symbols_file_path}' 不存在！")
        symbols = []

    if os.path.exists(results_file_path):
        try:
            results_df = pd.read_csv(results_file_path, index_col='symbol')
            if 'timestamp' not in results_df.columns:
                results_df['timestamp'] = pd.NaT
            results = results_df.to_dict('index')
        except Exception as e:
            st.error(f"读取结果文件时出错: {str(e)}")
            results = {}
    else:
        results = {}

    if len(symbols) == 0:
        st.warning("未在'symbols.txt'中找到有效的交易对")
    else:
        st.success("交易对加载成功!")
        while True:
            new_results = update_results(symbols, results)
            if new_results:
                display_results(new_results)
                pd.DataFrame(results).to_csv(results_file_path)
            time.sleep(14400)  # 每4小时运行一次