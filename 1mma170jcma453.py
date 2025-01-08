import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import requests
import pytz
import time

# 初始化 gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 北京时间
beijing_tz = pytz.timezone('Asia/Shanghai')

# 加载市场数据
def load_markets_with_retry():
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.write(f"网络错误，正在重试 ({attempt + 1}/3)...")
        except Exception as e:
            st.write(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

# 获取前300日交易额最大的交易对
def get_top_300_volume_symbols():
    symbols = []
    try:
        tickers = exchange.fetch_tickers()  # 获取所有交易对的 ticker 信息
        tickers_sorted = sorted(tickers.items(), key=lambda x: x[1].get('quoteVolume', 0), reverse=True)
        
        # 筛选出基准货币为 USDT 的交易对，选择前300个
        top_300 = [symbol for symbol, data in tickers_sorted if data['quote'] == 'USDT'][:300]
        return top_300
    except Exception as e:
        st.write(f"获取前300个交易对时出错: {str(e)}")
        return []

# 获取 OHLC 数据并计算 MA 指标
def fetch_data(symbol, timeframe='1m', max_bars=1000):
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(minutes=max_bars)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)

        if df.empty:
            st.write(f"{symbol} 在 {timeframe} 时间周期内没有数据。")
            return None

        # 计算移动平均线
        df['ma170'] = df['close'].rolling(window=170, min_periods=1).mean()
        df['ma453'] = df['close'].rolling(window=453, min_periods=1).mean()

        return df
    except Exception as e:
        st.write(f"抓取 {symbol} 的数据时出错: {str(e)}")
        return None

# 检测金叉和死叉条件
def check_cross_conditions(df):
    if df is None or len(df) < 453:
        return False, None, None

    # 检测最新的 9 根 K 线内是否发生金叉
    last_9 = df.iloc[-9:]
    gold_cross_time = None
    for i in range(1, len(last_9)):
        if last_9['ma170'].iloc[i - 1] < last_9['ma453'].iloc[i - 1] and \
           last_9['ma170'].iloc[i] >= last_9['ma453'].iloc[i]:
            gold_cross_time = last_9.index[i]
            gold_cross_value = last_9['ma170'].iloc[i]
            return True, "金叉", gold_cross_time

    # 检测是否发生死叉
    death_cross_time = None
    death_cross_value = None
    for i in range(1, len(last_9)):
        if last_9['ma170'].iloc[i - 1] > last_9['ma453'].iloc[i - 1] and \
           last_9['ma170'].iloc[i] <= last_9['ma453'].iloc[i]:
            death_cross_time = last_9.index[i]
            death_cross_value = last_9['ma170'].iloc[i]
            return True, "死叉", death_cross_time

    return False, None, None

# 播放音频
def play_alert_sound():
    audio_url = "http://121.36.79.185/wp-content/uploads/2024/12/alert.wav"
    audio_data = requests.get(audio_url).content
    with open('alert.wav', 'wb') as f:
        f.write(audio_data)

    # 使用 Streamlit 播放音频
    st.audio('alert.wav', format='audio/wav')

# 显示符合条件的交易对结果
def display_result(symbol_data):
    st.write(f"交易对: {symbol_data['symbol']}")
    st.write(f"满足条件的时间: {symbol_data['condition_time']}")
    st.write(f"信号类型: {symbol_data['signal_type']}")
    st.write(f"输出时间: {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("---")
    play_alert_sound()

# 监控交易对并累加显示符合条件的交易对
def monitor_symbols(symbols):
    # 初始化一个存储符合条件的交易对列表
    if 'valid_signals' not in st.session_state:
        st.session_state.valid_signals = []

    progress_bar = st.empty()
    status_text = st.empty()
    detected_text = st.empty()

    detected_text.markdown("### 当前检测状态：")
    
    while True:
        progress_bar.progress(0)
        status_text.text("检测进行中...")
        
        # 当前轮次符合条件的交易对
        current_valid_signals = []

        for index, symbol in enumerate(symbols):
            df = fetch_data(symbol, timeframe='1m', max_bars=1000)
            if df is not None and not df.empty:
                condition_met, signal_type, condition_time = check_cross_conditions(df)
                if condition_met:
                    signal_key = (symbol, condition_time.strftime('%Y-%m-%d %H:%M:%S'))
                    symbol_data = {
                        'symbol': symbol,
                        'condition_time': condition_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'signal_type': signal_type
                    }
                    # 只有新的交易对才会被添加
                    if signal_key not in [x['symbol'] for x in st.session_state.valid_signals]:
                        st.session_state.valid_signals.append(symbol_data)
                        current_valid_signals.append(symbol_data)
                        display_result(symbol_data)

            # 更新进度条
            progress_bar.progress((index + 1) / len(symbols))
            
            # 延迟请求，防止触发 API 限制
            time.sleep(9)

        # 显示所有符合条件的交易对
        if current_valid_signals:
            detected_text.markdown("### 累计符合条件的交易对：")
            for signal in st.session_state.valid_signals:
                detected_text.markdown(f"交易对: {signal['symbol']}, 满足条件时间: {signal['condition_time']}, 信号类型: {signal['signal_type']}")
        
        # 等待下一轮检测
        time.sleep(10)

# 主程序
def main():
    st.title('高交易额现货 MA170 金叉 MA453 筛选系统')

    symbols = get_top_300_volume_symbols()

    if not symbols:
        st.warning("未找到符合条件的交易对。")
    else:
        st.success(f"成功加载 {len(symbols)} 个交易对！")
        st.write(f"正在检测中，交易对总数: {len(symbols)}")
        monitor_symbols(symbols)  # 直接使用同步方法进行监控

if __name__ == "__main__":
    main()
