import ccxt
import pandas as pd
from datetime import datetime, timezone
import streamlit as st
import pytz
import time
import requests

# 初始化 gate.io API
api_key = 'YOUR_API_KEY'  # 无需替换即可运行
api_secret = 'YOUR_API_SECRET'  # 无需替换即可运行
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 30000,
    'rateLimit': 1000
})

# 北京时间
beijing_tz = pytz.timezone('Asia/Shanghai')

# 支持的交易周期配置
TIMEFRAMES = {
    '1m': {'interval': 60, 'max_bars': 500},
    '5m': {'interval': 300, 'max_bars': 500},
    '30m': {'interval': 1800, 'max_bars': 700},
    '4h': {'interval': 14400, 'max_bars': 500}
}

# 初始化 session state
def initialize_session_state():
    if 'valid_signals' not in st.session_state:
        st.session_state.valid_signals = {tf: [] for tf in TIMEFRAMES}
    if 'shown_signals' not in st.session_state:
        st.session_state.shown_signals = {tf: set() for tf in TIMEFRAMES}
    if 'displayed_signals' not in st.session_state:
        st.session_state.displayed_signals = {tf: set() for tf in TIMEFRAMES}
    if 'detection_round' not in st.session_state:
        st.session_state.detection_round = 0
    if 'new_signals_count' not in st.session_state:
        st.session_state.new_signals_count = {tf: 0 for tf in TIMEFRAMES}
    if 'result_containers' not in st.session_state:
        st.session_state.result_containers = {tf: st.empty() for tf in TIMEFRAMES}

# 加载市场数据（带缓存）
@st.cache_resource(ttl=3600)
def load_markets_with_retry():
    for attempt in range(3):
        try:
            return exchange.load_markets()
        except ccxt.NetworkError:
            time.sleep(2)
        except Exception as e:
            st.error(f"加载市场数据失败: {str(e)}")
            st.stop()
    return exchange.load_markets()

# 获取前168个有效交易对
def get_top_valid_symbols():
    try:
        markets = load_markets_with_retry()
        tickers = exchange.fetch_tickers()

        valid_symbols = []
        for symbol in tickers:
            market = markets.get(symbol)
            if market and market['active'] and market['quote'] == 'USDT' \
                    and market['type'] == 'spot' \
                    and not any(c in symbol for c in ['3L', '3S', '5L', '5S']):
                valid_symbols.append((symbol, tickers[symbol]['quoteVolume']))

        # 按交易量排序并取前168
        valid_symbols.sort(key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in valid_symbols[:168]]
    except Exception as e:
        st.error(f"获取交易对失败: {str(e)}")
        return []

# 优化后的数据获取函数
def fetch_ohlcv_with_retry(symbol, timeframe='1m'):
    config = TIMEFRAMES[timeframe]
    for attempt in range(3):
        try:
            now = exchange.milliseconds()
            since = now - config['max_bars'] * config['interval'] * 1000
            return exchange.fetch_ohlcv(symbol, timeframe, since)
        except ccxt.NetworkError:
            time.sleep(1)
        except ccxt.BadSymbol:
            return None
        except Exception as e:
            st.warning(f"{symbol} {timeframe} 数据获取失败: {str(e)}")
            time.sleep(2)
    return None

# 处理数据并计算指标
def process_data(ohlcvs, timeframe):
    min_bars = 453  # 所有周期保持相同的最小K线数量要求
    if not ohlcvs or len(ohlcvs) < min_bars:
        return None

    df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
    df.set_index('timestamp', inplace=True)

    # 计算移动平均线（保持原逻辑）
    df['ma170'] = df['close'].rolling(window=170, min_periods=1).mean()
    df['ma453'] = df['close'].rolling(window=453, min_periods=1).mean()
    df['ma34'] = df['close'].rolling(window=34, min_periods=1).mean()
    df['ma7'] = df['close'].rolling(window=7, min_periods=1).mean()
    return df

# 检测逻辑（保持原逻辑不变）
def check_cross_conditions(df):
    if df is None or len(df) < 31:
        return None, None

    last_31 = df.iloc[-31:]

    # 金叉检测
    ma7_cross_ma34 = any(last_31['ma7'].iloc[i] > last_31['ma34'].iloc[i] 
                       and last_31['ma7'].iloc[i-1] <= last_31['ma34'].iloc[i-1] 
                       for i in range(1, len(last_31)))
    
    ma7_cross_ma170 = any(last_31['ma7'].iloc[i] > last_31['ma170'].iloc[i] 
                        and last_31['ma7'].iloc[i-1] <= last_31['ma170'].iloc[i-1] 
                        for i in range(1, len(last_31)))
    
    ma7_cross_ma453 = any(last_31['ma7'].iloc[i] > last_31['ma453'].iloc[i] 
                        and last_31['ma7'].iloc[i-1] <= last_31['ma453'].iloc[i-1] 
                        for i in range(1, len(last_31)))

    # 死叉检测
    ma7_death_ma34 = any(last_31['ma7'].iloc[i] < last_31['ma34'].iloc[i] 
                       and last_31['ma7'].iloc[i-1] >= last_31['ma34'].iloc[i-1] 
                       for i in range(1, len(last_31)))
    
    ma7_death_ma170 = any(last_31['ma7'].iloc[i] < last_31['ma170'].iloc[i] 
                        and last_31['ma7'].iloc[i-1] >= last_31['ma170'].iloc[i-1] 
                        for i in range(1, len(last_31)))
    
    ma7_death_ma453 = any(last_31['ma7'].iloc[i] < last_31['ma453'].iloc[i] 
                        and last_31['ma7'].iloc[i-1] >= last_31['ma453'].iloc[i-1] 
                        for i in range(1, len(last_31)))

    signal_type = None
    if all([ma7_cross_ma34, ma7_cross_ma170, ma7_cross_ma453]):
        signal_type = 'MA7 金叉 MA34, MA170, MA453'
    elif all([ma7_death_ma34, ma7_death_ma170, ma7_death_ma453]):
        signal_type = 'MA7 死叉 MA34, MA170, MA453'

    condition_time = last_31.index[-1] if signal_type else None
    return signal_type, condition_time

# 播放提示音（使用 JavaScript 实现）
def play_alert_sound():
    audio_url = "http://121.36.79.185/wp-content/uploads/2024/12/alert.wav"
    js_code = f"""
    <audio id="alertAudio" src="{audio_url}" preload="auto"></audio>
    <script>
        var audio = document.getElementById("alertAudio");
        audio.play();
    </script>
    """
    st.components.v1.html(js_code, height=0)

# 动态追加新信号到展示容器
def append_new_signals(timeframe):
    container = st.session_state.result_containers[timeframe]
    new_signals = [
        s for s in st.session_state.valid_signals[timeframe] 
        if s['signal_id'] not in st.session_state.displayed_signals[timeframe]
    ]
    
    if new_signals:
        with container:
            for signal in new_signals:
                st.markdown(f"""
                **交易对**: {signal['symbol']}  
                **信号类型**: {signal['signal_type']}  
                **条件时间**: {signal['condition_time']}  
                **检测时间**: {signal['detect_time']}
                """)
                st.write("---")
            # 记录已展示信号
            st.session_state.displayed_signals[timeframe].update(s['signal_id'] for s in new_signals)

# 主监控逻辑
def monitor_symbols():
    cols = st.columns(4)
    for idx, (tf, col) in enumerate(zip(TIMEFRAMES, cols)):
        with col:
            st.subheader(f"{tf.upper()} 周期")
            st.session_state.result_containers[tf] = st.empty()

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    round_info = st.sidebar.empty()
    new_signals_info = st.sidebar.empty()

    while True:
        start_time = time.time()
        st.session_state.detection_round += 1
        current_round_new = {tf: 0 for tf in TIMEFRAMES}

        # 每轮检测前重新获取交易额前168的交易对
        symbols = get_top_valid_symbols()
        if not symbols:
            st.error("未找到有效交易对")
            return
        
        for idx, symbol in enumerate(symbols):
            progress = (idx + 1) / len(symbols)
            progress_bar.progress(progress)
            status_text.text(f"正在检测: {symbol}")

            for timeframe in TIMEFRAMES:
                try:
                    ohlcvs = fetch_ohlcv_with_retry(symbol, timeframe)
                    df = process_data(ohlcvs, timeframe)
                    signal_type, condition_time = check_cross_conditions(df)
                    
                    if signal_type and condition_time:
                        signal_id = f"{symbol}|{timeframe}|{condition_time.timestamp()}"
                        
                        if signal_id not in st.session_state.shown_signals[timeframe]:
                            detect_time = datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')
                            new_signal = {
                                'symbol': symbol,
                                'signal_type': signal_type,
                                'condition_time': condition_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'detect_time': detect_time,
                                'signal_id': signal_id
                            }
                            
                            st.session_state.valid_signals[timeframe].append(new_signal)
                            st.session_state.shown_signals[timeframe].add(signal_id)
                            current_round_new[timeframe] += 1
                            
                            # 立即更新展示
                            append_new_signals(timeframe)
                            play_alert_sound()

                except Exception as e:
                    continue

            time.sleep(0.5)  # 降低请求频率

        # 更新统计信息
        round_info.markdown(f"**检测轮次**: {st.session_state.detection_round}")
        new_signals_info.markdown("**本轮新增信号**")
        for tf in TIMEFRAMES:
            new_signals_info.markdown(f"- {tf.upper()}: {current_round_new[tf]}")

        elapsed = time.time() - start_time
        sleep_time = max(60 - elapsed, 15)  # 保证至少15秒间隔
        time.sleep(sleep_time)

# 主程序入口
def main():
    st.title('多周期MA交叉实时监控系统')
    
    with st.expander("当前监控交易对列表", expanded=False):
        st.write("每轮检测前重新获取交易额前168的交易对")
    
    st.sidebar.title("监控状态面板")
    initialize_session_state()  # 确保 session state 正确初始化
    monitor_symbols()

if __name__ == "__main__":
    main()
