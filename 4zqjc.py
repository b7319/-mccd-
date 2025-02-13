import ccxt
import pandas as pd
from datetime import datetime
import streamlit as st
import pytz
import time

# 初始化 gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 30000
})

# 配置参数
CONFIG = {
    'timeframes': {
        '1m': {'interval': 60, 'max_bars': 500},
        '5m': {'interval': 300, 'max_bars': 500},
        '30m': {'interval': 1800, 'max_bars': 700},
        '4h': {'interval': 14400, 'max_bars': 500}
    },
    'refresh_interval': 60  # 完整检测周期（秒）
}

# 初始化全局状态
def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.signal_history = {tf: [] for tf in CONFIG['timeframes']}
        st.session_state.processed_signals = {tf: set() for tf in CONFIG['timeframes']}
        st.session_state.detection_round = 0
        st.session_state.last_update = {tf: None for tf in CONFIG['timeframes']}
        st.session_state.symbols = []

# 加载市场数据
def load_markets():
    for _ in range(3):
        try:
            markets = exchange.load_markets()
            tickers = exchange.fetch_tickers()
            valid = [(s, tickers[s]['quoteVolume']) for s in tickers 
                    if markets[s].get('quote') == 'USDT' 
                    and not any(c in s for c in ['3L','3S','5L','5S'])]
            valid.sort(key=lambda x: x[1], reverse=True)
            st.session_state.symbols = [s[0] for s in valid[:168]]
            return True
        except Exception as e:
            st.error(f"加载市场数据失败: {str(e)}")
            time.sleep(2)
    return False

# 获取K线数据
def fetch_ohlcv(symbol, timeframe):
    cfg = CONFIG['timeframes'][timeframe]
    for _ in range(3):
        try:
            since = exchange.milliseconds() - cfg['max_bars'] * cfg['interval'] * 1000
            return exchange.fetch_ohlcv(symbol, timeframe, since=since)
        except Exception as e:
            st.warning(f"获取 {symbol} {timeframe} 数据失败: {str(e)}")
            time.sleep(1)
    return None

# 处理数据并检测信号
def process_data(symbol, timeframe, ohlcv):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    
    try:
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_convert(beijing_tz)
        df.set_index('timestamp', inplace=True)
        
        # 计算MA指标
        windows = [7, 34, 170, 453]
        for w in windows:
            df[f'ma{w}'] = df['close'].rolling(window=w).mean()
        
        # 检测交叉
        latest = df.iloc[-31:]
        golden_cross = all(
            (latest['ma7'].iloc[-1] > latest[f'ma{w}'].iloc[-1]) and
            (latest['ma7'].iloc[-2] <= latest[f'ma{w}'].iloc[-2])
            for w in [34, 170, 453]
        )
        
        death_cross = all(
            (latest['ma7'].iloc[-1] < latest[f'ma{w}'].iloc[-1]) and
            (latest['ma7'].iloc[-2] >= latest[f'ma{w}'].iloc[-2])
            for w in [34, 170, 453]
        )
        
        if golden_cross or death_cross:
            signal_id = f"{symbol}|{timeframe}|{df.index[-1].timestamp()}"
            if signal_id not in st.session_state.processed_signals[timeframe]:
                signal = {
                    'symbol': symbol,
                    'type': '金叉' if golden_cross else '死叉',
                    'timeframe': timeframe,
                    'condition_time': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'detect_time': datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S'),
                    'id': signal_id
                }
                st.session_state.signal_history[timeframe].append(signal)
                st.session_state.processed_signals[timeframe].add(signal_id)
                play_alert_sound()
    except Exception as e:
        st.error(f"处理 {symbol} {timeframe} 数据失败: {str(e)}")

# 提示音播放
def play_alert_sound():
    js = """
    <audio id="alert" src="https://assets.mixkit.co/active_storage/sfx/2860/2860-preview.mp3"></audio>
    <script>document.getElementById("alert").play();</script>
    """
    st.components.v1.html(js, height=0)

# 主监控逻辑
def monitor_symbols():
    if not st.session_state.symbols:
        if not load_markets():
            return
    
    for symbol in st.session_state.symbols:
        for timeframe in CONFIG['timeframes']:
            ohlcv = fetch_ohlcv(symbol, timeframe)
            if ohlcv:
                process_data(symbol, timeframe, ohlcv)
        time.sleep(0.3)  # 降低请求频率

    st.session_state.detection_round += 1
    st.session_state.last_update = {tf: datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%H:%M:%S') 
                                   for tf in CONFIG['timeframes']}

# 页面展示组件
def render_dashboard():
    st.title('MA多周期交叉监控系统')
    
    # 侧边栏状态显示
    with st.sidebar:
        st.header("监控状态")
        st.write(f"检测轮次: {st.session_state.detection_round}")
        st.write("最近更新:")
        for tf in CONFIG['timeframes']:
            last = st.session_state.last_update[tf]
            st.write(f"{tf.upper()}: {last if last else '暂无'}")

    # 主展示区
    tabs = st.tabs([f"{tf.upper()}周期" for tf in CONFIG['timeframes']])
    for idx, (tf, tab) in enumerate(zip(CONFIG['timeframes'], tabs)):
        with tab:
            if st.session_state.signal_history[tf]:
                for signal in reversed(st.session_state.signal_history[tf][-20:]):  # 显示最近20条
                    st.markdown(f"""
                    **交易对**: {signal['symbol']}  
                    **信号类型**: {signal['type']}  
                    **条件时间**: {signal['condition_time']}  
                    **检测时间**: {signal['detect_time']}
                    """)
                    st.write("---")

# 主程序
def main():
    init_session_state()
    
    # 执行监控逻辑
    monitor_symbols()
    
    # 渲染界面
    render_dashboard()
    
    # 定时刷新页面
    time.sleep(CONFIG['refresh_interval'])
    st.experimental_rerun()

if __name__ == "__main__":
    main()
