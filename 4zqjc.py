import ccxt
import pandas as pd
from datetime import datetime
import streamlit as st
import pytz
import time

# ================== 配置区 ==================
API_KEY = 'YOUR_API_KEY'  # 本地运行时直接填写
API_SECRET = 'YOUR_API_SECRET'
AUDIO_URL = "http://121.36.79.185/wp-content/uploads/2024/12/alert.wav"

TIMEFRAMES = {
    '1m': {'interval': 60, 'max_bars': 500},
    '5m': {'interval': 300, 'max_bars': 500},
    '30m': {'interval': 1800, 'max_bars': 700},
    '4h': {'interval': 14400, 'max_bars': 500}
}
# ============================================

# 初始化交易所连接
def init_exchange():
    try:
        return ccxt.gateio({
            'apiKey': st.secrets.get("GATEIO", {}).get("API_KEY") or API_KEY,
            'secret': st.secrets.get("GATEIO", {}).get("API_SECRET") or API_SECRET,
            'enableRateLimit': True,
            'timeout': 30000
        })
    except Exception as e:
        st.error(f"交易所初始化失败: {str(e)}")
        st.stop()

exchange = init_exchange()

# 初始化全局状态
def init_session_state():
    required_states = {
        'valid_signals': {tf: [] for tf in TIMEFRAMES},
        'shown_signals': {tf: set() for tf in TIMEFRAMES},
        'detection_round': 0,
        'last_symbols': []
    }
    for key, value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# 带缓存的交易对获取（每小时刷新）
@st.cache_resource(ttl=3600)
def get_top_symbols():
    try:
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()
        
        valid = []
        for symbol in tickers:
            m = markets.get(symbol)
            if m and m['active'] and m['quote'] == 'USDT' \
               and m['type'] == 'spot' and not any(c in symbol for c in ['3L','3S','5L','5S']):
                valid.append((symbol, tickers[symbol]['quoteVolume']))
        
        valid.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in valid[:168]]
    except Exception as e:
        st.error(f"获取交易对失败: {str(e)}")
        return []

# 数据获取与处理
def fetch_data(symbol, timeframe):
    config = TIMEFRAMES[timeframe]
    try:
        now = exchange.milliseconds()
        since = now - config['max_bars'] * config['interval'] * 1000
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
        return pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    except Exception as e:
        st.warning(f"{symbol} {timeframe} 数据获取失败: {str(e)}")
        return None

# 技术指标计算
def calculate_indicators(df):
    if df is None or len(df) < 453:
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_convert('Asia/Shanghai')
    df.set_index('timestamp', inplace=True)
    
    windows = [7, 34, 170, 453]
    for w in windows:
        df[f'ma{w}'] = df['close'].rolling(window=w, min_periods=1).mean()
    return df

# 信号检测逻辑
def check_cross(df):
    if df is None or len(df) < 31:
        return None
    
    latest = df.iloc[-31:]
    cross_up = all([
        any(latest['ma7'].iloc[i] > latest['ma34'].iloc[i] and latest['ma7'].iloc[i-1] <= latest['ma34'].iloc[i-1] for i in range(1,31)),
        any(latest['ma7'].iloc[i] > latest['ma170'].iloc[i] and latest['ma7'].iloc[i-1] <= latest['ma170'].iloc[i-1] for i in range(1,31)),
        any(latest['ma7'].iloc[i] > latest['ma453'].iloc[i] and latest['ma7'].iloc[i-1] <= latest['ma453'].iloc[i-1] for i in range(1,31))
    ])
    
    cross_down = all([
        any(latest['ma7'].iloc[i] < latest['ma34'].iloc[i] and latest['ma7'].iloc[i-1] >= latest['ma34'].iloc[i-1] for i in range(1,31)),
        any(latest['ma7'].iloc[i] < latest['ma170'].iloc[i] and latest['ma7'].iloc[i-1] >= latest['ma170'].iloc[i-1] for i in range(1,31)),
        any(latest['ma7'].iloc[i] < latest['ma453'].iloc[i] and latest['ma7'].iloc[i-1] >= latest['ma453'].iloc[i-1] for i in range(1,31))
    ])
    
    if cross_up: return '金叉'
    if cross_down: return '死叉'
    return None

# 音频提示组件
def audio_player():
    return st.markdown(f"""
    <audio id="alertAudio" src="{AUDIO_URL}" preload="auto"></audio>
    <script>
        function play() {{ document.getElementById('alertAudio').play(); }}
        setTimeout(play, 1000)  // 延迟1秒播放避免拦截
    </script>
    """, unsafe_allow_html=True)

# 界面组件
def main_interface():
    st.title('MA多周期交叉监控系统')
    audio_player()
    
    with st.sidebar:
        st.header("监控状态")
        progress = st.progress(0)
        status = st.empty()
        round_info = st.empty()
        new_signal_info = st.empty()
    
    cols = st.columns(4)
    containers = {tf: cols[i].container() for i, tf in enumerate(TIMEFRAMES)}
    
    while True:
        start = time.time()
        symbols = get_top_symbols()
        new_signals = {tf: 0 for tf in TIMEFRAMES}
        
        # 交易对变化检测
        if symbols != st.session_state.last_symbols:
            st.session_state.last_symbols = symbols
            st.session_state.detection_round += 1
        
        for idx, symbol in enumerate(symbols):
            progress.progress((idx+1)/len(symbols))
            status.text(f"检测中: {symbol}")
            
            for tf in TIMEFRAMES:
                try:
                    df = fetch_data(symbol, tf)
                    df = calculate_indicators(df)
                    signal = check_cross(df)
                    
                    if signal:
                        signal_id = f"{symbol}|{tf}|{df.index[-1].timestamp()}"
                        if signal_id not in st.session_state.shown_signals[tf]:
                            detect_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
                            entry = {
                                'symbol': symbol,
                                'signal': signal,
                                'time': detect_time,
                                'id': signal_id
                            }
                            
                            st.session_state.valid_signals[tf].append(entry)
                            st.session_state.shown_signals[tf].add(signal_id)
                            new_signals[tf] += 1
                            
                            with containers[tf]:
                                st.success(f"""
                                **交易对**: {symbol}  
                                **信号类型**: {signal}  
                                **发现时间**: {detect_time}
                                """)
                
                except Exception as e:
                    pass
            
            time.sleep(0.3)
        
        # 更新统计信息
        round_info.text(f"检测轮次: {st.session_state.detection_round}")
        new_signal_info.text("\n".join([f"{tf}: {new_signals[tf]}" for tf in TIMEFRAMES]))
        
        elapsed = time.time() - start
        time.sleep(max(60 - elapsed, 15))

if __name__ == "__main__":
    main_interface()
