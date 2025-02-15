import ccxt
import pandas as pd
from datetime import datetime, timezone
import streamlit as st
import pytz
import time
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置参数
CONFIG = {
    'refresh_interval': 60,  # 自动刷新间隔（秒）
    'max_workers': 6,        # 线程池大小
    'max_retries': 3,        # API重试次数
    'batch_size': 10         # 每批处理交易对数
}

# 交易周期配置
TIMEFRAMES = {
    '1m': {'interval': 60, 'max_bars': 500, 'cache_ttl': 45, 'window_size': 31},
    '5m': {'interval': 300, 'max_bars': 500, 'cache_ttl': 240, 'window_size': 31},
    '30m': {'interval': 1800, 'max_bars': 700, 'cache_ttl': 1500, 'window_size': 131},
    '4h': {'interval': 14400, 'max_bars': 700, 'cache_ttl': 14400, 'window_size': 131}
}

# 初始化 session state
def init_session_state():
    defaults = {
        'valid_signals': {tf: [] for tf in TIMEFRAMES},
        'shown_signals': {tf: set() for tf in TIMEFRAMES},
        'detection_round': 0,
        'new_signals_count': {tf: 0 for tf in TIMEFRAMES},
        'current_batch': 0,
        'audio_enabled': False,
        'pending_audio': False,
        'last_refresh': time.time(),
        'api_configured': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# 初始化 API 配置
def init_api():
    try:
        api_key = st.secrets["GATEIO_API_KEY"]
        api_secret = st.secrets["GATEIO_API_SECRET"]
        exchange = ccxt.gateio({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 30000,
            'rateLimit': 1000
        })
        st.session_state.api_configured = True
        return exchange
    except KeyError:
        st.error("API 密钥未配置，请在 Streamlit Secrets 中设置 GATEIO_API_KEY 和 GATEIO_API_SECRET")
        st.session_state.api_configured = False
        return None
    except Exception as e:
        st.error(f"API 初始化失败: {str(e)}")
        st.session_state.api_configured = False
        return None

# 音频处理
@st.cache_data
def get_audio_base64():
    try:
        with open("alert.mp3", "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"音频文件加载失败: {str(e)}")
        return None

def audio_player():
    if st.session_state.pending_audio and st.session_state.audio_enabled:
        audio_base64 = get_audio_base64()
        if audio_base64:
            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            st.components.v1.html(audio_html, height=0)
            st.session_state.pending_audio = False

# 界面组件
def setup_ui():
    st.title('📈 智能多周期趋势监控系统')
    st.markdown("### Gate.io现货市场实时MA交叉信号检测")
    
    with st.expander("⚙️ 系统配置", expanded=True):
        st.session_state.audio_enabled = st.checkbox("启用声音提示", value=st.session_state.audio_enabled)
        st.caption("首次使用需要点击上方复选框启用声音提示")
    
    st.sidebar.title("控制面板")
    st.sidebar.progress(st.session_state.current_batch/CONFIG['batch_size'], 
                       text="检测进度")
    
    if st.sidebar.button("🔄 立即刷新数据"):
        st.session_state.current_batch = 0
        st.session_state.last_refresh = time.time()
        st.rerun()

# 获取交易对
@st.cache_data(ttl=600)
def get_top_valid_symbols(exchange):
    try:
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()

        valid_symbols = []
        for symbol in tickers:
            if any(c in symbol for c in ['3L', '3S', '5L', '5S', '_']):
                continue

            market = markets.get(symbol)
            if not market:
                continue

            if (market['active'] and
                    market['quote'] == 'USDT' and
                    market['type'] == 'spot' and
                    market['spot'] and
                    market['percentage'] > 0):

                quote_volume = tickers[symbol].get('quoteVolume', 0)
                if isinstance(quote_volume, (int, float)) and quote_volume >= 10000:
                    valid_symbols.append((symbol, quote_volume))

        valid_symbols.sort(key=lambda x: -x[1])
        filtered = [s for s in valid_symbols if s[1] >= 100000][:86]
        remaining = 86 - len(filtered)
        if remaining > 0:
            filtered += [s for s in valid_symbols if s[1] < 100000][:remaining]

        return [s[0] for s in filtered]
    except Exception as e:
        st.error(f"获取交易对失败: {str(e)}")
        return []

# 主检测流程
def main_detection_cycle(exchange):
    # 自动刷新逻辑
    if time.time() - st.session_state.last_refresh > CONFIG['refresh_interval']:
        st.session_state.current_batch = 0
        st.session_state.last_refresh = time.time()
        st.rerun()

    symbols = get_top_valid_symbols(exchange)
    if not symbols:
        st.error("无法获取交易对列表")
        return
    
    total_batches = len(symbols) // CONFIG['batch_size'] + 1
    start = st.session_state.current_batch * CONFIG['batch_size']
    end = start + CONFIG['batch_size']
    current_batch = symbols[start:end]
    
    process_batch(current_batch, exchange)
    
    if end < len(symbols):
        st.session_state.current_batch += 1
        st.rerun()

# 结果展示
def display_results():
    tabs = st.tabs([f"{tf.upper()}周期" for tf in TIMEFRAMES])
    for idx, tf in enumerate(TIMEFRAMES):
        with tabs[idx]:
            if st.session_state.valid_signals[tf]:
                for signal in st.session_state.valid_signals[tf]:
                    st.success(f"""
                    🚩 **交易对**: {signal['symbol']}  
                    ⏰ **时间框架**: {tf.upper()}  
                    🔔 **信号类型**: {signal['signal_type']}  
                    📅 **信号时间**: {signal['condition_time']}
                    """)
            else:
                st.info("当前周期暂无有效信号")

def main():
    init_session_state()
    exchange = init_api()
    
    if not st.session_state.api_configured:
        st.stop()
    
    setup_ui()
    audio_player()
    
    with st.spinner("🔍 正在扫描市场机会..."):
        main_detection_cycle(exchange)
    
    display_results()
    
    st.sidebar.markdown(f"**检测轮次**: {st.session_state.detection_round}")
    st.sidebar.markdown("**最新信号统计**")
    for tf in TIMEFRAMES:
        st.sidebar.metric(f"{tf.upper()}周期", 
                         st.session_state.new_signals_count[tf])

if __name__ == "__main__":
    main()
