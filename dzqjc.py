import ccxt
import pandas as pd
import streamlit as st
import pytz
import time
import base64
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置参数
CONFIG = {
    'refresh_interval': 60,  # 自动刷新间隔（秒）
    'max_workers': 4,        # 线程池大小
    'batch_size': 8          # 每批处理交易对数
}

# 交易周期配置
TIMEFRAMES = {
    '5m': {'interval': 300, 'max_bars': 500, 'window_size': 31},
    '30m': {'interval': 1800, 'max_bars': 700, 'window_size': 131},
    '4h': {'interval': 14400, 'max_bars': 700, 'window_size': 131}
}

# 初始化 session state
def init_session_state():
    defaults = {
        'valid_signals': {tf: [] for tf in TIMEFRAMES},
        'shown_signals': {tf: set() for tf in TIMEFRAMES},
        'detection_round': 0,
        'current_batch': 0,
        'last_refresh': time.time(),
        'api_configured': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# 初始化 API 配置（增强兼容性版）
def init_api():
    try:
        # 检查Secrets配置
        if "GATEIO" not in st.secrets:
            st.error("请检查Secrets配置格式，需要使用 [GATEIO] 段落")
            return None
            
        api_key = st.secrets.GATEIO.get("API_KEY")
        api_secret = st.secrets.GATEIO.get("API_SECRET")
        
        if not api_key or not api_secret:
            st.error("API密钥未完整配置，请检查GATEIO段落下的API_KEY和API_SECRET")
            return None

        exchange = ccxt.gateio({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 15000,
            'rateLimit': 600
        })
        
        # 验证API连接
        exchange.fetch_balance()
        st.session_state.api_configured = True
        return exchange
        
    except ccxt.AuthenticationError:
        st.error("API密钥验证失败，请检查密钥是否正确")
        return None
    except Exception as e:
        st.error(f"API初始化失败: {str(e)}")
        return None

# 获取交易对（优化缓存版）
@st.cache_data(ttl=300, show_spinner=False)
def get_top_symbols(exchange):
    try:
        markets = exchange.load_markets()
        valid = [
            s for s in markets 
            if markets[s]['active'] 
            and markets[s]['quote'] == 'USDT' 
            and not any(c in s for c in ['3L','3S','5L','5S'])
        ]
        return valid[:50]  # 精简监控数量提高性能
    except Exception as e:
        st.error(f"获取交易对失败: {str(e)}")
        return []

# 数据处理核心逻辑
def analyze_symbol(symbol, exchange, timeframe):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=500)
        if len(data) < 100:
            return None
            
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
        df['ma7'] = df.close.rolling(7).mean()
        df['ma34'] = df.close.rolling(34).mean()
        
        # 检测逻辑
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        if latest['ma7'] > latest['ma34'] and prev['ma7'] <= prev['ma34']:
            return {'symbol': symbol, 'type': '金叉', 'timeframe': timeframe}
        elif latest['ma7'] < latest['ma34'] and prev['ma7'] >= prev['ma34']:
            return {'symbol': symbol, 'type': '死叉', 'timeframe': timeframe}
            
    except Exception:
        pass
    return None

# 主界面
def main_ui():
    st.title('📊 智能MA交叉监控系统')
    st.caption("Gate.io现货市场实时监控 | 每60秒自动刷新")
    
    if not st.session_state.api_configured:
        with st.expander("❓ 配置指南", expanded=True):
            st.markdown("""
            1. 在Streamlit Secrets中按以下格式配置API密钥：
               ```toml
               [GATEIO]
               API_KEY = "您的API密钥"
               API_SECRET = "您的密钥密码"
               ```
            2. 确保交易对为USDT现货交易对
            3. 首次加载可能需要1-2分钟初始化
            """)
        return
    
    # 控制面板
    with st.sidebar:
        st.header("控制面板")
        if st.button("🔄 立即刷新"):
            st.session_state.current_batch = 0
            st.rerun()
            
        st.progress(st.session_state.current_batch / CONFIG['batch_size'])
        st.write(f"检测轮次: {st.session_state.detection_round}")
        
    # 检测逻辑
    symbols = get_top_symbols(exchange)
    if not symbols:
        return
        
    batch_size = CONFIG['batch_size']
    start = st.session_state.current_batch * batch_size
    batch = symbols[start:start+batch_size]
    
    with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = []
        for symbol in batch:
            for tf in TIMEFRAMES:
                futures.append(executor.submit(analyze_symbol, symbol, exchange, tf))
        
        for future in futures:
            result = future.result()
            if result:
                signal_id = f"{result['symbol']}|{result['timeframe']}|{result['type']}"
                if signal_id not in st.session_state.shown_signals[result['timeframe']]:
                    st.session_state.valid_signals[result['timeframe']].append(result)
                    st.session_state.shown_signals[result['timeframe']].add(signal_id)
    
    # 显示结果
    for tf in TIMEFRAMES:
        with st.expander(f"{tf.upper()}周期信号 ({len(st.session_state.valid_signals[tf])})", expanded=True):
            if st.session_state.valid_signals[tf]:
                for sig in st.session_state.valid_signals[tf]:
                    color = "#00ff00" if sig['type'] == '金叉' else "#ff0000"
                    st.markdown(f"""
                    <div style="padding:10px;border-radius:5px;margin:5px 0;background:#1a1a1a">
                        🚩 <strong>{sig['symbol']}</strong> | 
                        <span style="color:{color}">{sig['type']}</span> | 
                        {pd.Timestamp.now().strftime('%H:%M:%S')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("暂无信号")

    # 自动刷新控制
    if time.time() - st.session_state.last_refresh > CONFIG['refresh_interval']:
        st.session_state.current_batch = 0
        st.session_state.detection_round += 1
        st.session_state.last_refresh = time.time()
        st.rerun()
    elif (start + batch_size) < len(symbols):
        st.session_state.current_batch += 1
        st.rerun()

if __name__ == "__main__":
    init_session_state()
    exchange = init_api()
    main_ui()
