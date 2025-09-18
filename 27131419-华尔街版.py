# integrated_streamlit_app_v2.py
import ccxt
import numpy as np
from datetime import datetime
import streamlit as st
import pytz
import time
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import logging
import queue
import html

# ========== 基础配置 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
beijing_tz = pytz.timezone('Asia/Shanghai')

TIMEFRAMES = {
    '1m': {'interval': 60, 'max_bars': 1000, 'cache_ttl': 30},
    '5m': {'interval': 300, 'max_bars': 1000, 'cache_ttl': 180},
    '30m': {'interval': 1800, 'max_bars': 1000, 'cache_ttl': 600},
    '4h': {'interval': 14400, 'max_bars': 1000, 'cache_ttl': 3600},
}

# 添加默认阈值配置
THRESHOLD_DEFAULTS = {
    '1m': 0.31,
    '5m': 0.86,
    '30m': 3.68,
    '4h': 4.86
}

STRATEGIES = {'ma34_peak': 'MA34有效波峰 ≤ MA170最低值'}

CONFIG = {
    'fetch_limit': 1000,
    'max_workers': 4,
    'play_sound': True,
    'top_n_symbols': 300,
    'peak_window': 315,  # 最新K线窗口数
    'peak_lr': 9,  # 波峰前后K线数量
    'thresholds': THRESHOLD_DEFAULTS.copy()  # 添加阈值配置
}

# Session 初始化
if 'valid_signals' not in st.session_state:
    st.session_state.valid_signals = defaultdict(list)
if 'shown_signals' not in st.session_state:
    st.session_state.shown_signals = defaultdict(set)
if 'detection_round' not in st.session_state:
    st.session_state.detection_round = 0
if 'failed_symbols' not in st.session_state:
    st.session_state.failed_symbols = set()
if 'symbols_to_monitor' not in st.session_state:
    st.session_state.symbols_to_monitor = []
if 'latest_signals_ticker' not in st.session_state:
    st.session_state.latest_signals_ticker = deque(maxlen=68)
if 'ticker_seen' not in st.session_state:
    st.session_state.ticker_seen = set()
if 'right_panel_placeholder' not in st.session_state:
    st.session_state.right_panel_placeholder = None
if 'right_panel_css_injected' not in st.session_state:
    st.session_state.right_panel_css_injected = False
# 修复：添加缺失的 result_containers 初始化
if 'result_containers' not in st.session_state:
    st.session_state.result_containers = {}

ohlcv_cache = {}


# ========== 工具函数 ==========
def escape_html(text):
    if text is None:
        return ""
    return html.escape(str(text))


def generate_signal_id(symbol, timeframe, sig_type, sig_time):
    ts = int(sig_time.timestamp())
    unique_str = f"{symbol}|{timeframe}|{ts}|{sig_type}"
    return hashlib.md5(unique_str.encode()).hexdigest()


def simplify_symbol(symbol):
    return symbol.split('/')[0].lower()


def generate_symbol_link(symbol):
    base_symbol = simplify_symbol(symbol)
    return f"https://www.aicoin.com/chart/gate_{base_symbol}swapusdt"


# ========== 音频处理 ==========
def get_audio_base64(file_path="D:\\pycharm_study\\y1314.wav"):
    try:
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
    except Exception:
        return None


def play_alert_sound():
    audio_base64 = get_audio_base64()
    if audio_base64:
        autoplay_script = f'''
        <audio autoplay>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        <script>
        document.querySelector('audio') && document.querySelector('audio').play().catch(error => {{
            console.log('自动播放受阻: ', error);
        }});
        </script>
        '''
        st.components.v1.html(autoplay_script, height=0)


# ========== 交易对 & OHLCV ==========
@st.cache_data(ttl=3600)
def get_valid_symbols(api_key, api_secret, top_n=300):
    try:
        exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True})
        markets = exchange.load_markets(True)
        tickers = exchange.fetch_tickers(params={'type': 'swap'})
        volume_data = []
        for symbol, ticker in tickers.items():
            market = markets.get(symbol)
            if not market:
                continue
            if market.get('type') == 'swap' and market.get('active') and ticker.get(
                    'quoteVolume') is not None and market.get('quote') == 'USDT':
                volume_data.append({'symbol': symbol, 'volume': ticker['quoteVolume']})
        volume_data.sort(key=lambda x: x['volume'], reverse=True)
        top = [x['symbol'] for x in volume_data[:top_n]]
        final = []
        for s in top:
            try:
                exchange.fetch_ohlcv(s, '1m', limit=2)
                final.append(s)
            except Exception:
                continue
        return final
    except Exception:
        return []


def get_cached_ohlcv(exchange, symbol, timeframe, failed_symbols):
    if symbol in failed_symbols:
        return None
    now = time.time()
    cache_key = (symbol, timeframe)
    cfg = TIMEFRAMES[timeframe]
    if cache_key in ohlcv_cache:
        data, ts = ohlcv_cache[cache_key]
        if now - ts < cfg['cache_ttl']:
            return data
        else:
            del ohlcv_cache[cache_key]
    for attempt in range(3):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, limit=600)
            if data and len(data) >= CONFIG['peak_window']:
                ohlcv_cache[cache_key] = (data, now)
                return data
            else:
                return None
        except Exception:
            time.sleep(1 + attempt)
    failed_symbols.add(symbol)
    return None


# ========== 新逻辑：MA34 波峰 ==========
def find_ma34_valid_peaks(ma34, left_right=9):
    peaks = []
    for i in range(left_right, len(ma34) - left_right):
        if ma34[i] == max(ma34[i - left_right:i + left_right + 1]):
            peaks.append((i, ma34[i]))
    return peaks


def detect_ma34_peak_signals(data, symbol, timeframe):
    if data is None or len(data['closes']) < CONFIG['peak_window']:
        return []

    ma34 = data['ma34']
    ma170 = data['ma170']
    timestamps = data['timestamps']
    closes = data['closes']

    # 最新窗口
    window_start = len(ma34) - CONFIG['peak_window']
    ma34_win = ma34[window_start:]
    ma170_win = ma170[window_start:]
    ts_win = timestamps[window_start:]

    peaks = find_ma34_valid_peaks(ma34_win, left_right=CONFIG['peak_lr'])
    if not peaks:
        return []

    min_peak_val = min([p[1] for p in peaks])
    min_peak_idx = [p[0] for p in peaks if p[1] == min_peak_val][0]
    min_peak_time = datetime.fromtimestamp(int(ts_win[min_peak_idx]) / 1000, tz=beijing_tz)
    min_ma170 = min(ma170_win)

    # 计算涨幅百分比
    increase_percent = ((min_ma170 - min_peak_val) / min_peak_val) * 100 if min_peak_val > 0 else 0

    # 获取当前时间框架的阈值
    threshold = CONFIG['thresholds'].get(timeframe, THRESHOLD_DEFAULTS[timeframe])

    # 检查是否满足阈值条件，并增加最新价格>=min_peak_ma34的条件
    if (min_peak_val <= min_ma170 and
        increase_percent <= threshold and
        closes[-1] >= min_peak_val):  # 新增条件：最新价格>=min_peak_ma34
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': 'MA34有效波峰 ≤ MA170最低值',
            'cross_time': min_peak_time,
            'cross_index': min_peak_idx,
            'current_price': float(closes[-1]),
            'ma34_min_peak': float(min_peak_val),
            'ma170_min': float(min_ma170),
            'increase_percent': float(increase_percent),
            'detect_time': datetime.now(beijing_tz)
        }
        return [signal]
    return []


def process_data_all(ohlcvs):
    if not ohlcvs or len(ohlcvs) < CONFIG['peak_window']:
        return None
    try:
        timestamps = np.array([x[0] for x in ohlcvs], dtype=np.int64)
        closes = np.array([float(x[4]) for x in ohlcvs], dtype=np.float64)

        ma34 = np.convolve(closes, np.ones(34) / 34, mode='valid')
        ma170 = np.convolve(closes, np.ones(170) / 170, mode='valid')

        min_len = min(len(ma34), len(ma170))
        ma34 = ma34[-min_len:]
        ma170 = ma170[-min_len:]
        closes = closes[-min_len:]
        timestamps = timestamps[-min_len:]

        return {
            'timestamps': timestamps,
            'closes': closes,
            'ma34': ma34,
            'ma170': ma170
        }
    except ValueError:
        return None


# ========== 渲染 ==========
def render_ma34_signal(tf, signal):
    direction_color = "#D4AF37"  # 金色
    cross_time = signal['cross_time']
    current_price = signal['current_price']
    min_peak_val = signal['ma34_min_peak']
    min_ma170 = signal['ma170_min']
    increase_percent = signal['increase_percent']

    symbol_simple = simplify_symbol(signal['symbol'])
    symbol_link = generate_symbol_link(signal['symbol'])

    content = (
        f"<div style='margin-bottom: 8px; padding: 8px; border-radius: 6px; "
        f"background: linear-gradient(145deg, #f8f0e0, #f5e7c8); box-shadow: 0 1px 4px rgba(0,0,0,0.1); "
        f"border: 1px solid #D4AF37;'>"
        f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>"
        f"<a href='{symbol_link}' target='_blank' style='text-decoration: none; color: #8B4513; font-weight: bold; font-size: 13px;'>"
        f"📈 {symbol_simple} [{tf.upper()}]</a>"
        f"<span style='background-color: #D4AF37; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px;'>"
        f"信号</span>"
        f"</div>"
        f"<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px; font-size: 12px;'>"
        f"<div><span style='color: #8B4513;'>MA34峰:</span> <span style='font-weight: bold;'>{min_peak_val:.4f}</span></div>"
        f"<div><span style='color: #8B4513;'>MA170低:</span> <span style='font-weight: bold;'>{min_ma170:.4f}</span></div>"
        f"<div><span style='color: #8B4513;'>现价:</span> <span style='font-weight: bold;'>{current_price:.4f}</span></div>"
        f"<div><span style='color: #8B4513;'>涨幅:</span> <span style='font-weight: bold; color: {direction_color};'>{increase_percent:.2f}%</span></div>"
        f"<div><span style='color: #8B4513;'>时间:</span> <span style='font-weight: bold;'>{cross_time.strftime('%H:%M:%S')}</span></div>"
        f"<div><span style='color: #8B4513;'>检测:</span> <span style='font-weight: bold;'>{signal['detect_time'].strftime('%H:%M:%S')}</span></div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(content, unsafe_allow_html=True)


# ========== 右侧固定展示栏 ==========
def _enqueue_latest(signal, tf, strategy, symbol, signal_id):
    if signal_id in st.session_state.ticker_seen:
        return
    st.session_state.ticker_seen.add(signal_id)

    entry = {
        'id': signal_id,
        'symbol': symbol,
        'timeframe': tf,
        'strategy': strategy,
        'detect_dt': signal.get('detect_time'),
        'current_price': signal.get('current_price'),
        'signal_type': signal.get('signal_type'),
        'cross_time': signal.get('cross_time'),
        'ma34_min_peak': signal.get('ma34_min_peak'),
        'ma170_min': signal.get('ma170_min'),
        'increase_percent': signal.get('increase_percent')
    }
    st.session_state.latest_signals_ticker.appendleft(entry)


def render_right_sidebar():
    if not st.session_state.right_panel_css_injected:
        st.markdown("""
        <style>
        #latest-fixed-panel {
            position: fixed;
            top: 80px;
            right: 12px;
            width: 413px;
            height: 1413px;
            overflow-y: auto;
            z-index: 1001;
            background: rgba(248, 240, 224, 0.95);
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border: 1px solid #D4AF37;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        #latest-fixed-panel h3 {
            color: #8B4513;
            margin-top: 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #D4AF37;
            font-weight: 600;
            text-align: center;
            font-size: 16px;
        }
        #latest-fixed-panel .item {
            font-size: 12px;
            padding: 8px;
            margin: 6px 0;
            background: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.2s;
            border-left: 2px solid #D4AF37;
        }
        #latest-fixed-panel .item:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        #latest-fixed-panel .symbol {
            font-weight: bold;
            color: #8B4513;
            margin-bottom: 3px;
            font-size: 13px;
        }
        #latest-fixed-panel .timeframe {
            background: #f5e7c8;
            color: #8B4513;
            padding: 1px 4px;
            border-radius: 3px;
            font-size: 10px;
            display: inline-block;
            margin-right: 4px;
        }
        #latest-fixed-panel .increase {
            color: #D4AF37;
            font-weight: bold;
        }
        #latest-fixed-panel .time {
            color: #8B4513;
            font-size: 10px;
            margin-top: 3px;
        }
        #latest-fixed-panel a {
            color: #8B4513;
            text-decoration: none;
        }
        #latest-fixed-panel a:hover {
            text-decoration: underline;
        }
        </style>
        """, unsafe_allow_html=True)
        st.session_state.right_panel_css_injected = True

    items = list(st.session_state.latest_signals_ticker)
    if not items:
        html_panel = "<div id='latest-fixed-panel'><h3>最新信号</h3><div style='color:#8B4513; text-align:center; padding:20px;'>暂无信号</div></div>"
        st.session_state.right_panel_placeholder = st.empty()
        st.session_state.right_panel_placeholder.markdown(html_panel, unsafe_allow_html=True)
        return

    lines = []
    for s in items:
        dt_str = s['detect_dt'].strftime('%H:%M:%S')
        symbol_link = generate_symbol_link(s['symbol'])
        symbol_simple = simplify_symbol(s['symbol'])
        line = (
            f"<div class='item'>"
            f"<div class='symbol'>"
            f"<a href='{symbol_link}' target='_blank'>{symbol_simple}</a> "
            f"<span class='timeframe'>{s['timeframe'].upper()}</span></div>"
            f"<div>MA34峰: <span style='color:#D4AF37;'>{s['ma34_min_peak']:.4f}</span> | MA170低: <span style='color:#D4AF37;'>{s['ma170_min']:.4f}</span></div>"
            f"<div>涨幅: <span class='increase'>{s['increase_percent']:.2f}%</span> | 现价: <span style='color:#D4AF37;'>{s['current_price']:.4f}</span></div>"
            f"<div class='time'>检测时间: {dt_str}</div>"
            f"</div>"
        )
        lines.append(line)

    html_panel = "<div id='latest-fixed-panel'><h3>最新信号</h3>" + "".join(lines) + "</div>"
    if st.session_state.right_panel_placeholder is None:
        st.session_state.right_panel_placeholder = st.empty()
    st.session_state.right_panel_placeholder.markdown(html_panel, unsafe_allow_html=True)


# ========== 监控 ==========
def process_symbol_timeframe(exchange, symbol, timeframe, failed_symbols):
    out_signals = []
    try:
        ohlcvs = get_cached_ohlcv(exchange, symbol, timeframe, failed_symbols)
        if ohlcvs is None:
            return symbol, out_signals
        data = process_data_all(ohlcvs)
        signals = detect_ma34_peak_signals(data, symbol, timeframe)
        for s in signals:
            out_signals.append(s)
        return symbol, out_signals
    except Exception:
        return symbol, out_signals


def build_tabs():
    # 确保 result_containers 存在
    if 'result_containers' not in st.session_state:
        st.session_state.result_containers = {}

    labels = []
    keys = []
    for tf in TIMEFRAMES:
        labels.append(f"{tf.upper()}·波峰策略")
        keys.append((tf, 'ma34_peak'))
    tabs = st.tabs(labels)
    for i, k in enumerate(keys):
        with tabs[i]:
            container = st.container()
            placeholder = st.empty()
            st.session_state.result_containers[k] = {'container': container, 'placeholder': placeholder}


def update_tab_content(tf, strategy):
    if (tf, strategy) not in st.session_state.result_containers:
        return

    container = st.session_state.result_containers[(tf, strategy)]['container']
    placeholder = st.session_state.result_containers[(tf, strategy)]['placeholder']
    with container:
        with placeholder.container():
            signals = st.session_state.valid_signals[(tf, strategy)][-868:][::-1]
            if not signals:
                st.markdown("<div style='text-align: center; padding: 20px; color: #8B4513;'>暂无信号</div>",
                            unsafe_allow_html=True)
                return
            for s in signals:
                render_ma34_signal(tf, s)


def monitor_symbols(api_key, api_secret):
    exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 30000})
    build_tabs()

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    stats = st.sidebar.empty()
    max_workers = CONFIG.get('max_workers', 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            start_time = time.time()
            st.session_state.detection_round += 1
            new_signals = defaultdict(int)
            symbols = [s for s in st.session_state.symbols_to_monitor if s not in st.session_state.failed_symbols]
            if not symbols:
                time.sleep(30)
                continue
            failed_copy = st.session_state.failed_symbols.copy()
            futures = []
            for s in symbols:
                for tf in TIMEFRAMES:
                    futures.append(executor.submit(process_symbol_timeframe, exchange, s, tf, failed_copy))

            for i, fut in enumerate(as_completed(futures)):
                progress = (i + 1) / len(futures)
                progress_bar.progress(progress)
                status_text.text(f"检测进度: {progress * 100:.1f}%")
                symbol, signals = fut.result()
                play_sound = False
                for sig in signals:
                    tf = sig['timeframe']
                    signal_id = generate_signal_id(sig['symbol'], tf, sig['signal_type'], sig['cross_time'])
                    if signal_id not in st.session_state.shown_signals[(tf, 'ma34_peak')]:
                        st.session_state.valid_signals[(tf, 'ma34_peak')].append(sig)
                        st.session_state.shown_signals[(tf, 'ma34_peak')].add(signal_id)
                        new_signals[(tf, 'ma34_peak')] += 1
                        play_sound = True
                        update_tab_content(tf, 'ma34_peak')
                        _enqueue_latest(sig, tf, 'ma34_peak', symbol, signal_id)
                        render_right_sidebar()
                if play_sound and CONFIG.get('play_sound', True):
                    play_alert_sound()
            stats.markdown(
                f"轮次: {st.session_state.detection_round} | 新信号: {dict(new_signals)} | 失败: {len(st.session_state.failed_symbols)}")
            elapsed = time.time() - start_time
            sleep_time = max(45 - elapsed, 30)
            time.sleep(sleep_time)


# ========== 入口 ==========
def main():
    # 应用全局样式
    st.markdown("""
    <style>
    /* 主标题样式 */
    .title {
        color: #8B4513;
        font-weight: 700;
        font-size: 28px;
        margin-bottom: 20px;
        text-align: center;
        background: linear-gradient(to right, #D4AF37, #8B4513);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
        border-bottom: 2px solid #D4AF37;
    }

    /* 整体背景 */
    body {
        background-color: #f8f0e0;
        color: #5D4037;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background-color: #f5e7c8;
        padding: 20px 15px;
        border-right: 1px solid #D4AF37;
    }

    .stSidebar .sidebar-content h1, 
    .stSidebar .sidebar-content h2, 
    .stSidebar .sidebar-content h3, 
    .stSidebar .sidebar-content h4 {
        color: #8B4513 !important;
    }

    .stSidebar .sidebar-content .stNumberInput label,
    .stSidebar .sidebar-content .stSelectbox label,
    .stSidebar .sidebar-content .stTextInput label,
    .stSidebar .sidebar-content .stCheckbox label {
        color: #8B4513 !important;
    }

    /* 输入框和按钮样式 */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #f8f0e0;
        color: #8B4513;
        border: 1px solid #D4AF37;
        border-radius: 4px;
        padding: 8px 12px;
    }

    .stButton>button {
        background-color: #D4AF37;
        color: #5D4037;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }

    .stButton>button:hover {
        background-color: #C5A028;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(212, 175, 55, 0.4);
    }

    /* 标签页样式 */
    .stTabs [role="tab"] {
        background-color: #f5e7c8;
        color: #8B4513;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s;
        border: 1px solid #D4AF37;
        margin: 0 5px;
        font-size: 14px;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #D4AF37;
        color: #5D4037;
        font-weight: 600;
    }

    .stTabs [role="tab"]:hover {
        background-color: #f0d8a8;
    }

    /* 进度条样式 */
    .stProgress>div>div>div {
        background-color: #D4AF37;
    }

    /* 文本样式 */
    .stMarkdown, .stText, .stCaption {
        color: #8B4513;
    }

    /* 容器样式 */
    .stContainer {
        background-color: #f5e7c8;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #D4AF37;
    }

    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f5e7c8;
    }

    ::-webkit-scrollbar-thumb {
        background: #D4AF37;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #C5A028;
    }

    /* 表格样式 */
    .stDataFrame {
        background-color: #f8f0e0;
        border: 1px solid #D4AF37;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="title">华尔街金融量化监控系统</h1>', unsafe_allow_html=True)
    st.caption("专业量化策略监控 | MA34有效波峰 vs MA170最低值", unsafe_allow_html=True)

    st.sidebar.header('运行参数')
    st.sidebar.subheader('监控设置')
    CONFIG['max_workers'] = st.sidebar.number_input('并发线程数', 1, 12, 4, 1)
    CONFIG['play_sound'] = st.sidebar.checkbox('新信号播放声音', True)
    CONFIG['top_n_symbols'] = st.sidebar.selectbox('按交易额获取交易对数量', [50, 100, 150, 200, 300], index=4)

    # 参数设置
    st.sidebar.subheader('策略参数')
    CONFIG['peak_window'] = st.sidebar.number_input('最新K线窗口数', min_value=50, max_value=1000, value=315, step=5)
    CONFIG['peak_lr'] = st.sidebar.number_input('MA34波峰前后K线数', min_value=3, max_value=30, value=9, step=1)

    # 添加阈值设置
    st.sidebar.subheader('涨幅阈值设置 (%)')
    for tf in TIMEFRAMES:
        default_val = THRESHOLD_DEFAULTS[tf]
        CONFIG['thresholds'][tf] = st.sidebar.number_input(
            f'{tf.upper()}级别阈值',
            min_value=0.01,
            max_value=20.0,
            value=default_val,
            step=0.01,
            format="%.2f"
        )

    st.sidebar.subheader('API & 控制')
    api_key = st.sidebar.text_input('Gate.io API Key', value=API_KEY)
    api_secret = st.sidebar.text_input('Gate.io API Secret', value=API_SECRET, type='password')
    start_btn = st.sidebar.button('开始监控', key="start_monitor")

    if start_btn:
        if not api_key or not api_secret:
            st.sidebar.error('请填写 API Key/Secret')
            return
        with st.spinner('正在加载交易对数据...'):
            st.session_state.symbols_to_monitor = get_valid_symbols(api_key, api_secret, CONFIG['top_n_symbols'])
            # 展示监控标的
            st.sidebar.subheader(f"监控标的 ({len(st.session_state.symbols_to_monitor)}个)")
            symbols_container = st.sidebar.container(height=300)
            symbols_container.write(", ".join([simplify_symbol(s) for s in st.session_state.symbols_to_monitor]))
        monitor_symbols(api_key, api_secret)
    else:
        st.info('配置完成后点击侧栏的【开始监控】按钮以启动检测。')


if __name__ == '__main__':
    main()