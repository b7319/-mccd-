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
from collections import defaultdict
import logging
import queue
import threading
import os

# ========== 基础配置 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 替换为你的 Gate.io API（建议只读）
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"

beijing_tz = pytz.timezone('Asia/Shanghai')

# 增加30m和4h级别
TIMEFRAMES = {
'1m': {'interval': 60, 'max_bars': 1000, 'cache_ttl': 30},
'5m': {'interval': 300, 'max_bars': 1000, 'cache_ttl': 180},
'30m': {'interval': 1800, 'max_bars': 1000, 'cache_ttl': 600},
'4h': {'interval': 14400, 'max_bars': 1000, 'cache_ttl': 3600},
}

# 使用更简洁的策略名称
STRATEGIES = {'cluster': '密集', 'cross': '交叉'}

# 可由侧栏覆盖的运行配置
CONFIG = {
'density_threshold_pct_by_tf': {'1m': 0.13, '5m': 0.3, '30m': 1.0, '4h': 3.0}, # 每个时间级别的密集阈值
'price_diff_threshold_pct_by_tf': {'1m': 0.3, '5m': 1.0, '30m': 1.5, '4h': 3.0}, # 价格差百分比阈值
'cluster_recent_bars': 13, # 最近判定窗口（X）
'cluster_check_window': 86, # 检查窗口（Y）
'cross_cooldown_multiplier': 5, # 交叉冷却倍数
'cross_unique_window': 686, # 均线交叉密度：在最新的686根K线内只有这一个此类型的均线交叉（默认值改为686）
'cross_short_term_window': 3, # 短期确认信号窗口：最近3根K线内有MA7/MA34同向交叉
'fetch_limit': 1000,
'max_workers': 4,
'play_sound': True
}

# Session 初始化
if 'valid_signals' not in st.session_state:
    st.session_state.valid_signals = defaultdict(list)
if 'shown_signals' not in st.session_state:
    st.session_state.shown_signals = defaultdict(set)
if 'detection_round' not in st.session_state:
    st.session_state.detection_round = 0
if 'last_signal_times' not in st.session_state:
    st.session_state.last_signal_times = {}
if 'result_containers' not in st.session_state:
    st.session_state.result_containers = {}
if 'failed_symbols' not in st.session_state:
    st.session_state.failed_symbols = set()
if 'signal_queue' not in st.session_state:
    st.session_state.signal_queue = queue.Queue()
if 'symbols_cache' not in st.session_state:
    st.session_state.symbols_cache = {'symbols': [], 'timestamp': 0}
if 'symbols_to_monitor' not in st.session_state:
    st.session_state.symbols_to_monitor = []
if 'play_audio' not in st.session_state:
    st.session_state.play_audio = False

ohlcv_cache = {}


# ========== 工具函数 ==========
def generate_signal_id_cross(symbol, timeframe, cross_time, signal_type):
    ts = int(cross_time.timestamp())
    unique_str = f"{symbol}|{timeframe}|{ts}|{signal_type}|cross"
    return hashlib.md5(unique_str.encode()).hexdigest()


def generate_signal_id_cluster(symbol, timeframe, detect_time):
    ts = int(detect_time.timestamp())
    unique_str = f"{symbol}|{timeframe}|{ts}|cluster"
    return hashlib.md5(unique_str.encode()).hexdigest()


# 简化交易对名称
def simplify_symbol(symbol):
    return symbol.split('/')[0].lower()


# 生成交易对链接
def generate_symbol_link(symbol):
    base_symbol = simplify_symbol(symbol)
    return f"https://www.aicoin.com/chart/gate_{base_symbol}swapusdt"


# ========== 音频处理函数 ==========
def get_audio_base64(file_path="D:\\pycharm_study\\y1314.wav"):
    """获取音频文件的Base64编码"""
    try:
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error("警报音频文件未找到，请确认alert.wav文件存在")
        return None
    except Exception as e:
        logging.error(f"加载音频文件失败: {str(e)}")
        return None


def play_alert_sound():
    """播放警报声音"""
    audio_base64 = get_audio_base64()
    if audio_base64:
        autoplay_script = f'''
        <audio autoplay>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        <script>
        document.querySelector('audio').play().catch(error => {{
                        console.log('自动播放受阻: ', error);
                    }});
        </script>
        '''
        st.components.v1.html(autoplay_script, height=0)


# ========== 交易对列表与 OHLCV ==========
@st.cache_data(ttl=3600)
def get_valid_symbols(api_key, api_secret):
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
        top = [x['symbol'] for x in volume_data[:300]]
        final = []
        for s in top:
            try:
                exchange.fetch_ohlcv(s, '1m', limit=2)
                final.append(s)
            except Exception:
                logging.debug(f"排除交易对: {s}")
        return final
    except Exception as e:
        logging.error(f"获取交易对失败: {e}")
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
            since = exchange.milliseconds() - (cfg['max_bars'] * cfg['interval'] * 1000)
            data = exchange.fetch_ohlcv(symbol, timeframe, since, limit=cfg['max_bars'])
            # 保留原来较高的数据量要求以保证 MA453 可计算
            if data and len(data) >= max(513, 966):
                ohlcv_cache[cache_key] = (data, now)
                return data
            else:
                logging.debug(f"{symbol} {timeframe} 数据量不足: {len(data) if data else 0}")
                return None
        except Exception as e:
            logging.debug(f"获取OHLCV错误({symbol},{timeframe}): {e}")
            time.sleep(1 + attempt)
    failed_symbols.add(symbol)
    return None


def process_data_all(ohlcvs):
    if not ohlcvs or len(ohlcvs) < 513:
        return None
    timestamps = np.array([x[0] for x in ohlcvs], dtype=np.int64)
    closes = np.array([x[4] for x in ohlcvs], dtype=np.float64)
    ma7 = np.convolve(closes, np.ones(7) / 7, mode='valid')
    ma34 = np.convolve(closes, np.ones(34) / 34, mode='valid')
    ma170 = np.convolve(closes, np.ones(170) / 170, mode='valid')
    ma453 = np.convolve(closes, np.ones(453) / 453, mode='valid')
    min_len = min(len(ma7), len(ma34), len(ma170), len(ma453))
    ma7 = ma7[-min_len:]
    ma34 = ma34[-min_len:]
    ma170 = ma170[-min_len:]
    ma453 = ma453[-min_len:]
    closes = closes[-min_len:]
    timestamps = timestamps[-min_len:]
    return {'timestamps': timestamps, 'closes': closes, 'ma7': ma7, 'ma34': ma34, 'ma170': ma170, 'ma453': ma453}


# ========== 三均线密集（新版规则） ==========
def find_cluster_indices(data, pct_threshold):
    # 返回满足密集条件的索引列表（相对于 data 数组的索引）
    if data is None:
        return []
    ma34, ma170, ma453 = data['ma34'], data['ma170'], data['ma453']
    n = len(ma34)
    idxs = []
    for i in range(n):
        try:
            m34 = ma34[i];
            m170 = ma170[i];
            m453 = ma453[i]
            mx = max(m34, m170, m453)
            mn = min(m34, m170, m453)
            if mx == 0:
                continue
            if (mx - mn) / mx <= (pct_threshold / 100.0):
                idxs.append(i)
        except Exception:
            continue
    return idxs


def detect_cluster_signals(data, symbol, timeframe):
    # 新规则：在最近 X 根内发生，且在最近 Y 根内恰好只有一次发生
    res = []
    if data is None or len(data['closes']) < max(CONFIG['cluster_check_window'], CONFIG['cluster_recent_bars']):
        return res

    # 获取当前时间框架的密集阈值
    pct = CONFIG['density_threshold_pct_by_tf'][timeframe]

    idxs = find_cluster_indices(data, pct)
    if not idxs:
        return res
    n = len(data['closes'])
    recent_window_start = max(0, n - CONFIG['cluster_recent_bars'])
    check_window_start = max(0, n - CONFIG['cluster_check_window'])
    idxs_in_recent = [i for i in idxs if i >= recent_window_start]
    idxs_in_check = [i for i in idxs if i >= check_window_start]
    # 必须：最近 X 根内有发生，并且最近 Y 根内恰好只有一个发生
    if len(idxs_in_recent) >= 1 and len(idxs_in_check) == 1:
        cluster_idx = idxs_in_check[0]
        detect_time = datetime.fromtimestamp(int(data['timestamps'][cluster_idx]) / 1000, tz=beijing_tz)
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': '三均线密集排列',
            'detect_time': detect_time,
            'index': cluster_idx,
            'current_price': float(data['closes'][-1]),
            'ma34': float(data['ma34'][cluster_idx]),
            'ma170': float(data['ma170'][cluster_idx]),
            'ma453': float(data['ma453'][cluster_idx]),
            'density_percent': ((max(data['ma34'][cluster_idx], data['ma170'][cluster_idx],
                                 data['ma453'][cluster_idx]) - min(data['ma34'][cluster_idx],
                                                                   data['ma170'][cluster_idx],
                                                                   data['ma453'][cluster_idx])) / max(
                data['ma34'][cluster_idx], data['ma170'][cluster_idx], data['ma453'][cluster_idx])) * 100
        }
        res.append(signal)
    return res


def render_cluster_signal(tf, signal):
    position = "价格在均线之间"
    current_price = signal['current_price']
    max_ma = max(signal['ma34'], signal['ma170'], signal['ma453'])
    min_ma = min(signal['ma34'], signal['ma170'], signal['ma453'])
    if current_price > max_ma:
        position = "价格在均线上方"
        position_color = "green"
    elif current_price < min_ma:
        position_color = "red"
    else:
        position_color = "orange"
    density_percent = signal['density_percent']
    if density_percent < 0.1:
        density_color = "purple"
    elif density_percent < 0.2:
        density_color = "blue"
    else:
        density_color = "darkblue"

    # 简化交易对名称并添加超链接
    symbol_simple = simplify_symbol(signal['symbol'])
    symbol_link = generate_symbol_link(signal['symbol'])

    content = (
        f"<div style='margin-bottom: 10px; border-left: 4px solid {position_color}; padding-left: 8px;'>"
        f"<a href='{symbol_link}' target='_blank' style='text-decoration: none; color: {position_color}; font-weight: bold;'>"
        f"🔍🔍🔍🔍 {symbol_simple} [{tf.upper()}] {position}</a> | "
        f"密集度: <span style='color: {density_color}; font-weight: bold;'>{density_percent:.3f}%</span> | "
        f"现价: {signal['current_price']:.4f} | MA34: {signal['ma34']:.4f} | MA170: {signal['ma170']:.4f} | MA453: {signal['ma453']:.4f} | 时间: {signal['detect_time'].strftime('%H:%M:%S')}"
        f"</div>"
    )
    st.markdown(content, unsafe_allow_html=True)


# ========== 双均线交叉（修改为支持4h级别） ==========
def detect_cross_signals(data, timeframe):
    if data is None or len(data['closes']) < 513:
        return [], None
    ma7, ma34, ma170, ma453 = data['ma7'], data['ma34'], data['ma170'], data['ma453']
    closes, timestamps = data['closes'], data['timestamps']
    current_price = float(closes[-1])
    golden_cross_170_453 = (ma170 > ma453) & (np.roll(ma170, 1) <= np.roll(ma453, 1))
    golden_cross_34_170 = (ma34 > ma170) & (np.roll(ma34, 1) <= np.roll(ma170, 1))
    golden_cross_34_453 = (ma34 > ma453) & (np.roll(ma34, 1) <= np.roll(ma453, 1))
    golden_cross_7_34 = (ma7 > ma34) & (np.roll(ma7, 1) <= np.roll(ma34, 1))
    death_cross_170_453 = (ma170 < ma453) & (np.roll(ma170, 1) >= np.roll(ma453, 1))
    death_cross_34_170 = (ma34 < ma170) & (np.roll(ma34, 1) >= np.roll(ma170, 1))
    death_cross_34_453 = (ma34 < ma453) & (np.roll(ma34, 1) >= np.roll(ma453, 1))
    death_cross_7_34 = (ma7 < ma34) & (np.roll(ma7, 1) >= np.roll(ma34, 1))

    valid_signals = []
    recent_indices = np.arange(len(closes))[-86:]
    # 使用配置参数控制短期确认信号窗口
    short_term_window = CONFIG['cross_short_term_window']
    short_term_indices = np.arange(len(closes))[-short_term_window:]
    cross_unique_window = CONFIG['cross_unique_window']  # 获取交叉唯一性窗口大小

    # 获取当前时间框架的价格差百分比阈值
    price_diff_threshold = CONFIG['price_diff_threshold_pct_by_tf'][timeframe] / 100.0

    short_term_signals = []
    for idx in short_term_indices:
        if golden_cross_7_34[idx]:
            cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz)
            short_term_signals.append(('多头', cross_time, closes[idx]))
        elif death_cross_7_34[idx]:
            cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz)
            short_term_signals.append(('空头', cross_time, closes[idx]))

    for idx in recent_indices:
        try:
            if golden_cross_170_453[idx]:
                signal_type = 'MA170金叉MA453';
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz);
                cross_price = closes[idx]
                if current_price > cross_price and np.sum(golden_cross_170_453[-cross_unique_window:]) == 1:
                    # 检查短期信号：方向匹配、时间匹配、价格差在阈值内
                    has_match = any(
                        d == '多头' and
                        st_t >= cross_time and
                        abs(st_p - cross_price) / cross_price <= price_diff_threshold
                        for d, st_t, st_p in short_term_signals
                    )
                    if has_match:
                        valid_signals.append(('多头', signal_type, cross_time, cross_price))
            elif golden_cross_34_170[idx]:
                signal_type = 'MA34金叉MA170';
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz);
                cross_price = closes[idx]
                if current_price > cross_price and np.sum(golden_cross_34_170[-cross_unique_window:]) == 1:
                    has_match = any(
                        d == '多头' and
                        st_t >= cross_time and
                        abs(st_p - cross_price) / cross_price <= price_diff_threshold
                        for d, st_t, st_p in short_term_signals
                    )
                    if has_match:
                        valid_signals.append(('多头', signal_type, cross_time, cross_price))
            elif golden_cross_34_453[idx]:
                signal_type = 'MA34金叉MA453';
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz);
                cross_price = closes[idx]
                if current_price > cross_price and np.sum(golden_cross_34_453[-cross_unique_window:]) == 1:
                    has_match = any(
                        d == '多头' and
                        st_t >= cross_time and
                        abs(st_p - cross_price) / cross_price <= price_diff_threshold
                        for d, st_t, st_p in short_term_signals
                    )
                    if has_match:
                        valid_signals.append(('多头', signal_type, cross_time, cross_price))
            elif death_cross_170_453[idx]:
                signal_type = 'MA170死叉MA453';
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz);
                cross_price = closes[idx]
                if current_price < cross_price and np.sum(death_cross_170_453[-cross_unique_window:]) == 1:
                    has_match = any(
                        d == '空头' and
                        st_t >= cross_time and
                        abs(st_p - cross_price) / cross_price <= price_diff_threshold
                        for d, st_t, st_p in short_term_signals
                    )
                    if has_match:
                        valid_signals.append(('空头', signal_type, cross_time, cross_price))
            elif death_cross_34_170[idx]:
                signal_type = 'MA34死叉MA170';
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz);
                cross_price = closes[idx]
                if current_price < cross_price and np.sum(death_cross_34_170[-cross_unique_window:]) == 1:
                    has_match = any(
                        d == '空头' and
                        st_t >= cross_time and
                        abs(st_p - cross_price) / cross_price <= price_diff_threshold
                        for d, st_t, st_p in short_term_signals
                    )
                    if has_match:
                        valid_signals.append(('空头', signal_type, cross_time, cross_price))
            elif death_cross_34_453[idx]:
                signal_type = 'MA34死叉MA453';
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz);
                cross_price = closes[idx]
                if current_price < cross_price and np.sum(death_cross_34_453[-cross_unique_window:]) == 1:
                    has_match = any(
                        d == '空头' and
                        st_t >= cross_time and
                        abs(st_p - cross_price) / cross_price <= price_diff_threshold
                        for d, st_t, st_p in short_term_signals
                    )
                    if has_match:
                        valid_signals.append(('空头', signal_type, cross_time, cross_price))
        except Exception:
            continue

    return valid_signals, current_price


def render_cross_signal(tf, signal):
    direction, signal_type, cross_time, cross_price, current_price = (
        signal['direction'], signal['signal_type'], signal['cross_time'], signal['cross_price'], signal['current_price']
    )
    price_change = ((current_price - cross_price) / cross_price) * 100
    direction_color = "green" if direction == '多头' else "red"
    signal_icon = "⏳⏳⏳⏳⏳⏳⏳⏳⏳"
    price_change_color = "green" if price_change > 0 else "red"
    price_change_arrow = "↑" if price_change > 0 else "↓"

    # 简化交易对名称并添加超链接
    symbol_simple = simplify_symbol(signal['symbol'])
    symbol_link = generate_symbol_link(signal['symbol'])

    content = (
        f"<div style='margin-bottom: 10px; border-left: 4px solid {direction_color}; padding-left: 8px;'>"
        f"<a href='{symbol_link}' target='_blank' style='text-decoration: none; color: {direction_color}; font-weight: bold;'>"
        f"{signal_icon} {symbol_simple} [{tf.upper()}] {direction}</a> | "
        f"{signal_type} | 交叉价: {cross_price:.4f} | 现价: {current_price:.4f} | "
        f"<span style='color: {price_change_color};'>变化: {price_change:.2f}% {price_change_arrow}</span> | 时间: {cross_time.strftime('%H:%M:%S')} | 检测: {signal['detect_time']}"
        f"</div>"
    )
    st.markdown(content, unsafe_allow_html=True)


# ========== UI：标签页 ==========
def build_tabs():
    # 优化：将标签导航栏固定在页面底部中间位置
    st.markdown(
        """
        <style>
        .stTabs > div > div:first-child {
            position: fixed;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background-color: var(--default-background-color);
            padding: 8px 16px;
            border-radius: 8px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .stTabs > div > div:last-child {
            margin-bottom: 60px; /* 为底部标签栏留出空间 */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    labels = [];
    keys = []
    for tf in TIMEFRAMES:
        for sk, sn in STRATEGIES.items():
            # 使用更简洁的标签格式：时间框架 + 策略简称
            labels.append(f"{tf.upper()}·{sn}");
            keys.append((tf, sk))
    tabs = st.tabs(labels)
    for i, k in enumerate(keys):
        with tabs[i]:
            container = st.container();
            placeholder = st.empty()
            st.session_state.result_containers[k] = {'container': container, 'placeholder': placeholder}


def update_tab_content(tf, strategy):
    container = st.session_state.result_containers[(tf, strategy)]['container']
    placeholder = st.session_state.result_containers[(tf, strategy)]['placeholder']
    with container:
        with placeholder.container():
            signals = st.session_state.valid_signals[(tf, strategy)][-80:]
            if not signals:
                st.markdown("暂无信号")
                return
            for s in signals:
                if strategy == 'cluster':
                    render_cluster_signal(tf, s)
                else:
                    render_cross_signal(tf, s)


# ========== 处理与监控 ==========
def process_symbol_timeframe(exchange, symbol, timeframe, failed_symbols):
    out_cluster = [];
    out_cross = []
    try:
        ohlcvs = get_cached_ohlcv(exchange, symbol, timeframe, failed_symbols)
        if ohlcvs is None:
            return symbol, out_cluster, out_cross
        data = process_data_all(ohlcvs)
        cluster_list = detect_cluster_signals(data, symbol, timeframe)
        for s in cluster_list:
            out_cluster.append(s)
        cross_list, current_price = detect_cross_signals(data, timeframe)
        detect_time = datetime.now(beijing_tz)
        for (direction, signal_type, cross_time, cross_price) in cross_list:
            out_cross.append({
                'symbol': symbol, 'timeframe': timeframe, 'signal_type': f"双均线组合（{signal_type}）",
                'direction': direction, 'cross_time': cross_time, 'cross_price': cross_price,
                'current_price': current_price, 'detect_time': detect_time.strftime('%H:%M:%S')
            })
        return symbol, out_cluster, out_cross
    except Exception as e:
        logging.debug(f"处理 {symbol}({timeframe}) 异常: {e}")
        return symbol, out_cluster, out_cross


def monitor_symbols(api_key, api_secret):
    exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 30000})

    build_tabs()
    st.sidebar.subheader("待监控交易对（按成交量前300）")
    if not st.session_state.symbols_to_monitor:
        st.session_state.symbols_to_monitor = get_valid_symbols(api_key, api_secret)
    cols = st.sidebar.columns(4)
    syms = st.session_state.symbols_to_monitor
    per_col = max(1, len(syms) // 4)
    for i, c in enumerate(cols):
        start = i * per_col;
        end = start + per_col
        if i == len(cols) - 1: end = len(syms)
        with c:
            for s in syms[start:end]:
                color = 'red' if s in st.session_state.failed_symbols else 'green'
                # 简化交易对名称显示
                simple_s = simplify_symbol(s)
                st.markdown(f"<span style='color:{color};'>• {simple_s}</span>", unsafe_allow_html=True)
    progress_bar = st.sidebar.progress(0);
    status_text = st.sidebar.empty();
    stats = st.sidebar.empty()
    max_workers = CONFIG.get('max_workers', 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            start_time = time.time();
            st.session_state.detection_round += 1
            new_signals = defaultdict(int)
            symbols = [s for s in st.session_state.symbols_to_monitor if s not in st.session_state.failed_symbols]
            if not symbols:
                time.sleep(30);
                continue
            failed_copy = st.session_state.failed_symbols.copy()
            futures = []
            for s in symbols:
                for tf in TIMEFRAMES:
                    futures.append(executor.submit(process_symbol_timeframe, exchange, s, tf, failed_copy))
            for i, fut in enumerate(as_completed(futures)):
                symbol, cluster_signals, cross_signals = fut.result()
                progress = (i + 1) / len(futures);
                progress_bar.progress(progress);
                status_text.text(f"检测进度: {progress * 100:.1f}%")
                for sig in cluster_signals:
                    st.session_state.signal_queue.put(('cluster', sig['timeframe'], symbol, sig))
                for sig in cross_signals:
                    st.session_state.signal_queue.put(('cross', sig['timeframe'], symbol, sig))
            play_sound = False
            while not st.session_state.signal_queue.empty():
                strat, tf, symbol, signal = st.session_state.signal_queue.get()
                if tf is None: continue
                if strat == 'cluster':
                    # 使用发生时点的唯一 ID，保证单次展示
                    signal_id = generate_signal_id_cluster(symbol, tf, signal['detect_time'])
                    if signal_id in st.session_state.shown_signals[(tf, 'cluster')]:
                        continue
                    st.session_state.valid_signals[(tf, 'cluster')].append(signal)
                    st.session_state.shown_signals[(tf, 'cluster')].add(signal_id)
                    new_signals[(tf, 'cluster')] += 1;
                    play_sound = True
                else:  # cross
                    cross_time = signal['cross_time']
                    signal_id = generate_signal_id_cross(symbol, tf, cross_time, signal['signal_type'])
                    key = (symbol, tf, 'cross');
                    last_time = st.session_state.last_signal_times.get(key)
                    interval = TIMEFRAMES[tf]['interval'] * CONFIG.get('cross_cooldown_multiplier', 5)
                    if last_time and (cross_time - last_time).total_seconds() < interval:
                        continue
                    if signal_id in st.session_state.shown_signals[(tf, 'cross')]:
                        continue
                    st.session_state.valid_signals[(tf, 'cross')].append(signal)
                    st.session_state.shown_signals[(tf, 'cross')].add(signal_id)
                    st.session_state.last_signal_times[key] = cross_time
                    new_signals[(tf, 'cross')] += 1;
                    play_sound = True
            if play_sound and CONFIG.get('play_sound', True):
                try:
                    # 播放警报声音
                    play_alert_sound()
                    # 显示视觉提示
                    st.sidebar.success("检测到新信号！")
                except Exception as e:
                    logging.debug(f"播放声音失败: {e}")
            for tf in TIMEFRAMES:
                update_tab_content(tf, 'cluster');
                update_tab_content(tf, 'cross')
            stats.markdown(
                f"轮次: {st.session_state.detection_round} | 新信号: {dict(new_signals)} | 失败交易对: {len(st.session_state.failed_symbols)}")
            elapsed = time.time() - start_time;
            sleep_time = max(45 - elapsed, 30);
            time.sleep(sleep_time)


# ========== 入口：侧栏参数 ==========
def main():
    st.title('一体化监控（v2） - 三均线密集 & 双均线交叉（1m/5m/30m/4h）')
    st.sidebar.header('运行参数（可调整）')

    # 密集阈值配置 - 每个时间框架单独设置
    st.sidebar.subheader('密集阈值配置（%）')
    CONFIG['density_threshold_pct_by_tf']['1m'] = st.sidebar.number_input(
        '1m级别', min_value=0.01, max_value=5.0, value=0.13, step=0.01, key='density_1m')
    CONFIG['density_threshold_pct_by_tf']['5m'] = st.sidebar.number_input(
        '5m级别', min_value=0.01, max_value=5.0, value=0.3, step=0.01, key='density_5m')
    CONFIG['density_threshold_pct_by_tf']['30m'] = st.sidebar.number_input(
        '30m级别', min_value=0.01, max_value=5.0, value=1.0, step=0.01, key='density_30m')
    CONFIG['density_threshold_pct_by_tf']['4h'] = st.sidebar.number_input(
        '4h级别', min_value=0.01, max_value=5.0, value=3.0, step=0.01, key='density_4h')

    # 价格差百分比阈值配置 - 每个时间框架单独设置
    st.sidebar.subheader('价格差百分比阈值（%）')
    CONFIG['price_diff_threshold_pct_by_tf']['1m'] = st.sidebar.number_input(
        '1m级别价格差阈值', min_value=0.01, max_value=5.0, value=0.3, step=0.01, key='price_diff_1m')
    CONFIG['price_diff_threshold_pct_by_tf']['5m'] = st.sidebar.number_input(
        '5m级别价格差阈值', min_value=0.01, max_value=5.0, value=1.0, step=0.01, key='price_diff_5m')
    CONFIG['price_diff_threshold_pct_by_tf']['30m'] = st.sidebar.number_input(
        '30m级别价格差阈值', min_value=0.01, max_value=5.0, value=1.5, step=0.01, key='price_diff_30m')
    CONFIG['price_diff_threshold_pct_by_tf']['4h'] = st.sidebar.number_input(
        '4h级别价格差阈值', min_value=0.01, max_value=5.0, value=3.0, step=0.01, key='price_diff_4h')

    # 其他参数配置
    st.sidebar.subheader('其他参数')
    CONFIG['cluster_recent_bars'] = st.sidebar.number_input(
        '密集判定最近K线数 (X)', min_value=3, max_value=100, value=13, step=1)
    CONFIG['cluster_check_window'] = st.sidebar.number_input(
        '密集检查窗口 (Y)', min_value=10, max_value=200, value=86, step=1)
    # 关键修改1：均线交叉密度默认值改为686
    CONFIG['cross_unique_window'] = st.sidebar.number_input(
        '均线交叉密度（唯一性窗口）', min_value=10, max_value=1000, value=686, step=1)  # 默认值686
    # 关键修改2：添加短期确认信号参数
    CONFIG['cross_short_term_window'] = st.sidebar.number_input(
        '短期确认信号窗口', min_value=1, max_value=10, value=3, step=1,
        help="最近多少根K线内有MA7/MA34同向交叉（默认3根）")
    CONFIG['cross_cooldown_multiplier'] = st.sidebar.number_input(
        '双均线交叉冷却倍数 (interval * X)', min_value=1, max_value=20, value=5, step=1)
    CONFIG['fetch_limit'] = st.sidebar.number_input(
        '拉取K线数量 (fetch limit)', min_value=600, max_value=2000, value=968, step=1)
    CONFIG['max_workers'] = st.sidebar.number_input(
        '并发线程数', min_value=1, max_value=12, value=4, step=1)
    CONFIG['play_sound'] = st.sidebar.checkbox('新信号播放声音', value=True)

    for tf in TIMEFRAMES:
        TIMEFRAMES[tf]['max_bars'] = int(CONFIG['fetch_limit'])
        TIMEFRAMES[tf]['cache_ttl'] = 30 if tf == '1m' else (180 if tf == '5m' else 600)

    st.sidebar.markdown('---')
    st.sidebar.subheader('API & 控制')
    api_key = st.sidebar.text_input('Gate.io API Key', value=API_KEY)
    api_secret = st.sidebar.text_input('Gate.io API Secret', value=API_SECRET, type='password')
    start_btn = st.sidebar.button('开始监控')
    st.sidebar.markdown(
        '提示：三均线密集的判定为：在最近 X 根K线内发生，且在最近 Y 根内唯一，从而排除反复窄幅盘整的噪声。'.format())
    with st.expander('筛选规则说明（简要）', expanded=False):
        st.markdown('''- 三均线密集：MA34/MA170/MA453 在某一根 K 线处最大最小差 <= 密集阈值，且该密集发生位置满足：
1) 出现在最近 X 根 K 线内（X 可调，默认为 13）
2) 在最近 Y 根 K 线内（Y 可调，默认为 86）恰好只有一次密集发生 —— 用来排除反复盘整造成的噪声
- 双均线交叉：沿用原 513.py 的双均线组合 + MA7 与 MA34 的短期确认
- 冷却：三均线密集 -> 每一次具体发生的信号只展示一次（使用发生时点唯一ID）；双均线交叉 -> interval * 冷却倍数秒内不重复''')
    if start_btn:
        if not api_key or not api_secret:
            st.sidebar.error('请填写 API Key/Secret')
            return
        st.session_state.symbols_to_monitor = get_valid_symbols(api_key, api_secret)
        monitor_symbols(api_key, api_secret)
    else:
        st.info('配置完成后点击侧栏的【开始监控】按钮以启动检测。')


if __name__ == '__main__':
    main()