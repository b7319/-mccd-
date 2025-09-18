import ccxt
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import pytz
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import logging
import queue
import html
import sqlite3
import json
import requests
import os

# ========== 基础配置 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 从环境变量获取 API 密钥，优先于硬编码
API_KEY = os.getenv('GATEIO_API_KEY', "YOUR_API_KEY")
API_SECRET = os.getenv('GATEIO_API_SECRET', "YOUR_API_SECRET")

beijing_tz = pytz.timezone('Asia/Shanghai')

# 时间框架配置
TIMEFRAMES = {
    '1m': {'interval': 60, 'max_bars': 1000, 'cache_ttl': 30},
    '5m': {'interval': 300, 'max_bars': 1000, 'cache_ttl': 180},
    '30m': {'interval': 1800, 'max_bars': 1000, 'cache_ttl': 600},
    '4h': {'interval': 14400, 'max_bars': 1000, 'cache_ttl': 3600},
}

STRATEGIES = {'cluster': '密集', 'cross': '交叉'}

# 运行配置
CONFIG = {
    'density_threshold_pct_by_tf': {'1m': 0.13, '5m': 0.3, '30m': 1.0, '4h': 3.0},
    'price_diff_threshold_pct_by_tf': {'1m': 0.86, '5m': 1.86, '30m': 3.86, '4h': 4.86},
    'cluster_recent_bars': 13,
    'cluster_check_window': 86,
    'cross_cooldown_multiplier': 5,
    'cross_unique_window': 513,
    'cross_short_term_window': 3,
    'fetch_limit': 1000,
    'max_workers': 4,
    'top_n_symbols': 300,
    #'heartbeat_url': "https://cg141319.streamlit.app/health"  # 替换为你的 Streamlit 应用健康检查 URL
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
if 'all_signals' not in st.session_state:
    st.session_state.all_signals = []
if 'monitoring_thread' not in st.session_state:
    st.session_state.monitoring_thread = None
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = 0
if 'latest_signals_ticker' not in st.session_state:
    st.session_state.latest_signals_ticker = deque(maxlen=68)
if 'ticker_seen' not in st.session_state:
    st.session_state.ticker_seen = set()
if 'right_panel_placeholder' not in st.session_state:
    st.session_state.right_panel_placeholder = None
if 'right_panel_css_injected' not in st.session_state:
    st.session_state.right_panel_css_injected = False

ohlcv_cache = {}

# 设置页面配置
st.set_page_config(
    page_title="一体化监控 v2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 数据库初始化 ==========
def init_db():
    conn = sqlite3.connect('signals.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id TEXT PRIMARY KEY,
            symbol TEXT,
            timeframe TEXT,
            strategy TEXT,
            signal_data TEXT,
            detect_time TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_signal_to_db(signal_id, symbol, timeframe, strategy, signal_data):
    try:
        conn = sqlite3.connect('signals.db')
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO signals (id, symbol, timeframe, strategy, signal_data, detect_time) VALUES (?, ?, ?, ?, ?, ?)',
                  (signal_id, symbol, timeframe, strategy, json.dumps(signal_data, default=str), signal_data.get('detect_time')))
        conn.commit()
        conn.close()
        logging.info(f"信号保存到数据库: {symbol} {timeframe} {strategy}")
    except Exception as e:
        logging.error(f"保存信号到数据库失败: {e}")

def load_signals_from_db():
    try:
        conn = sqlite3.connect('signals.db')
        c = conn.cursor()
        c.execute('SELECT id, symbol, timeframe, strategy, signal_data FROM signals')
        rows = c.fetchall()
        for row in rows:
            signal_id, symbol, timeframe, strategy, signal_data = row
            signal = json.loads(signal_data)
            # 确保 detect_time 是 datetime 对象
            if isinstance(signal['detect_time'], str):
                try:
                    signal['detect_time'] = datetime.fromisoformat(signal['detect_time'])
                except:
                    signal['detect_time'] = datetime.now(beijing_tz)
            st.session_state.valid_signals[(timeframe, strategy)].append(signal)
            st.session_state.shown_signals[(timeframe, strategy)].add(signal_id)
            if strategy == 'cross':
                cross_time = signal.get('cross_time')
                if isinstance(cross_time, str):
                    try:
                        cross_time = datetime.fromisoformat(cross_time)
                    except:
                        cross_time = datetime.now(beijing_tz)
                st.session_state.last_signal_times[(symbol, timeframe, strategy)] = cross_time
            _enqueue_latest(signal, timeframe, strategy, symbol, signal_id)
        conn.close()
        logging.info("历史信号从数据库加载完成")
    except Exception as e:
        logging.error(f"加载信号失败: {e}")

def clean_old_signals(max_age_hours=24):
    try:
        conn = sqlite3.connect('signals.db')
        c = conn.cursor()
        threshold = (datetime.now(beijing_tz) - timedelta(hours=max_age_hours)).isoformat()
        c.execute('DELETE FROM signals WHERE detect_time < ?', (threshold,))
        conn.commit()
        conn.close()
        logging.info(f"清理了超过 {max_age_hours} 小时的旧信号")
    except Exception as e:
        logging.error(f"清理旧信号失败: {e}")

# ========== 工具函数 ==========
def escape_html(text):
    if text is None:
        return ""
    return html.escape(str(text))

def generate_signal_id_cross(symbol, timeframe, cross_time, signal_type):
    ts = int(cross_time.timestamp())
    unique_str = f"{symbol}|{timeframe}|{ts}|{signal_type}|cross"
    return hashlib.md5(unique_str.encode()).hexdigest()

def generate_signal_id_cluster(symbol, timeframe, detect_time):
    ts = int(detect_time.timestamp())
    unique_str = f"{symbol}|{timeframe}|{ts}|cluster"
    return hashlib.md5(unique_str.encode()).hexdigest()

def simplify_symbol(symbol):
    return symbol.split('/')[0].lower()

def generate_symbol_link(symbol):
    base_symbol = simplify_symbol(symbol)
    return f"https://www.aicoin.com/chart/gate_{base_symbol}swapusdt"

# ========== 交易对列表与 OHLCV ==========
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
            if market.get('type') == 'swap' and market.get('active') and ticker.get('quoteVolume') is not None and market.get('quote') == 'USDT':
                volume_data.append({'symbol': symbol, 'volume': ticker['quoteVolume']})
        volume_data.sort(key=lambda x: x['volume'], reverse=True)
        top = [x['symbol'] for x in volume_data[:top_n]]
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
    try:
        timestamps = np.array([x[0] for x in ohlcvs], dtype=np.int64)
        closes = np.array([float(x[4]) for x in ohlcvs], dtype=np.float64)
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
    except ValueError:
        logging.error(f"无效的收盘价数据: {ohlcvs}")
        return None

# ========== 三均线密集 ==========
def find_cluster_indices(data, pct_threshold):
    if data is None:
        return []
    ma34, ma170, ma453 = data['ma34'], data['ma170'], data['ma453']
    n = len(ma34)
    idxs = []
    for i in range(n):
        try:
            m34 = ma34[i]
            m170 = ma170[i]
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
    res = []
    if data is None or len(data['closes']) < max(CONFIG['cluster_check_window'], CONFIG['cluster_recent_bars']):
        return res
    pct = CONFIG['density_threshold_pct_by_tf'][timeframe]
    idxs = find_cluster_indices(data, pct)
    if not idxs:
        return res
    n = len(data['closes'])
    recent_window_start = max(0, n - CONFIG['cluster_recent_bars'])
    check_window_start = max(0, n - CONFIG['cluster_check_window'])
    idxs_in_recent = [i for i in idxs if i >= recent_window_start]
    idxs_in_check = [i for i in idxs if i >= check_window_start]
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
    cluster_price = min(signal['ma34'], signal['ma170'], signal['ma453'])
    price_change = ((current_price - cluster_price) / cluster_price) * 100
    price_change_color = "green" if price_change > 0 else "red"
    price_change_arrow = "↑" if price_change > 0 else "↓"
    symbol_simple = simplify_symbol(signal['symbol'])
    symbol_link = generate_symbol_link(signal['symbol'])
    content = (
        f"<div style='margin-bottom: 10px; border-left: 4px solid {position_color}; padding-left: 8px;'>"
        f"<a href='{symbol_link}' target='_blank' style='text-decoration: none; color: {position_color}; font-weight: bold;'>"
        f"🔍 {symbol_simple} [{tf.upper()}] {position}</a> | "
        f"密集度: <span style='color: {density_color}; font-weight: bold;'>{density_percent:.3f}%</span> | "
        f"现价: {signal['current_price']:.4f} | "
        f"涨跌幅: <span style='color: {price_change_color}; font-weight: bold;'>{price_change:.2f}% {price_change_arrow}</span> | "
        f"MA34: {signal['ma34']:.4f} | MA170: {signal['ma170']:.4f} | MA453: {signal['ma453']:.4f} | 时间: {signal['detect_time'].strftime('%H:%M:%S')}"
        f"</div>"
    )
    st.markdown(content, unsafe_allow_html=True)

# ========== 双均线交叉 ==========
def detect_cross_signals(data, timeframe, symbol):
    if data is None or len(data['closes']) < 513:
        return [], None
    ma7, ma34, ma170, ma453 = data['ma7'], data['ma34'], data['ma170'], data['ma453']
    closes, timestamps = data['closes'], data['timestamps']
    current_price = float(closes[-1])
    golden_cross_170_453 = (ma170 > ma453) & (np.roll(ma170, 1) <= np.roll(ma453, 1))
    golden_cross_34_170 = (ma34 > ma170) & (np.roll(ma34, 1) <= np.roll(ma170, 1))
    golden_cross_34_453 = (ma34 > ma453) & (np.roll(ma34, 1) <= np.roll(ma453, 1))
    death_cross_170_453 = (ma170 < ma453) & (np.roll(ma170, 1) >= np.roll(ma453, 1))
    death_cross_34_170 = (ma34 < ma170) & (np.roll(ma34, 1) >= np.roll(ma170, 1))
    death_cross_34_453 = (ma34 < ma453) & (np.roll(ma34, 1) >= np.roll(ma453, 1))
    golden_cross_7_34 = (ma7 > ma34) & (np.roll(ma7, 1) <= np.roll(ma34, 1))
    death_cross_7_34 = (ma7 < ma34) & (np.roll(ma7, 1) >= np.roll(ma34, 1))
    valid_signals = []
    recent_indices = np.arange(len(closes))[-86:]
    short_term_window = CONFIG['cross_short_term_window']
    short_term_indices = np.arange(len(closes))[-short_term_window:]
    cross_unique_window = CONFIG['cross_unique_window']
    density_threshold = CONFIG['price_diff_threshold_pct_by_tf'][timeframe] / 100.0
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
                signal_type = 'MA170金叉MA453'
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz)
                cross_price = closes[idx]
                if (np.sum(golden_cross_170_453[-cross_unique_window:]) == 1 and
                        np.sum(death_cross_170_453[-cross_unique_window:]) == 0):
                    for d, st_t, st_p in short_term_signals:
                        if d == '多头' and st_t >= cross_time:
                            high_price = max(st_p, cross_price)
                            low_price = min(st_p, cross_price)
                            density_ratio = (high_price - low_price) / low_price
                            if density_ratio <= density_threshold:
                                density_percent = density_ratio * 100
                                valid_signals.append(('多头', signal_type, cross_time, cross_price,
                                                      current_price, density_percent))
                                break
            elif golden_cross_34_170[idx]:
                signal_type = 'MA34金叉MA170'
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz)
                cross_price = closes[idx]
                if (np.sum(golden_cross_34_170[-cross_unique_window:]) == 1 and
                        np.sum(death_cross_34_170[-cross_unique_window:]) == 0):
                    for d, st_t, st_p in short_term_signals:
                        if d == '多头' and st_t >= cross_time:
                            high_price = max(st_p, cross_price)
                            low_price = min(st_p, cross_price)
                            density_ratio = (high_price - low_price) / low_price
                            if density_ratio <= density_threshold:
                                density_percent = density_ratio * 100
                                valid_signals.append(('多头', signal_type, cross_time, cross_price,
                                                      current_price, density_percent))
                                break
            elif golden_cross_34_453[idx]:
                signal_type = 'MA34金叉MA453'
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz)
                cross_price = closes[idx]
                if (np.sum(golden_cross_34_453[-cross_unique_window:]) == 1 and
                        np.sum(death_cross_34_453[-cross_unique_window:]) == 0):
                    for d, st_t, st_p in short_term_signals:
                        if d == '多头' and st_t >= cross_time:
                            high_price = max(st_p, cross_price)
                            low_price = min(st_p, cross_price)
                            density_ratio = (high_price - low_price) / low_price
                            if density_ratio <= density_threshold:
                                density_percent = density_ratio * 100
                                valid_signals.append(('多头', signal_type, cross_time, cross_price,
                                                      current_price, density_percent))
                                break
            elif death_cross_170_453[idx]:
                signal_type = 'MA170死叉MA453'
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz)
                cross_price = closes[idx]
                if (np.sum(death_cross_170_453[-cross_unique_window:]) == 1 and
                        np.sum(golden_cross_170_453[-cross_unique_window:]) == 0):
                    for d, st_t, st_p in short_term_signals:
                        if d == '空头' and st_t >= cross_time:
                            high_price = max(st_p, cross_price)
                            low_price = min(st_p, cross_price)
                            density_ratio = (high_price - low_price) / low_price
                            if density_ratio <= density_threshold:
                                density_percent = density_ratio * 100
                                valid_signals.append(('空头', signal_type, cross_time, cross_price,
                                                      current_price, density_percent))
                                break
            elif death_cross_34_170[idx]:
                signal_type = 'MA34死叉MA170'
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz)
                cross_price = closes[idx]
                if (np.sum(death_cross_34_170[-cross_unique_window:]) == 1 and
                        np.sum(golden_cross_34_170[-cross_unique_window:]) == 0):
                    for d, st_t, st_p in short_term_signals:
                        if d == '空头' and st_t >= cross_time:
                            high_price = max(st_p, cross_price)
                            low_price = min(st_p, cross_price)
                            density_ratio = (high_price - low_price) / low_price
                            if density_ratio <= density_threshold:
                                density_percent = density_ratio * 100
                                valid_signals.append(('空头', signal_type, cross_time, cross_price,
                                                      current_price, density_percent))
                                break
            elif death_cross_34_453[idx]:
                signal_type = 'MA34死叉MA453'
                cross_time = datetime.fromtimestamp(int(timestamps[idx]) / 1000, tz=beijing_tz)
                cross_price = closes[idx]
                if (np.sum(death_cross_34_453[-cross_unique_window:]) == 1 and
                        np.sum(golden_cross_34_453[-cross_unique_window:]) == 0):
                    for d, st_t, st_p in short_term_signals:
                        if d == '空头' and st_t >= cross_time:
                            high_price = max(st_p, cross_price)
                            low_price = min(st_p, cross_price)
                            density_ratio = (high_price - low_price) / low_price
                            if density_ratio <= density_threshold:
                                density_percent = density_ratio * 100
                                valid_signals.append(('空头', signal_type, cross_time, cross_price,
                                                      current_price, density_percent))
                                break
        except Exception as e:
            logging.error(f"处理交叉信号时出错: {e}")
            continue
    formatted_signals = []
    for direction, signal_type, cross_time, cross_price, current_price, density_percent in valid_signals:
        price_change = ((current_price - cross_price) / cross_price) * 100
        formatted_signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'signal_type': signal_type,
            'cross_time': cross_time,
            'cross_price': cross_price,
            'current_price': current_price,
            'price_change': price_change,
            'density_percent': density_percent,
            'detect_time': datetime.now(beijing_tz).strftime('%H:%M:%S')
        })
    return formatted_signals, current_price

def render_cross_signal(tf, signal):
    direction, signal_type, cross_time, cross_price, current_price, price_change, density_percent = (
        signal['direction'], signal['signal_type'], signal['cross_time'],
        signal['cross_price'], signal['current_price'], signal['price_change'], signal['density_percent']
    )
    direction_color = "green" if direction == '多头' else "red"
    signal_icon = "⏳"
    price_change_color = "green" if price_change > 0 else "red"
    price_change_arrow = "↑" if price_change > 0 else "↓"
    if density_percent < 0.1:
        density_color = "purple"
    elif density_percent < 0.2:
        density_color = "blue"
    else:
        density_color = "darkblue"
    symbol_simple = simplify_symbol(signal['symbol'])
    symbol_link = generate_symbol_link(signal['symbol'])
    content = (
        f"<div style='margin-bottom: 10px; border-left: 4px solid {direction_color}; padding-left: 8px;'>"
        f"<a href='{symbol_link}' target='_blank' style='text-decoration: none; color: {direction_color}; font-weight: bold;'>"
        f"{signal_icon} {symbol_simple} [{tf.upper()}] {direction}</a> | "
        f"{signal_type} | 交叉价: {cross_price:.4f} | 现价: {current_price:.4f} | "
        f"<span style='color: {price_change_color};'>涨跌幅: {price_change:.2f}% {price_change_arrow}</span> | "
        f"密集度: <span style='color: {density_color}; font-weight: bold;'>{density_percent:.3f}%</span> | "
        f"时间: {cross_time.strftime('%H:%M:%S')} | 检测: {signal['detect_time']}"
        f"</div>"
    )
    st.markdown(content, unsafe_allow_html=True)

# ========== 右侧固定展示栏 ==========
def _normalize_detect_dt(val):
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=beijing_tz)
    if isinstance(val, str):
        try:
            h, m, s = map(int, val.split(':'))
            now = datetime.now(beijing_tz)
            return datetime(now.year, now.month, now.day, h, m, s, tzinfo=beijing_tz)
        except:
            return datetime.now(beijing_tz)
    return datetime.now(beijing_tz)

def clean_signal_data(signal):
    for key in ['current_price', 'ma34', 'ma170', 'ma453', 'density_percent', 'price_change']:
        if key in signal:
            try:
                signal[key] = float(signal[key])
            except (TypeError, ValueError):
                signal[key] = 0.0
    if 'detect_time' in signal:
        signal['detect_time'] = _normalize_detect_dt(signal['detect_time'])
    if 'cross_time' in signal and isinstance(signal['cross_time'], str):
        try:
            signal['cross_time'] = datetime.fromisoformat(signal['cross_time'])
        except:
            signal['cross_time'] = datetime.now(beijing_tz)
    return signal

def _enqueue_latest(signal, tf, strategy, symbol, signal_id):
    if signal_id in st.session_state.ticker_seen:
        return
    st.session_state.ticker_seen.add(signal_id)
    entry = {
        'id': signal_id,
        'symbol': symbol,
        'timeframe': tf,
        'strategy': strategy,
        'detect_dt': _normalize_detect_dt(signal.get('detect_time')),
        'current_price': signal.get('current_price'),
        'signal_type': signal.get('signal_type'),
        'direction': signal.get('direction'),
        'cross_time': signal.get('cross_time'),
        'ma34': signal.get('ma34'),
        'ma170': signal.get('ma170'),
        'ma453': signal.get('ma453'),
        'density_percent': signal.get('density_percent'),
        'price_change': signal.get('price_change')
    }
    entry = clean_signal_data(entry)
    st.session_state.latest_signals_ticker.appendleft(entry)
    logging.info(f"加入右侧栏队列: {symbol} {tf} {strategy}")

def render_right_sidebar():
    try:
        if not st.session_state.right_panel_css_injected:
            st.markdown("""
            <style>
            #latest-fixed-panel {
                position: fixed;
                top: 80px;
                right: 12px;
                width: 360px;
                height: 1314px;
                overflow-y: auto;
                z-index: 1001;
                background: rgba(255,255,255,0.85);
                backdrop-filter: blur(6px);
                border: 1px solid rgba(0,0,0,0.08);
                border-radius: 12px;
                box-shadow: 0 6px 18px rgba(0,0,0,0.08);
                padding: 10px 12px;
            }
            #latest-fixed-panel .hdr {
                font-weight: 700;
                font-size: 14px;
                margin-bottom: 6px;
            }
            #latest-fixed-panel .sub {
                font-size: 12px;
                color: #666;
                margin-bottom: 8px;
            }
            #latest-fixed-panel .item {
                font-size: 13px;
                line-height: 1.35;
                padding: 6px 8px;
                margin: 4px 0;
                border-left: 4px solid #ddd;
                background: rgba(0,0,0,0.02);
                border-radius: 8px;
                word-break: break-all;
            }
            #latest-fixed-panel .item a { text-decoration:none; }
            @media (max-width: 1200px) {
                #latest-fixed-panel { display: none; }
            }
            </style>
            """, unsafe_allow_html=True)
            st.session_state.right_panel_css_injected = True
        if st.session_state.right_panel_placeholder is None:
            st.session_state.right_panel_placeholder = st.empty()
        items = list(st.session_state.latest_signals_ticker)
        if not items:
            html = """
            <div id="latest-fixed-panel">
                <div class="hdr">📰 最新信号（最多68条）</div>
                <div class="sub">暂无信号</div>
            </div>
            """
            st.session_state.right_panel_placeholder.markdown(html, unsafe_allow_html=True)
            return
        lines = []
        for s in items:
            try:
                tf = s['timeframe']
                symbol_simple = escape_html(simplify_symbol(s['symbol']))
                symbol_link = escape_html(generate_symbol_link(s['symbol']))
                dt_str = escape_html(s['detect_dt'].strftime('%H:%M:%S'))
                if s['strategy'] == 'cluster':
                    max_ma = max(s['ma34'], s['ma170'], s['ma453'])
                    min_ma = min(s['ma34'], s['ma170'], s['ma453'])
                    if s['current_price'] > max_ma:
                        edge_color = "green"
                        pos_text = "价格在均线上方"
                    elif s['current_price'] < min_ma:
                        edge_color = "red"
                        pos_text = "价格在均线下方"
                    else:
                        edge_color = "orange"
                        pos_text = "价格在均线之间"
                    density = s.get('density_percent')
                    density_txt = escape_html(f"{density:.3f}%" if density is not None else "--")
                    try:
                        cluster_price = min(s['ma34'], s['ma170'], s['ma453'])
                        price_change = ((s['current_price'] - cluster_price) / cluster_price) * 100
                    except Exception:
                        price_change = 0
                    price_change_color = "green" if price_change > 0 else "red"
                    price_change_arrow = "↑" if price_change > 0 else "↓"
                    line = (
                        f"<div class='item' style='border-left-color:{edge_color};'>"
                        f"<a href='{symbol_link}' target='_blank' style='color:{edge_color};font-weight:600;'>📊📊 {symbol_simple} [{tf.upper()}]</a> "
                        f"{pos_text} | 密集度 {density_txt} | 现价 {escape_html(f'{s['current_price']:.4f}')} | "
                        f"<span style='color:{price_change_color};'>涨跌幅: {escape_html(f'{price_change:.2f}%')} {price_change_arrow}</span> "
                        f"<span style='float:right;color:#555;'>{dt_str}</span>"
                        f"</div>"
                    )
                    lines.append(line)
                else:
                    direction = s.get('direction') or ''
                    edge_color = "green" if direction == '多头' else "red"
                    ct = s.get('cross_time')
                    ct_str = escape_html(ct.strftime('%H:%M:%S') if isinstance(ct, datetime) else dt_str)
                    price_change = s.get('price_change')
                    if price_change is None:
                        try:
                            price_change = ((s['current_price'] - s['cross_price']) / s['cross_price']) * 100
                        except Exception:
                            price_change = 0
                    price_change_color = "green" if price_change > 0 else "red"
                    price_change_arrow = "↑" if price_change > 0 else "↓"
                    density = s.get('density_percent', 0)
                    if density < 0.1:
                        density_color = "purple"
                    elif density < 0.2:
                        density_color = "blue"
                    else:
                        density_color = "darkblue"
                    density_txt = escape_html(f"{density:.3f}%")
                    line = (
                        f"<div class='item' style='border-left-color:{edge_color};'>"
                        f"<a href='{symbol_link}' target='_blank' style='color:{edge_color};font-weight:600;'>⏳ {symbol_simple} [{tf.upper()}] {direction}</a> "
                        f"{s.get('signal_type', '')} | 现价 {escape_html(f'{s['current_price']:.4f}')} | "
                        f"<span style='color:{price_change_color};'>涨跌幅: {escape_html(f'{price_change:.2f}%')} {price_change_arrow}</span> | "
                        f"密集度: <span style='color:{density_color};'>{density_txt}</span> "
                        f"<span style='float:right;color:#555;'>{ct_str}</span>"
                        f"</div>"
                    )
                    lines.append(line)
            except Exception as e:
                logging.error(f"渲染右侧栏条目失败: {str(e)}")
                continue
        html = (
            "<div id='latest-fixed-panel'>"
            "<div class='hdr'>📰最新信号（最多68条）</div>"
            "<div class='sub'>按检测时间降序排列，最新的在最上方</div>"
            + "".join(lines) +
            "</div>"
        )
        st.session_state.right_panel_placeholder.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"渲染右侧栏失败: {str(e)}")
        st.session_state.right_panel_placeholder = None
        st.session_state.right_panel_css_injected = False
        st.experimental_rerun()

# ========== UI：标签页 ==========
def build_tabs():
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
            margin-bottom: 60px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    labels = []
    keys = []
    for tf in TIMEFRAMES:
        for sk, sn in STRATEGIES.items():
            labels.append(f"{tf.upper()}·{sn}")
            keys.append((tf, sk))
    tabs = st.tabs(labels)
    for i, k in enumerate(keys):
        with tabs[i]:
            container = st.container()
            placeholder = st.empty()
            st.session_state.result_containers[k] = {'container': container, 'placeholder': placeholder}

def update_tab_content(tf, strategy):
    container = st.session_state.result_containers[(tf, strategy)]['container']
    placeholder = st.session_state.result_containers[(tf, strategy)]['placeholder']
    with container:
        placeholder.empty()
        signals = st.session_state.valid_signals.get((tf, strategy), [])
        if not signals:
            placeholder.markdown("暂无信号")
            return
        placeholder.markdown(f"**{tf.upper()} {STRATEGIES[strategy]}信号: {len(signals)}个**")
        for s in signals[-868:][::-1]:
            if strategy == 'cluster':
                render_cluster_signal(tf, s)
            else:
                render_cross_signal(tf, s)

# ========== 处理与监控 ==========
def process_symbol_timeframe(exchange, symbol, timeframe, failed_symbols):
    out_cluster = []
    out_cross = []
    try:
        ohlcvs = get_cached_ohlcv(exchange, symbol, timeframe, failed_symbols)
        if ohlcvs is None:
            return symbol, out_cluster, out_cross
        data = process_data_all(ohlcvs)
        cluster_list = detect_cluster_signals(data, symbol, timeframe)
        for s in cluster_list:
            out_cluster.append(s)
        cross_list, current_price = detect_cross_signals(data, timeframe, symbol)
        detect_time = datetime.now(beijing_tz)
        for signal in cross_list:
            signal['detect_time'] = detect_time.strftime('%H:%M:%S')
            out_cross.append(signal)
        return symbol, out_cluster, out_cross
    except Exception as e:
        logging.debug(f"处理 {symbol}({timeframe}) 异常: {e}")
        return symbol, out_cluster, out_cross

def monitor_symbols(api_key, api_secret):
    exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 30000})
    build_tabs()
    render_right_sidebar()
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"待监控交易对（按成交量前{CONFIG['top_n_symbols']}）")
    if not st.session_state.symbols_to_monitor:
        st.session_state.symbols_to_monitor = get_valid_symbols(api_key, api_secret, CONFIG['top_n_symbols'])
    cols = st.sidebar.columns(4)
    syms = st.session_state.symbols_to_monitor
    per_col = max(1, len(syms) // 4)
    for i, c in enumerate(cols):
        start = i * per_col
        end = start + per_col
        if i == len(cols) - 1: end = len(syms)
        with c:
            for s in syms[start:end]:
                color = 'red' if s in st.session_state.failed_symbols else 'green'
                simple_s = simplify_symbol(s)
                st.markdown(f"<span style='color:{color};'>• {simple_s}</span>", unsafe_allow_html=True)
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    stats = st.sidebar.empty()
    max_workers = CONFIG.get('max_workers', 4)
    last_cleanup_time = time.time()

    while st.session_state.monitoring_active:
        try:
            start_time = time.time()
            st.session_state.detection_round += 1
            new_signals = defaultdict(int)
            symbols = [s for s in st.session_state.symbols_to_monitor if s not in st.session_state.failed_symbols]
            if not symbols:
                time.sleep(30)
                continue
            failed_copy = st.session_state.failed_symbols.copy()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for s in symbols:
                    for tf in TIMEFRAMES:
                        futures.append(executor.submit(process_symbol_timeframe, exchange, s, tf, failed_copy))
                for i, fut in enumerate(as_completed(futures)):
                    progress = (i + 1) / len(futures)
                    progress_bar.progress(progress)
                    status_text.text(f"检测进度: {progress * 100:.1f}%")
                    symbol, cluster_signals, cross_signals = fut.result()
                    for sig in cluster_signals:
                        tf = sig['timeframe']
                        signal_id = generate_signal_id_cluster(sig['symbol'], tf, sig['detect_time'])
                        if signal_id not in st.session_state.shown_signals[(tf, 'cluster')]:
                            st.session_state.valid_signals[(tf, 'cluster')].append(sig)
                            st.session_state.shown_signals[(tf, 'cluster')].add(signal_id)
                            new_signals[(tf, 'cluster')] += 1
                            update_tab_content(tf, 'cluster')
                            _enqueue_latest(sig, tf, 'cluster', symbol, signal_id)
                            save_signal_to_db(signal_id, symbol, tf, 'cluster', sig)
                    for sig in cross_signals:
                        tf = sig['timeframe']
                        cross_time = sig['cross_time']
                        signal_id = generate_signal_id_cross(sig['symbol'], tf, cross_time, sig['signal_type'])
                        key = (symbol, tf, 'cross')
                        last_time = st.session_state.last_signal_times.get(key)
                        interval = TIMEFRAMES[tf]['interval'] * CONFIG.get('cross_cooldown_multiplier', 5)
                        if not (last_time and (cross_time - last_time).total_seconds() < interval) and \
                                signal_id not in st.session_state.shown_signals[(tf, 'cross')]:
                            st.session_state.valid_signals[(tf, 'cross')].append(sig)
                            st.session_state.shown_signals[(tf, 'cross')].add(signal_id)
                            st.session_state.last_signal_times[key] = cross_time
                            new_signals[(tf, 'cross')] += 1
                            update_tab_content(tf, 'cross')
                            _enqueue_latest(sig, tf, 'cross', symbol, signal_id)
                            save_signal_to_db(signal_id, symbol, tf, 'cross', sig)
                    render_right_sidebar()
            stats.markdown(
                f"轮次: {st.session_state.detection_round} | 新信号: {dict(new_signals)} | 失败交易对: {len(st.session_state.failed_symbols)}")
            # 每小时清理一次旧信号
            if time.time() - last_cleanup_time > 3600:
                clean_old_signals(max_age_hours=24)
                last_cleanup_time = time.time()
            # 发送心跳请求以防止休眠
            try:
                requests.get(CONFIG['heartbeat_url'], timeout=5)
                logging.info("发送心跳请求以保持应用活跃")
            except Exception as e:
                logging.warning(f"心跳请求失败: {e}")
            elapsed = time.time() - start_time
            sleep_time = max(45 - elapsed, 30)
            time.sleep(sleep_time)
        except Exception as e:
            logging.error(f"监控循环异常: {e}")
            time.sleep(10)
            continue

def start_monitoring(api_key, api_secret):
    if st.session_state.monitoring_thread is None or not st.session_state.monitoring_thread.is_alive():
        st.session_state.monitoring_active = True
        st.session_state.monitoring_thread = threading.Thread(
            target=monitor_symbols, args=(api_key, api_secret), daemon=True
        )
        st.session_state.monitoring_thread.start()
        st.sidebar.success("监控已启动")

def stop_monitoring():
    st.session_state.monitoring_active = False
    if st.session_state.monitoring_thread and st.session_state.monitoring_thread.is_alive():
        st.session_state.monitoring_thread.join(timeout=5)
    st.sidebar.info("监控已停止")

# ========== 入口：侧栏参数 ==========
def main():
    init_db()
    load_signals_from_db()
    st.title('一体化监控（v2） - 三均线密集 & 双均线交叉（1m/5m/30m/4h）')
    st.sidebar.header('运行参数（可调整）')
    st.sidebar.subheader('三均线密集阈值配置（%）')
    CONFIG['density_threshold_pct_by_tf']['1m'] = st.sidebar.number_input(
        '1m级别', min_value=0.01, max_value=5.0, value=0.13, step=0.01, key='density_1m')
    CONFIG['density_threshold_pct_by_tf']['5m'] = st.sidebar.number_input(
        '5m级别', min_value=0.01, max_value=5.0, value=0.3, step=0.01, key='density_5m')
    CONFIG['density_threshold_pct_by_tf']['30m'] = st.sidebar.number_input(
        '30m级别', min_value=0.01, max_value=5.0, value=1.0, step=0.01, key='density_30m')
    CONFIG['density_threshold_pct_by_tf']['4h'] = st.sidebar.number_input(
        '4h级别', min_value=0.01, max_value=5.0, value=3.0, step=0.01, key='density_4h')
    st.sidebar.subheader('长短交叉点密集度阈值（%）')
    CONFIG['price_diff_threshold_pct_by_tf']['1m'] = st.sidebar.number_input(
        '1m级别密集度阈值', min_value=0.01, max_value=5.0, value=0.86, step=0.01, key='price_diff_1m')
    CONFIG['price_diff_threshold_pct_by_tf']['5m'] = st.sidebar.number_input(
        '5m级别密集度阈值', min_value=0.01, max_value=5.0, value=1.86, step=0.01, key='price_diff_5m')
    CONFIG['price_diff_threshold_pct_by_tf']['30m'] = st.sidebar.number_input(
        '30m级别密集度阈值', min_value=0.01, max_value=5.0, value=3.86, step=0.01, key='price_diff_30m')
    CONFIG['price_diff_threshold_pct_by_tf']['4h'] = st.sidebar.number_input(
        '4h级别密集度阈值', min_value=0.01, max_value=5.0, value=4.86, step=0.01, key='price_diff_4h')
    st.sidebar.subheader('其他参数')
    CONFIG['cluster_recent_bars'] = st.sidebar.number_input(
        '密集判定最近K线数 (X)', min_value=3, max_value=100, value=13, step=1)
    CONFIG['cluster_check_window'] = st.sidebar.number_input(
        '密集检查窗口 (Y)', min_value=10, max_value=200, value=86, step=1)
    CONFIG['cross_unique_window'] = st.sidebar.number_input(
        '均线交叉密度（唯一性窗口）', min_value=10, max_value=1000, value=513, step=1)
    CONFIG['cross_short_term_window'] = st.sidebar.number_input(
        '短期确认信号窗口', min_value=1, max_value=10, value=3, step=1,
        help="最近多少根K线内有MA7/MA34同向交叉（默认3根）")
    CONFIG['cross_cooldown_multiplier'] = st.sidebar.number_input(
        '双均线交叉冷却倍数 (interval * X)', min_value=1, max_value=20, value=5, step=1)
    CONFIG['fetch_limit'] = st.sidebar.number_input(
        '拉取K线数量 (fetch limit)', min_value=600, max_value=2000, value=968, step=1)
    CONFIG['max_workers'] = st.sidebar.number_input(
        '并发线程数', min_value=1, max_value=12, value=4, step=1)
    CONFIG['top_n_symbols'] = st.sidebar.selectbox(
        '按交易额获取交易对数量', [50, 100, 150, 200, 300], index=4)
    for tf in TIMEFRAMES:
        TIMEFRAMES[tf]['max_bars'] = int(CONFIG['fetch_limit'])
        TIMEFRAMES[tf]['cache_ttl'] = 30 if tf == '1m' else (180 if tf == '5m' else 600)
    st.sidebar.markdown('---')
    st.sidebar.subheader('API & 控制')
    api_key = st.sidebar.text_input('Gate.io API Key', value=API_KEY)
    api_secret = st.sidebar.text_input('Gate.io API Secret', value=API_SECRET, type='password')
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_btn = st.button('开始监控')
    with col2:
        stop_btn = st.button('停止监控')
    if start_btn:
        if not api_key or not api_secret:
            st.sidebar.error('请填写 API Key/Secret')
            return
        st.session_state.symbols_to_monitor = get_valid_symbols(api_key, api_secret, CONFIG['top_n_symbols'])
        start_monitoring(api_key, api_secret)
    if stop_btn:
        stop_monitoring()
    st.sidebar.markdown(
        '提示：三均线密集的判定为：在最近 X 根K线内发生，且在最近 Y 根内唯一，从而排除反复窄幅盘整的噪声。')
    with st.expander('筛选规则说明（简要）', expanded=False):
        st.markdown('''- 三均线密集：MA34/MA170/MA453 在某一根 K 线处最大最小差 <= 密集阈值，且该密集发生位置满足：
                    1) 出现在最近 X 根 K 线内（X 可调，默认为 13）
                    2) 在最近 Y 根 K 线内（Y 可调，默认为 86）恰好只有一次密集发生 —— 用来排除反复盘整造成的噪声
                    - 双均线交叉：沿用原 513.py 的双均线组合 + MA7 与 MA34 的短期确认
                    - 冷却：三均线密集 -> 每一次具体发生的信号只展示一次（使用发生时点唯一ID）；双均线交叉 -> interval * 冷却倍数秒内不重复''')
    st.info('配置完成后点击侧栏的【开始监控】按钮以启动检测。')

if __name__ == '__main__':
    main()

