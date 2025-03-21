import ccxt
import pandas as pd
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 30000,
    'rateLimit': 500
})

# 北京时间配置
beijing_tz = pytz.timezone('Asia/Shanghai')

# 交易周期配置
TIMEFRAMES = {
    '1m': {'interval': 60, 'max_bars': 600, 'cache_ttl': 30},  # 增加 max_bars 至 600
    '5m': {'interval': 300, 'max_bars': 600, 'cache_ttl': 180},
    '30m': {'interval': 1800, 'max_bars': 600, 'cache_ttl': 1200},
    '4h': {'interval': 14400, 'max_bars': 600, 'cache_ttl': 10800}
}

# 初始化 session state
if 'valid_signals' not in st.session_state:
    st.session_state.valid_signals = defaultdict(list)
if 'shown_signals' not in st.session_state:
    st.session_state.shown_signals = defaultdict(set)
if 'detection_round' not in st.session_state:
    st.session_state.detection_round = 0
if 'last_signal_times' not in st.session_state:
    st.session_state.last_signal_times = {}
if 'result_containers' not in st.session_state:
    st.session_state.result_containers = {tf: {'container': None, 'placeholder': None} for tf in TIMEFRAMES}
if 'failed_symbols' not in st.session_state:
    st.session_state.failed_symbols = set()
if 'signal_queue' not in st.session_state:
    st.session_state.signal_queue = queue.Queue()

# 全局缓存
ohlcv_cache = {}

# MACD计算函数
def calculate_macd(close, fast=12, slow=26, signal=9):
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# 音频处理
def get_audio_base64(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error("警报音频文件未找到")
        return None

def play_alert_sound():
    audio_base64 = get_audio_base64("alert.wav")
    if audio_base64:
        st.components.v1.html(f'<audio autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>', height=0)

# 信号ID生成
def generate_signal_id(symbol, timeframe, ma_cross_time, macd_cross_time, signal_type):
    ma_ts = int(ma_cross_time.timestamp())
    macd_ts = int(macd_cross_time.timestamp())
    unique_str = f"{symbol}|{timeframe}|{ma_ts}|{macd_ts}|{signal_type}"
    return hashlib.md5(unique_str.encode()).hexdigest()

# 获取有效交易对
@st.cache_data(ttl=3600)
def get_valid_symbols():
    try:
        with open("D:\\pycharm_study\\HY1413.txt", "r") as file:
            symbols = [line.strip() for line in file.readlines() if line.strip()]
        markets = exchange.load_markets()
        valid_symbols = []
        for symbol in symbols:
            if symbol in markets and markets[symbol]['active']:
                try:
                    exchange.fetch_ohlcv(symbol, '1m', limit=10)
                    valid_symbols.append(symbol)
                except Exception as e:
                    logging.warning(f"交易对 {symbol} 无数据或不可用: {str(e)}")
                    st.session_state.failed_symbols.add(symbol)
        return valid_symbols
    except Exception as e:
        logging.error(f"获取交易对失败: {str(e)}")
        return []

# 数据获取
def get_cached_ohlcv(symbol, timeframe, failed_symbols):
    if symbol in failed_symbols:
        return None

    now = time.time()
    cache_key = (symbol, timeframe)
    config = TIMEFRAMES[timeframe]

    if cache_key in ohlcv_cache:
        data, timestamp = ohlcv_cache[cache_key]
        if now - timestamp < config['cache_ttl']:
            return data

    for attempt in range(3):
        try:
            since = exchange.milliseconds() - (config['max_bars'] * config['interval'] * 1000)
            data = exchange.fetch_ohlcv(symbol, timeframe, since, limit=config['max_bars'])
            if data and len(data) >= 584:  # 453 (MA453) + 131 (检测窗口)
                ohlcv_cache[cache_key] = (data, now)
                return data
            else:
                logging.warning(f"数据不足: {symbol}, {timeframe}, 获取到 {len(data)} 条")
                return None
        except ccxt.RateLimitExceeded as e:
            logging.warning(f"请求频率超限 ({symbol}, {timeframe}): {str(e)}")
            time.sleep(2 ** attempt + 5)
        except ccxt.NetworkError as e:
            logging.warning(f"网络错误 ({symbol}, {timeframe}): {str(e)}")
            time.sleep(2 ** attempt)
        except ccxt.BadSymbol:
            failed_symbols.add(symbol)
            logging.error(f"无效交易对: {symbol}")
            return None
        except Exception as e:
            logging.error(f"数据获取失败 ({symbol}, {timeframe}): {str(e)}")
            return None
    failed_symbols.add(symbol)
    logging.error(f"多次尝试后数据获取失败: {symbol}, {timeframe}")
    return None

# 数据处理
def process_data(ohlcvs, timeframe):
    if not ohlcvs or len(ohlcvs) < 584:
        return None

    df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(beijing_tz)
    df.set_index('timestamp', inplace=True)

    df['ma34'] = df['close'].rolling(34, min_periods=34).mean()
    df['ma453'] = df['close'].rolling(453, min_periods=453).mean()
    df['macd_line'], df['signal_line'] = calculate_macd(df['close'])

    return df.iloc[453:]  # 从第 453 根开始，确保 MA453 有效

# 信号检测
def detect_signals(df, timeframe):
    if df is None or len(df) < 131:
        logging.debug(f"数据不足以检测信号: {timeframe}, 长度 {len(df)}")
        return None, None, None

    ma34, ma453 = df['ma34'], df['ma453']
    golden_cross = (ma34 > ma453) & (ma34.shift(1) <= ma453.shift(1))
    death_cross = (ma34 < ma453) & (ma34.shift(1) >= ma453.shift(1))

    recent_131 = df.iloc[-131:]
    golden_in_131 = golden_cross[-131:]
    death_in_131 = death_cross[-131:]

    # 金叉检测
    golden_crosses = golden_in_131[golden_in_131]
    if len(golden_crosses) > 0:  # 放宽条件，检测所有金叉
        for idx in golden_crosses.index:
            if idx >= df.index[-31]:  # 最近 31 根 K 线
                macd_cross_time = find_macd_cross(df, window=5, cross_type='golden')  # 扩展窗口至 5
                if macd_cross_time and (macd_cross_time - idx).total_seconds() <= TIMEFRAMES[timeframe]['interval'] * 3:
                    logging.info(f"检测到金叉信号: {timeframe}, MA交叉时间 {idx}, MACD确认时间 {macd_cross_time}")
                    return 'MA金叉-MACD确认', idx, macd_cross_time

    # 死叉检测
    death_crosses = death_in_131[death_in_131]
    if len(death_crosses) > 0:  # 放宽条件，检测所有死叉
        for idx in death_crosses.index:
            if idx >= df.index[-31]:  # 最近 31 根 K 线
                macd_cross_time = find_macd_cross(df, window=5, cross_type='death')  # 扩展窗口至 5
                if macd_cross_time and (macd_cross_time - idx).total_seconds() <= TIMEFRAMES[timeframe]['interval'] * 3:
                    logging.info(f"检测到死叉信号: {timeframe}, MA交叉时间 {idx}, MACD确认时间 {macd_cross_time}")
                    return 'MA死叉-MACD确认', idx, macd_cross_time

    return None, None, None

def find_macd_cross(df, window=5, cross_type='golden'):
    recent_df = df.iloc[-window:]
    macd_diff = recent_df['macd_line'] - recent_df['signal_line']
    crosses = (macd_diff > 0) & (macd_diff.shift(1) <= 0) if cross_type == 'golden' else (macd_diff < 0) & (macd_diff.shift(1) >= 0)
    if crosses.any():
        return recent_df.index[crosses][-1]
    return None

# 界面更新
def update_tab_content(tf):
    container = st.session_state.result_containers[tf]['container']
    placeholder = st.session_state.result_containers[tf]['placeholder']
    with container:
        with placeholder.container():
            for signal in st.session_state.valid_signals[tf][-10:]:
                st.markdown(f"**{signal['symbol']} [{tf.upper()}]** | {signal['signal_type']} | MA: {signal['ma_cross_time'].strftime('%H:%M:%S')} | MACD: {signal['macd_cross_time'].strftime('%H:%M:%S')} | 检测: {signal['detect_time']}")

# 主监控逻辑
def monitor_symbols():
    tabs = st.tabs([f"{tf.upper()} 周期" for tf in TIMEFRAMES])
    for idx, tf in enumerate(TIMEFRAMES):
        with tabs[idx]:
            container = st.container()
            placeholder = st.empty()
            st.session_state.result_containers[tf] = {'container': container, 'placeholder': placeholder}

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    stats = st.sidebar.empty()

    with ThreadPoolExecutor(max_workers=6) as executor:
        while True:
            start_time = time.time()
            st.session_state.detection_round += 1
            new_signals = defaultdict(int)

            symbols = [s for s in get_valid_symbols() if s not in st.session_state.failed_symbols]
            if not symbols:
                logging.error("无有效交易对可监控")
                break

            failed_symbols_copy = st.session_state.failed_symbols.copy()
            futures = [
                executor.submit(process_symbol_task, symbol, timeframe, (i + 1) / len(symbols), failed_symbols_copy)
                for i, symbol in enumerate(symbols)
                for timeframe in TIMEFRAMES
            ]

            for future in as_completed(futures):
                symbol, timeframe, progress, signal = future.result()
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"检测: {symbol} ({timeframe})")
                if signal:
                    st.session_state.signal_queue.put(signal)

            play_sound = False
            while not st.session_state.signal_queue.empty():
                signal = st.session_state.signal_queue.get()
                tf = signal['timeframe']
                signal_id = signal['signal_id']
                key = (signal['symbol'], tf)
                cross_time = signal['ma_cross_time']

                interval = TIMEFRAMES[tf]['interval'] * 5
                last_time = st.session_state.last_signal_times.get(key)
                if last_time and (cross_time - last_time).total_seconds() < interval:
                    continue

                if signal_id not in st.session_state.shown_signals[tf]:
                    st.session_state.valid_signals[tf].append(signal)
                    st.session_state.shown_signals[tf].add(signal_id)
                    st.session_state.last_signal_times[key] = cross_time
                    new_signals[tf] += 1
                    play_sound = True

            if play_sound:
                play_alert_sound()

            for tf in TIMEFRAMES:
                update_tab_content(tf)

            stats.markdown(f"轮次: {st.session_state.detection_round} | 新信号: {dict(new_signals)} | 失败交易对: {len(st.session_state.failed_symbols)}")
            elapsed = time.time() - start_time
            time.sleep(max(30 - elapsed, 15))

def process_symbol_task(symbol, timeframe, progress, failed_symbols):
    try:
        ohlcvs = get_cached_ohlcv(symbol, timeframe, failed_symbols)
        if ohlcvs is None:
            return symbol, timeframe, progress, None

        df = process_data(ohlcvs, timeframe)
        signal_type, ma_cross_time, macd_cross_time = detect_signals(df, timeframe)

        if signal_type:
            detect_time = datetime.now(beijing_tz)
            ma_cross_time = ma_cross_time.tz_convert(beijing_tz)
            macd_cross_time = macd_cross_time.tz_convert(beijing_tz)

            signal_id = generate_signal_id(symbol, timeframe, ma_cross_time, macd_cross_time, signal_type.split('-')[0])
            return symbol, timeframe, progress, {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal_type': signal_type,
                'ma_cross_time': ma_cross_time,
                'macd_cross_time': macd_cross_time,
                'detect_time': detect_time.strftime('%H:%M:%S'),
                'signal_id': signal_id
            }
    except Exception as e:
        logging.debug(f"处理 {symbol} ({timeframe}) 出错: {str(e)}")
    return symbol, timeframe, progress, None

# 主函数
def main():
    st.title('MA-MACD实时监控系统（V9.4）')
    with st.expander("筛选条件说明", expanded=False):
        st.markdown("""
        **核心筛选条件**：
        - **金叉**: 131根K线内出现MA34金叉MA453，最近31根K线发生，5根K线内MACD金叉。
        - **死叉**: 131根K线内出现MA34死叉MA453，最近31根K线发生，5根K线内MACD死叉。
        - **规则**: 同交易对同周期5个时间单位内不重复报警，北京时间，多周期并行。
        """)
    st.sidebar.title("监控面板")
    monitor_symbols()

if __name__ == "__main__":
    main()