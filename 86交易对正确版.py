import ccxt
import pandas as pd
from datetime import datetime, timezone
import streamlit as st
import pytz
import time
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化 gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 30000,
    'rateLimit': 1000
})

# 北京时间配置
beijing_tz = pytz.timezone('Asia/Shanghai')

# 交易周期配置
TIMEFRAMES = {
    '1m': {'interval': 60, 'max_bars': 500, 'cache_ttl': 45, 'window_size': 31},
    '5m': {'interval': 300, 'max_bars': 500, 'cache_ttl': 240, 'window_size': 31},
    '30m': {'interval': 1800, 'max_bars': 700, 'cache_ttl': 1500, 'window_size': 131},
    '4h': {'interval': 14400, 'max_bars': 700, 'cache_ttl': 14400, 'window_size': 131}
}

# 初始化 session state
if 'valid_signals' not in st.session_state:
    st.session_state.valid_signals = {tf: [] for tf in TIMEFRAMES}
if 'shown_signals' not in st.session_state:
    st.session_state.shown_signals = {tf: set() for tf in TIMEFRAMES}
if 'detection_round' not in st.session_state:
    st.session_state.detection_round = 0
if 'new_signals_count' not in st.session_state:
    st.session_state.new_signals_count = {tf: 0 for tf in TIMEFRAMES}

# 全局缓存
ohlcv_cache = {}
symbol_cache = {'symbols': [], 'timestamp': 0}

# 结果展示容器
if 'result_containers' not in st.session_state:
    st.session_state.result_containers = {
        tf: {'container': None, 'placeholder': None}
        for tf in TIMEFRAMES
    }

# 实时交易对列表状态
if 'current_symbols' not in st.session_state:
    st.session_state.current_symbols = []


# 音频处理
def get_audio_base64(file_path):
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')


def play_alert_sound():
    audio_base64 = get_audio_base64("alert.wav")
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
    </audio>
    """
    st.components.v1.html(audio_html, height=0)


# 信号ID生成
def generate_signal_id(symbol, timeframe, condition_time, signal_type):
    unique_str = f"{symbol}|{timeframe}|{condition_time}|{signal_type}"
    return hashlib.md5(unique_str.encode()).hexdigest()


# 交易对获取
@st.cache_data(ttl=600)
def get_top_valid_symbols():
    try:
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()

        valid_symbols = []
        for symbol in tickers:
            if any(c in symbol for c in ['3L', '3S', '5L', '5S', '_']):
                continue

            # 屏蔽 USDC/USDT 交易对
            if symbol == 'USDC/USDT':
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


# 带缓存的数据获取
def get_cached_ohlcv(symbol, timeframe):
    now = time.time()
    cache_key = (symbol, timeframe)
    config = TIMEFRAMES[timeframe]

    if cache_key in ohlcv_cache:
        data, timestamp = ohlcv_cache[cache_key]
        if now - timestamp < config['cache_ttl']:
            return data

    for _ in range(3):
        try:
            since = exchange.milliseconds() - (config['max_bars'] * config['interval'] * 1000)
            data = exchange.fetch_ohlcv(symbol, timeframe, since)
            if data and len(data) >= 453:
                ohlcv_cache[cache_key] = (data, now)
                return data
        except ccxt.NetworkError:
            time.sleep(1)
        except ccxt.BadSymbol:
            return None
    return None


# 数据处理
def process_data(ohlcvs, timeframe):
    windows = [7, 34, 170, 453]
    if not ohlcvs or len(ohlcvs) < max(windows):
        return None

    df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(beijing_tz)
    df.set_index('timestamp', inplace=True)

    # 严格计算移动平均线
    close = df['close']
    for window in windows:
        df[f'ma{window}'] = close.rolling(window=window, min_periods=window).mean()

    return df.dropna()


# 交叉检测逻辑
def check_cross_conditions(df, timeframe):
    config = TIMEFRAMES[timeframe]
    window_size = config['window_size']

    if df is None or len(df) < window_size:
        return None, None

    df_window = df.iloc[-window_size:]
    required_columns = ['ma7', 'ma34', 'ma170', 'ma453']

    if not all(col in df_window.columns for col in required_columns):
        return None, None
    if df_window[required_columns].isnull().any().any():
        return None, None

    signal_type = None
    condition_time = None

    # 初始化交叉检测矩阵
    cross_matrix = {
        'gold': {ma: (df_window['ma7'] > df_window[ma]) & (df_window['ma7'].shift(1) <= df_window[ma].shift(1))
                 for ma in ['ma34', 'ma170', 'ma453']},
        'death': {ma: (df_window['ma7'] < df_window[ma]) & (df_window['ma7'].shift(1) >= df_window[ma].shift(1))
                  for ma in ['ma34', 'ma170', 'ma453']}
    }

    # 检测金叉条件
    gold_times = []
    for ma in ['ma34', 'ma170', 'ma453']:
        cross_points = df_window.index[cross_matrix['gold'][ma]]
        if not cross_points.empty:
            gold_times.append(cross_points[-1])
        else:
            gold_times = []
            break

    if len(gold_times) == 3:
        last_gold_time = max(gold_times)
        subsequent_data = df_window.loc[last_gold_time:]

        death_occurred = False
        for ma in ['ma34', 'ma170', 'ma453']:
            if cross_matrix['death'][ma][subsequent_data.index].any():
                death_occurred = True
                break

        if not death_occurred:
            signal_type = 'MA7 金叉 MA34, MA170, MA453'
            condition_time = last_gold_time

    # 检测死叉条件（仅当金叉未触发时）
    if not signal_type:
        death_times = []
        for ma in ['ma34', 'ma170', 'ma453']:
            cross_points = df_window.index[cross_matrix['death'][ma]]
            if not cross_points.empty:
                death_times.append(cross_points[-1])
            else:
                death_times = []
                break

        if len(death_times) == 3:
            last_death_time = max(death_times)
            subsequent_data = df_window.loc[last_death_time:]

            gold_occurred = False
            for ma in ['ma34', 'ma170', 'ma453']:
                if cross_matrix['gold'][ma][subsequent_data.index].any():
                    gold_occurred = True
                    break

            if not gold_occurred:
                signal_type = 'MA7 死叉 MA34, MA170, MA453'
                condition_time = last_death_time

    return signal_type, condition_time


# 界面更新函数
def update_symbol_list(symbols):
    st.session_state.current_symbols = symbols
    with symbol_list.container():
        st.write(f"总数量：{len(symbols)} 个")
        cols = 3
        col_items = [[] for _ in range(cols)]
        for i, symbol in enumerate(symbols):
            col_items[i % cols].append(symbol)

        cols = st.columns(cols)
        for i, col in enumerate(cols):
            with col:
                for symbol in col_items[i]:
                    st.write(f"• {symbol}")


def update_tab_content(tf):
    container = st.session_state.result_containers[tf]['container']
    placeholder = st.session_state.result_containers[tf]['placeholder']

    with container:
        with placeholder.container():
            for signal in st.session_state.valid_signals[tf]:
                st.markdown(f"""
                **交易对**: {signal['symbol']}【{tf.upper()}】  
                **信号类型**: {signal['signal_type']}  
                **条件时间**: {signal['condition_time']}  
                **检测时间**: {signal['detect_time']}
                """)
                st.write("---")


# 主监控逻辑
def monitor_symbols():
    tabs = st.tabs([f"{tf.upper()} 周期" for tf in TIMEFRAMES])
    for idx, tf in enumerate(TIMEFRAMES):
        with tabs[idx]:
            container = st.container()
            placeholder = st.empty()
            st.session_state.result_containers[tf] = {
                'container': container,
                'placeholder': placeholder
            }
            update_tab_content(tf)

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    round_info = st.sidebar.empty()
    new_signals_info = st.sidebar.empty()

    global symbol_list
    symbol_list = st.sidebar.expander("当前监控交易对列表（前86）", expanded=False)

    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            start_time = time.time()
            st.session_state.detection_round += 1
            current_round_new = {tf: 0 for tf in TIMEFRAMES}

            symbols = get_top_valid_symbols()
            update_symbol_list(symbols)

            futures = []
            for idx, symbol in enumerate(symbols):
                for timeframe in TIMEFRAMES:
                    progress = (idx + 1) / len(symbols)
                    futures.append(
                        executor.submit(
                            process_symbol_task,
                            symbol,
                            timeframe,
                            progress
                        )
                    )

            for future in as_completed(futures):
                symbol, timeframe, progress, signal = future.result()
                progress_bar.progress(progress)
                status_text.text(f"正在检测: {symbol} ({timeframe})")

                if signal:
                    tf = signal['timeframe']
                    signal_id = signal['signal_id']
                    if signal_id not in st.session_state.shown_signals[tf]:
                        st.session_state.valid_signals[tf].append(signal)
                        st.session_state.shown_signals[tf].add(signal_id)
                        current_round_new[tf] += 1
                        play_alert_sound()
                        update_tab_content(tf)

            round_info.markdown(f"**检测轮次**: {st.session_state.detection_round}")
            new_signals_info.markdown("**本轮新增信号**")
            for tf in TIMEFRAMES:
                new_signals_info.markdown(f"- {tf.upper()}: {current_round_new[tf]}")

            elapsed = time.time() - start_time
            sleep_time = max(60 - elapsed, 15)
            time.sleep(sleep_time)


def process_symbol_task(symbol, timeframe, progress):
    try:
        ohlcvs = get_cached_ohlcv(symbol, timeframe)
        df = process_data(ohlcvs, timeframe)
        signal_type, condition_time = check_cross_conditions(df, timeframe)

        if signal_type and condition_time:
            detect_time = datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')
            signal_id = generate_signal_id(
                symbol,
                timeframe,
                condition_time.strftime('%Y-%m-%d %H:%M:%S'),
                signal_type
            )
            return symbol, timeframe, progress, {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal_type': signal_type,
                'condition_time': condition_time.strftime('%Y-%m-%d %H:%M:%S'),
                'detect_time': detect_time,
                'signal_id': signal_id
            }
    except Exception as e:
        pass
    return symbol, timeframe, progress, None


def main():
    st.title('多周期MA交叉实时监控系统（线程优化版）')
    with st.expander("核心修正说明", expanded=True):
        st.markdown("""
        **关键修复点**：
        1. **条件时间精准计算**  
           现在严格保证只取同一信号类型（金叉/死叉）中三个交叉的最后发生时间
        2. **信号互斥检测**  
           金叉和死叉检测现在为互斥逻辑，避免同一周期内同时触发两种信号
        3. **数据验证增强**  
           新增移动平均线数据完整性检查，排除计算不完整的情况
        """)
    st.sidebar.title("智能监控面板")
    monitor_symbols()


if __name__ == "__main__":
    main()
