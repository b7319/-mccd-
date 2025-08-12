import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import pytz
import time
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import logging
import queue
import os
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Gate.io API
api_key = os.getenv('GATEIO_API_KEY', 'YOUR_API_KEY')
api_secret = os.getenv('GATEIO_API_SECRET', 'YOUR_API_SECRET')
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
    '1m': {'interval': 60, 'max_bars': 1000, 'cache_ttl': 30},
    '5m': {'interval': 300, 'max_bars': 1000, 'cache_ttl': 180},
    '30m': {'interval': 1800, 'max_bars': 1000, 'cache_ttl': 1200},
    '4h': {'interval': 14400, 'max_bars': 1000, 'cache_ttl': 10800}
}

# 初始化 session state
def init_session_state():
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
    if 'symbols_cache' not in st.session_state:
        st.session_state.symbols_cache = {'symbols': [], 'timestamp': 0}
    if 'symbols_to_monitor' not in st.session_state:
        st.session_state.symbols_to_monitor = []
    if 'audio_base64' not in st.session_state:
        st.session_state.audio_base64 = None

# 全局缓存
ohlcv_cache = {}

def get_audio_base64(file_path="alert.wav"):
    """获取音频文件的Base64编码"""
    try:
        if st.session_state.audio_base64:
            return st.session_state.audio_base64
            
        with open(file_path, "rb") as audio_file:
            base64_data = base64.b64encode(audio_file.read()).decode('utf-8')
            st.session_state.audio_base64 = base64_data
            return base64_data
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

def generate_signal_id(symbol, timeframe, detect_time):
    ts = int(detect_time.timestamp())
    unique_str = f"{symbol}|{timeframe}|{ts}"
    return hashlib.md5(unique_str.encode()).hexdigest()

@st.cache_data(ttl=3600)
def get_valid_symbols():
    try:
        # 如果缓存有效且未过期（1小时），直接使用缓存
        if st.session_state.symbols_cache['symbols'] and time.time() - st.session_state.symbols_cache[
            'timestamp'] < 3600:
            return st.session_state.symbols_cache['symbols']

        # 专门获取永续合约市场
        markets = exchange.load_markets(True)  # 重新加载市场数据
        tickers = exchange.fetch_tickers(params={'type': 'swap'})  # 指定获取永续合约行情

        valid_symbols = []
        volume_data = []

        # 只处理永续合约交易对
        for symbol, ticker in tickers.items():
            market = markets.get(symbol)
            if not market:
                continue

            # 过滤条件：永续合约市场、活跃状态、USDT交易对
            if (market.get('type') == 'swap' and
                    market.get('active') and
                    ticker.get('quoteVolume') is not None and
                    market.get('quote') == 'USDT'):
                volume_data.append({
                    'symbol': symbol,
                    'volume': ticker['quoteVolume'],
                    'last': ticker['last']
                })
                logging.debug(f"有效交易对: {symbol}, 成交量: {ticker['quoteVolume']}")

        if not volume_data:
            logging.error("未找到任何有效的永续合约交易对")
            return []

        volume_data.sort(key=lambda x: x['volume'], reverse=True)
        top_symbols = [item['symbol'] for item in volume_data[:300]]  # 取成交量前300

        final_symbols = []
        for symbol in top_symbols:
            try:
                # 验证交易对是否可以获取K线数据
                exchange.fetch_ohlcv(symbol, '1m', limit=2)
                final_symbols.append(symbol)
                if len(final_symbols) >= 300:
                    break
            except ccxt.BadSymbol:
                logging.warning(f"无效交易对: {symbol}")
                st.session_state.failed_symbols.add(symbol)
            except Exception as e:
                logging.warning(f"交易对 {symbol} 验证失败: {str(e)}")
                st.session_state.failed_symbols.add(symbol)

        logging.info(f"获取到 {len(final_symbols)} 个有效永续合约交易对")

        # 更新缓存
        st.session_state.symbols_cache = {
            'symbols': final_symbols,
            'timestamp': time.time()
        }

        return final_symbols

    except Exception as e:
        logging.error(f"获取交易对失败: {str(e)}")
        return st.session_state.symbols_cache.get('symbols', [])

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
        else:
            del ohlcv_cache[cache_key]

    for attempt in range(3):
        try:
            since = exchange.milliseconds() - (config['max_bars'] * config['interval'] * 1000)
            data = exchange.fetch_ohlcv(symbol, timeframe, since, limit=config['max_bars'])
            if data and len(data) >= 500:
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

def process_data(ohlcvs, timeframe):
    if not ohlcvs or len(ohlcvs) < 500:
        return None

    # 使用NumPy数组代替Pandas DataFrame以提高性能
    timestamps = np.array([x[0] for x in ohlcvs], dtype=np.int64)
    opens = np.array([x[1] for x in ohlcvs], dtype=np.float64)
    highs = np.array([x[2] for x in ohlcvs], dtype=np.float64)
    lows = np.array([x[3] for x in ohlcvs], dtype=np.float64)
    closes = np.array([x[4] for x in ohlcvs], dtype=np.float64)
    volumes = np.array([x[5] for x in ohlcvs], dtype=np.float64)

    # 计算移动平均线
    ma34 = np.convolve(closes, np.ones(34) / 34, mode='valid')
    ma170 = np.convolve(closes, np.ones(170) / 170, mode='valid')
    ma453 = np.convolve(closes, np.ones(453) / 453, mode='valid')

    # 对齐数据长度
    min_length = min(len(ma34), len(ma170), len(ma453))
    ma34 = ma34[-min_length:]
    ma170 = ma170[-min_length:]
    ma453 = ma453[-min_length:]
    closes = closes[-min_length:]
    timestamps = timestamps[-min_length:]

    return {
        'timestamps': timestamps,
        'closes': closes,
        'ma34': ma34,
        'ma170': ma170,
        'ma453': ma453
    }

def check_ma_cluster(ma34, ma170, ma453, pct_threshold=0.003):
    """检查三条均线是否在指定百分比内密集排列"""
    try:
        # 获取当前均线值
        current_ma34 = ma34[-1]
        current_ma170 = ma170[-1]
        current_ma453 = ma453[-1]
        
        # 计算最大值和最小值
        max_ma = max(current_ma34, current_ma170, current_ma453)
        min_ma = min(current_ma34, current_ma170, current_ma453)
        
        # 计算密集度
        if max_ma == 0:
            return False
        return (max_ma - min_ma) / max_ma <= pct_threshold
    except Exception:
        return False

def detect_signals(data, timeframe):
    if data is None or len(data['closes']) < 500:
        return []

    # 检查三均线密集排列条件（区间 <= 0.3%）
    if not check_ma_cluster(data['ma34'], data['ma170'], data['ma453']):
        return []

    # 获取当前价格
    current_price = data['closes'][-1]
    
    # 获取当前均线值
    current_ma34 = data['ma34'][-1]
    current_ma170 = data['ma170'][-1]
    current_ma453 = data['ma453'][-1]
    
    # 确定价格相对于均线的位置
    max_ma = max(current_ma34, current_ma170, current_ma453)
    min_ma = min(current_ma34, current_ma170, current_ma453)
    
    if current_price > max_ma:
        position = "价格在均线上方"
    elif current_price < min_ma:
        position = "价格在均线下方"
    else:
        position = "价格在均线之间"

    # 计算密集度百分比
    density_percent = ((max_ma - min_ma) / max_ma) * 100
    
    # 创建信号
    signal = {
        'signal_type': "三均线密集排列",
        'position': position,
        'detect_time': datetime.fromtimestamp(data['timestamps'][-1] / 1000, tz=beijing_tz),
        'current_price': current_price,
        'ma34': current_ma34,
        'ma170': current_ma170,
        'ma453': current_ma453,
        'density_percent': density_percent
    }
    
    return [signal]

def update_tab_content(tf):
    container = st.session_state.result_containers[tf]['container']
    placeholder = st.session_state.result_containers[tf]['placeholder']
    with container:
        with placeholder.container():
            signals = st.session_state.valid_signals[tf][-50:]
            if signals:
                for signal in signals:
                    # 设置样式
                    position = signal['position']
                    if "上方" in position:
                        position_color = "green"
                    elif "下方" in position:
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

                    content = (
                        f"<div style='margin-bottom: 10px; border-left: 4px solid {position_color}; padding-left: 8px;'>"
                        f"<span style='color: {position_color}; font-weight: bold;'>🔍 {signal['symbol']} [{tf.upper()}] {position}</span> | "
                        f"密集度: <span style='color: {density_color}; font-weight: bold;'>{density_percent:.3f}%</span> | "
                        f"现价: {signal['current_price']:.4f} | "
                        f"MA34: {signal['ma34']:.4f} | MA170: {signal['ma170']:.4f} | MA453: {signal['ma453']:.4f} | "
                        f"时间: {signal['detect_time'].strftime('%H:%M:%S')}"
                        f"</div>"
                    )
                    st.markdown(content, unsafe_allow_html=True)
            else:
                st.markdown("暂无信号")

def process_symbol_batch(symbol, failed_symbols):
    signals = []
    for timeframe in TIMEFRAMES:
        try:
            ohlcvs = get_cached_ohlcv(symbol, timeframe, failed_symbols)
            if ohlcvs is None:
                continue

            data = process_data(ohlcvs, timeframe)
            signal_list = detect_signals(data, timeframe)

            if signal_list:
                detect_time = datetime.now(beijing_tz)
                for signal_info in signal_list:
                    signal_id = generate_signal_id(
                        symbol, timeframe, signal_info['detect_time']
                    )

                    signals.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'signal_type': signal_info['signal_type'],
                        'position': signal_info['position'],
                        'detect_time': signal_info['detect_time'],
                        'current_price': signal_info['current_price'],
                        'ma34': signal_info['ma34'],
                        'ma170': signal_info['ma170'],
                        'ma453': signal_info['ma453'],
                        'density_percent': signal_info['density_percent'],
                        'signal_id': signal_id
                    })
        except Exception as e:
            logging.debug(f"处理 {symbol} ({timeframe}) 出错: {str(e)}")
    return symbol, signals

def monitor_symbols():
    """监控交易对"""
    init_session_state()
    
    st.title('三均线密集排列实时监控系统（永续合约成交量前300版）')
    
    with st.expander("筛选条件说明", expanded=False):
        st.markdown("""
        **核心筛选条件**：

        ### 三均线密集排列：
        - **均线组合**：MA34, MA170, MA453
        - **密集度要求**：三条均线的最大值与最小值之间的差值不超过0.3%
          （即 (max_ma - min_ma) / max_ma <= 0.003）
        - **价格位置**：
          - 价格在均线上方（多头趋势）
          - 价格在均线下方（空头趋势）
          - 价格在均线之间（震荡趋势）
        - **警报规则**：相同交易对信号在半小时内只警报显示一次

        **交易对来源**：Gate.io实时成交量前300的USDT永续合约交易对

        **显示信息**：
        - <span style='color:green; font-weight:bold;'>绿色</span>：价格在均线上方（多头趋势）
        - <span style='color:red; font-weight:bold;'>红色</span>：价格在均线下方（空头趋势）
        - <span style='color:orange; font-weight:bold;'>橙色</span>：价格在均线之间（震荡趋势）
        - 密集度百分比（颜色越深表示越密集）
        - 当前价格和三条均线的最新值

        **规则**: 同交易对半小时内不重复报警，北京时间，多周期并行
        """, unsafe_allow_html=True)
    
    tabs = st.tabs([f"{tf.upper()} 周期" for tf in TIMEFRAMES])
    for idx, tf in enumerate(TIMEFRAMES):
        with tabs[idx]:
            container = st.container()
            placeholder = st.empty()
            st.session_state.result_containers[tf] = {'container': container, 'placeholder': placeholder}

    # 在监控面板下方展示待检测的300个目标交易对
    st.sidebar.subheader("待监控的300个交易对")

    # 获取一次交易对列表（如果尚未获取）
    if not st.session_state.symbols_to_monitor:
        st.session_state.symbols_to_monitor = get_valid_symbols()

    # 展示交易对列表（分列显示）
    cols = st.sidebar.columns(4)  # 分4列显示
    per_col = max(1, len(st.session_state.symbols_to_monitor) // len(cols))

    for i, col in enumerate(cols):
        start = i * per_col
        end = start + per_col
        if i == len(cols) - 1:
            # 最后一列显示剩余的所有交易对
            end = len(st.session_state.symbols_to_monitor)
        symbols = st.session_state.symbols_to_monitor[start:end]
        with col:
            for symbol in symbols:
                # 判断交易对当前状态
                if symbol in st.session_state.failed_symbols:
                    color = "red"  # 红色表示失败状态
                else:
                    color = "green"  # 绿色表示正常运行
                # 显示交易对名称并添加颜色标签
                st.markdown(f"<span style='color:{color};'>• {symbol}</span>", unsafe_allow_html=True)

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    stats = st.sidebar.empty()

    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            start_time = time.time()
            st.session_state.detection_round += 1
            new_signals = defaultdict(int)

            symbols = [s for s in get_valid_symbols() if s not in st.session_state.failed_symbols]
            if not symbols:
                logging.error("无有效交易对可监控")
                time.sleep(60)
                continue

            failed_symbols_copy = st.session_state.failed_symbols.copy()

            # 分批处理任务
            batch_size = 40
            futures = []
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                for symbol in batch_symbols:
                    futures.append(executor.submit(process_symbol_batch, symbol, failed_symbols_copy))

            # 处理结果
            for i, future in enumerate(as_completed(futures)):
                symbol, signals = future.result()
                progress = (i + 1) / len(futures)
                progress_bar.progress(progress)
                status_text.text(f"检测进度: {progress * 100:.1f}%")

                for signal in signals:
                    st.session_state.signal_queue.put(signal)

            play_sound = False
            while not st.session_state.signal_queue.empty():
                signal = st.session_state.signal_queue.get()
                tf = signal['timeframe']
                symbol = signal['symbol']
                signal_id = signal['signal_id']
                detect_time = signal['detect_time']
                
                # 检查相同交易对是否在半小时内已有信号
                last_time = st.session_state.last_signal_times.get(symbol)
                if last_time and (detect_time - last_time).total_seconds() < 1800:
                    continue  # 跳过半小时内的重复信号
                
                # 检查信号是否已显示
                if signal_id not in st.session_state.shown_signals[tf]:
                    st.session_state.valid_signals[tf].append(signal)
                    st.session_state.shown_signals[tf].add(signal_id)
                    st.session_state.last_signal_times[symbol] = detect_time
                    new_signals[tf] += 1
                    play_sound = True

            if play_sound:
                try:
                    play_alert_sound()
                except Exception as e:
                    logging.error(f"播放声音失败: {str(e)}")

            for tf in TIMEFRAMES:
                update_tab_content(tf)

            stats.markdown(
                f"轮次: {st.session_state.detection_round} | 新信号: {dict(new_signals)} | 失败交易对: {len(st.session_state.failed_symbols)}")
            elapsed = time.time() - start_time
            sleep_time = max(45 - elapsed, 30)
            time.sleep(sleep_time)

# 主函数
def main():
    st.set_page_config(
        page_title="三均线密集排列监控系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    monitor_symbols()

if __name__ == "__main__":
    main()
