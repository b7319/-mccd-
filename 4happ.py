import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st

# 初始化 Gate.io API
api_key = 'YOUR_API_KEY'  # 替换为你的 API Key
api_secret = 'YOUR_API_SECRET'  # 替换为你的 API Secret
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 初始化交易市场数据，加载市场列表
def load_markets_with_retry():
    """尝试加载交易所市场，最多重试 3 次"""
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.write(f"网络错误，正在重试 ({attempt + 1}/3)...")
        except Exception as e:
            st.write(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

# 获取 USDT 现货交易对，日交易额超过 100 万 USDT 的交易对
def get_active_usdt_symbols(min_volume=1000000):
    """获取日交易额超过指定金额的 USDT 交易对"""
    try:
        tickers = exchange.fetch_tickers()  # 获取所有交易对的行情信息
        active_symbols = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and 'spot' in ticker.get('type', ''):
                # 检查交易额是否超过指定值
                if ticker.get('quoteVolume', 0) > min_volume:
                    active_symbols.append(symbol)
        return active_symbols
    except Exception as e:
        st.write(f"获取交易对时出错: {str(e)}")
        return []

# 获取指定交易对的历史数据
def fetch_data(symbol, timeframe='4h', days=68):
    """
    从交易所获取交易对的历史 K 线数据
    参数：
        symbol: 交易对名称，例如 'BTC/USDT'
        timeframe: 时间周期，例如 '4h'
        days: 获取最近多少天的数据
    返回：
        pandas.DataFrame 格式的 K 线数据，包含时间、开高低收和成交量
    """
    if symbol not in exchange.symbols:
        return None
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)

        if df.empty:
            st.write(f"{symbol} 在 {timeframe} 时间周期内没有数据。")
            return None

        # 计算移动平均线
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma34'] = df['close'].rolling(window=34).mean()

        return df
    except Exception as e:
        st.write(f"抓取 {symbol} 的数据时出错: {str(e)}")
        return None

# 识别顶分型
def is_top_fractal(df):
    """检查是否存在顶分型"""
    if len(df) < 3:
        return False

    recent_candles = df.iloc[-3:]
    return (recent_candles['high'][0] < recent_candles['high'][1] > recent_candles['high'][2] and
            recent_candles['low'][0] < recent_candles['low'][1] > recent_candles['low'][2])

# 计算密集成交区（VWAP）
def calculate_vwap(df, days=68):
    recent_data = df[-days * 6:]
    if recent_data.empty:
        return None
    vwap = (recent_data['close'] * recent_data['volume']).sum() / recent_data['volume'].sum()
    return round(vwap, 13)

# 检查底分型信号，并满足上方有顶分型的条件
def check_conditions(df):
    if len(df) < 3:
        return False, None, None, None

    recent_candles = df.iloc[-3:]
    low_fractal_formed = (
            recent_candles['low'].iloc[1] < recent_candles['low'].iloc[0] and
            recent_candles['low'].iloc[1] < recent_candles['low'].iloc[2] and
            recent_candles['high'].iloc[1] < recent_candles['high'].iloc[0] and
            recent_candles['high'].iloc[1] < recent_candles['high'].iloc[2]
    )

    if not low_fractal_formed:
        return False, None, None, None

    # 计算密集成交区的 VWAP 值
    vwap = calculate_vwap(df)
    if vwap is None:
        return False, None, None, None

    # 检查条件：底分型最低价或 MA7 波谷距离 VWAP 小于等于 3%
    ma7_valley = df['ma7'].min()  # 获取波谷的最小值
    low_distance_condition = abs(recent_candles['low'].iloc[1] - vwap) / vwap <= 0.03
    valley_distance_condition = abs(ma7_valley - vwap) / vwap <= 0.03

    if not (low_distance_condition or valley_distance_condition):
        return False, None, None, None

    # 检查底分型上方是否存在顶分型
    top_fractal_exists = any(is_top_fractal(df.iloc[i-2:i+1]) for i in range(len(df)-3))

    if not top_fractal_exists:
        return False, None, None, None

    return True, df.index[-1], df['ma7'].iloc[-1], vwap

# 存储已触发的条件和上次信号的时间
last_trigger_times = {}

def display_result(res):
    st.write(f"交易对: {res['symbol']}")
    st.write(f"时间周期: {res['timeframe']}")
    st.write(f"底分型对应时间点: {res['condition_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"密集成交区价格（VWAP）: {res['vwap']:.13f}")
    st.write(f"MA7最新值: {res['ma7_value']:.13f}")
    st.write(f"检测到信号时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("---")

    # 在页面中嵌入音频
    audio_file_path = "y1314.wav"
    with open(audio_file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/wav")

def monitor_symbols(symbols):
    st.title('底分型与密集成交区检测 (4小时)')
    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, symbol in enumerate(symbols):
        current_time = datetime.now()
        df = fetch_data(symbol)
        if df is not None and not df.empty:
            condition_met, condition_time, ma7_value, vwap = check_conditions(df)
            if condition_met:
                last_trigger_time = last_trigger_times.get(symbol, None)
                if last_trigger_time is None or (current_time - last_trigger_time).total_seconds() > 1800:
                    last_trigger_times[symbol] = current_time
                    symbol_data = {
                        'symbol': symbol,
                        'timeframe': '4小时',
                        'condition_time': condition_time,
                        'ma7_value': ma7_value,
                        'vwap': vwap
                    }
                    display_result(symbol_data)

        progress_bar.progress((index + 1) / len(symbols))
        status_text.text(f"正在检测交易对: {symbol}")

if __name__ == "__main__":
    # 动态获取符合条件的交易对
    st.title("交易对加载中...")
    symbols = get_active_usdt_symbols()
    if symbols:
        st.success(f"加载到 {len(symbols)} 个符合条件的交易对")
        monitor_symbols(symbols)
    else:
        st.warning("未找到符合条件的交易对")
