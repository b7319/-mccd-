import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import smtplib
from email.mime.text import MIMEText
import asyncio

# 初始化 gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# 139 邮箱配置
SMTP_SERVER = 'smtp.139.com'
SMTP_PORT = 465  # 使用 SSL 加密
EMAIL_ADDRESS = '15762703145@139.com'  # 替换为你的 139 邮箱
EMAIL_PASSWORD = 'GANen131419'  # 替换为你的 139 邮箱密码
RECIPIENT_EMAIL = '13964531331@139.com'  # 收件人邮箱

# 初始化交易市场
def load_markets_with_retry():
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.write(f"网络错误，正在重试 ({attempt + 1}/3)...")
            asyncio.sleep(5)
        except Exception as e:
            st.write(f"加载市场数据时出错: {str(e)}")
            st.stop()

load_markets_with_retry()

# 获取日交易额大于 500 万 USDT 的交易对
def filter_symbols_by_volume():
    try:
        tickers = exchange.fetch_tickers()
        symbols = [
            symbol for symbol, data in tickers.items()
            if data['quoteVolume'] and data['quoteVolume'] >= 5000000
        ]
        return symbols
    except Exception as e:
        st.write(f"获取交易对数据时出错: {str(e)}")
        return []

# 获取数据并计算 MA7、MA34、MA453 和 VWAP
def fetch_data(symbol, timeframe='30m', max_bars=500):
    if symbol not in exchange.symbols:
        return None

    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(minutes=max_bars * 30)).isoformat())
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)

        if df.empty:
            st.write(f"{symbol} 在 {timeframe} 时间周期内没有数据。")
            return None

        # 计算 MA7、MA34、MA453
        df['ma7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['ma34'] = df['close'].rolling(window=34, min_periods=1).mean()
        df['ma453'] = df['close'].rolling(window=453, min_periods=1).mean()

        # 计算 VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()

        return df
    except Exception as e:
        st.write(f"抓取 {symbol} 的数据时出错: {str(e)}")
        return None

# 检查是否满足筛选条件
def check_conditions(df):
    if df is None or len(df) < 453:
        return False, None

    # 检查 MA34 金叉 MA453
    last_13 = df.iloc[-13:]
    ma34 = last_13['ma34']
    ma453 = last_13['ma453']

    crossover = False
    for i in range(1, len(ma34)):
        if ma34.iloc[i - 1] < ma453.iloc[i - 1] and ma34.iloc[i] >= ma453.iloc[i]:
            crossover = True
            crossover_time = last_13.index[i]
            break

    if not crossover:
        return False, None

    # 检查 MA7 波谷向上
    last_3 = df.iloc[-3:]
    ma7 = last_3['ma7']
    ma7_above_ma453 = (ma7 > last_3['ma453']).all()

    if not ma7_above_ma453:
        return False, None

    ma7_diff = ma7.diff()
    if ma7_diff.iloc[1] < 0 and ma7_diff.iloc[2] > 0:
        return True, last_3.index[2]

    return False, None

# 发送邮件功能
def send_email(subject, body):
    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")

# 检测交易对
async def monitor_symbols(symbols, progress_bar, status_text):
    valid_signals = set()
    while True:
        for index, symbol in enumerate(symbols):
            df = fetch_data(symbol)
            if df is not None and not df.empty:
                condition_met, condition_time = check_conditions(df)
                if condition_met:
                    signal_key = (symbol, condition_time.strftime('%Y-%m-%d %H:%M:%S'))
                    if signal_key not in valid_signals:
                        valid_signals.add(signal_key)
                        message = f"""
                        交易对: {symbol}
                        时间周期: 30分钟
                        满足条件时间: {condition_time.strftime('%Y-%m-%d %H:%M:%S')}
                        VWAP: {df.loc[condition_time, 'vwap']:.2f}
                        """
                        send_email(f"筛选结果通知 - {symbol}", message)

            progress_bar.progress((index + 1) / len(symbols))
            status_text.text(f"正在检测交易对: {symbol}")
            await asyncio.sleep(13)  # 每检测一个交易对后等待 6 秒

# 主函数
async def main():
    st.title('筛选条件: MA34金叉MA453 + MA7波谷向上')
    symbols = filter_symbols_by_volume()

    if not symbols:
        st.warning("没有符合条件的交易对")
    else:
        st.success("交易对加载成功！")
        progress_bar = st.progress(0)
        status_text = st.empty()
        await monitor_symbols(symbols, progress_bar, status_text)

if __name__ == "__main__":
    asyncio.run(main())
