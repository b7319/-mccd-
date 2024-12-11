import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import smtplib
from email.mime.text import MIMEText
import asyncio

# Initialize gate.io API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
exchange = ccxt.gateio({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'timeout': 20000})

# Email Configuration
SMTP_SERVER = 'smtp.139.com'
SMTP_PORT = 465
EMAIL_ADDRESS = '15762703145@139.com'
EMAIL_PASSWORD = 'GANen131419'
RECIPIENT_EMAIL = '13964531331@139.com'

# Initialize markets
def load_markets_with_retry():
    for attempt in range(3):
        try:
            exchange.load_markets()
            break
        except ccxt.NetworkError:
            st.write(f"Network error, retrying ({attempt + 1}/3)...")
            asyncio.sleep(5)
        except Exception as e:
            st.write(f"Error loading market data: {str(e)}")
            st.stop()

load_markets_with_retry()

# Get symbols with daily volume > 5M USDT
def filter_symbols_by_volume():
    try:
        tickers = exchange.fetch_tickers()
        symbols = [
            symbol for symbol, data in tickers.items()
            if data['quoteVolume'] and data['quoteVolume'] >= 5000000
        ]
        return symbols
    except Exception as e:
        st.write(f"Error fetching symbols: {str(e)}")
        return []

# Fetch data and calculate indicators
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
            return None

        # Calculate MA7, MA34, MA170, MA453
        df['ma7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['ma34'] = df['close'].rolling(window=34, min_periods=1).mean()
        df['ma170'] = df['close'].rolling(window=170, min_periods=1).mean()
        df['ma453'] = df['close'].rolling(window=453, min_periods=1).mean()

        # Calculate MACD (12, 26, 9)
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        return df
    except Exception as e:
        st.write(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Check if conditions are met
def check_conditions(df):
    if df is None or len(df) < 453:
        return False, None

    # Check MA7 trough in last 9 candles
    last_9 = df.iloc[-9:]
    ma7_diff = last_9['ma7'].diff()
    trough = (ma7_diff.shift(-1) > 0) & (ma7_diff < 0)

    if not trough.any():
        return False, None

    # Check MACD crossover in last 9 candles
    macd_cross = (last_9['macd'] > last_9['signal']) & (last_9['macd'].shift(1) <= last_9['signal'].shift(1))
    if not macd_cross.any():
        return False, None

    # Check MA34 crossover with MA170 or MA453 in last 31 candles
    last_31 = df.iloc[-31:]
    ma34 = last_31['ma34']
    ma170 = last_31['ma170']
    ma453 = last_31['ma453']

    crossover_170 = (ma34.shift(1) < ma170.shift(1)) & (ma34 >= ma170)
    crossover_453 = (ma34.shift(1) < ma453.shift(1)) & (ma34 >= ma453)

    if not (crossover_170.any() or crossover_453.any()):
        return False, None

    return True, last_9.index[-1]

# Send email notification
def send_email(subject, body):
    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Monitor symbols
async def monitor_symbols(symbols, progress_bar, status_text):
    valid_signals = set()
    detected_text = st.empty()

    while True:
        detected_signals = []
        for index, symbol in enumerate(symbols):
            df = fetch_data(symbol)
            if df is not None and not df.empty:
                condition_met, condition_time = check_conditions(df)
                if condition_met:
                    signal_key = (symbol, condition_time.strftime('%Y-%m-%d %H:%M:%S'))
                    if signal_key not in valid_signals:
                        valid_signals.add(signal_key)
                        message = f"""
                        Symbol: {symbol}
                        Timeframe: 30m
                        Condition Met At: {condition_time.strftime('%Y-%m-%d %H:%M:%S')}
                        VWAP: {df.loc[condition_time, 'vwap']:.2f}
                        """
                        send_email(f"Signal Notification - {symbol}", message)
                        detected_signals.append(message)

            progress_bar.progress((index + 1) / len(symbols))
            status_text.text(f"Monitoring symbol: {symbol}")
            await asyncio.sleep(9)  # Delay of 9 seconds between symbols

        detected_text.text("\n".join(detected_signals))

# Main function
async def main():
    st.title('Condition Monitor: MA34 Crossovers + MA7 Trough + MACD')
    symbols = filter_symbols_by_volume()

    if not symbols:
        st.warning("No symbols meet the volume criteria")
    else:
        st.success("Symbols loaded successfully!")
        progress_bar = st.progress(0)
        status_text = st.empty()
        await monitor_symbols(symbols, progress_bar, status_text)

if __name__ == "__main__":
    asyncio.run(main())
