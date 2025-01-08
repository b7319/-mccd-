def monitor_symbols(symbols):
    if 'valid_signals' not in st.session_state:
        st.session_state.valid_signals = []

    progress_bar = st.empty()
    status_text = st.empty()
    detected_text = st.empty()

    detected_text.markdown("### 当前检测状态：")
    
    while True:
        progress_bar.progress(0)
        status_text.text("检测进行中...")
        
        current_valid_signals = []

        for index, symbol in enumerate(symbols):
            df = fetch_data(symbol, timeframe='1m', max_bars=1000)
            if df is not None and not df.empty:
                condition_met, signal_type, condition_time = check_cross_conditions(df)
                if condition_met:
                    signal_key = (symbol, condition_time.strftime('%Y-%m-%d %H:%M:%S'))
                    symbol_data = {
                        'symbol': symbol,
                        'condition_time': condition_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'signal_type': signal_type
                    }
                    if signal_key not in [x['symbol'] for x in st.session_state.valid_signals]:
                        st.session_state.valid_signals.append(symbol_data)
                        current_valid_signals.append(symbol_data)
                        display_result(symbol_data)

            progress_bar.progress((index + 1) / len(symbols))
            time.sleep(1)

        if current_valid_signals:
            detected_text.markdown("### 累计符合条件的交易对：")
            for signal in st.session_state.valid_signals:
                detected_text.markdown(f"交易对: {signal['symbol']}, 满足条件时间: {signal['condition_time']}, 信号类型: {signal['signal_type']}")
        
        time.sleep(10)
