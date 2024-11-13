import ccxt
import os

# 使用API密钥初始化gate.io API
api_key = 'c8e2fb89d031ca42a30ed7b674cb06dc'
api_secret = 'fab0bc8aeebeb31e46238eda033e2b6258e9c9185f262f74d4472489f9f03219'
exchange = ccxt.gateio({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 20000
})

# 加载市场数据
exchange.load_markets()

# 获取所有的USDT交易对
usdt_pairs = [symbol for symbol in exchange.symbols if symbol.endswith('/USDT')]

# 文件路径
output_file_path = 'usdt_pairs_over_1m.txt'


def fetch_and_filter_usdt_pairs():
    valid_pairs = []
    for symbol in usdt_pairs:
        try:
            # 获取过去24小时的交易信息
            ticker = exchange.fetch_ticker(symbol)
            volume_quote = ticker['quoteVolume']

            # 检查交易对的日交易量是否大于100万USDT
            if volume_quote > 500000:
                valid_pairs.append(symbol)

        except Exception as e:
            print(f"获取 {symbol} 数据时出错: {str(e)}")

    # 输出符合条件的交易对至文本文件
    with open(output_file_path, 'w') as file:
        for pair in valid_pairs:
            file.write(f"{pair}\n")

    print(f"已将符合条件的交易对写入 '{output_file_path}' 文件中。")


# 运行函数
fetch_and_filter_usdt_pairs()
