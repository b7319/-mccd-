def remove_duplicates_from_file(file_path):
    try:
        # 读取文件中的所有行，并去除每行的空白字符
        with open(file_path, 'r') as file:
            symbols = file.read().splitlines()

        # 使用集合去除重复的交易对
        unique_symbols = set(symbols)

        # 将去除重复后的数据写回文件
        with open(file_path, 'w') as file:
            for symbol in unique_symbols:
                file.write(symbol + '\n')

    except Exception as e:
        print(f"An error occurred: {e}")


# 调用函数，传入symbols.txt文件的路径
remove_duplicates_from_file('symbols.txt')
