import streamlit as st
import requests
from bs4 import BeautifulSoup

# 设置 Streamlit 页面标题
st.title("百度网盘公开资源搜索工具")

# 搜索框输入
keyword = st.text_input("请输入关键词进行搜索：")

# 搜索按钮
if st.button("搜索"):
    if keyword.strip():
        st.write(f"正在搜索与 **{keyword}** 相关的公开资源，请稍候...")
        
        # 搜索逻辑
        try:
            # 模拟爬取公开的资源网站
            search_url = f"https://www.baidu.com/s?wd={keyword}+site:pan.baidu.com"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # 从搜索结果中提取资源标题和链接
                results = []
                for result in soup.find_all("a", href=True):
                    link = result["href"]
                    title = result.get_text(strip=True)
                    if "pan.baidu.com" in link:  # 只获取百度网盘链接
                        results.append({"title": title, "link": link})
                
                # 显示结果
                if results:
                    st.write(f"找到以下与 **{keyword}** 相关的资源：")
                    for item in results:
                        st.markdown(f"- **[{item['title']}]({item['link']})**")
                else:
                    st.write("未找到相关资源，请尝试其他关键词。")
            else:
                st.error("搜索失败，请稍后重试！")

        except Exception as e:
            st.error(f"发生错误：{e}")
    else:
        st.warning("请输入有效的关键词！")
