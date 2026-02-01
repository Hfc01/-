import streamlit as st
import torch
import torch.nn as nn
import jieba
import pandas as pd
import os
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置与模型定义
# ==========================================
MAX_LEN = 50
EMBEDDING_DIM = 100
HIDDEN_DIM = 128

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=2):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text) 
        output, (hidden, cell) = self.lstm(embedded)
        final_hidden = hidden[-1] 
        return self.fc(final_hidden)

@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'ChnSentiCorp_htl_all.csv')
    model_path = os.path.join(current_dir, 'sentiment_model.pth')

    if not os.path.exists(model_path): return None, None, None, "找不到模型文件！"
    
    try:
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path)
            except:
                df = pd.read_csv(data_path, encoding='gbk')
            df = df.dropna(subset=['review'])
            all_words = [word for text in df['review'].astype(str) for word in jieba.lcut(text)]
        else:
            all_words = ["好", "差"] 
            
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in Counter(all_words).most_common(5000):
            vocab[word] = len(vocab)
    except:
        return None, None, None, "数据加载失败"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, vocab, data_path, "SUCCESS"

# ==========================================
# 2. 工具函数
# ==========================================
def clear_cache():
    if 'result_df' in st.session_state: del st.session_state['result_df']
    if 'analyzed' in st.session_state: del st.session_state['analyzed']
    if 'csv_data' in st.session_state: del st.session_state['csv_data']

def generate_wordcloud(text_list, title):
    if not text_list: return None
    stop_words = {"的", "是", "了", "在", "我", "我们", "你", "有", "和", "就", "不", "人", "都", "一个", "上", "也", "很", "到", "说", "去", "会", "着", "没有", "但是", "因为", "还是", "这", "那", "个", "住", "对", "让", "给", "把", "被", "跟", "与", "为", "等", "酒店", "宾馆", "感觉", "觉得"}
    
    full_text = " ".join([str(t) for t in text_list])
    words = jieba.lcut(full_text)
    clean_words = [w for w in words if w not in stop_words and len(w) > 1]
    if not clean_words: return None
    cut_text = " ".join(clean_words)
    
    font_path = "simhei.ttf" 
    if os.path.exists("C:/Windows/Fonts/simhei.ttf"): font_path = "C:/Windows/Fonts/simhei.ttf"
    elif os.path.exists("C:/Windows/Fonts/msyh.ttc"): font_path = "C:/Windows/Fonts/msyh.ttc"
    
    wc = WordCloud(font_path=font_path, background_color='white', width=1000, height=800, max_words=100, font_step=2, collocations=False).generate(cut_text)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off') 
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    return fig

def analyze_aspect(text):
    aspects = {
        "卫生/设施": ["脏", "乱", "臭", "味道", "旧", "破", "坏", "干净", "整洁", "马桶", "床"],
        "服务态度": ["服务", "前台", "态度", "热情", "冷淡", "效率", "慢"],
        "位置/交通": ["位置", "交通", "地铁", "公交", "偏僻", "方便", "吵", "隔音"],
        "价格/性价比": ["价格", "贵", "便宜", "性价比", "划算", "值"]
    }
    text = str(text)
    detected = [k for k, v in aspects.items() if any(kw in text for kw in v)]
    return ", ".join(detected) if detected else "其他/未提及"

# ==========================================
# 3. 主程序入口
# ==========================================
def main():
    st.set_page_config(page_title="电商评论分析系统Pro", page_icon="🛍️", layout="wide")
    st.title("🛍️ 电商评论情感分析系统")

    with st.spinner('正在初始化 AI 大脑...'):
        model, vocab, default_data_path, status = load_resources()
    if status != "SUCCESS": st.error(status); return

    st.sidebar.header("🕹️ 控制台")
    app_mode = st.sidebar.radio("选择模式", ["单条分析", "批量分析"], on_change=clear_cache)

    # === 单条分析 ===
    if app_mode == "单条分析":
        st.subheader("📝 单条评论预测")
        col1, col2 = st.columns([3, 2])
        with col1:
            user_input = st.text_area("输入评论:", height=150)
            if st.button("分析", type="primary"):
                if user_input.strip():
                    words = jieba.lcut(user_input)
                    ids = [vocab.get(w, 1) for w in words]
                    ids = ids[:MAX_LEN] if len(ids) > MAX_LEN else ids + [0]*(MAX_LEN-len(ids))
                    tensor_input = torch.tensor([ids], dtype=torch.long)
                    with torch.no_grad():
                        prob = torch.nn.functional.softmax(model(tensor_input), dim=1)
                        pred_class = torch.argmax(prob).item()
                        conf = prob[0][pred_class].item()
                    aspect_info = analyze_aspect(user_input)
                    with col2:
                        st.markdown("### 结果")
                        if pred_class == 1: st.success(f"**😊 好评** ({conf:.2%})")
                        else: st.error(f"**😡 差评** ({conf:.2%})")
                        st.info(f"维度：{aspect_info}")

    # === 批量分析 (极简版) ===
    elif app_mode == "批量分析":
        st.subheader("📊 批量数据处理")
        
        data_source = st.radio("数据来源:", ["📂 上传 CSV", "🎁 演示数据"], horizontal=True, on_change=clear_cache)
        
        df = None
        
        if data_source == "📂 上传 CSV":
            uploaded_file = st.file_uploader("上传文件", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='gbk')
        else:
            if st.button("加载演示数据"):
                if default_data_path and os.path.exists(default_data_path):
                    try:
                        df = pd.read_csv(default_data_path).sample(1000)
                        st.session_state['temp_df'] = df 
                        clear_cache()
                    except: st.error("读取失败")
            if 'temp_df' in st.session_state and data_source == "🎁 演示数据":
                df = st.session_state['temp_df']

        if df is not None:
            # ✨ 核心修改：完全自动化列名识别，不让用户选 ✨
            cols = df.columns.tolist()
            # 优先级关键词列表
            keywords = ['review', '评论', 'content', 'text', '内容', 'category', 'feedback']
            text_col = cols[0] # 默认第一列，防崩
            
            # 智能匹配
            for col in cols:
                if any(k in col.lower() for k in keywords):
                    text_col = col
                    break
            
            # 仅仅展示一行小字告知用户
            st.info(f"✅ 已自动识别文本列：**{text_col}**")
            
            if st.button("🚀 开始分析", type="primary"):
                with st.spinner("处理中..."):
                    texts = df[text_col].astype(str).tolist()
                    input_ids = []
                    for text in texts:
                        words = jieba.lcut(text)
                        ids = [vocab.get(w, 1) for w in words]
                        ids = ids[:MAX_LEN] if len(ids) > MAX_LEN else ids + [0]*(MAX_LEN-len(ids))
                        input_ids.append(ids)
                    
                    tensor_input = torch.tensor(input_ids, dtype=torch.long)
                    with torch.no_grad():
                        preds = torch.argmax(model(tensor_input), dim=1).tolist()
                    
                    df['预测结果'] = ['好评' if p==1 else '差评' for p in preds]
                    df['涉及维度'] = df[text_col].apply(analyze_aspect)
                    
                    csv_data = df.to_csv(index=False).encode('utf-8-sig')
                    st.session_state['result_df'] = df
                    st.session_state['csv_data'] = csv_data
                    st.session_state['analyzed'] = True
                    st.rerun() 

        # 结果区
        if st.session_state.get('analyzed') and 'result_df' in st.session_state:
            res_df = st.session_state['result_df']
            
            st.markdown("---")
            
            # 筛选
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                filter_sentiment = st.multiselect("情感筛选:", ["好评", "差评"], default=["好评", "差评"])
            with col_filter2:
                filter_keyword = st.text_input("关键词搜索:")
            
            filtered_df = res_df.copy()
            # 这里的 text_col 需要重新获取一下，或者简单粗暴遍历所有列，
            # 但为了简单，我们直接用 session_state 里的数据，不用管列名了
            # 为了筛选关键词，我们假设包含'review'或'评论'的列，或者直接搜全表
            if filter_sentiment: filtered_df = filtered_df[filtered_df['预测结果'].isin(filter_sentiment)]
            if filter_keyword: 
                # 简单粗暴搜索所有列，省去麻烦
                mask = filtered_df.astype(str).apply(lambda x: x.str.contains(filter_keyword, case=False)).any(axis=1)
                filtered_df = filtered_df[mask]

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("数量", f"{len(filtered_df)}")
            kpi2.metric("好评", f"{len(filtered_df[filtered_df['预测结果']=='好评'])}")
            kpi3.metric("差评", f"{len(filtered_df[filtered_df['预测结果']=='差评'])}")
            
            c1, c2 = st.columns(2)
            with c1:
                if not filtered_df.empty: st.bar_chart(filtered_df['预测结果'].value_counts())
            with c2:
                # 重新寻找文本列用于画图
                cols = filtered_df.columns.tolist()
                keywords = ['review', '评论', 'content', 'text', 'category']
                target_col = cols[0]
                for col in cols:
                    if any(k in col.lower() for k in keywords):
                        target_col = col
                        break
                if not filtered_df.empty:
                    fig = generate_wordcloud(filtered_df[target_col].tolist(), "")
                    if fig: st.pyplot(fig)
            
            if 'csv_data' in st.session_state:
                st.download_button("📥 下载结果 (CSV)", st.session_state['csv_data'], 'result.csv', 'text/csv', type='primary')
            
            with st.expander("详细数据"):
                st.dataframe(filtered_df)

if __name__ == "__main__":
    main()