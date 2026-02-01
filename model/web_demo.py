import streamlit as st
import torch
import torch.nn as nn
import jieba
import pandas as pd
import os
from collections import Counter

# ==========================================
# 1. 核心配置 (必须与训练时一致)
# ==========================================
MAX_LEN = 50
EMBEDDING_DIM = 100
HIDDEN_DIM = 128

# ==========================================
# 2. 定义模型架构
# ==========================================
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

# ==========================================
# 3. 加载资源 (使用缓存，网页不卡顿)
# ==========================================
@st.cache_resource
def load_resources():
    # --- A. 获取路径 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'ChnSentiCorp_htl_all.csv')
    model_path = os.path.join(current_dir, 'sentiment_model.pth')

    # --- B. 重建词典 ---
    if not os.path.exists(data_path):
        return None, None, "找不到数据文件，请检查 data 文件夹！"
    
    df = pd.read_csv(data_path).dropna(subset=['review'])
    texts = df['review'].astype(str).tolist()
    all_words = []
    for text in texts:
        all_words.extend(jieba.lcut(text))
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in Counter(all_words).most_common(5000):
        vocab[word] = len(vocab)

    # --- C. 加载模型 ---
    if not os.path.exists(model_path):
        return None, None, "找不到模型文件，请确认 sentiment_model.pth 在 model 文件夹里！"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, vocab, "SUCCESS"

# ==========================================
# 4. 网页界面主程序
# ==========================================
def main():
    # 设置网页标题
    st.set_page_config(page_title="电商评论情感分析系统", page_icon="🛒")
    
    st.title("🛒 基于深度学习的电商评论情感分析")
    st.markdown("这是你的毕业设计演示系统。输入一段评论，AI 将自动识别是好评还是差评。")

    # 加载模型 (显示加载状态)
    with st.spinner('正在初始化 AI 大脑...'):
        model, vocab, status = load_resources()

    if status != "SUCCESS":
        st.error(status)
        return

    # 左边输入，右边展示
    col1, col2 = st.columns([2, 1])

    with col1:
        # 输入框
        user_input = st.text_area("在此输入评论内容：", height=150, placeholder="例如：东西很好，物流很快，下次还来！")
        predict_btn = st.button("开始分析 🚀", type="primary")

    with col2:
        st.write("---")
        if predict_btn and user_input:
            if not user_input.strip():
                st.warning("请输入有效内容！")
            else:
                # --- 预测逻辑 ---
                words = jieba.lcut(user_input)
                ids = [vocab.get(w, 1) for w in words]
                if len(ids) > MAX_LEN:
                    ids = ids[:MAX_LEN]
                else:
                    ids = ids + [0] * (MAX_LEN - len(ids))
                
                tensor_input = torch.tensor([ids], dtype=torch.long)
                
                with torch.no_grad():
                    output = model(tensor_input)
                    probability = torch.nn.functional.softmax(output, dim=1)
                    pred_class = torch.argmax(probability).item()
                    confidence = probability[0][pred_class].item()

                # --- 结果展示 ---
                if pred_class == 1:
                    st.success("## 😊 分析结果：好评")
                    st.metric("置信度 (AI有多确定)", f"{confidence:.2%}")
                    st.balloons() # 放个气球庆祝一下
                else:
                    st.error("## 😡 分析结果：差评")
                    st.metric("置信度 (AI有多确定)", f"{confidence:.2%}")



# ==========================================
# 4. 网页界面主程序 
# ==========================================
def main():
    st.set_page_config(page_title="电商评论情感分析系统", page_icon="🛒", layout="wide")
    
    st.title("🛒 基于 LSTM 的电商评论情感分析系统")
    
    # 加载模型
    with st.spinner('正在初始化 AI 大脑...'):
        model, vocab, status = load_resources()

    if status != "SUCCESS":
        st.error(status)
        return

    # --- 侧边栏：功能选择 ---
    st.sidebar.title("功能菜单")
    app_mode = st.sidebar.radio("请选择模式", ["单条测试 (演示用)", "批量分析 (实战用)"])

    # ===================================
    # 模式一：单条测试 
    # ===================================
    if app_mode == "单条测试 (演示用)":
        st.header("👤 单条评论实时分析")
        st.markdown("这里模拟的是 **客服人员** 收到一条投诉时的场景。")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            user_input = st.text_area("输入评论内容：", height=150, placeholder="例如：物流太慢了，包装也破了！")
            predict_btn = st.button("开始分析 🚀", type="primary")
        
        with col2:
            st.write("---")
            if predict_btn and user_input:
                if not user_input.strip():
                    st.warning("请输入有效内容！")
                else:
                    # 预测逻辑
                    words = jieba.lcut(user_input)
                    ids = [vocab.get(w, 1) for w in words]
                    if len(ids) > MAX_LEN: ids = ids[:MAX_LEN]
                    else: ids = ids + [0] * (MAX_LEN - len(ids))
                    
                    tensor_input = torch.tensor([ids], dtype=torch.long)
                    with torch.no_grad():
                        output = model(tensor_input)
                        prob = torch.nn.functional.softmax(output, dim=1)
                        pred_class = torch.argmax(prob).item()
                        conf = prob[0][pred_class].item()

                    if pred_class == 1:
                        st.success("## 😊 好评")
                        st.metric("置信度", f"{conf:.2%}")
                    else:
                        st.error("## 😡 差评")
                        st.metric("置信度", f"{conf:.2%}")

   
    # 模式二：批量分析 
    
    elif app_mode == "批量分析 (实战用)":
        st.header("📊 海量数据自动化处理")
        st.markdown("这里模拟的是 **后台系统** 自动处理成千上万条历史评论的场景。")
        
        uploaded_file = st.file_uploader("上传 CSV 文件 (需包含 'review' 列)", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"✅ 成功读取文件，共 {len(df)} 条数据。前 5 条预览：")
            st.dataframe(df.head())
            
            if 'review' not in df.columns:
                st.error("❌ 文件里必须有一列叫 'review' 哦！")
            else:
                if st.button("开始批量分析 (可能会花一点时间)"):
                    # 进度条
                    progress_bar = st.progress(0)
                    results = []
                    probs = []
                    
                    # 批量预测
                    total = len(df)
                    # 为了演示不卡顿，我们只取前100条或者全部 (如果电脑快的话)
                    # 这里演示处理全部数据
                    texts = df['review'].astype(str).tolist()
                    
                    input_ids = []
                    for text in texts:
                        words = jieba.lcut(text)
                        ids = [vocab.get(w, 1) for w in words]
                        if len(ids) > MAX_LEN: ids = ids[:MAX_LEN]
                        else: ids = ids + [0] * (MAX_LEN - len(ids))
                        input_ids.append(ids)
                    
                    # 转 Tensor
                    tensor_input = torch.tensor(input_ids, dtype=torch.long)
                    
                    # 预测
                    with torch.no_grad():
                        outputs = model(tensor_input)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        predictions = torch.argmax(probabilities, dim=1).tolist()
                        max_probs = torch.max(probabilities, dim=1).values.tolist()
                    
                    # 把结果写回表格
                    df['预测结果'] = ['好评' if p==1 else '差评' for p in predictions]
                    df['置信度'] = [f"{p:.2%}" for p in max_probs]
                    
                    progress_bar.progress(100)
                    st.success("🎉 分析完成！")
                    
                    # 展示统计图表
                    st.subheader("分析报告")
                    count_data = df['预测结果'].value_counts()
                    st.bar_chart(count_data)
                    
                    st.write("详细结果预览：")
                    st.dataframe(df[['review', '预测结果', '置信度']])

if __name__ == "__main__":
    main()