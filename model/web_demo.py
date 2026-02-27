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
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import hashlib

# ==========================================
# 0. 用户认证与数据库模块
# ==========================================
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

def create_usertable():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

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

    if not os.path.exists(model_path): 
        return None, {"<PAD>": 0, "<UNK>": 1}, data_path, "未检测到模型文件，系统将以演示模式运行。"
    
    try:
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path)
            except:
                df = pd.read_csv(data_path, encoding='gbk')
            df = df.dropna(subset=['review'])
            all_words = [word for text in df['review'].astype(str) for word in jieba.lcut(text)]
        else:
            all_words = ["好", "差", "服务", "环境", "干净"] 
            
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in Counter(all_words).most_common(5000):
            vocab[word] = len(vocab)
    except Exception as e:
        return None, None, None, f"数据加载失败: {str(e)}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        status = "SUCCESS"
    except Exception as e:
        model = None
        status = f"模型权重加载失败: {str(e)}"
    
    return model, vocab, data_path, status

# ==========================================
# 2. 核心分析与可视化工具函数
# ==========================================
def analyze_aspect(text):
    aspects = {
        "卫生与设施": ["脏", "乱", "臭", "味道", "旧", "破", "坏", "干净", "整洁", "马桶", "床", "设施", "硬件"],
        "服务体验": ["服务", "前台", "态度", "热情", "冷淡", "效率", "慢", "保洁", "保安"],
        "位置与交通": ["位置", "交通", "地铁", "公交", "偏僻", "方便", "吵", "隔音", "周边", "商场"],
        "价格与性价比": ["价格", "贵", "便宜", "性价比", "划算", "值", "收费"]
    }
    text = str(text)
    detected = [k for k, v in aspects.items() if any(kw in text for kw in v)]
    return detected if detected else ["其他/未提及"]

def generate_wordcloud(text_list, custom_stop_words=""):
    if not text_list: return None
    base_stop_words = {"的", "是", "了", "在", "我", "我们", "你", "有", "和", "就", "不", "人", "都", "一个", "上", "也", "很", "到", "说", "去", "会", "着", "没有", "但是", "因为", "还是", "这", "那", "个", "住", "对", "让", "给", "把", "被", "跟", "与", "为", "等", "感觉", "觉得"}
    
    if custom_stop_words:
        base_stop_words.update(set(custom_stop_words.replace("，", ",").split(",")))
    
    full_text = " ".join([str(t) for t in text_list])
    words = jieba.lcut(full_text)
    clean_words = [w for w in words if w not in base_stop_words and len(w) > 1]
    if not clean_words: return None
    cut_text = " ".join(clean_words)
    
    font_path = "simhei.ttf" 
    if os.path.exists("C:/Windows/Fonts/simhei.ttf"): font_path = "C:/Windows/Fonts/simhei.ttf"
    elif os.path.exists("C:/Windows/Fonts/msyh.ttc"): font_path = "C:/Windows/Fonts/msyh.ttc"
    
    wc = WordCloud(font_path=font_path, background_color='white', width=800, height=400, max_words=100, font_step=2, collocations=False).generate(cut_text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off') 
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    return fig

def predict_sentiment(texts, model, vocab):
    if not model:
        return ["好评" if np.random.rand() > 0.4 else "差评" for _ in texts]
    
    input_ids = []
    for text in texts:
        words = jieba.lcut(str(text))
        ids = [vocab.get(w, 1) for w in words]
        ids = ids[:MAX_LEN] if len(ids) > MAX_LEN else ids + [0]*(MAX_LEN-len(ids))
        input_ids.append(ids)
    
    tensor_input = torch.tensor(input_ids, dtype=torch.long)
    device = next(model.parameters()).device
    tensor_input = tensor_input.to(device)
    
    with torch.no_grad():
        preds = torch.argmax(model(tensor_input), dim=1).cpu().tolist()
    return ["好评" if p == 1 else "差评" for p in preds]

# ==========================================
# 3. 页面渲染模块
# ==========================================
def render_dashboard(df):
    st.markdown("### 📈 舆情数据监控大屏")
    
    total = len(df)
    pos_count = len(df[df['预测结果'] == '好评'])
    neg_count = len(df[df['预测结果'] == '差评'])
    pos_rate = pos_count / total if total > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总处理评论数", f"{total:,} 条", "+更新")
    with col2:
        st.metric("总体好评数", f"{pos_count:,} 条", f"{pos_rate:.1%} 占比")
    with col3:
        st.metric("总体差评数", f"{neg_count:,} 条", f"{1-pos_rate:.1%} 占比", delta_color="inverse")
    with col4:
        st.metric("分析模型状态", "在线 (LSTM)", "正常运行")
    
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        fig_pie = px.pie(
            names=['好评', '差评'], 
            values=[pos_count, neg_count], 
            hole=0.4,
            title="情感极性总体分布",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        all_aspects = [aspect for sublist in df['维度列表'] for aspect in sublist]
        aspect_counts = pd.Series(all_aspects).value_counts().reset_index()
        aspect_counts.columns = ['维度', '提及频次']
        
        fig_bar = px.bar(
            aspect_counts, 
            x='提及频次', 
            y='维度', 
            orientation='h',
            title="消费者核心关注维度Top排行",
            color='提及频次',
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 🎯 核心维度情感交叉分析")
    aspect_sentiment_data = []
    for _, row in df.iterrows():
        for aspect in row['维度列表']:
            aspect_sentiment_data.append({'维度': aspect, '情感': row['预测结果']})
    
    cross_df = pd.DataFrame(aspect_sentiment_data)
    if not cross_df.empty:
        cross_table = pd.crosstab(cross_df['维度'], cross_df['情感']).reset_index()
        for col in ['好评', '差评']:
            if col not in cross_table.columns: cross_table[col] = 0
            
        fig_stack = go.Figure(data=[
            go.Bar(name='差评', x=cross_table['维度'], y=cross_table['差评'], marker_color='#e74c3c'),
            go.Bar(name='好评', x=cross_table['维度'], y=cross_table['好评'], marker_color='#2ecc71')
        ])
        fig_stack.update_layout(barmode='stack', title="各维度情感倾向构成比", xaxis_title="评价维度", yaxis_title="评论数量")
        st.plotly_chart(fig_stack, use_container_width=True)

# ==========================================
# 4. 登录页面与主程序入口
# ==========================================
def login_page():
    # 为了让登录页面更简洁，居中显示
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 系统登录")
        st.markdown("欢迎访问电商评论智能分析平台。")
        
        create_usertable()
        
        tab1, tab2 = st.tabs(["🔑 身份验证", "📝 注册账号"])
        
        with tab1:
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type='password', key="login_pass")
            if st.button("登录系统", type="primary", use_container_width=True):
                hashed_pswd = make_hashes(password)
                result = login_user(username, hashed_pswd)
                if result:
                    st.session_state['logged_in'] = True
                    st.session_state['current_user'] = username
                    st.success("验证成功，正在进入系统...")
                    st.rerun()
                else:
                    st.error("验证失败：用户名或密码错误。")

        with tab2:
            new_user = st.text_input("设定用户名", key="reg_user")
            new_password = st.text_input("设定密码", type='password', key="reg_pass")
            if st.button("注册新用户", use_container_width=True):
                if new_user and new_password:
                    add_userdata(new_user, make_hashes(new_password))
                    st.success("账号注册成功，请切换至登录标签进行操作。")
                else:
                    st.warning("注册失败：需填写完整信息。")

def main():
    if 'global_stop_words' not in st.session_state:
        st.session_state['global_stop_words'] = "酒店,宾馆,入住"

    with st.spinner('系统内核初始化中...'):
        model, vocab, default_data_path, status = load_resources()
        
    st.sidebar.title("🏢 文本挖掘系统")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("系统功能导航", [
        "📊 监控大屏 (Dashboard)", 
        "📝 单条诊断 (Single Test)", 
        "📂 批量挖掘 (Batch Mining)",
        "⚙️ 系统设置 (Settings)"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Model Status: " + ("✅ Online" if model else "⚠️ Demo Mode"))
    st.sidebar.write(f"操作员: **{st.session_state.get('current_user', 'Admin')}**")
    if st.sidebar.button("🚪 安全退出"):
        st.session_state['logged_in'] = False
        st.rerun()

    if app_mode == "⚙️ 系统设置 (Settings)":
        st.title("⚙️ 系统参数配置")
        with st.form("config_form"):
            st.subheader("文本预处理配置")
            stop_words_input = st.text_area("自定义停用词 (以英文逗号分隔):", value=st.session_state['global_stop_words'])
            submitted = st.form_submit_button("保存系统配置")
            if submitted:
                st.session_state['global_stop_words'] = stop_words_input
                st.success("系统配置已更新并生效。")

    elif app_mode == "📝 单条诊断 (Single Test)":
        st.title("📝 单文本情感诊断")
        with st.form("single_analysis_form"):
            user_input = st.text_area("📄 待测文本输入区:", height=150)
            submit_btn = st.form_submit_button("运行分析预测")
            
        if submit_btn and user_input.strip():
            with st.spinner("神经网络推断中..."):
                aspects = analyze_aspect(user_input)
                if model:
                    words = jieba.lcut(user_input)
                    ids = [vocab.get(w, 1) for w in words]
                    ids = ids[:MAX_LEN] if len(ids) > MAX_LEN else ids + [0]*(MAX_LEN-len(ids))
                    tensor_input = torch.tensor([ids], dtype=torch.long)
                    with torch.no_grad():
                        prob = torch.nn.functional.softmax(model(tensor_input), dim=1)
                        pred_class = torch.argmax(prob).item()
                        conf = prob[0][pred_class].item()
                    res_label = "好评" if pred_class == 1 else "差评"
                else:
                    res_label = "好评" if "好" in user_input else "差评"
                    conf = 0.95
                
            st.markdown("### 诊断报告")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                if res_label == "好评": st.success(f"**判定极性：正向**")
                else: st.error(f"**判定极性：负向**")
            with col_res2: st.info(f"**模型置信度：{conf:.2%}**")
            with col_res3: st.warning(f"**涉及维度：{', '.join(aspects)}**")

    elif app_mode == "📂 批量挖掘 (Batch Mining)":
        st.title("📂 数据批量导入与挖掘")
        data_source = st.radio("选择数据源方式:", ["📂 本地上传文件", "🎁 加载系统演示数据集"], horizontal=True)
        
        df = None
        if data_source == "📂 本地上传文件":
            uploaded_file = st.file_uploader("请选择 CSV 格式数据文件", type=["csv"])
            if uploaded_file:
                try: df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='gbk')
        else:
            if st.button("一键加载测试样本"):
                if default_data_path and os.path.exists(default_data_path):
                    st.session_state['temp_demo_df'] = pd.read_csv(default_data_path).sample(500)
                else:
                    st.error("未找到演示数据集。")
            if 'temp_demo_df' in st.session_state and data_source == "🎁 加载系统演示数据集":
                df = st.session_state['temp_demo_df']
                
        if df is not None:
            cols = df.columns.tolist()
            keywords = ['review', '评论', 'content', 'text', '内容']
            text_col = cols[0] 
            for col in cols:
                if any(k in col.lower() for k in keywords):
                    text_col = col; break
            
            st.info(f"💡 识别分析对象列：`[{text_col}]`")
            
            if st.button("🚀 启动全量深度分析", type="primary"):
                progress_bar = st.progress(0)
                with st.spinner("执行分析中..."):
                    texts = df[text_col].astype(str).tolist()
                    df['预测结果'] = predict_sentiment(texts, model, vocab)
                    progress_bar.progress(50)
                    df['维度列表'] = df[text_col].apply(analyze_aspect)
                    df['涉及维度'] = df['维度列表'].apply(lambda x: ", ".join(x))
                    progress_bar.progress(100)
                    st.session_state['master_df'] = df
                    st.session_state['text_col'] = text_col
                    st.success("分析完成！请前往「监控大屏」查看。")
            
        if 'master_df' in st.session_state:
            st.markdown("### 数据导出")
            res_df = st.session_state['master_df']
            csv_data = res_df.drop(columns=['维度列表'], errors='ignore').to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 导出分析结果 (CSV)", csv_data, 'output.csv', 'text/csv')

    elif app_mode == "📊 监控大屏 (Dashboard)":
        st.title("📊 全局数据监控与可视化")
        if 'master_df' not in st.session_state:
            st.warning("暂无数据。请先执行批量挖掘任务。")
        else:
            df = st.session_state['master_df']
            render_dashboard(df)
            
            st.markdown("### ☁️ 高频词云提取")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**正向特征**")
                pos_texts = df[df['预测结果'] == '好评'][st.session_state.get('text_col', 'review')].tolist()
                fig_pos = generate_wordcloud(pos_texts, st.session_state['global_stop_words'])
                if fig_pos: st.pyplot(fig_pos)
            with c2:
                st.markdown("**负向特征**")
                neg_texts = df[df['预测结果'] == '差评'][st.session_state.get('text_col', 'review')].tolist()
                fig_neg = generate_wordcloud(neg_texts, st.session_state['global_stop_words'])
                if fig_neg: st.pyplot(fig_neg)

if __name__ == "__main__":
    # 确保此为程序的首个 Streamlit 指令
    st.set_page_config(page_title="电商评论情感分析系统", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        # 隐藏左侧边栏，营造纯粹的登录界面
        st.markdown("<style>[data-testid='collapsedControl'] {display: none;}</style>", unsafe_allow_html=True)
        login_page()
    else:
        main()