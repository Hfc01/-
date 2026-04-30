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
from run_model import run_training

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

def load_resources(model_filename='sentiment_model.pth'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'ChnSentiCorp_htl_all.csv')
    model_path = os.path.join(current_dir, model_filename)

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
def analyze_aspect(text, domain="电商通用"):
    domain_aspects = {
        "外卖餐饮": {
            "餐饮口感与分量": ["好吃", "难吃", "口味", "味道", "咸", "淡", "辣", "新鲜", "变质", "馊", "分量", "太少", "吃不饱", "口感", "肉", "菜", "饭", "冷", "凉", "热乎", "变味", "骨头", "汤"],
            "外卖配送与骑手": ["骑手", "外卖员", "小哥", "送餐", "超时", "送错", "撒", "洒", "漏", "提前", "准时", "送上楼", "没按门铃", "态度差", "电话"]
        },
        "酒店住宿": {
            "酒店设施与卫生": ["卫生", "干净", "脏", "乱", "臭", "异味", "床", "马桶", "热水", "花洒", "空调", "设施", "硬件", "旧", "破", "被子", "毛巾", "虫", "蟑螂", "发霉", "洗澡"],
            "住宿位置与环境": ["位置", "交通", "地铁", "公交", "隔音", "吵", "安静", "周边", "商场", "便利店", "偏僻", "市中心", "风景", "海景", "隔壁"]
        },
        "汽车出行": {
            "汽车动力与操控": ["动力", "起步", "加速", "肉", "推背感", "操控", "方向盘", "底盘", "减震", "刹车", "顿挫", "换挡", "平顺", "马力", "超车"],
            "空间内饰与能耗": ["空间", "宽敞", "拥挤", "后排", "后备箱", "内饰", "塑料感", "异响", "油耗", "省油", "费油", "续航", "电耗", "充电", "车机", "屏幕", "大屏", "死机", "真皮"]
        },
        "电商通用": {
            "商品质量与做工": ["质量", "正品", "材质", "做工", "坏", "好用", "差", "破损", "假", "瑕疵", "掉色", "粗糙", "精致", "结实", "耐用", "假货", "劣质", "用料", "手感", "脱线", "起球"],
            "物流发货与包装": ["物流", "快递", "包装", "速度", "发货", "慢", "快", "顺丰", "驿站", "签收", "送门", "送货", "漏发", "错发", "压扁", "箱子", "严实", "暴力", "隔天"],
            "客户服务与售后": ["客服", "态度", "退换", "售后", "回复", "解决", "热情", "冷淡", "维权", "投诉", "催", "理人", "退款", "运费险", "服务", "敷衍", "保养", "4S"],
            "价格与性价比": ["价格", "贵", "便宜", "性价比", "划算", "值", "降价", "优惠", "折扣", "活动", "满减", "薅羊毛", "差价", "不值", "物美价廉", "实惠", "宰客", "坑钱"],
            "外观颜值与设计": ["外观", "颜值", "漂亮", "难看", "丑", "色差", "款式", "设计", "好看", "颜色", "精美", "高级", "廉价", "可爱", "时尚", "土", "车漆", "造型"]
        }
    }

    general_keywords = [
        "推荐", "不错", "很好", "满意", "喜欢", "绝了", "挺好", "还行", "可以", "买", "赞", "棒", "爱了", "好评", "无敌", "yyds",
        "失望", "垃圾", "避雷", "坑", "一般般", "无语", "差评", "恶心", "服了", "千万别", "后悔", "上当", "退避三舍", "踩雷", "骗"
    ]

    text = str(text)
    current_aspects = domain_aspects.get(domain, domain_aspects["电商通用"])
    detected = [k for k, v in current_aspects.items() if any(kw in text for kw in v)]

    if not detected:
        if any(kw in text for kw in general_keywords):
            return ["综合整体评价"]
        else:
            return ["综合整体评价"]

    return detected

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
    possible_font_paths = [
        "simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc"
    ]

    for path in possible_font_paths:
        if os.path.exists(path):
            font_path = path
            break

    wc = WordCloud(font_path=font_path, background_color='#f5f5f7', width=800, height=400, max_words=100, font_step=2, collocations=False).generate(cut_text)
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
    total = len(df)
    pos_count = len(df[df['预测结果'] == '好评'])
    neg_count = len(df[df['预测结果'] == '差评'])
    pos_rate = pos_count / total if total > 0 else 0

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.metric("总处理评论数", f"{total:,} 条")
    with col2:
        st.metric("总体好评数", f"{pos_count:,} 条", f"{pos_rate:.1%}")
    with col3:
        st.metric("总体差评数", f"{neg_count:,} 条", f"{1-pos_rate:.1%}")
    with col4:
        st.metric("模型状态", "在线运行")

    col_chart1, col_chart2 = st.columns([1, 1])
    with col_chart1:
        fig_pie = px.pie(
            names=['好评', '差评'],
            values=[pos_count, neg_count],
            hole=0.4,
            title="情感极性分布",
            color_discrete_sequence=['#0071e3', '#ff3b30']
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#f5f5f7', width=3))
        )
        fig_pie.update_layout(
            height=350,
            font=dict(family="SF Pro Text", size=14),
            title=dict(font=dict(size=20, color='#1d1d1f')),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        all_aspects = [aspect for sublist in df['维度列表'] for aspect in sublist]
        aspect_counts = pd.Series(all_aspects).value_counts().reset_index()
        aspect_counts.columns = ['维度', '提及频次']

        fig_bar = px.bar(
            aspect_counts,
            x='提及频次',
            y='维度',
            orientation='h',
            title="消费者核心关注维度",
            color='提及频次',
            color_continuous_scale='blues'
        )
        fig_bar.update_traces(
            marker=dict(line=dict(color='#f5f5f7', width=1)),
            hovertemplate='<b>%{y}</b><br>提及频次: %{x}<extra></extra>'
        )
        fig_bar.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=350,
            font=dict(family="SF Pro Text", size=14),
            title=dict(font=dict(size=20, color='#1d1d1f')),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(title="频次")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

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
            go.Bar(name='差评', x=cross_table['维度'], y=cross_table['差评'], marker_color='#ff3b30'),
            go.Bar(name='好评', x=cross_table['维度'], y=cross_table['好评'], marker_color='#0071e3')
        ])
        fig_stack.update_layout(
            barmode='stack',
            title=dict(text="各维度情感构成", font=dict(size=20, color='#1d1d1f')),
            xaxis_title="评价维度",
            yaxis_title="评论数量",
            height=400,
            font=dict(family="SF Pro Text", size=14),
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_stack, use_container_width=True)

# ==========================================
# 4. 登录页面与主程序入口
# ==========================================
def login_page():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Apple Design System Colors */
    :root {
        --apple-black: #000000;
        --apple-gray: #f5f5f7;
        --apple-near-black: #1d1d1f;
        --apple-blue: #0071e3;
        --apple-blue-hover: #0077ed;
        --apple-red: #ff3b30;
        --apple-white: #ffffff;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {display: none;}



    /* Full Page Background */
    .stApp {
        background-color: var(--apple-gray);
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
    }

    /* Title Styling */
    .login-title {
        font-size: 32px;
        font-weight: 600;
        color: var(--apple-near-black);
        text-align: center;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
        line-height: 1.07;
    }

    .login-subtitle {
        font-size: 15px;
        color: #86868b;
        text-align: center;
        margin-bottom: 32px;
        line-height: 1.47;
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: var(--apple-gray);
        border: none;
        border-radius: 8px;
        padding: 14px 16px;
        font-size: 15px;
        transition: all 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        background-color: var(--apple-white);
        box-shadow: none;
        outline: 2px solid var(--apple-blue);
    }

    /* Primary Button */
    .stButton > button {
        background-color: var(--apple-blue);
        color: var(--apple-white);
        border: none;
        border-radius: 8px;
        padding: 8px 15px;
        font-size: 17px;
        font-weight: 400;
        width: 100%;
        transition: background-color 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--apple-blue-hover);
    }

    /* Secondary Button */
    .stButton > button[kind="secondary"],
    .stButton > button:not([type="primary"]) {
        background-color: var(--apple-gray);
        color: var(--apple-blue);
    }

    /* Tabs */
    .stTabs > div > div {
        border-bottom: 1px solid #d2d2d7;
    }

    .stTabs > div > div > button {
        font-size: 14px;
        font-weight: 500;
        color: #86868b;
        background: transparent;
        border: none;
        padding: 12px 20px;
        border-bottom: 2px solid transparent;
        margin-right: 20px;
    }

    .stTabs > div > div > button[data-testid="stTabActive"]:not([class*="st-"]) {
        color: var(--apple-near-black);
        border-bottom-color: var(--apple-blue);
    }

    /* Success/Error Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        padding: 12px 16px;
    }
    </style>
    """, unsafe_allow_html=True)



    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<h1 class="login-title">登录</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">电商评论智能分析平台</p>', unsafe_allow_html=True)

        create_usertable()

        tab1, tab2 = st.tabs(["身份验证", "注册账号"])

        with tab1:
            username = st.text_input("用户名", placeholder="请输入用户名", key="login_user")
            password = st.text_input("密码", type='password', placeholder="请输入密码", key="login_pass")
            if st.button("登录系统", type="primary"):
                if not username or not password:
                    st.warning("请填写完整的登录信息")
                else:
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
            new_user = st.text_input("设定用户名", placeholder="请设定用户名", key="reg_user")
            new_password = st.text_input("设定密码", type='password', placeholder="请设定密码", key="reg_pass")
            if st.button("注册新用户"):
                if new_user and new_password:
                    add_userdata(new_user, make_hashes(new_password))
                    st.success("账号注册成功，请切换至登录标签进行操作。")
                else:
                    st.warning("注册失败：需填写完整信息。")

        st.markdown('<p style="text-align: center; margin-top: 32px; color: var(--apple-white); font-size: 12px;">© 2026 电商评论情感分析系统</p>', unsafe_allow_html=True)

def main():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    :root {
        --apple-black: #000000;
        --apple-gray: #f5f5f7;
        --apple-near-black: #1d1d1f;
        --apple-blue: #0071e3;
        --apple-blue-hover: #0077ed;
        --apple-red: #ff3b30;
        --apple-green: #34c759;
        --apple-white: #ffffff;
    }

    /* Background */
    .stApp {
        background-color: var(--apple-gray);
    }

    /* Hide Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {display: none;}

    /* Page Title */
    h1 {
        font-size: 32px !important;
        font-weight: 600 !important;
        color: var(--apple-near-black) !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 24px !important;
        line-height: 1.07 !important;
    }

    h2 {
        font-size: 28px !important;
        font-weight: 400 !important;
        color: var(--apple-near-black) !important;
        letter-spacing: -0.3px !important;
        margin-top: 24px !important;
        line-height: 1.14 !important;
    }

    h3 {
        font-size: 21px !important;
        font-weight: 600 !important;
        color: var(--apple-near-black) !important;
        letter-spacing: 0.231px !important;
        line-height: 1.19 !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: var(--apple-white);
        border: none;
        border-radius: 8px;
        padding: 20px;
        box-shadow: rgba(0, 0, 0, 0.22) 3px 5px 30px 0px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 12px;
        font-weight: 500;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
        color: var(--apple-near-black);
    }

    [data-testid="stMetricDelta"] {
        font-size: 13px;
        color: #86868b;
    }

    /* Primary Button */
    .stButton > button[type="primary"],
    [data-testid="stMainBlockContainer"] button[data-testid="stBaseButton-primary"] {
        background-color: var(--apple-blue) !important;
        color: var(--apple-white) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 15px !important;
        font-size: 17px !important;
        font-weight: 400 !important;
        transition: background-color 0.2s ease !important;
    }

    .stButton > button[type="primary"]:hover,
    [data-testid="stMainBlockContainer"] button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--apple-blue-hover) !important;
    }

    /* Secondary Button */
    .stButton > button:not([type="primary"]) {
        background-color: var(--apple-white) !important;
        color: var(--apple-near-black) !important;
        border: 1px solid #d2d2d7 !important;
        border-radius: 8px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--apple-white);
        backdrop-filter: saturate(180%) blur(20px);
    }

    [data-testid="stSidebarNav"] {
        background-color: var(--apple-white);
    }

    /* Radio */
    .stRadio > label {
        font-size: 14px;
        color: var(--apple-near-black);
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: var(--apple-white);
        border: 1px solid #d2d2d7;
        border-radius: 8px;
    }

    /* Text Area */
    .stTextArea > div > div > textarea {
        background-color: var(--apple-white);
        border: 1px solid #d2d2d7;
        border-radius: 8px;
        padding: 14px;
        transition: all 0.2s ease;
    }

    textarea:focus {
        box-shadow: none !important;
        outline: 2px solid var(--apple-blue) !important;
        border: 1px solid var(--apple-blue) !important;
    }

    /* Also remove Streamlit's default focus styling */
    .stTextArea textarea:focus {
        box-shadow: none !important;
        outline: 2px solid var(--apple-blue) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--apple-white);
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: var(--apple-white);
        border-radius: 8px;
    }

    /* Tabs */
    .stTabs > div > div {
        border-bottom: 1px solid #d2d2d7;
    }

    .stTabs button {
        font-size: 14px;
        font-weight: 500;
        color: #86868b;
        background: transparent;
        border: none;
        padding: 12px 20px;
        border-bottom: 2px solid transparent;
        margin-right: 20px;
    }

    .stTabs button[data-testid="stTabActive"] {
        color: var(--apple-near-black);
        border-bottom-color: var(--apple-blue);
    }

    /* Success/Info/Warning/Error */
    .stSuccess {
        background-color: rgba(52, 199, 89, 0.12);
        color: #28a745;
        border-radius: 8px;
        padding: 12px 16px;
    }

    .stError {
        background-color: rgba(255, 59, 48, 0.12);
        color: #dc3545;
        border-radius: 8px;
        padding: 12px 16px;
    }

    .stWarning {
        background-color: rgba(255, 149, 0, 0.12);
        color: #e67e22;
        border-radius: 8px;
        padding: 12px 16px;
    }

    .stInfo {
        background-color: rgba(0, 113, 227, 0.12);
        color: var(--apple-blue);
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Link Button */
    .link-button {
        display: inline-block;
        padding: 8px 16px;
        background-color: transparent;
        color: var(--apple-blue);
        border: 1px solid var(--apple-blue);
        border-radius: 980px;
        font-size: 14px;
        font-weight: 400;
        text-decoration: none;
        transition: all 0.2s ease;
    }

    .link-button:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Header
    st.sidebar.markdown('<div style="padding: 20px 0;"><h1 style="font-size: 24px; font-weight: 600; color: var(--apple-near-black); margin: 0; letter-spacing: -0.5px;">电商评论情感预测系统</h1></div>', unsafe_allow_html=True)

    # Usage Guide
    st.sidebar.markdown("""
    <div style="background: #f5f5f7; border-radius: 12px; padding: 16px; margin-bottom: 20px;">
        <p style="font-size: 12px; font-weight: 600; color: #86868b; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.5px;">使用流程</p>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: #0071e3; color: #fff; font-size: 11px; font-weight: 600; margin-right: 10px; flex-shrink: 0;">1</span>
            <span style="font-size: 13px; color: #1d1d1f;">进入「批量挖掘」上传数据</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: #0071e3; color: #fff; font-size: 11px; font-weight: 600; margin-right: 10px; flex-shrink: 0;">2</span>
            <span style="font-size: 13px; color: #1d1d1f;">点击「启动全量深度分析」</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: #0071e3; color: #fff; font-size: 11px; font-weight: 600; margin-right: 10px; flex-shrink: 0;">3</span>
            <span style="font-size: 13px; color: #1d1d1f;">前往「监控大屏」查看结果</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.sidebar.markdown('<div style="margin: 20px 0;">', unsafe_allow_html=True)
    app_mode = st.sidebar.radio(
        "", 
        ["批量挖掘", "监控大屏", "单条诊断", "模型训练"],
        label_visibility="collapsed"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Model Management
    st.sidebar.markdown('<div style="margin: 30px 0;"><h3 style="font-size: 17px; font-weight: 600; color: var(--apple-near-black); margin: 0 0 16px 0;">模型管理</h3></div>', unsafe_allow_html=True)
    
    # 扫描当前目录下的模型文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_files = [f for f in os.listdir(current_dir) if f.endswith('.pth')]
    
    # 模型选择方式
    model_select_mode = st.sidebar.radio("选择模型方式", ["使用当前目录模型", "上传模型文件"], horizontal=True)
    
    # 确保selected_model总是被定义
    selected_model = "sentiment_model.pth"  # 默认值
    
    if model_select_mode == "使用当前目录模型":
        if model_files:
            selected_model = st.sidebar.selectbox("选择模型文件", model_files, index=0)
        else:
            st.sidebar.warning("未检测到模型文件，请先训练模型")
    else:
        # 允许用户上传模型文件
        uploaded_model = st.sidebar.file_uploader("上传模型文件", type=["pth"])
        if uploaded_model:
            # 保存上传的模型文件
            with open(os.path.join(current_dir, uploaded_model.name), "wb") as f:
                f.write(uploaded_model.getbuffer())
            selected_model = uploaded_model.name
            st.sidebar.success(f"模型文件 {uploaded_model.name} 上传成功！")

    domain = st.sidebar.selectbox("选择业务领域", ["电商通用", "外卖餐饮", "酒店住宿", "汽车出行"])

    if 'global_stop_words' not in st.session_state:
        st.session_state['global_stop_words'] = "酒店,宾馆,入住"

    if 'model' not in st.session_state or 'vocab' not in st.session_state:
        with st.spinner('系统内核初始化中...'):
            model, vocab, default_data_path, status = load_resources(selected_model)
            st.session_state['model'] = model
            st.session_state['vocab'] = vocab
            st.session_state['default_data_path'] = default_data_path
            st.session_state['status'] = status
    else:
        model = st.session_state['model']
        vocab = st.session_state['vocab']
        default_data_path = st.session_state['default_data_path']
        status = st.session_state['status']

    if model:
        st.sidebar.success("模型加载成功")
    else:
        st.sidebar.warning(f"{status}")

    # Advanced Settings
    with st.sidebar.expander("高级设置"):
        st.info("自定义停用词用于过滤无意义词汇，提高分析准确性。")
        stop_words_input = st.text_area("停用词 (逗号分隔)", value=st.session_state['global_stop_words'], label_visibility="collapsed")
        if st.button("保存设置"):
            st.session_state['global_stop_words'] = stop_words_input
            st.success("设置已更新。")

    # Footer
    st.sidebar.markdown('<div style="position: absolute; bottom: 30px; width: 80%;">', unsafe_allow_html=True)
    st.sidebar.markdown('---')
    st.sidebar.caption("Model Status: " + ("Online" if model else "Demo Mode"))
    st.sidebar.write(f"操作员: **{st.session_state.get('current_user', 'Admin')}**")
    if st.sidebar.button("安全退出", use_container_width=True):
        st.session_state['logged_in'] = False
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    if app_mode == "批量挖掘":
        st.title("数据批量导入与挖掘")

        # Data Source Selection
        st.markdown('<div style="margin: 30px 0;">', unsafe_allow_html=True)
        data_source = st.radio("选择数据源", ["本地上传文件", "加载系统演示数据集"], horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

        df = None
        if data_source == "本地上传文件":
            uploaded_file = st.file_uploader("请选择 CSV 格式数据文件", type=["csv"])
            if uploaded_file:
                try: df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='gbk')
        else:
            if st.button("一键加载测试样本", use_container_width=True):
                if default_data_path and os.path.exists(default_data_path):
                    st.session_state['temp_demo_df'] = pd.read_csv(default_data_path).sample(500)
                else:
                    st.error("未找到演示数据集。")
            if 'temp_demo_df' in st.session_state and data_source == "加载系统演示数据集":
                df = st.session_state['temp_demo_df']

        if df is not None:
            cols = df.columns.tolist()
            keywords = ['review', '评论', 'content', 'text', '内容']
            text_col = cols[0]
            for col in cols:
                if any(k in col.lower() for k in keywords):
                    text_col = col; break

            st.info(f"识别分析对象列：`[{text_col}]`")
            st.markdown("#### 数据预览")
            st.dataframe(df.head(5), use_container_width=True)

            if st.button("启动全量深度分析", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                with st.spinner("执行分析中..."):
                    try:
                        texts = df[text_col].astype(str).tolist()
                        df['预测结果'] = predict_sentiment(texts, model, vocab)
                        progress_bar.progress(50)
                        df['维度列表'] = df[text_col].apply(lambda x: analyze_aspect(x, domain))
                        df['涉及维度'] = df['维度列表'].apply(lambda x: ", ".join(x))
                        progress_bar.progress(100)
                        st.session_state['master_df'] = df
                        st.session_state['text_col'] = text_col
                        st.success("分析完成！请前往「监控大屏」查看。")
                    except Exception as e:
                        st.error(f"分析失败：{str(e)}")

        if 'master_df' in st.session_state:
            st.markdown("#### 数据导出")
            res_df = st.session_state['master_df']
            csv_data = res_df.drop(columns=['维度列表'], errors='ignore').to_csv(index=False).encode('utf-8-sig')
            st.download_button("导出分析结果 (CSV)", csv_data, 'output.csv', 'text/csv')

    elif app_mode == "监控大屏":
        st.title("全局数据监控与可视化")
        if 'master_df' not in st.session_state:
            st.warning("暂无数据。请先执行批量挖掘任务。")
            st.info("操作步骤：\n1. 点击左侧「批量挖掘」\n2. 上传或加载数据\n3. 启动分析\n4. 回到此页面查看结果")
        else:
            df = st.session_state['master_df']
            render_dashboard(df)

            st.markdown("#### 高频词云提取")
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

    elif app_mode == "单条诊断":
        st.title("单文本情感诊断")

        user_input = st.text_area("待测文本输入区:", height=150, key="single_text_input")

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            submit_btn = st.button("运行分析预测", type="primary", use_container_width=True)

        if submit_btn and user_input.strip():
            with st.spinner("神经网络推断中..."):
                aspects = analyze_aspect(user_input, domain)
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

            st.markdown("#### 文本分词结果")
            words = jieba.lcut(user_input)
            st.write(' '.join(words))

    elif app_mode == "模型训练":
        st.title("模型训练与评估")
        
        # Data Source
        st.markdown('<div style="margin: 30px 0;"><h2>数据来源</h2></div>', unsafe_allow_html=True)
        
        # 数据来源选择
        data_source = st.radio("选择训练数据", ["使用默认数据集", "上传自定义数据集"], horizontal=True)
        
        user_df = None
        
        if data_source == "上传自定义数据集":
            uploaded_file = st.file_uploader("请选择 CSV 格式数据文件", type=["csv"])
            if uploaded_file:
                try:
                    user_df = pd.read_csv(uploaded_file)
                    st.success("数据上传成功！")
                    st.dataframe(user_df.head(), use_container_width=True)
                    
                    # 测试数据集功能
                    with st.expander("测试数据集"):
                        st.markdown("### 数据集验证")
                        
                        if st.button("验证数据集"):
                            # 检查必要的列
                            required_columns = ['review', 'label']
                            missing_columns = [col for col in required_columns if col not in user_df.columns]
                            
                            if missing_columns:
                                st.error(f"数据集缺少必要的列：{', '.join(missing_columns)}")
                                st.info("请确保数据集包含 'review'（评论文本）和 'label'（情感标签，0表示差评，1表示好评）列")
                            else:
                                # 检查数据类型和质量
                                st.success("数据集包含所有必要的列！")
                                
                                # 显示数据集基本信息
                                st.markdown("### 数据集信息")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("总行数", len(user_df))
                                with col2:
                                    st.metric("总列数", len(user_df.columns))
                                with col3:
                                    st.metric("非空评论数", user_df['review'].notna().sum())
                                
                                # 检查标签分布
                                if 'label' in user_df.columns:
                                    label_counts = user_df['label'].value_counts()
                                    st.markdown("### 标签分布")
                                    st.bar_chart(label_counts)
                                    
                                    # 检查标签值是否合理
                                    unique_labels = user_df['label'].unique()
                                    if all(label in [0, 1] for label in unique_labels):
                                        st.success("标签值正确（0表示差评，1表示好评）")
                                    else:
                                        st.warning("标签值可能不正确，建议使用 0 表示差评，1 表示好评")
                            
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    user_df = pd.read_csv(uploaded_file, encoding='gbk')
                    st.success("数据上传成功！")
                    st.dataframe(user_df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"数据读取失败：{str(e)}")
        
        # Model Training Configuration
        st.markdown('<div style="margin: 30px 0;"><h2>模型训练配置</h2></div>', unsafe_allow_html=True)
        
        # 训练配置选项
        epochs = st.slider("训练轮次", min_value=5, max_value=50, value=10, step=5)
        batch_size = st.selectbox("批处理大小", [32, 64, 128], index=1)
        st.caption("批处理大小：每次训练时同时处理的样本数量，影响训练速度和内存使用")
        
        # Training Results Visualization
        st.markdown('<div style="margin: 30px 0;"><h2>训练结果可视化</h2></div>', unsafe_allow_html=True)
        
        if st.button("开始训练模型", type="primary", use_container_width=True):
            with st.spinner("模型训练中，请耐心等待..."):
                try:
                    # 调用训练函数
                    history, best_acc = run_training(user_df)
                    
                    st.success(f"训练完成！最佳准确率：{best_acc:.2f}%")
                    
                    # 构建训练历史数据框
                    train_df = pd.DataFrame(history)
                    
                    # 绘制损失曲线
                    fig_loss = px.line(train_df, x='epochs', y='loss', 
                                     title='训练损失曲线',
                                     color_discrete_sequence=['#0071e3'])
                    fig_loss.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_font=dict(size=16, color='#1d1d1f'),
                        xaxis=dict(title='轮次', gridcolor='#f5f5f7'),
                        yaxis=dict(title='损失值', gridcolor='#f5f5f7')
                    )
                    
                    # 绘制准确率曲线
                    fig_acc = px.line(train_df, x='epochs', y='accuracy', 
                                    title='验证准确率曲线',
                                    color_discrete_sequence=['#34c759'])
                    fig_acc.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_font=dict(size=16, color='#1d1d1f'),
                        xaxis=dict(title='轮次', gridcolor='#f5f5f7'),
                        yaxis=dict(title='准确率 (%)', gridcolor='#f5f5f7')
                    )
                    
                    # 显示图表
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_loss, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_acc, use_container_width=True)
                    
                    # 显示训练指标
                    st.markdown("### 训练指标汇总")
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    with col_metrics1:
                        st.metric("最佳准确率", f"{best_acc:.2f}%")
                    with col_metrics2:
                        st.metric("最终损失值", f"{history['loss'][-1]:.4f}")
                    with col_metrics3:
                        st.metric("训练轮次", f"{len(history['epochs'])}")
                    
                    st.info("模型已保存到 sentiment_model.pth，您可以在其他功能中使用新训练的模型。")
                    
                except Exception as e:
                    st.error(f"训练失败：{str(e)}")

if __name__ == "__main__":
    st.set_page_config(page_title="电商评论情感分析系统", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        st.markdown("<style>[data-testid='collapsedControl'] {display: none;}</style>", unsafe_allow_html=True)
        login_page()
    else:
        main()
