"""
电商评论情感分析系统 (Flask + Tailwind CSS)
基于 UI-UX-Pro-Max 设计规范重构
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import base64
import hashlib
import io
import json
import logging
import random
import sqlite3
from collections import Counter

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import jieba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from wordcloud import WordCloud

from run_model import run_training
from saa_engine import get_saa, SAAEngine, ASPECT_DICT

app = Flask(__name__)
app.secret_key = 'sentimentai_secret_key_2026'

# ==========================================
# 配置
# ==========================================
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MAX_LEN = 50
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
MAX_SAA_SAMPLES = 300
MAX_BATCH_SAMPLES = 2000
DEMO_SAMPLE_SIZE = 500

# ==========================================
# 模型定义
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
        return self.fc(hidden[-1])

# ==========================================
# 认证模块
# ==========================================
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def create_usertable():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT PRIMARY KEY, password TEXT)')
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username = ?', (username,))
    if c.fetchone() is not None:
        conn.close()
        return False
    c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', (username, password))
    conn.commit()
    conn.close()
    return True

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

# ==========================================
# 模型加载
# ==========================================
def load_model(model_filename='sentiment_model.pth'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '..', 'data', 'ChnSentiCorp_htl_all.csv')
    if not os.path.exists(data_file):
        data_file = os.path.join(current_dir, '..', 'ChnSentiCorp_htl_all.csv')
    model_path = os.path.join(current_dir, model_filename)

    if not os.path.exists(model_path):
        return None, {"<PAD>": 0, "<UNK>": 1}, data_file

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        vocab_size = cfg.get("vocab_size", len(checkpoint["vocab"]))
        embed_dim = cfg.get("embed_dim", EMBEDDING_DIM)
        hidden_dim = cfg.get("hidden_dim", HIDDEN_DIM)
        output_dim = cfg.get("output_dim", 2)
        vocab = checkpoint["vocab"]

        model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, output_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, vocab, data_file
    except Exception as e:
        return None, {"<PAD>": 0, "<UNK>": 1}, data_file

# 全局模型缓存
_model_cache = {}

def get_model(model_filename='sentiment_model.pth'):
    if model_filename not in _model_cache:
        _model_cache[model_filename] = load_model(model_filename)
    return _model_cache[model_filename]

# ==========================================
# 预测函数
# ==========================================
def rule_based_predict(text):
    positive_words = ["好", "满意", "喜欢", "推荐", "不错", "优秀", "划算", "值得", "干净", "方便"]
    negative_words = ["差", "失望", "垃圾", "难用", "不推荐", "糟糕", "脏", "慢", "贵", "后悔"]
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    return "好评" if pos_count >= neg_count else "差评"

def predict_sentiment(texts, model, vocab):
    if not model:
        labels = [rule_based_predict(str(t)) for t in texts]
        confidences = [0.60] * len(texts)
        return labels, confidences

    input_ids = []
    for text in texts:
        words = jieba.lcut(str(text))
        ids = [vocab.get(w, 1) for w in words]
        ids = ids[:MAX_LEN] if len(ids) > MAX_LEN else ids + [0] * (MAX_LEN - len(ids))
        input_ids.append(ids)

    tensor_input = torch.tensor(input_ids, dtype=torch.long)
    device = next(model.parameters()).device
    tensor_input = tensor_input.to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.nn.functional.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)

    labels = ["好评" if p == 1 else "差评" for p in preds.cpu().tolist()]
    confidences = [round(c, 4) for c in max_probs.cpu().tolist()]
    return labels, confidences

def analyze_aspect(text, domain="电商通用"):
    domain_aspects = {
        "外卖餐饮": {
            "餐饮口感与分量": ["好吃", "难吃", "口味", "味道", "咸", "淡", "辣", "新鲜", "变质", "馊", "分量", "太少", "吃不饱", "口感", "肉", "菜", "饭", "冷", "凉", "热乎"],
            "外卖配送与骑手": ["骑手", "外卖员", "小哥", "送餐", "超时", "送错", "撒", "洒", "漏", "提前", "准时", "送上楼"]
        },
        "酒店住宿": {
            "酒店设施与卫生": ["卫生", "干净", "脏", "乱", "臭", "异味", "床", "马桶", "热水", "花洒", "空调", "设施", "硬件", "旧", "破", "被子", "毛巾"],
            "住宿位置与环境": ["位置", "交通", "地铁", "公交", "隔音", "吵", "安静", "周边", "商场", "便利店", "偏僻", "市中心", "风景"]
        },
        "电商通用": {
            "商品质量与做工": ["质量", "正品", "材质", "做工", "坏", "好用", "差", "破损", "假", "瑕疵", "掉色", "粗糙", "精致", "结实", "耐用"],
            "物流发货与包装": ["物流", "快递", "包装", "速度", "发货", "慢", "快", "顺丰", "驿站", "签收"],
            "客户服务与售后": ["客服", "态度", "退换", "售后", "回复", "解决", "热情", "冷淡", "维权", "投诉"],
            "价格与性价比": ["价格", "贵", "便宜", "性价比", "划算", "值", "降价", "优惠", "折扣", "活动"],
            "外观颜值与设计": ["外观", "颜值", "漂亮", "难看", "丑", "色差", "款式", "设计", "好看", "颜色"]
        }
    }
    text = str(text)
    current_aspects = domain_aspects.get(domain, domain_aspects["电商通用"])
    detected = [k for k, v in current_aspects.items() if any(kw in text for kw in v)]
    return detected if detected else ["综合整体评价"]

def _get_font_path():
    """跨平台检测中文字体路径"""
    for p in [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]:
        if os.path.exists(p):
            return p
    return None


def generate_wordcloud_base64(text_list, stop_words=""):
    if not text_list:
        return None
    base_stop = {"的", "是", "了", "在", "我", "我们", "你", "有", "和", "就", "不", "人", "都",
                 "一个", "上", "也", "很", "到", "说", "去", "会", "着", "没有", "但是", "因为",
                 "还是", "这", "那", "个", "住", "对", "让", "给", "把", "被", "跟", "与", "为",
                 "等", "感觉", "觉得", "酒店", "宾馆", "入住"}
    if stop_words:
        base_stop.update(set(stop_words.replace("，", ",").split(",")))

    full_text = " ".join([str(t) for t in text_list])
    words = jieba.lcut(full_text)
    clean_words = [w for w in words if w not in base_stop and len(w) > 1]
    if not clean_words:
        return None
    cut_text = " ".join(clean_words)

    font_path = _get_font_path()
    if font_path is None:
        return None

    wc = WordCloud(font_path=font_path, background_color='white', width=600, height=300,
                   max_words=80, font_step=2, collocations=False,
                   color_func=lambda *args, **kwargs: (37, 99, 235)).generate(cut_text)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

# ==========================================
# 路由
# ==========================================

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return redirect(url_for('batch'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    create_usertable()
    error = None
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        if action == 'login':
            hashed = make_hashes(password)
            result = login_user(username, hashed)
            if result:
                session['logged_in'] = True
                session['current_user'] = username
                return redirect(url_for('batch'))
            else:
                error = "用户名或密码错误"
        elif action == 'register':
            if username and password:
                success = add_userdata(username, make_hashes(password))
                if success:
                    error = "注册成功，请登录"
                else:
                    error = "用户名已存在"
            else:
                error = "请填写完整信息"
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---- 页面路由 ----
@app.route('/batch')
def batch():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('batch.html', user=session.get('current_user'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session.get('current_user'))

@app.route('/diagnose')
def diagnose():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('diagnose.html', user=session.get('current_user'))

@app.route('/train')
def train_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('train.html', user=session.get('current_user'))

# ---- 工具函数 ----
def _detect_text_col(cols):
    """自动检测 DataFrame 中的文本列"""
    text_keywords = ['review', '评论', 'content', 'text', '内容']
    for col in cols:
        if any(k in col.lower() for k in text_keywords):
            return col
    non_label = [c for c in cols if 'label' not in c.lower() and '标签' not in c and '情感' not in c]
    return non_label[0] if non_label else cols[0]


def _run_full_analysis(df, domain):
    """完整的分析流水线：预测 + 词云 + SAA → 统一结果 dict"""
    cols = df.columns.tolist()
    text_col = _detect_text_col(cols)
    model, vocab, _ = get_model()
    texts = df[text_col].astype(str).tolist()
    labels, confs = predict_sentiment(texts, model, vocab)
    aspects = [analyze_aspect(t, domain) for t in texts]

    total = len(labels)
    pos_count = labels.count('好评')
    neg_count = labels.count('差评')

    all_aspects = [a for sub in aspects for a in sub]
    aspect_counter = Counter(all_aspects)

    cross_data = {}
    for asps, label in zip(aspects, labels):
        for a in asps:
            cross_data.setdefault(a, {'好评': 0, '差评': 0})
            cross_data[a][label] += 1

    pos_texts = [t for t, l in zip(texts, labels) if l == '好评']
    neg_texts = [t for t, l in zip(texts, labels) if l == '差评']
    pos_wc = generate_wordcloud_base64(pos_texts)
    neg_wc = generate_wordcloud_base64(neg_texts)

    # SAA 情感归因分析（含模型融合）
    try:
        # 过滤时保留原始索引，避免 labels/confs 张冠李戴
        filtered = [(i, t) for i, t in enumerate(texts) if len(str(t).strip()) >= 5]
        if len(filtered) > MAX_SAA_SAMPLES:
            random.seed(42)
            sampled = sorted(random.sample(filtered, MAX_SAA_SAMPLES))
            texts_for_saa = [t for _, t in sampled]
            model_preds = [(labels[i], confs[i]) for i, _ in sampled]
        else:
            texts_for_saa = [t for _, t in filtered]
            model_preds = [(labels[i], confs[i]) for i, _ in filtered]
        saa_report = get_saa().analyze(texts_for_saa, model_preds=model_preds)
    except Exception as e:
        app.logger.exception("SAA 分析失败")
        saa_report = {'error': str(e)}

    return {
        'total': total,
        'pos_count': pos_count,
        'neg_count': neg_count,
        'pos_rate': round(pos_count / total * 100, 1) if total else 0,
        'text_col': text_col,
        'aspects': [{'name': k, 'count': v} for k, v in aspect_counter.most_common(10)],
        'cross_data': cross_data,
        'pos_wordcloud': pos_wc,
        'neg_wordcloud': neg_wc,
        'saa_report': saa_report,
        'data': df.to_dict('records'),
        'columns': cols,
        'labels': labels,
        'confidences': confs,
        'detail_aspects': [', '.join(a) for a in aspects],
    }


# ---- API 路由 ----
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if not session.get('logged_in'):
        return jsonify({'error': '未登录'}), 401

    file = request.files.get('file')
    domain = request.form.get('domain', '电商通用')

    if not file:
        return jsonify({'error': '请上传文件'}), 400

    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding='gbk')
    except Exception as e:
        return jsonify({'error': f'文件读取失败: {str(e)}'}), 400

    # 限制批量大小
    if len(df) > MAX_BATCH_SAMPLES:
        df = df.sample(MAX_BATCH_SAMPLES, random_state=42).reset_index(drop=True)

    return jsonify(_run_full_analysis(df, domain))

@app.route('/api/diagnose', methods=['POST'])
def api_diagnose():
    if not session.get('logged_in'):
        return jsonify({'error': '未登录'}), 401

    text = request.json.get('text', '')
    domain = request.json.get('domain', '电商通用')

    if not text.strip():
        return jsonify({'error': '请输入文本'}), 400

    model, vocab, _ = get_model()
    words = jieba.lcut(text)
    aspects = analyze_aspect(text, domain)
    labels, confs = predict_sentiment([text], model, vocab)

    return jsonify({
        'label': labels[0],
        'confidence': confs[0],
        'aspects': aspects,
        'words': ' '.join(words),
        'word_count': len(words)
    })

@app.route('/api/load_demo', methods=['POST'])
def api_load_demo():
    """加载系统预设演示数据集"""
    if not session.get('logged_in'):
        return jsonify({'error': '未登录'}), 401

    current_dir = os.path.dirname(os.path.abspath(__file__))
    demo_paths = [
        os.path.join(current_dir, '..', 'data', 'ChnSentiCorp_htl_all.csv'),
        os.path.join(current_dir, '..', 'ChnSentiCorp_htl_all.csv'),
        os.path.join(current_dir, 'ChnSentiCorp_htl_all.csv'),
    ]

    demo_file = None
    for p in demo_paths:
        if os.path.exists(p):
            demo_file = p
            break

    if not demo_file:
        return jsonify({'error': '未找到预设数据集文件 (ChnSentiCorp_htl_all.csv)'}), 404

    try:
        df = pd.read_csv(demo_file)
    except Exception as e:
        return jsonify({'error': f'数据集读取失败: {str(e)}'}), 500

    # 采样作为演示
    if len(df) > DEMO_SAMPLE_SIZE:
        df = df.sample(DEMO_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    # 找文本列
    cols = df.columns.tolist()
    text_keywords = ['review', '评论', 'content', 'text', '内容']
    text_col = None
    for col in cols:
        if any(k in col.lower() for k in text_keywords):
            text_col = col
            break
    if text_col is None:
        non_label = [c for c in cols if 'label' not in c.lower() and '标签' not in c and '情感' not in c]
        text_col = non_label[0] if non_label else cols[0]

    # 预览数据（前5行）
    preview = df.head(5).to_dict('records')
    preview_cols = [c for c in cols if 'label' not in c.lower()]

    return jsonify({
        'success': True,
        'total_rows': len(df),
        'text_col': text_col,
        'columns': cols,
        'preview_cols': preview_cols,
        'preview': preview,
        'message': f'已加载预设数据集，共 {len(df)} 条评论（从 {os.path.basename(demo_file)} 随机采样）'
    })

@app.route('/api/analyze_demo', methods=['POST'])
def api_analyze_demo():
    """分析预设演示数据集"""
    if not session.get('logged_in'):
        return jsonify({'error': '未登录'}), 401

    domain = request.form.get('domain', '电商通用')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    demo_paths = [
        os.path.join(current_dir, '..', 'data', 'ChnSentiCorp_htl_all.csv'),
        os.path.join(current_dir, '..', 'ChnSentiCorp_htl_all.csv'),
        os.path.join(current_dir, 'ChnSentiCorp_htl_all.csv'),
    ]

    demo_file = None
    for p in demo_paths:
        if os.path.exists(p):
            demo_file = p
            break

    if not demo_file:
        return jsonify({'error': '未找到预设数据集文件'}), 404

    try:
        df = pd.read_csv(demo_file)
    except Exception as e:
        return jsonify({'error': f'数据集读取失败: {str(e)}'}), 500

    # 采样
    if len(df) > DEMO_SAMPLE_SIZE:
        df = df.sample(DEMO_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    return jsonify(_run_full_analysis(df, domain))

@app.route('/api/model_status')
def api_model_status():
    model, vocab, data_file = get_model()
    model_files = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__)))
                   if f.endswith('.pth')]
    # 尝试读取模型元数据（训练时保存）
    meta_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_meta.json')
    accuracy = None
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                accuracy = meta.get('best_val_acc')
        except Exception as e:
            app.logger.warning(f"读取 model_meta.json 失败: {e}")
    return jsonify({
        'model_loaded': model is not None,
        'model_files': model_files,
        'data_file_exists': os.path.exists(data_file) if data_file else False,
        'accuracy': accuracy
    })

@app.route('/api/train', methods=['POST'])
def api_train():
    if not session.get('logged_in'):
        return jsonify({'error': '未登录'}), 401

    epochs = int(request.form.get('epochs', 10))
    batch_size = int(request.form.get('batch_size', 64))
    file = request.files.get('file')

    user_df = None
    if file:
        try:
            user_df = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            user_df = pd.read_csv(file, encoding='gbk')

    try:
        history, best_acc, test_metrics = run_training(user_df, epochs=epochs, batch_size=batch_size)
        # 刷新模型缓存
        _model_cache.clear()
        return jsonify({
            'success': True,
            'best_acc': best_acc,
            'history': history,
            'test_metrics': test_metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# 情感归因分析 (SAA) 路由
# ==========================================

@app.route('/api/saa/diagnose', methods=['POST'])
def api_saa_diagnose():
    """单条文本的情感归因诊断"""
    if not session.get('logged_in'):
        return jsonify({'error': '未登录'}), 401

    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': '请输入文本'}), 400

        engine = get_saa()
        result = engine.analyze_single(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'SAA分析失败: {str(e)}'}), 500


@app.route('/api/saa/analyze', methods=['POST'])
def api_saa_analyze():
    """批量文本的情感归因分析"""
    if not session.get('logged_in'):
        return jsonify({'error': '未登录'}), 401

    try:
        # 支持上传文件和直接提交文本
        texts = []

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '未选择文件'}), 400

            ext = os.path.splitext(file.filename)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(file)
            elif ext in ('.xlsx', '.xls'):
                df = pd.read_excel(file)
            elif ext == '.txt':
                texts = [line.strip() for line in file.read().decode('utf-8').split('\n') if line.strip()]
            else:
                return jsonify({'error': f'不支持的文件格式: {ext}'}), 400

            if not texts:
                # 自动检测文本列
                for col in df.columns:
                    if any(k in str(col).lower() for k in ['评论', 'review', 'text', '内容', 'comment']):
                        texts = [str(t) for t in df[col].dropna().tolist() if str(t).strip()]
                        break

        elif request.is_json:
            data = request.get_json()
            if 'texts' in data:
                texts = data['texts']
            elif 'text' in data:
                texts = [data['text']]

        if not texts:
            return jsonify({'error': '未提取到有效文本'}), 400

        # 限制分析数量
        if len(texts) > MAX_BATCH_SAMPLES:
            texts = random.sample(texts, MAX_BATCH_SAMPLES)

        engine = get_saa()
        report = engine.analyze(texts)

        # 转换关键词格式（前端友好）
        report['keywords'] = [{'word': w, 'score': s} for w, s in report['keywords']]

        return jsonify(report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'SAA分析失败: {str(e)}'}), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    create_usertable()
    get_model()
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='127.0.0.1', port=5000)
