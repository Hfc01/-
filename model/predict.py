import torch
import torch.nn as nn
import jieba
import pandas as pd
import os
from collections import Counter

# ==========================================
# 1. 必须保持和训练时完全一致的配置
# ==========================================
MAX_LEN = 50
EMBEDDING_DIM = 100
HIDDEN_DIM = 128

# ==========================================
# 2. 定义模型 (必须和训练代码里的长得一模一样)
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
# 3. 核心功能：加载模型并预测
# ==========================================
def predict_sentiment():
    print("⏳ 正在初始化系统，请稍候...")
    
    # --- 第一步：重建词典 (为了保证和训练时对应的数字一样) ---
    # 这里我们快速重读一遍数据来生成词典
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'ChnSentiCorp_htl_all.csv')
    df = pd.read_csv(data_path).dropna(subset=['review'])
    texts = df['review'].astype(str).tolist()
    
    all_words = []
    for text in texts:
        all_words.extend(jieba.lcut(text))
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in Counter(all_words).most_common(5000):
        vocab[word] = len(vocab)
    print("✅ 词典加载完毕！")

    # --- 第二步：加载模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)
    
    # 加载训练好的参数
    model_path = os.path.join(current_dir, 'sentiment_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 成功加载训练好的模型！")
    else:
        print("❌ 错误：找不到模型文件 sentiment_model.pth")
        return

    model.eval() # 开启评估模式

    # --- 第三步：循环让用户输入 ---
    print("\n" + "="*40)
    print("🤖 情感分析机器人已就绪！")
    print("输入评论后回车，输入 'q' 退出")
    print("="*40)

    while True:
        text = input("\n请输入测试评论: ")
        if text.lower() == 'q':
            break
        
        if not text.strip():
            continue

        # 预处理输入
        words = jieba.lcut(text)
        ids = [vocab.get(w, 1) for w in words]
        
        # 填充/截断
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        else:
            ids = ids + [0] * (MAX_LEN - len(ids))
            
        # 转为 Tensor 并预测
        tensor_input = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            output = model(tensor_input)
            probability = torch.nn.functional.softmax(output, dim=1)
            # 获取预测结果 (0是差评, 1是好评)
            pred_class = torch.argmax(probability).item()
            confidence = probability[0][pred_class].item()

        # 打印结果
        if pred_class == 1:
            print(f"👉 预测结果：【好评 😊】 (置信度: {confidence:.2%})")
        else:
            print(f"👉 预测结果：【差评 😡】 (置信度: {confidence:.2%})")

if __name__ == "__main__":
    predict_sentiment()