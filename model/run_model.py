import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import jieba
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time

# ==========================================
# 🛠️ 配置参数 (你可以修改这里)
# ==========================================
MAX_LEN = 50          # 句子的最大长度
BATCH_SIZE = 64       # 每次喂给模型多少条数据
EMBEDDING_DIM = 100   # 每个词用多少维的向量表示
HIDDEN_DIM = 128      # 神经网络隐藏层神经元数量
EPOCHS = 10           # 训练多少轮 (建议 5-10 轮)
LEARNING_RATE = 0.001 # 学习率

# ==========================================
# 1. 数据读取与处理 (保持不变)
# ==========================================
def load_and_process_data():
    # --- 定位文件 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'ChnSentiCorp_htl_all.csv')
    
    print(f"📂 正在读取数据：{data_path}")
    if not os.path.exists(data_path):
        print("❌ 错误：找不到数据文件！")
        return None, None, None

    # --- 读取清洗 ---
    df = pd.read_csv(data_path).dropna(subset=['review'])
    texts = df['review'].astype(str).tolist()
    labels = df['label'].tolist()
    print(f"✅ 读取成功！共 {len(texts)} 条数据")

    # --- 构建词典 ---
    print("🔨 正在构建词典 (只保留最常见的5000词)...")
    all_words = []
    for text in texts:
        all_words.extend(jieba.lcut(text))
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in Counter(all_words).most_common(5000):
        vocab[word] = len(vocab)
    
    # --- 数字化 ---
    print("🔢 正在将文本转为数字...")
    input_ids = []
    for text in texts:
        words = jieba.lcut(text)
        ids = [vocab.get(w, 1) for w in words]
        # 填充或截断
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        else:
            ids = ids + [0] * (MAX_LEN - len(ids))
        input_ids.append(ids)
    
    # --- 转为 Tensor ---
    X = torch.tensor(input_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    
    return X, y, len(vocab)

# ==========================================
# 🧠 2. 定义神经网络模型 (LSTM)
# ==========================================
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=2):
        super(SentimentLSTM, self).__init__()
        # 1. 嵌入层：把数字变成向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 2. LSTM层：提取语义特征
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # 3. 全连接层：分类 (好评/差评)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text形状: [batch_size, max_len]
        embedded = self.embedding(text) 
        # embedded形状: [batch_size, max_len, embed_dim]
        
        # LSTM 输出
        output, (hidden, cell) = self.lstm(embedded)
        # 我们只取最后一步的隐藏状态作为句子的代表
        final_hidden = hidden[-1] 
        
        # 分类
        return self.fc(final_hidden)

# ==========================================
#  3. 训练与评估函数
# ==========================================
def train_model():
    # 1. 准备数据
    X, y, vocab_size = load_and_process_data()
    if X is None: return

    # 拆分训练集和测试集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 包装成 DataLoader (方便批量训练)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    # 2. 初始化模型
    print(f"\n🧠 初始化模型 (词表大小: {vocab_size})...")
    model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss() # 分类任务标准损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. 开始训练循环
    print("🚀 开始训练... (请耐心等待，每轮大概几秒钟)")
    print("-" * 50)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train() # 开启训练模式
        total_loss = 0
        correct = 0
        total = 0
        
        for texts, labels in train_loader:
            optimizer.zero_grad()           # 清空梯度
            predictions = model(texts)      # 前向传播 (预测)
            loss = criterion(predictions, labels) # 计算误差
            loss.backward()                 # 反向传播 (求导)
            optimizer.step()                # 更新参数
            
            total_loss += loss.item()
            # 计算准确率
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # --- 每轮结束后测试一下 ---
        model.eval() # 开启评估模式
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for texts, labels in test_loader:
                outputs = model(texts)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"耗时: {time.time()-start_time:.1f}s | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"训练准确率: {train_acc:.2f}% | "
              f"测试准确率: {test_acc:.2f}%")

    print("-" * 50)
    print("🎉 训练结束！模型已经学会了区分好评和差评！")
    
    # 保存模型 (毕设需要)
    torch.save(model.state_dict(), 'sentiment_model.pth')
    print("💾 模型参数已保存为 sentiment_model.pth")

if __name__ == "__main__":
    train_model()