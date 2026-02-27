import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import time
import random
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ==========================================
# 实验配置 (Experiment Configuration)
# ==========================================
class Config:
    """集中管理超参数，方便实验调整"""
    SEED = 42                # 随机种子，保证结果可复现
    MAX_LEN = 50             # 文本截断/填充长度
    BATCH_SIZE = 64          # 批大小
    EMBED_DIM = 100          # 词向量维度
    HIDDEN_DIM = 128         # LSTM 隐藏层维度
    EPOCHS = 10              # 训练轮次
    LR = 0.001               # 学习率
    VOCAB_SIZE = 5000        # 词表容量
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 文件路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'ChnSentiCorp_htl_all.csv')
    SAVE_PATH = 'sentiment_model.pth'

def seed_everything(seed):
    """锁定所有随机种子，确保毕设实验的可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# 数据处理流水线 (Data Pipeline)
# ==========================================
def load_and_vectorize():
    """读取数据并转换为 Tensor"""
    print(f"[Info] Loading data from {Config.DATA_PATH}...")
    
    if not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError("数据文件未找到，请检查路径设置！")

    # 读取 CSV，处理可能的空值
    try:
        df = pd.read_csv(Config.DATA_PATH).dropna(subset=['review'])
    except UnicodeDecodeError:
        df = pd.read_csv(Config.DATA_PATH, encoding='gbk').dropna(subset=['review'])
        
    texts = df['review'].astype(str).tolist()
    labels = df['label'].tolist()
    
    # 1. 构建词表 (Tokenization & Vocab Building)
    print("[Info] Building vocabulary...")
    all_tokens = []
    for t in texts:
        all_tokens.extend(jieba.lcut(t))
    
    # 保留高频词，其余设为 UNK
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in Counter(all_tokens).most_common(Config.VOCAB_SIZE):
        vocab[word] = len(vocab)
        
    # 2. 序列数字化 (Vectorization)
    print("[Info] Converting text to sequences...")
    input_ids = []
    for t in texts:
        words = jieba.lcut(t)
        ids = [vocab.get(w, 1) for w in words]
        
        # Padding / Truncating
        if len(ids) > Config.MAX_LEN:
            ids = ids[:Config.MAX_LEN]
        else:
            ids += [0] * (Config.MAX_LEN - len(ids))
        input_ids.append(ids)
        
    # 转换为 PyTorch Tensor
    X = torch.tensor(input_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    
    return X, y, len(vocab)

# ==========================================
# 模型架构 (Model Architecture)
# ==========================================
class TextClassificationModel(nn.Module):
    """
    标准的 Embedding + LSTM + FC 结构。
    适用于短文本情感分类任务。
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=2):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # batch_first=True 使得输入维度为 (batch, seq, feature)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x) 
        output, (hidden, cell) = self.lstm(embedded)
        # 取最后一个时间步的输出作为句子的特征表示
        return self.fc(hidden[-1])

# ==========================================
# 训练主流程 (Main Loop)
# ==========================================
def run_training():
    seed_everything(Config.SEED)
    
    # 1. 准备数据
    X, y, vocab_size = load_and_vectorize()
    # 划分训练集和验证集 (8:2)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=Config.SEED)
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE)
    
    # 2. 初始化模型
    print(f"[Info] Initializing model on {Config.DEVICE}...")
    model = TextClassificationModel(vocab_size, Config.EMBED_DIM, Config.HIDDEN_DIM).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    # 3. 循环训练
    print(f"[Info] Start training for {Config.EPOCHS} epochs...")
    best_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 验证集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(Config.DEVICE), val_y.to(Config.DEVICE)
                outputs = model(val_x)
                _, predicted = torch.max(outputs.data, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()
        
        acc = 100 * correct / total
        time_elapsed = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] | "
              f"Time: {time_elapsed:.1f}s | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {acc:.2f}%")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), Config.SAVE_PATH)
            
    print(f"[Done] Training finished. Best Accuracy: {best_acc:.2f}%")
    print(f"[Info] Model saved to {Config.SAVE_PATH}")

if __name__ == "__main__":
    run_training()