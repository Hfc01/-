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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    VOCAB_SIZE = 5000        # 高频词保留数量，实际词表大小 = VOCAB_SIZE + 2（含<PAD>和<UNK>）
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 文件路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'ChnSentiCorp_htl_all.csv')
    SAVE_PATH = os.path.join(BASE_DIR, 'sentiment_model.pth')  # 基于BASE_DIR的绝对路径

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
def load_and_vectorize(df=None):
    """
    读取数据并转换为 Tensor。
    返回 X, y, vocab (词表字典，非词表大小)。
    """
    if df is None:
        print(f"[Info] Loading data from {Config.DATA_PATH}...")

        if not os.path.exists(Config.DATA_PATH):
            raise FileNotFoundError("数据文件未找到，请检查路径设置！")

        # 读取 CSV，处理可能的空值
        try:
            df = pd.read_csv(Config.DATA_PATH).dropna(subset=['review'])
        except UnicodeDecodeError:
            df = pd.read_csv(Config.DATA_PATH, encoding='gbk').dropna(subset=['review'])

    else:
        print("[Info] Using user-uploaded dataset...")
        df = df.dropna(subset=['review'])

    texts = df['review'].astype(str).tolist()
    labels = df['label'].tolist()

    # 1. 构建词表 (Tokenization & Vocab Building)
    print("[Info] Building vocabulary...")
    all_tokens = []
    for t in texts:
        all_tokens.extend(jieba.lcut(t))

    # <PAD>=0 用于序列填充，<UNK>=1 用于未登录词
    # VOCAB_SIZE=5000 表示高频词保留数量，实际词表大小为 5002
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in Counter(all_tokens).most_common(Config.VOCAB_SIZE):
        vocab[word] = len(vocab)

    # 2. 序列数字化 (Vectorization)
    print("[Info] Converting text to sequences...")
    input_ids = []
    for t in texts:
        words = jieba.lcut(t)
        ids = [vocab.get(w, 1) for w in words]  # 未登录词映射为<UNK>=1

        # Padding / Truncating
        if len(ids) > Config.MAX_LEN:
            ids = ids[:Config.MAX_LEN]
        else:
            ids += [0] * (Config.MAX_LEN - len(ids))
        input_ids.append(ids)

    # 转换为 PyTorch Tensor
    X = torch.tensor(input_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y, vocab  # 返回完整词表字典而非len(vocab)

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
def run_training(df=None, epochs=None, batch_size=None):
    """训练模型并返回 history, best_acc, test_metrics"""
    seed_everything(Config.SEED)

    # 使用传入参数或默认配置
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE

    # 1. 准备数据：load_and_vectorize 现在返回 X, y, vocab
    X, y, vocab = load_and_vectorize(df)

    # 三层数据集划分：训练集70%、验证集15%、测试集15%
    # 先按8:2划分train和temp，再将temp平分为val和test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=Config.SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=Config.SEED, stratify=y_temp
    )
    print(f"[Info] Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)

    # 2. 初始化模型
    print(f"[Info] Initializing model on {Config.DEVICE}...")
    model = TextClassificationModel(len(vocab), Config.EMBED_DIM, Config.HIDDEN_DIM).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    # 3. 循环训练
    print(f"[Info] Start training for {num_epochs} epochs...")
    best_acc = 0.0

    # 记录训练历史（保留 loss/accuracy/epochs 字段，兼容 web_demo.py 原有训练曲线）
    history = {
        'loss': [],
        'accuracy': [],
        'epochs': []
    }

    for epoch in range(num_epochs):
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

        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        time_elapsed = time.time() - start_time

        # 记录历史数据
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        history['epochs'].append(epoch + 1)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Time: {time_elapsed:.1f}s | "
              f"Loss: {avg_loss:.4f} | "
              f"Val Acc: {acc:.2f}%")

        # 保存最佳 checkpoint（包含完整信息，不仅 state_dict）
        if acc > best_acc:
            best_acc = acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "config": {
                    "max_len": Config.MAX_LEN,
                    "embed_dim": Config.EMBED_DIM,
                    "hidden_dim": Config.HIDDEN_DIM,
                    "output_dim": 2,
                    "vocab_size": len(vocab)
                },
                "best_val_acc": best_acc
            }
            torch.save(checkpoint, Config.SAVE_PATH)

    print(f"[Done] Training finished. Best Val Accuracy: {best_acc:.2f}%")

    # 4. 加载最佳 checkpoint，在独立测试集上评估
    print("[Info] Loading best checkpoint for test evaluation...")
    best_checkpoint = torch.load(Config.SAVE_PATH, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(Config.DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(Config.DEVICE), test_y.to(Config.DEVICE)
            outputs = model(test_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(test_y.cpu().tolist())

    # 计算测试集指标（label=1即"好评"作为正类）
    test_acc = accuracy_score(all_labels, all_preds) * 100
    test_precision = precision_score(all_labels, all_preds, average='binary') * 100
    test_recall = recall_score(all_labels, all_preds, average='binary') * 100
    test_f1 = f1_score(all_labels, all_preds, average='binary') * 100
    test_cm = confusion_matrix(all_labels, all_preds)

    test_metrics = {
        "accuracy": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1,
        "confusion_matrix": test_cm.tolist()
    }

    print(f"[Test Results] Acc={test_acc:.2f}% | Prec={test_precision:.2f}% | "
          f"Recall={test_recall:.2f}% | F1={test_f1:.2f}%")
    print(f"[Confusion Matrix] TN={test_cm[0][0]}, FP={test_cm[0][1]}, "
          f"FN={test_cm[1][0]}, TP={test_cm[1][1]}")

    # 5. 将 test_metrics 写入 checkpoint 后重新保存
    best_checkpoint["test_metrics"] = test_metrics
    torch.save(best_checkpoint, Config.SAVE_PATH)
    print(f"[Info] Model saved to {Config.SAVE_PATH} with full checkpoint")

    return history, best_acc, test_metrics

if __name__ == "__main__":
    run_training()
