import sys
import os

# ==========================================
# 👇 核心修复代码：强行把当前目录加入系统路径
# ==========================================
# 1. 获取 data_process.py 所在的文件夹路径 (即 model 文件夹)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 把这个路径告诉 Python
sys.path.append(current_dir)

import torch
import jieba
from collections import Counter
# 引用你刚才写的 dataset_loader
from dataset_loader import load_data

# === 基础配置 ===
MAX_LEN = 50       # 规定每句话最长处理多少个词（酒店评论通常不长，50够了）
BATCH_SIZE = 64    # 一次训练多少条数据

# === 1. 构建词典 (Vocabulary) ===
class Vocab:
    def __init__(self, texts):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1} # 0是填充位，1是未知词
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
        print("正在构建词典，请稍候...")
        all_words = []
        for text in texts:
            # 使用结巴分词
            words = jieba.lcut(text)
            all_words.extend(words)
        
        # 统计词频，只保留出现次数最多的前 10000 个词 (减少噪音)
        counter = Counter(all_words)
        common_words = counter.most_common(10000)
        
        for word, _ in common_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"✅ 词典构建完成！词表大小: {len(self.word2idx)}")

    def text_to_ids(self, text):
        # 把一句话变成数字列表
        words = jieba.lcut(text)
        ids = [self.word2idx.get(w, 1) for w in words] # 找不到的词就用1(<UNK>)代替
        
        # 统一长度处理 (Padding / Truncating)
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN] # 截断
        else:
            ids = ids + [0] * (MAX_LEN - len(ids)) # 补0
            
        return ids

# === 2. 核心处理函数 (给后续训练调用) ===
def get_processed_data():
    # 1. 加载原始数据
    texts, labels = load_data()
    
    if texts is None:
        return None, None, None

    # 2. 构建词典
    vocab = Vocab(texts)
    
    # 3. 把所有文本转成数字
    print("正在把文本转化为数字序列...")
    input_ids = []
    for text in texts:
        ids = vocab.text_to_ids(text)
        input_ids.append(ids)
    
    # 4. 转成 PyTorch 需要的 Tensor 格式
    X = torch.tensor(input_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    
    print(f"✅ 数据处理完毕！输入形状: {X.shape}, 标签形状: {y.shape}")
    return X, y, vocab

if __name__ == "__main__":
    # 测试代码
    get_processed_data()