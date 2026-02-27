import torch
import torch.nn as nn
import jieba
import pandas as pd
import os
from collections import Counter

# 引入必要的类型提示，提升代码可读性
from typing import Tuple, Dict

# ==========================================
# 模型定义 (需保持一致)
# ==========================================
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=2):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x) 
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden[-1])

# ==========================================
# 推理引擎类
# ==========================================
class InferenceEngine:
    """处理模型加载、预处理和推理的封装类"""
    
    def __init__(self):
        self.max_len = 50
        self.embed_dim = 100
        self.hidden_dim = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab = None
        
    def initialize(self):
        """初始化资源：词典和模型权重"""
        print(">>> 正在初始化推理引擎...")
        
        # 1. 快速重建词典 (在实际工程中通常会把 vocab 保存为 json，这里为了简化依赖重新构建)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'ChnSentiCorp_htl_all.csv')
        
        try:
            df = pd.read_csv(data_path).dropna(subset=['review'])
            corpus = df['review'].astype(str).tolist()
            print(">>> 数据文件读取成功，正在构建词表...")
        except FileNotFoundError:
            print("!!! 警告：找不到原始数据文件，将使用空词表 (仅用于测试代码逻辑)")
            corpus = []

        tokens = []
        for text in corpus:
            tokens.extend(jieba.lcut(text))
            
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in Counter(tokens).most_common(5000):
            self.vocab[word] = len(self.vocab)
            
        # 2. 加载权重
        model_path = os.path.join(current_dir, 'sentiment_model.pth')
        self.model = TextClassificationModel(len(self.vocab), self.embed_dim, self.hidden_dim)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
            print(">>> 模型权重加载完毕！")
        else:
            print(f"!!! 错误：找不到模型文件 {model_path}")
            return False
            
        return True

    def predict(self, text: str) -> Tuple[int, float]:
        """对输入文本进行预测，返回 (类别, 置信度)"""
        if not text or not text.strip():
            return -1, 0.0
            
        words = jieba.lcut(text)
        ids = [self.vocab.get(w, 1) for w in words]
        
        # Padding
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [0] * (self.max_len - len(ids))
            
        tensor_in = torch.tensor([ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs).item()
            conf = probs[0][pred].item()
            
        return pred, conf

# ==========================================
# 交互入口
# ==========================================
if __name__ == "__main__":
    engine = InferenceEngine()
    if engine.initialize():
        print("\n" + "="*40)
        print("   电商评论情感分析控制台 (CLI Demo)")
        print("   输入 'q' 或 'exit' 退出")
        print("="*40 + "\n")
        
        while True:
            try:
                user_input = input("User> ")
                if user_input.lower() in ['q', 'exit']:
                    print("Bye!")
                    break
                
                label_idx, confidence = engine.predict(user_input)
                
                if label_idx == -1:
                    continue
                    
                result_str = "好评 (Positive)" if label_idx == 1 else "差评 (Negative)"
                print(f"AI Output> {result_str} | 置信度: {confidence:.2%}\n")
                
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break