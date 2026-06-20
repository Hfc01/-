"""
SAA - 情感归因分析引擎 (Sentiment Attribution Algorithm)
三层架构：
  第一层: jieba + TF-IDF 关键词提取（词性过滤）
  第二层: jieba.posseg + 规则引擎（属性-情感对抽取）
  第三层: 情感归因报告生成

Author: 电商评论情感分析
"""

import os
import re
import json
import math
from collections import Counter, defaultdict

import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# 常量配置
# ============================================================

# 电商领域维度词典（属性词 → 所属维度）
ASPECT_DICT = {
    # 服务维度
    '服务': '服务体验', '客服': '服务体验', '态度': '服务体验', '售后': '服务体验',
    '回复': '服务体验', '投诉': '服务体验', '沟通': '服务体验', '接待': '服务体验',
    '热线': '服务体验', '退款': '服务体验', '退货': '服务体验', '换货': '服务体验',
    # 物流维度
    '物流': '物流配送', '快递': '物流配送', '发货': '物流配送', '配送': '物流配送',
    '速度': '物流配送', '包装': '物流配送', '送货': '物流配送', '到货': '物流配送',
    '收货': '物流配送', '运输': '物流配送', '运送': '物流配送',
    # 质量维度
    '质量': '商品质量', '材质': '商品质量', '做工': '商品质量', '面料': '商品质量',
    '手感': '商品质量', '味道': '商品质量', '口感': '商品质量', '品质': '商品质量',
    '效果': '商品质量', '功能': '商品质量', '性能': '商品质量', '续航': '商品质量',
    '屏幕': '商品质量', '音质': '商品质量', '画质': '商品质量', '电池': '商品质量',
    '外观': '商品质量', '颜值': '商品质量', '重量': '商品质量', '容量': '商品质量',
    # 价格维度
    '价格': '价格性价比', '性价比': '价格性价比', '价钱': '价格性价比',
    '优惠': '价格性价比', '便宜': '价格性价比', '贵': '价格性价比',
    '折扣': '价格性价比', '活动': '价格性价比', '划算': '价格性价比',
    # 描述维度
    '描述': '描述相符', '图片': '描述相符', '实物': '描述相符', '色差': '描述相符',
    '尺寸': '描述相符', '大小': '描述相符', '颜色': '描述相符', '款式': '描述相符',
    '规格': '描述相符', '型号': '描述相符',
    # 使用维度
    '使用': '使用体验', '体验': '使用体验', '方便': '使用体验', '简单': '使用体验',
    '操作': '使用体验', '舒适': '使用体验', '安装': '使用体验', '穿': '使用体验',
    '戴': '使用体验', '携带': '使用体验', '清洗': '使用体验',
    # 住宿维度
    '房间': '住宿环境', '卫生': '住宿环境', '环境': '住宿环境', '隔音': '住宿环境',
    '空调': '住宿环境', '热水': '住宿环境', 'wifi': '住宿环境', '网络': '住宿环境',
    '设施': '住宿环境', '床': '住宿环境', '浴室': '住宿环境', '卫生间': '住宿环境',
    '大堂': '住宿环境', '停车场': '住宿环境', '电梯': '住宿环境', '窗户': '住宿环境',
    '通风': '住宿环境', '采光': '住宿环境', '气味': '住宿环境',
    '打扫': '住宿环境', '整洁': '住宿环境', '毛巾': '住宿环境', '床单': '住宿环境',
    '位置': '地理位置', '地理位置': '地理位置', '地段': '地理位置', '交通': '地理位置',
    '出行': '地理位置', '地铁': '地理位置', '公交': '地理位置', '周边': '地理位置',
    '附近': '地理位置', '景点': '地理位置', '商圈': '地理位置', '火车站': '地理位置',
    '机场': '地理位置', '酒店': '住宿环境',
    # 餐饮维度
    '早餐': '餐饮体验', '餐': '餐饮体验', '菜品': '餐饮体验', '餐饮': '餐饮体验',
    '食物': '餐饮体验', '晚餐': '餐饮体验', '自助': '餐饮体验', '餐厅': '餐饮体验',
    # 拍照维度
    '拍照': '拍照效果', '相机': '拍照效果', '照片': '拍照效果', '像素': '拍照效果',
    '摄像头': '拍照效果', '自拍': '拍照效果',
}

# 非属性词黑名单（即使词性是 n/v 也不应该被当作属性词的词）
ASPECT_BLACKLIST = {
    '有点', '一点', '一下', '一个', '一种', '一次', '一天', '一会',
    '觉得', '感觉', '认为', '希望', '可能', '应该', '可以', '需要',
    '什么', '怎么', '为什么', '哪个', '哪里', '多少', '这个', '那个',
    '时候', '地方', '东西', '事情', '问题', '情况', '方面',
}

# 正向情感词
POSITIVE_WORDS = {
    '好', '不错', '棒', '赞', '满意', '喜欢', '推荐', '快', '完美', '优秀',
    '舒服', '方便', '漂亮', '好吃', '实用', '良心', '惊喜', '实惠', '划算',
    '值', '细心', '耐心', '热情', '周到', '及时', '迅速', '认真', '负责',
    '放心', '干净', '整洁', '精致', '大气', '高端', '耐用', '柔软', '舒适',
    '清晰', '流畅', '稳定', '安全', '贴心', '给力', '超值', '好评', '靠谱',
    '完好', '完整', '严实', '到位', '一流', '出色', '专业', '厚道',
}

# 负向情感词
NEGATIVE_WORDS = {
    '差', '不好', '烂', '糟糕', '失望', '慢', '差劲', '垃圾', '坑', '坏',
    '难吃', '难用', '丑', '假', '劣质', '忽悠', '骗', '后悔', '不值',
    '粗糙', '敷衍', '恶劣', '怠慢', '拖延', '破损', '损坏', '变形', '褪色',
    '缩水', '起球', '开线', '掉色', '异味', '噪音', '卡顿', '发热', '死机',
    '划痕', '瑕疵', '缺陷', '故障', '松动', '漏发', '错发', '过期', '变质',
    '不新鲜', '太小', '太大', '太薄', '太厚', '太硬', '太软', '麻烦', '复杂',
}

# 否定词（反转情感）
NEGATION_WORDS = {'不', '没', '无', '未', '非', '别', '莫', '勿', '否'}

# 组合否定模式: "没有 + 形容词"、"谈不上 + 形容词" 等
COMPOUND_NEGATION_PATTERNS = [('没有', ''), ('谈不上', ''), ('算不上', ''), ('称不上', '')]

# 程度副词分级映射
DEGREE_MAP = {
    '极其': 1.8, '极度': 1.8, '非常': 1.8, '十分': 1.8, '格外': 1.8, '最': 1.8,
    '很': 1.5, '特别': 1.5, '太': 1.5, '真': 1.5, '相当': 1.5,
    '比较': 1.2, '挺': 1.2, '蛮': 1.2, '还': 1.2,
    '有点': 0.7, '稍微': 0.7, '略': 0.7, '一点': 0.7,
}

# 停用词（精简版）
STOP_WORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '人', '都', '一',
    '一个', '上', '也', '到', '说', '要', '去', '你', '会', '着',
    '看', '自己', '这', '他', '她', '它', '们', '那', '些',
    '这个', '那个', '可以', '还是', '所以', '因为', '但是', '而且', '虽然',
    '如果', '然后', '已经', '真的',
    '吧', '吗', '呢', '啊', '哦', '嗯', '呀', '嘛', '哈', '哇',
}

# 小句切分符号
CLAUSE_DELIMITERS = {'，', '。', '；', '！', '？', '!', '?', ';'}

# 转折词（后面的内容为核心观点，权重更高）
CONTRASTIVE_WORDS = {'但', '但是', '不过', '然而', '可是', '只是', '却', '可'}


# ============================================================
# 第一层: TF-IDF 关键词提取
# ============================================================

class Layer1_KeywordExtractor:
    """jieba 分词 + TF-IDF + 词性过滤 → 候选关键词"""

    def __init__(self, max_features=200):
        self.max_features = max_features
        self.vectorizer = None
        self.feature_names = None

    def _tokenize(self, text):
        """jieba 分词 + 词性过滤"""
        words = []
        for word, flag in pseg.cut(text):
            word = word.strip()
            if len(word) < 2:
                continue
            if word in STOP_WORDS:
                continue
            # 只保留名词、动词、形容词、副词的组合
            if flag.startswith(('n', 'v', 'a', 'd')):
                words.append(word)
        return ' '.join(words)

    def fit_transform(self, texts):
        """对文本列表做 TF-IDF 提取"""
        tokenized = [self._tokenize(t) for t in texts]
        # 过滤空文档
        valid = [(i, tok) for i, tok in enumerate(tokenized) if tok.strip()]
        if len(valid) < 2:
            return {}  # 样本太少

        indices, docs = zip(*valid)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            token_pattern=r'(?u)\b\w+\b',
        )
        tfidf_matrix = self.vectorizer.fit_transform(docs)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # 返回每个词的平均 TF-IDF 分
        scores = tfidf_matrix.mean(axis=0).A1
        return {word: round(float(score), 4) for word, score in zip(self.feature_names, scores)}

    def get_top_keywords(self, texts, top_k=30):
        """获取 top-K 关键词（按 TF-IDF 均值排序）"""
        scores = self.fit_transform(texts)
        if not scores:
            return []
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_k]


# ============================================================
# 第二层: 属性-情感对抽取（规则引擎）
# ============================================================

class Layer2_AspectExtractor:
    """基于 jieba.posseg 词性标注 + 规则引擎抽取 (aspect, sentiment) 对"""

    # 规则模板: (置信度, 前词性模式, 后词性模式, 方向)
    # 方向: 'forward' → 前=属性, 后=情感; 'backward' → 后=属性, 前=情感
    # 置信度越高，优先匹配，命中后 token 被消费
    RULES = [
        (10, ('n',), ('a',), 'forward'),         # 名词+形容词 — 最强依存
        (10, ('n',), ('d', 'a'), 'forward'),       # 名词+副词+形容词
        (9,  ('v',), ('a',), 'forward'),           # 动词+形容词
        (9,  ('v',), ('d', 'a'), 'forward'),       # 动词+副词+形容词
        (8,  ('n',), ('d', 'd', 'a'), 'forward'),  # 名词+副词+不+形容词
        (7,  ('a', 'u'), ('n',), 'backward'),      # 形容词+的+名词
        (6,  ('d', 'a'), ('n',), 'backward'),      # 不+形容词+名词
    ]

    # 否定词回溯窗口大小
    NEGATION_WINDOW = 3

    def __init__(self):
        self.aspect_dict = ASPECT_DICT
        self.pos_words = POSITIVE_WORDS
        self.neg_words = NEGATIVE_WORDS
        self.negation_words = NEGATION_WORDS
        self.degree_map = DEGREE_MAP

    def _is_aspect_word(self, word):
        """判断是否为属性词（在维度词典中，不在黑名单，或者是名词/动词且长度≥2）"""
        if word in ASPECT_BLACKLIST:
            return False
        if len(word) < 2:
            return False
        return word in self.aspect_dict

    def _get_dimension(self, word):
        """获取属性词所属维度"""
        return self.aspect_dict.get(word, '其他')

    def _sentiment_polarity(self, word):
        """判断情感词极性: 1=正向, -1=负向, 0=中性"""
        if word in self.pos_words:
            return 1
        if word in self.neg_words:
            return -1
        # 启发式: 形容词结尾 + 正面常见字 → 正向
        if word.endswith(('好', '快', '棒', '赞', '值')):
            return 1
        if word.endswith(('差', '慢', '烂', '贵')):
            return -1
        return 0

    def _count_negations_in_window(self, sentiment_candidates, sentiment_idx):
        """回溯窗口内统计否定词数量（兼容组合否定"没有"等）"""
        neg_count = 0
        window_end = sentiment_idx
        window_start = max(0, sentiment_idx - self.NEGATION_WINDOW)
        for k in range(window_start, window_end):
            w = sentiment_candidates[k][0]
            if w in self.negation_words:
                neg_count += 1
            # 组合否定: "没有"、"谈不上" 等作为整体
            for pfx, _ in COMPOUND_NEGATION_PATTERNS:
                if w == pfx:
                    neg_count += 1
                    break
        return neg_count

    def _get_degree_intensity(self, sentiment_candidates, sentiment_idx):
        """扫描情感词前的程度词，返回强度系数"""
        intensity = 1.0
        window_end = sentiment_idx
        window_start = max(0, sentiment_idx - self.NEGATION_WINDOW)
        for k in range(window_start, window_end):
            w = sentiment_candidates[k][0]
            if w in self.degree_map:
                intensity = max(intensity, self.degree_map[w])  # 取最强程度词
        return intensity

    def _find_sentiment_in_candidates(self, sentiment_candidates):
        """在候选列表中定位情感词，返回 (index, word, 否定计数, 程度强度)"""
        for j, (w, pos) in enumerate(sentiment_candidates):
            if w in self.negation_words:
                continue
            if w in self.degree_map:
                continue
            if pos.startswith('a') or pos.startswith('v'):
                neg_count = self._count_negations_in_window(sentiment_candidates, j)
                intensity = self._get_degree_intensity(sentiment_candidates, j)
                return j, w, neg_count, intensity
        return None, None, 0, 1.0

    def _extract_pairs_rule(self, words_with_pos):
        """规则引擎抽取（含否定回溯窗口 + 程度分级 + token 消费 + 置信度排序）"""
        pairs = []
        n = len(words_with_pos)
        consumed = set()  # 被消费的 token 索引

        # 按置信度降序排列规则
        sorted_rules = sorted(self.RULES, key=lambda r: r[0], reverse=True)

        for confidence, prefix_pattern, suffix_pattern, direction in sorted_rules:
            plen = len(prefix_pattern)
            slen = len(suffix_pattern)

            for i in range(n - plen - slen + 1):
                # 检查是否已被消费
                match_range = set(range(i, i + plen + slen))
                if match_range & consumed:
                    continue

                # 匹配前缀词性
                prefix_words = words_with_pos[i:i + plen]
                prefix_pos = tuple(w[1] for w in prefix_words)
                if not all(p.startswith(pat) for p, pat in zip(prefix_pos, prefix_pattern)):
                    continue

                # 匹配后缀词性
                suffix_words = words_with_pos[i + plen:i + plen + slen]
                suffix_pos = tuple(w[1] for w in suffix_words)
                if not all(p.startswith(pat) for p, pat in zip(suffix_pos, suffix_pattern)):
                    continue

                # 按方向分配候选
                if direction == 'forward':
                    aspect_candidates = [(w, pos) for w, pos in prefix_words]
                    sentiment_candidates = [(w, pos) for w, pos in suffix_words]
                else:
                    aspect_candidates = [(w, pos) for w, pos in suffix_words]
                    sentiment_candidates = [(w, pos) for w, pos in prefix_words]

                # 筛选属性词
                aspect_word = None
                for w, pos in aspect_candidates:
                    if self._is_aspect_word(w):
                        aspect_word = w
                        break
                if not aspect_word:
                    for w, pos in aspect_candidates:
                        if pos.startswith('n') and len(w) >= 2 and w not in ASPECT_BLACKLIST:
                            aspect_word = w
                            break
                if not aspect_word:
                    continue

                # 定位情感词（含否定窗口 + 程度分级）
                s_idx, sentiment_word, neg_count, intensity = self._find_sentiment_in_candidates(sentiment_candidates)
                if not sentiment_word:
                    continue

                polarity = self._sentiment_polarity(sentiment_word)
                if polarity == 0:
                    continue

                # 奇数否定 → 极性反转
                if neg_count % 2 == 1:
                    polarity = -polarity

                pairs.append((aspect_word, sentiment_word, polarity, intensity))
                consumed |= match_range  # 消费 token

        return pairs

    def _extract_pairs_distance(self, words_with_pos):
        """距离衰减配对: 小句内所有 (名词候选, 形容词) 按距离打分，取最高分配对"""
        pairs = []
        n = len(words_with_pos)
        if n < 2:
            return pairs

        # 收集候选: 名词/动词候选 + 形容词/动词候选
        aspect_indices = []
        sentiment_indices = []

        for i, (w, pos) in enumerate(words_with_pos):
            if pos.startswith('n') and len(w) >= 2 and w not in ASPECT_BLACKLIST:
                aspect_indices.append((i, w))
            elif pos.startswith('v') and len(w) >= 2 and w not in ASPECT_BLACKLIST:
                aspect_indices.append((i, w))
            elif pos.startswith('a') and len(w) >= 1:
                sentiment_indices.append((i, w))
            elif pos.startswith('v') and len(w) >= 2:
                # 动词也可能是情感词 (如 "破损", "卡顿")
                sentiment_indices.append((i, w))

        # 配对：每个属性词找最近的未匹配情感词，按距离衰减
        used_sentiment = set()
        for ai, aw in aspect_indices:
            best_score = -999
            best_pair = None
            for si, sw in sentiment_indices:
                if si in used_sentiment:
                    continue
                # 跳过否定词和程度词
                if sw in self.negation_words or sw in self.degree_map:
                    continue
                pol = self._sentiment_polarity(sw)
                if pol == 0:
                    continue

                distance = abs(ai - si)
                # 超出 6 个词忽略
                if distance > 6:
                    continue

                # 检测属性词和情感词之间的否定词
                neg_count = 0
                for k in range(min(ai, si) + 1, max(ai, si)):
                    wk = words_with_pos[k][0]
                    if wk in self.negation_words:
                        neg_count += 1

                # 检测程度词
                intensity = 1.0
                for k in range(min(ai, si) + 1, max(ai, si)):
                    wk = words_with_pos[k][0]
                    if wk in self.degree_map:
                        intensity = max(intensity, self.degree_map[wk])

                if neg_count % 2 == 1:
                    pol = -pol

                # 距离衰减权重
                decay = 1.0 / (1.0 + distance)
                score = pol * intensity * decay

                if score > best_score:
                    best_score = score
                    best_pair = (aw, sw, pol, intensity * decay)

            if best_pair:
                pairs.append(best_pair)

        return pairs

    def _segment_by_punct(self, text):
        """按标点符号切分文本为片段"""
        segments = []
        last = 0
        for i, ch in enumerate(text):
            if ch in CLAUSE_DELIMITERS:
                segments.append(text[last:i])
                last = i + 1
        if last < len(text):
            segments.append(text[last:])
        return segments

    def _split_clauses(self, text):
        """将长句按标点和转折词切成小句，返回 [(clause_text, weight)]

        - 转折词后面的小句为核心观点，权重 ×1.3
        - 每个小句独立抽取，互不污染
        """
        segments = self._segment_by_punct(text)
        result = []
        after_contrast = False
        for seg in segments:
            stripped = seg.strip()
            if not stripped:
                continue
            # 检查是否以转折词开头
            is_contrast = any(stripped.startswith(w) for w in CONTRASTIVE_WORDS)
            if is_contrast:
                after_contrast = True
                # 去掉转折词前缀
                for w in sorted(CONTRASTIVE_WORDS, key=len, reverse=True):
                    if stripped.startswith(w):
                        stripped = stripped[len(w):].strip()
                        break
            weight = 1.3 if after_contrast else 1.0
            # 转折权重仅对紧跟转折词的那一个小句生效，之后立即复位
            if after_contrast:
                after_contrast = False
            if stripped:
                result.append((stripped, weight))
        return result

    def extract(self, text):
        """对单条文本做属性-情感对抽取（小句切分 + 规则引擎 + 距离衰减）"""
        all_pairs = []
        clauses = self._split_clauses(text)
        for clause_text, clause_weight in clauses:
            words_with_pos = [(w, f) for w, f in pseg.cut(clause_text)
                              if w.strip() and w not in STOP_WORDS]
            # 规则引擎
            rule_pairs = self._extract_pairs_rule(words_with_pos)
            # 距离衰减补充（仅对未被规则覆盖的词）
            distance_pairs = self._extract_pairs_distance(words_with_pos)
            # 合并
            clause_pairs = rule_pairs + distance_pairs
            for pair in clause_pairs:
                aw, sw, pol, intensity = pair
                all_pairs.append((aw, sw, pol, intensity * clause_weight))
        return list(set(all_pairs))

    def extract_batch(self, texts):
        """批量抽取"""
        all_pairs = []
        for text in texts:
            pairs = self.extract(text)
            all_pairs.append(pairs)
        return all_pairs


# ============================================================
# 第三层: 情感归因报告
# ============================================================

class Layer3_ReportGenerator:
    """汇总各维度情感得分，生成归因报告"""

    def __init__(self, aspect_dict=None):
        self.aspect_dict = aspect_dict or ASPECT_DICT

    def generate(self, texts, all_pairs, keywords=None):
        """生成完整归因报告"""
        # ---- 维度统计 (intensity 加权) ----
        dim_stats = defaultdict(lambda: {'pos': 0.0, 'neg': 0.0, 'total': 0})

        for pairs in all_pairs:
            for entry in pairs:
                # 兼容三元组和四元组
                if len(entry) == 4:
                    aspect_word, sentiment_word, polarity, intensity = entry
                else:
                    aspect_word, sentiment_word, polarity = entry
                    intensity = 1.0
                dim = self.aspect_dict.get(aspect_word, '其他')
                dim_stats[dim]['total'] += 1
                if polarity > 0:
                    dim_stats[dim]['pos'] += intensity
                else:
                    dim_stats[dim]['neg'] += intensity

        # ---- 维度得分（-100 ~ +100）----
        aspect_scores = {}
        for dim, counts in dim_stats.items():
            total = counts['total']
            if total > 0:
                score = round((counts['pos'] - counts['neg']) / total * 100, 1)
            else:
                score = 0.0
            aspect_scores[dim] = {
                'score': score,
                'positive_count': counts['pos'],
                'negative_count': counts['neg'],
                'total_mentions': total,
            }

        # ---- 排序 ----
        sorted_pain = sorted(
            [(k, v) for k, v in aspect_scores.items() if v['score'] < 0],
            key=lambda x: x[1]['score']
        )
        sorted_highlight = sorted(
            [(k, v) for k, v in aspect_scores.items() if v['score'] > 0],
            key=lambda x: x[1]['score'], reverse=True
        )

        # ---- 整体情感分布 ----
        total_pos = sum(c['pos'] for c in dim_stats.values())
        total_neg = sum(c['neg'] for c in dim_stats.values())
        total_all = total_pos + total_neg

        # ---- 关键特征词（从 pairs 中统计高频属性词） ----
        aspect_counter = Counter()
        sentiment_counter = Counter()
        for pairs in all_pairs:
            for entry in pairs:
                aw, sw = entry[0], entry[1]
                aspect_counter[aw] += 1
                sentiment_counter[sw] += 1

        # ---- 组装报告 ----
        report = {
            'summary': {
                'total_texts': len(texts),
                'total_aspect_mentions': total_all,
                'positive_mentions': total_pos,
                'negative_mentions': total_neg,
                'overall_sentiment': round((total_pos - total_neg) / max(total_all, 1) * 100, 1),
            },
            'dimension_scores': aspect_scores,
            'pain_points': [(dim, data['score'], data['total_mentions']) for dim, data in sorted_pain],
            'highlights': [(dim, data['score'], data['total_mentions']) for dim, data in sorted_highlight],
            'top_aspect_words': aspect_counter.most_common(15),
            'top_sentiment_words': sentiment_counter.most_common(15),
            'keywords': keywords or [],
        }

        return report


# ============================================================
# 一体式 SAA 引擎
# ============================================================

class SAAEngine:
    """情感归因分析引擎（三层一体 + 模型融合）"""

    # 模型融合权重: α * 规则得分 + (1-α) * LSTM 得分
    FUSION_ALPHA = 0.4
    MODEL_CONFIDENCE_THRESHOLD = 0.85  # 模型置信度高于此值时强行采用模型结果

    def __init__(self):
        self.layer1 = Layer1_KeywordExtractor()
        self.layer2 = Layer2_AspectExtractor()
        self.layer3 = Layer3_ReportGenerator()

    def analyze(self, texts, top_k_keywords=30, model_preds=None):
        """
        完整分析流程
        输入: texts (list of str), model_preds 可选: [(label, confidence), ...]
        输出: dict (完整归因报告)
        """
        # 第一层
        keywords = self.layer1.get_top_keywords(texts, top_k=top_k_keywords)

        # 第二层
        all_pairs = self.layer2.extract_batch(texts)

        # 第三层
        report = self.layer3.generate(texts, all_pairs, keywords)

        # 模型融合（如果有 LSTM 预测结果）
        if model_preds and len(model_preds) == len(texts):
            self._fuse_model_scores(report, model_preds)

        return report

    def _fuse_model_scores(self, report, model_preds):
        """将 LSTM 模型预测与规则得分进行加权融合

        Args:
            report: Layer3 生成的报告 dict
            model_preds: [(label_str, confidence_float), ...]
                         例: [('好评', 0.92), ('差评', 0.67), ...]
        """
        # 规则整体得分 S_r (-100 ~ 100)
        s_r = report['summary']['overall_sentiment']

        # 模型得分 S_m: 计算 (-100 ~ 100)
        total = len(model_preds)
        if total == 0:
            return
        pos_weight = sum(conf if label == '好评' else 1 - conf
                         for label, conf in model_preds)
        neg_weight = sum(conf if label == '差评' else 1 - conf
                         for label, conf in model_preds)
        s_m = round((pos_weight - neg_weight) / total * 100, 1)

        # 判断是否有高置信度冲突
        model_avg_conf = sum(c for _, c in model_preds) / total
        rule_pos = report['summary']['positive_mentions']
        rule_neg = report['summary']['negative_mentions']
        rule_agrees_model = (s_r >= 0 and s_m >= 0) or (s_r < 0 and s_m < 0)

        if not rule_agrees_model and model_avg_conf > self.MODEL_CONFIDENCE_THRESHOLD:
            # 规则和模型冲突且模型置信度高 → 以模型为准
            alpha = 0.0
        else:
            alpha = self.FUSION_ALPHA

        fused = round(alpha * s_r + (1 - alpha) * s_m, 1)

        # 写入报告
        report['summary']['rule_sentiment'] = s_r
        report['summary']['model_sentiment'] = s_m
        report['summary']['overall_sentiment'] = fused
        report['summary']['fusion_alpha'] = alpha

    def analyze_single(self, text):
        """单条文本的维度情感分析"""
        pairs = self.layer2.extract(text)
        if not pairs:
            return {
                'text': text,
                'aspects': [],
                'dimensions': {},
                'summary': '未检测到明确的情感维度',
            }

        # 维度汇总
        dims = defaultdict(lambda: {'pos': 0.0, 'neg': 0.0})
        aspect_details = []
        for entry in pairs:
            if len(entry) == 4:
                aw, sw, pol, intensity = entry
            else:
                aw, sw, pol = entry
                intensity = 1.0
            dim = ASPECT_DICT.get(aw, '其他')
            aspect_details.append({
                'aspect_word': aw,
                'sentiment_word': sw,
                'polarity': '正向' if pol > 0 else '负向',
                'dimension': dim,
                'intensity': intensity,
            })
            if pol > 0:
                dims[dim]['pos'] += intensity
            else:
                dims[dim]['neg'] += intensity

        # 维度得分
        dim_scores = {}
        for dim, counts in dims.items():
            t = counts['pos'] + counts['neg']
            dim_scores[dim] = round((counts['pos'] - counts['neg']) / t * 100, 1) if t > 0 else 0

        # 总结
        pos_total = sum(1 for p in pairs if p[2] > 0)
        neg_total = sum(1 for p in pairs if p[2] < 0)
        if pos_total > neg_total:
            summary = '整体偏正向'
        elif neg_total > pos_total:
            summary = '整体偏负向'
        else:
            summary = '情感中性'

        return {
            'text': text,
            'aspects': aspect_details,
            'dimension_scores': dim_scores,
            'summary': summary,
            'positive_aspects': pos_total,
            'negative_aspects': neg_total,
        }


# ============================================================
# 全局单例
# ============================================================

_saa_instance = None


def get_saa():
    """获取 SAA 引擎单例（懒加载）"""
    global _saa_instance
    if _saa_instance is None:
        _saa_instance = SAAEngine()
    return _saa_instance


# ============================================================
# 命令行测试入口
# ============================================================

if __name__ == '__main__':
    test_texts = [
        "酒店服务态度非常好，房间也很干净，但是价格有点贵",
        "快递速度太慢了，包装也破损了，非常失望",
        "这个手机性价比很高，拍照效果很棒，续航也不错",
        "客服回复很慢态度也很差，物流倒是挺快的",
        "衣服质量很好，穿起来很舒服，值得购买",
    ]

    print("=" * 60)
    print("SAA 情感归因分析引擎 - 测试")
    print("=" * 60)

    engine = SAAEngine()

    # 完整分析
    print("\n>>> 第一层: TF-IDF 关键词")
    keywords = engine.layer1.get_top_keywords(test_texts, top_k=15)
    for w, s in keywords:
        print(f"  {w}: {s:.4f}")

    print("\n>>> 第二层: 属性-情感对抽取")
    for text in test_texts:
        pairs = engine.layer2.extract(text)
        print(f"\n  原文: {text}")
        if pairs:
            for entry in pairs:
                if len(entry) == 4:
                    aw, sw, pol, intensity = entry
                else:
                    aw, sw, pol = entry
                    intensity = 1.0
                label = '+' if pol > 0 else '-'
                print(f"    {label} [{aw}] → [{sw}] ({'正向' if pol > 0 else '负向'}, 强度={intensity})")
        else:
            print(f"    (无匹配)")

    print("\n>>> 第三层: 归因报告")
    report = engine.analyze(test_texts)
    print(f"  总评论文本: {report['summary']['total_texts']}")
    print(f"  属性提及: {report['summary']['total_aspect_mentions']} (正{report['summary']['positive_mentions']}/负{report['summary']['negative_mentions']})")
    print(f"  整体情感: {report['summary']['overall_sentiment']}")

    print("\n  维度得分:")
    for dim, data in sorted(report['dimension_scores'].items(), key=lambda x: x[1]['score']):
        bar = '█' * max(1, abs(int(data['score'])) // 5)
        sign = '+' if data['score'] >= 0 else ''
        print(f"    {dim:8s}  {sign}{data['score']:5.1f}  {bar}  (提及{data['total_mentions']}次)")

    print("\n  痛点 (最需改进):")
    for dim, score, mentions in report['pain_points']:
        print(f"    {dim}: {score} (提及{mentions}次)")

    print("\n  亮点 (保持优势):")
    for dim, score, mentions in report['highlights']:
        print(f"    {dim}: +{score} (提及{mentions}次)")

    print("\n>>> 单条诊断")
    result = engine.analyze_single(test_texts[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))
