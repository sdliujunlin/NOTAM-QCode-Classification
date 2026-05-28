"""
BERT 模型定义与数据集类

技术路径说明：
    使用哈工大讯飞联合发布的中文 BERT-wwm-ext 预训练模型
    (hfl/chinese-bert-wwm-ext) 进行 NOTAM E 项文本的 Q 代码分类。
    该方法属于基于预训练语言模型微调的深度学习方案。
"""

import torch
from torch.utils.data import Dataset


class NOTAMDataset(Dataset):
    """NOTAM 文本数据集类，用于 BERT 模型的训练与推理。

    Args:
        texts: NOTAM E 项文本列表
        labels: Q 代码标签列表（已数值编码）
        tokenizer: BERT 分词器
        max_len: 最大序列长度
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        enc = self.tokenizer(
            str(self.texts[item]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[item]),
        }