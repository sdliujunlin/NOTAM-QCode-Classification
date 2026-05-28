# NOTAM Q-Code Classification

基于自然语言处理的航行通告（NOTAM）Q 代码自动分类系统。

## 项目概述

本项目实现了两种技术路径对 NOTAM 的 E 项文本进行 Q 代码分类：

| 方案 | 技术路径 | 核心方法 | 特点 |
|------|---------|---------|------|
| **方案 A** | 深度学习 | Chinese BERT-wwm-ext 微调 | 准确率高（~96.8%），需要 GPU |
| **方案 B** | 传统机器学习 | jieba 分词 + TF-IDF + 逻辑回归 | 训练快速、资源需求低（~96.3%） |

## 项目结构

```
NOTAM-QCode-Classification/
├── README.md                        # 项目说明文档
├── requirements.txt                 # Python 依赖包列表
├── data/                            # 数据集目录
│   ├── train_data.csv               # 训练集（80%）
│   ├── val_data.csv                 # 验证集（10%）
│   └── test_data.csv                # 测试集（10%）
├── notebooks/                       # Jupyter Notebook 实验记录
│   ├── 01_data_preparation.ipynb    # 数据划分脚本
│   ├── 02_bert_classification.ipynb # BERT 方案完整流程
│   ├── 03_tfidf_classification.ipynb# TF-IDF 方案完整流程
│   └── 04_gpu_check.ipynb          # GPU 环境检测
├── src/                             # 模块化源代码
│   ├── bert/                        # BERT 深度学习方案
│   │   ├── __init__.py
│   │   ├── model.py                 # 数据集类定义
│   │   ├── train.py                 # 训练流程
│   │   └── predict.py               # 预测接口
│   └── tfidf/                       # TF-IDF 传统 ML 方案
│       ├── __init__.py
│       ├── preprocess.py            # 文本预处理与特征提取
│       ├── train.py                 # 逻辑回归训练
│       └── predict.py               # 预测接口
└── utils/                           # 工具模块
    ├── __init__.py
    └── gpu_check.py                 # GPU 环境检测
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

原始数据来源于 `NOTAM_test_processed.xlsx`，包含 `e_text`（E 项文本）和 `q_code`（Q 代码标签）两列。

使用 `notebooks/01_data_preparation.ipynb` 可完成数据集的 80/10/10 分层划分。

### 3. 方案 A：BERT 深度学习

```python
from src.bert.train import BERTTrainer

trainer = BERTTrainer(
    model_name="hfl/chinese-bert-wwm-ext",
    max_len=128,
    batch_size=32,
    epochs=3,
    learning_rate=2e-5,
)
model, label_encoder = trainer.train(
    train_path="data/train_data.csv",
    val_path="data/val_data.csv",
    test_path="data/test_data.csv",
    model_save_path="best_model.bin",
)
trainer.evaluate_detailed()
```

### 4. 方案 B：TF-IDF + 逻辑回归

```python
from src.tfidf.preprocess import extract_features
from src.tfidf.train import train_logistic_regression

# 特征提取
X_train, y_train, X_val, y_val, X_test, y_test, tfidf, le = extract_features(
    train_path="data/train_data.csv",
    val_path="data/val_data.csv",
    test_path="data/test_data.csv",
    max_features=2000,
    output_dir="./models",
)

# 模型训练
model = train_logistic_regression(
    train_features_path="./models/train_features.pkl",
    val_features_path="./models/val_features.pkl",
    test_features_path="./models/test_features.pkl",
    label_encoder_path="./models/label_encoder.pkl",
    model_save_path="./models/qcode_classifier_model.pkl",
)
```

### 5. 交互式预测

```python
# BERT 方案预测
from src.bert.predict import load_model, interactive_predict
model, tokenizer, device = load_model("best_model.bin", num_labels=13)
interactive_predict(model, tokenizer, label_encoder, device)

# TF-IDF 方案预测
from src.tfidf.predict import interactive_predict
interactive_predict(model, tfidf, label_encoder)
```

## 数据集说明

- **输入**: NOTAM E 项文本（中英混合的航行通告描述）
- **输出**: Q 代码（13 个类别：A, F, I, K, L, M, N, O, P, R, S, W, X）
- **训练集**: 10,591 条 | **验证集**: 1,324 条 | **测试集**: 1,324 条

## 技术路径对比

| 维度 | BERT (方案 A) | TF-IDF + LR (方案 B) |
|------|--------------|---------------------|
| 文本表示 | 动态词向量（上下文感知） | 静态词频统计 |
| 中文处理 | BERT 自带分词 | jieba 分词 |
| 分类器 | BERT 顶层 + 全连接 | 逻辑回归 |
| 硬件需求 | GPU 推荐 | CPU 即可 |
| 训练时间 | 较长（需微调预训练模型） | 快速（秒级） |
| 测试准确率 | ~96.83% | ~96.30% |
| 可解释性 | 较低 | 较高（可查看特征权重） |

## 模型文件

训练过程中生成的模型文件（`.bin`、`.pkl`）已加入 `.gitignore`，需要时本地运行训练脚本生成。