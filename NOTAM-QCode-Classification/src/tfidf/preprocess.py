"""
文本预处理与特征提取模块

技术路径说明：
    采用 jieba 中文分词 + TF-IDF 词频-逆文档频率特征提取的传统 NLP 方案。
    与 BERT 方案不同，该方法不依赖预训练语言模型，计算资源需求低。
"""

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib


def preprocess_text(text):
    """NOTAM 文本预处理函数

    处理步骤：
        1. 转换为小写（针对英文部分，如跑道代号 RWY 等）
        2. 使用 jieba 进行中文分词
        3. 过滤掉单字字符（去除标点、空格等）

    Args:
        text: 原始 NOTAM E 项文本

    Returns:
        空格连接的分词结果字符串
    """
    text = str(text).lower()
    words = jieba.lcut(text)
    words = [word for word in words if len(word) > 1]
    return " ".join(words)


def extract_features(
    train_path,
    val_path,
    test_path,
    max_features=2000,
    output_dir="./",
):
    """完整的特征提取流水线

    Args:
        train_path: 训练集 CSV 路径
        val_path: 验证集 CSV 路径
        test_path: 测试集 CSV 路径
        max_features: TF-IDF 最大特征数
        output_dir: 输出目录（保存 pkl 文件）

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, tfidf, label_encoder)
    """
    print("正在加载数据集...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    print(f"训练集样本数: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

    print("正在进行文本预处理与分词...")
    train_df["processed_text"] = train_df["e_text"].apply(preprocess_text)
    val_df["processed_text"] = val_df["e_text"].apply(preprocess_text)
    test_df["processed_text"] = test_df["e_text"].apply(preprocess_text)

    print(f"\n预处理后的文本样本（前3条）：")
    print(train_df[["e_text", "processed_text"]].head(3))

    print(f"\n正在提取 TF-IDF 特征（max_features={max_features}）...")
    tfidf = TfidfVectorizer(max_features=max_features)
    X_train = tfidf.fit_transform(train_df["processed_text"])
    X_val = tfidf.transform(val_df["processed_text"])
    X_test = tfidf.transform(test_df["processed_text"])
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    print("\n正在对标签进行编码...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["q_code"])
    y_val = label_encoder.transform(val_df["q_code"])
    y_test = label_encoder.transform(test_df["q_code"])
    print(f"标签类别数量: {len(label_encoder.classes_)}")

    print("\n正在保存预处理结果...")
    joblib.dump(tfidf, f"{output_dir}/tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, f"{output_dir}/label_encoder.pkl")
    joblib.dump((X_train, y_train), f"{output_dir}/train_features.pkl")
    joblib.dump((X_val, y_val), f"{output_dir}/val_features.pkl")
    joblib.dump((X_test, y_test), f"{output_dir}/test_features.pkl")
    print(f"预处理结果已保存至: {output_dir}")

    return X_train, y_train, X_val, y_val, X_test, y_test, tfidf, label_encoder