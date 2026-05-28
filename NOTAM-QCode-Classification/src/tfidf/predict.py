"""
TF-IDF + 逻辑回归模型预测模块

提供交互式命令行预测功能，输入 E 项文本即可输出对应的 Q 代码。
"""

import jieba
import joblib


def preprocess_text(text):
    """文本预处理（与训练时保持一致）"""
    text = str(text).lower()
    words = jieba.lcut(text)
    words = [word for word in words if len(word) > 1]
    return " ".join(words)


def predict_qcode(e_text, model, tfidf, label_encoder):
    """预测单条 E 项文本对应的 Q 代码

    Args:
        e_text: NOTAM E 项文本
        model: 训练好的 LogisticRegression 模型
        tfidf: TF-IDF 向量化器
        label_encoder: 标签编码器

    Returns:
        Q 代码字符串
    """
    processed = preprocess_text(e_text)
    features = tfidf.transform([processed])
    pred_num = model.predict(features)[0]
    q_code = label_encoder.inverse_transform([pred_num])[0]
    return q_code


def interactive_predict(model, tfidf, label_encoder):
    """交互式预测模式"""
    print("=" * 60)
    print("TF-IDF + 逻辑回归模型预测 - 输入 E 项文本，输出 Q 代码")
    print("=" * 60)

    e_text = input("请输入/粘贴 E 项文本：")
    q_code = predict_qcode(e_text, model, tfidf, label_encoder)

    print("\n预测完成！")
    print(f"对应的 Q 代码为：【{q_code}】")
    print("=" * 60)
    return q_code