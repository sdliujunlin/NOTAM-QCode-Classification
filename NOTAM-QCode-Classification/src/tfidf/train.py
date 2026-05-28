"""
传统机器学习模型训练模块

技术路径说明：
    使用逻辑回归 (LogisticRegression) 作为分类器，
    在 TF-IDF 特征矩阵上进行训练，快速建立基线模型。
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def train_logistic_regression(
    train_features_path,
    val_features_path,
    test_features_path,
    label_encoder_path,
    model_save_path="qcode_classifier_model.pkl",
    max_iter=1000,
    random_state=42,
):
    """训练逻辑回归分类器并评估

    Args:
        train_features_path: 训练集特征文件路径
        val_features_path: 验证集特征文件路径
        test_features_path: 测试集特征文件路径
        label_encoder_path: 标签编码器路径
        model_save_path: 模型保存路径
        max_iter: 最大迭代次数
        random_state: 随机种子

    Returns:
        训练好的模型
    """
    print("正在加载特征数据...")
    X_train, y_train = joblib.load(train_features_path)
    X_val, y_val = joblib.load(val_features_path)
    X_test, y_test = joblib.load(test_features_path)
    label_encoder = joblib.load(label_encoder_path)
    print(f"数据加载完成！训练集样本数: {X_train.shape[0]}")

    print("\n正在训练逻辑回归模型...")
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    print("模型训练完成！")

    print("\n正在验证集上评估...")
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"验证集准确率: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")

    print("\n详细分类报告：")
    print(
        classification_report(
            y_val, y_val_pred, target_names=label_encoder.classes_
        )
    )

    print("\n正在测试集上最终测试...")
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"测试集最终准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    joblib.dump(model, model_save_path)
    print(f"\n模型已保存至：{model_save_path}")

    return model