"""
BERT 模型预测模块

提供交互式命令行预测功能，输入 E 项文本即可输出对应的 Q 代码。
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification


def load_model(
    model_path="best_model.bin",
    model_name="hfl/chinese-bert-wwm-ext",
    num_labels=13,
    device=None,
):
    """加载训练好的 BERT 模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_qcode(e_text, model, tokenizer, label_encoder, device, max_len=128):
    """预测单条 E 项文本对应的 Q 代码"""
    inputs = tokenizer(
        e_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            inputs["input_ids"].to(device),
            inputs["attention_mask"].to(device),
        )

    pred_label = torch.argmax(outputs.logits, dim=1).cpu().item()
    q_code = label_encoder.inverse_transform([pred_label])[0]
    return q_code


def interactive_predict(model, tokenizer, label_encoder, device):
    """交互式预测模式"""
    print("=" * 60)
    print("BERT 模型预测 - 输入 E 项文本，输出 Q 代码")
    print("=" * 60)

    e_text = input("请输入/粘贴 E 项文本：")
    q_code = predict_qcode(e_text, model, tokenizer, label_encoder, device)

    print("\n预测完成！")
    print(f"对应的 Q 代码为：【{q_code}】")
    print("=" * 60)
    return q_code