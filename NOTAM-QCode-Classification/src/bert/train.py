"""
BERT 模型训练模块

训练流程：
    1. 加载中文 BERT-wwm-ext 预训练模型
    2. 使用 NOTAM 数据集进行微调
    3. 每轮在验证集上评估并保存最优模型
    4. 最终在测试集上评估模型性能
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd

from .model import NOTAMDataset


class BERTTrainer:
    """BERT 模型训练器"""

    def __init__(
        self,
        model_name="hfl/chinese-bert-wwm-ext",
        max_len=128,
        batch_size=32,
        epochs=3,
        learning_rate=2e-5,
    ):
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.optimizer = None
        self.scheduler = None

    def load_data(self, train_path, val_path, test_path):
        """加载并预处理数据"""
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)

        # 标签编码
        self.label_encoder = LabelEncoder()
        self.train_df["label"] = self.label_encoder.fit_transform(
            self.train_df["q_code"]
        )
        self.val_df["label"] = self.label_encoder.transform(self.val_df["q_code"])
        self.test_df["label"] = self.label_encoder.transform(self.test_df["q_code"])

        # 初始化分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

    def _create_dataloaders(self):
        """创建数据加载器"""
        self.train_loader = DataLoader(
            NOTAMDataset(
                self.train_df["e_text"],
                self.train_df["label"],
                self.tokenizer,
                self.max_len,
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            NOTAMDataset(
                self.val_df["e_text"],
                self.val_df["label"],
                self.tokenizer,
                self.max_len,
            ),
            batch_size=self.batch_size,
        )
        self.test_loader = DataLoader(
            NOTAMDataset(
                self.test_df["e_text"],
                self.test_df["label"],
                self.tokenizer,
                self.max_len,
            ),
            batch_size=self.batch_size,
        )

    def _setup_model(self):
        """初始化模型与优化器"""
        num_labels = len(self.label_encoder.classes_)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        ).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 0, len(self.train_loader) * self.epochs
        )

    def _run_epoch(self, loader, mode="train"):
        """运行单个 epoch 的训练或验证"""
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        correct, total_loss = 0, 0
        with torch.set_grad_enabled(mode == "train"):
            for batch in loader:
                outputs = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    labels=batch["label"].to(self.device),
                )
                correct += (
                    (torch.max(outputs.logits, 1)[1] == batch["label"].to(self.device))
                    .sum()
                    .item()
                )
                total_loss += outputs.loss.item()

                if mode == "train":
                    outputs.loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

        accuracy = correct / len(loader.dataset)
        avg_loss = total_loss / len(loader)
        return accuracy, avg_loss

    def train(self, train_path, val_path, test_path, model_save_path="best_model.bin"):
        """完整训练流程"""
        print("=" * 60)
        print("BERT 模型训练 - NOTAM Q 代码分类")
        print(f"设备: {self.device}")
        print("=" * 60)

        self.load_data(train_path, val_path, test_path)
        self._create_dataloaders()
        self._setup_model()

        best_acc = 0.0
        for epoch in range(self.epochs):
            train_acc, train_loss = self._run_epoch(
                self.train_loader, mode="train"
            )
            val_acc, val_loss = self._run_epoch(self.val_loader, mode="val")
            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), model_save_path)
                print(f"  -> 保存最优模型 (Val Acc: {best_acc:.4f})")

        # 加载最优模型并在测试集评估
        self.model.load_state_dict(
            torch.load(model_save_path, map_location=self.device)
        )
        test_acc, _ = self._run_epoch(self.test_loader, mode="val")
        print(f"\n最终测试集准确率: {test_acc:.4f}")

        return self.model, self.label_encoder

    def evaluate_detailed(self):
        """输出详细评估指标（精确率、召回率、F1）"""
        y_true, y_pred = [], []
        self.model.eval()

        with torch.no_grad():
            for batch in self.test_loader:
                outputs = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                predictions = torch.argmax(outputs.logits, dim=1)
                y_true.extend(batch["label"].cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        print("\n" + "=" * 60)
        print("模型详细评估指标（精确率 / 召回率 / F1分数）")
        print("=" * 60)
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.label_encoder.classes_,
            digits=4,
        )
        print(report)
        return report