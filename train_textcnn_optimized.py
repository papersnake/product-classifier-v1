'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 17:30:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 17:27:03
FilePath: \\myproject\\train_textcnn_optimized.py
Description: 优化的TextCNN训练脚本 - 针对CPU训练进行性能优化

Copyright (c) 2026 by papersnake, All Rights Reserved.
'''

from embedding_manager import Vocabulary
from text_cnn import create_textcnn_from_vocab
from text_dataset import create_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time
import os
import sys
from typing import Tuple


# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class OptimizedTextCNNTrainer:
    """
    优化的TextCNN模型训练器 - 针对CPU性能优化
    """

    def __init__(self,
                 batch_size: int = 64,  # 增大批大小以提高CPU利用率
                 max_seq_len: int = 11,
                 learning_rate: float = 0.001,
                 num_epochs: int = 50,
                 patience: int = 5,
                 num_workers: int = 4,  # 增加数据加载工作线程
                 use_amp: bool = True,  # 尝试混合精度训练
                 model_save_path: str = "data/textcnn_optimized.pth",
                 vocab_path: str = "data/vocab.pkl",
                 embedding_matrix_path: str = "data/embedding_matrix.npy",
                 cat_mapping_path: str = "data/cat_id_mapping.csv"):
        """
        初始化优化的训练器

        Parameters:
        batch_size: 批大小（建议64-128以充分利用CPU）
        max_seq_len: 最大序列长度
        learning_rate: 学习率
        num_epochs: 训练轮数
        patience: 早停耐心值
        num_workers: 数据加载工作线程数
        use_amp: 是否使用混合精度训练
        model_save_path: 模型保存路径
        vocab_path: 词汇表路径
        embedding_matrix_path: 嵌入矩阵路径
        cat_mapping_path: 类别映射路径
        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.model_save_path = model_save_path
        self.vocab_path = vocab_path
        self.embedding_matrix_path = embedding_matrix_path
        self.cat_mapping_path = cat_mapping_path

        # 设备设置
        self.device = torch.device('cpu')

        # 训练历史
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []

        # 最佳准确率
        self.best_test_acc = 0.0

        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

        # 数据加载器
        self.train_loader = None
        self.test_loader = None

        # 混合精度训练的梯度缩放器（CPU上禁用）
        self.scaler = None  # CPU上不使用梯度缩放

        print("优化训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"批大小: {batch_size}")
        print(f"数据加载工作线程: {num_workers}")
        print(f"混合精度训练: {use_amp}")

    def setup_data(self):
        """设置数据加载器"""
        print("=== 设置数据集 ===")

        # 创建数据加载器
        self.train_loader, self.test_loader = create_data_loaders(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            num_workers=self.num_workers  # 使用更多工作线程
        )

        print(f"训练数据加载器: {len(self.train_loader)}个批次")
        print(f"测试数据加载器: {len(self.test_loader)}个批次")

    def setup_model(self):
        """设置模型、优化器和损失函数"""
        print("=== 设置模型 ===")

        # 加载词汇表
        vocab = Vocabulary()
        vocab.load(self.vocab_path)
        vocab_size = vocab.get_vocab_size()
        num_classes = len(pd.read_csv(self.cat_mapping_path, encoding='utf-8-sig')
                          ) if os.path.exists(self.cat_mapping_path) else 246

        print(f"词汇表大小: {vocab_size}")
        print(f"类别数量: {num_classes}")

        # 创建模型
        self.model = create_textcnn_from_vocab(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_matrix_path=self.embedding_matrix_path,
            trainable_embeddings=True  # 训练嵌入层
        )
        self.model.to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )

        print(f"模型: {type(self.model).__name__}")
        print(f"优化器: Adam (lr={self.learning_rate})")
        print(f"损失函数: {type(self.criterion).__name__}")
        print("学习率调度器: ReduceLROnPlateau")

    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_save_path):
            try:
                checkpoint = torch.load(
                    self.model_save_path, map_location=self.device)

                # 检查词汇表大小是否匹配
                if self.model is None:
                    self.setup_model()

                # 恢复模型
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(
                    checkpoint.get('scheduler_state_dict', {}))

                # 恢复训练历史
                self.train_loss_history = checkpoint.get(
                    'train_loss_history', [])
                self.train_acc_history = checkpoint.get(
                    'train_acc_history', [])
                self.test_loss_history = checkpoint.get(
                    'test_loss_history', [])
                self.test_acc_history = checkpoint.get('test_acc_history', [])

                self.best_test_acc = max(
                    self.test_acc_history) if self.test_acc_history else 0.0

                print(f"模型已加载，最佳准确率: {self.best_test_acc:.4f}")
                return True
            except Exception as e:
                print(f"加载模型时出错: {e}")
                print("将从头开始训练")
                return False
        else:
            print("未找到已保存的模型，将从头开始训练")
            return False

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        start_time = time.time()

        for batch_idx, (texts, labels) in enumerate(self.train_loader):
            # 移动到设备
            texts = texts.to(self.device)
            labels = labels.squeeze(1).to(self.device)  # 去除多余的维度

            # 标准训练流程（CPU上禁用混合精度）
            outputs = self.model(texts)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item() * texts.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += texts.size(0)

            # 每50个批次打印进度
            if batch_idx % 50 == 0:
                batch_acc = (predicted == labels).sum().item() / texts.size(0)
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * \
                    self.batch_size / elapsed if elapsed > 0 else 0

                print(f"  批次 {batch_idx:4d}/{len(self.train_loader)}: "
                      f"损失={loss.item():.4f}, 准确率={batch_acc:.4f}, "
                      f"速度={samples_per_sec:.1f}样本/秒")

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        epoch_time = time.time() - start_time
        print(
            f"  Epoch {epoch}训练完成: 损失={avg_loss:.4f}, 准确率={avg_acc:.4f}, 时间={epoch_time:.2f}秒")

        return avg_loss, avg_acc

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for texts, labels in data_loader:
                texts = texts.to(self.device)
                labels = labels.squeeze(1).to(self.device)

                if self.use_amp:
                    with torch.amp.autocast('cpu', enabled=True):
                        outputs = self.model(texts)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(texts)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * texts.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += texts.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def train(self) -> float:
        """训练主循环"""
        print("=== 开始优化训练 ===")

        # 设置数据和模型
        self.setup_data()
        self.setup_model()

        # 尝试加载现有模型
        self.load_model()

        # 早停计数器
        patience_counter = 0

        # 训练循环
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}:")

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            # 评估
            test_loss, test_acc = self.evaluate(self.test_loader)
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_acc)

            print(
                f"  Epoch {epoch}测试结果: 损失={test_loss:.4f}, 准确率={test_acc:.4f}")

            # 更新学习率
            self.scheduler.step(test_acc)

            # 保存最佳模型
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                patience_counter = 0
                self.save_model(epoch, f"epoch_{epoch}_acc_{test_acc:.4f}")
                print(f"  [OK] 新的最佳模型已保存! 准确率: {test_acc:.4f}")
            else:
                patience_counter += 1
                print(f"  [WAIT] 早停计数器: {patience_counter}/{self.patience}")

            # 检查早停
            if patience_counter >= self.patience:
                print(f"\n[WARN] 早停触发! 连续{self.patience}个epoch没有改善")
                print(f"最佳准确率: {self.best_test_acc:.4f}")
                break

        # 最终保存
        self.save_model(epoch, "final_model")

        print("\n训练完成!")
        print(f"最终准确率: {test_acc:.4f}")
        print(f"最佳准确率: {self.best_test_acc:.4f}")

        return self.best_test_acc

    def save_model(self, epoch: int, tag: str = ""):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss_history': self.train_loss_history,
            'train_acc_history': self.train_acc_history,
            'test_loss_history': self.test_loss_history,
            'test_acc_history': self.test_acc_history,
            'best_test_acc': self.best_test_acc,
            'config': {
                'batch_size': self.batch_size,
                'max_seq_len': self.max_seq_len,
                'learning_rate': self.learning_rate,
                'num_workers': self.num_workers,
                'use_amp': self.use_amp
            },
            'tag': tag
        }

        torch.save(checkpoint, self.model_save_path)
        print(f"  模型已保存到: {self.model_save_path} (tag: {tag})")

    def evaluate_on_test_set(self) -> float:
        """在测试集上评估模型"""
        if self.test_loader is None:
            self.setup_data()

        if self.model is None:
            if not self.load_model():
                raise ValueError("没有可评估的模型")

        print("\n=== 在测试集上评估 ===")
        test_loss, test_acc = self.evaluate(self.test_loader)
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")

        return test_acc

    def predict_single(self, text: str):
        """单个文本预测"""
        if self.model is None:
            if not self.load_model():
                raise ValueError("没有可用的模型")

        # 加载词汇表
        vocab = Vocabulary()
        vocab.load(self.vocab_path)

        # 预处理文本
        sequence = vocab.text_to_sequence(text, self.max_seq_len)
        text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)

        # 预测
        self.model.eval()
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cpu', enabled=True):
                    outputs = self.model(text_tensor)
            else:
                outputs = self.model(text_tensor)

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        pred_id = predicted.item()

        # 获取类别名称
        try:
            import pandas as pd
            cat_mapping = pd.read_csv(
                self.cat_mapping_path, encoding='utf-8-sig')
            cat_name = cat_mapping.loc[
                cat_mapping['cat_id'] == pred_id, 'ClassName'
            ].values[0] if pred_id in cat_mapping['cat_id'].values else "未知"
        except Exception:
            cat_name = f"类别_{pred_id}"

        return pred_id, cat_name, probabilities.squeeze().cpu().numpy()


def main():
    """主函数"""
    print("优化的TextCNN训练")
    print("=" * 60)
    print("配置:")
    print("  批大小: 64 (默认32)")
    print("  数据加载工作线程: 4 (默认0)")
    print("  混合精度训练: 是")
    print("  早停耐心值: 5")
    print("  最大训练轮数: 50")
    print("=" * 60)

    # 创建训练器
    trainer = OptimizedTextCNNTrainer(
        batch_size=64,
        max_seq_len=11,
        learning_rate=0.001,
        num_epochs=50,
        patience=5,
        num_workers=4,
        use_amp=True,
        model_save_path="data/textcnn_optimized.pth",
        vocab_path="data/vocab.pkl",
        embedding_matrix_path="data/embedding_matrix.npy",
        cat_mapping_path="data/cat_id_mapping.csv"
    )

    # 开始训练
    best_acc = trainer.train()

    # 在测试集上评估
    test_acc = trainer.evaluate_on_test_set()

    print("\n训练完成!")
    print(f"最佳准确率: {best_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")

    # 测试单个预测
    print("\n=== 测试单个预测 ===")
    test_text = "99990017 尊贵 年货 礼盒 盒 9999"

    try:
        pred_id, pred_name, probs = trainer.predict_single(test_text)

        print(f"输入文本: {test_text}")
        print(f"预测类别ID: {pred_id}")
        print(f"预测类别名称: {pred_name}")

        # 获取Top-3预测
        import numpy as np
        top3_indices = np.argsort(probs)[-3:][::-1]
        for idx in top3_indices:
            print(f"  - 类别 {idx}: {probs[idx]*100:.2f}%")
    except Exception as e:
        print(f"预测时出错: {e}")

    return trainer


if __name__ == "__main__":
    main()
