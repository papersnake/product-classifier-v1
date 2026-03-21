'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 21:30:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-14 21:30:00
FilePath: \\myproject\\train_extended.py
Description: 扩展训练PyTorch TextCNN模型以达到目标准确率(94.78%)

Copyright (c) 2026 by papersnake, All Rights Reserved. 
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
import os
import sys
from typing import Tuple, Dict, List
import pickle

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from text_dataset import create_data_loaders
from text_cnn import create_textcnn_from_vocab
from embedding_manager import Vocabulary


class ExtendedTextCNNTrainer:
    """
    扩展的TextCNN模型训练器 - 添加save_best_model方法并支持继续训练
    """
    
    def __init__(self,
                 batch_size: int = 64,
                 max_seq_len: int = 11,
                 learning_rate: float = 0.001,
                 num_epochs: int = 30,
                 patience: int = 8,
                 num_workers: int = 4,
                 use_amp: bool = False,
                 model_save_path: str = "data/textcnn_extended.pth",
                 vocab_path: str = "data/vocab.pkl",
                 embedding_matrix_path: str = "data/embedding_matrix.npy",
                 cat_mapping_path: str = "data/cat_id_mapping.csv",
                 start_from_checkpoint: str = None):
        """
        初始化训练器
        
        Parameters:
        start_from_checkpoint: 从指定检查点继续训练
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
        self.start_from_checkpoint = start_from_checkpoint
        
        # 设备设置
        self.device = torch.device('cpu')
        
        # 训练历史
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []
        
        # 最佳准确率
        self.best_test_acc = 0.0
        self.best_epoch = 0
        
        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # 数据加载器
        self.train_loader = None
        self.test_loader = None
        
        print("扩展训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"批大小: {batch_size}")
        print(f"数据加载工作线程: {num_workers}")
        print(f"混合精度训练: {use_amp}")
        print(f"最大训练轮数: {num_epochs}")
        print(f"早停耐心值: {patience}")
        if start_from_checkpoint:
            print(f"从检查点继续训练: {start_from_checkpoint}")
    
    def setup_data(self):
        """设置数据加载器"""
        print("=== 设置数据集 ===")
        
        # 创建数据加载器
        self.train_loader, self.test_loader = create_data_loaders(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            num_workers=self.num_workers
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
        num_classes = len(pd.read_csv(self.cat_mapping_path, encoding='utf-8-sig')) if os.path.exists(self.cat_mapping_path) else 246
        
        print(f"词汇表大小: {vocab_size}")
        print(f"类别数量: {num_classes}")
        
        # 创建模型
        self.model = create_textcnn_from_vocab(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_matrix_path=self.embedding_matrix_path,
            trainable_embeddings=True
        )
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        print(f"模型: {type(self.model).__name__}")
        print(f"优化器: Adam (lr={self.learning_rate})")
        print(f"损失函数: {type(self.criterion).__name__}")
        print(f"学习率调度器: ReduceLROnPlateau")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # 检查词汇表大小是否匹配
                if self.model is None:
                    self.setup_model()
                
                # 恢复模型
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
                
                # 恢复训练历史
                self.train_loss_history = checkpoint.get('train_loss_history', [])
                self.train_acc_history = checkpoint.get('train_acc_history', [])
                self.test_loss_history = checkpoint.get('test_loss_history', [])
                self.test_acc_history = checkpoint.get('test_acc_history', [])
                
                self.best_test_acc = max(self.test_acc_history) if self.test_acc_history else 0.0
                self.best_epoch = checkpoint.get('epoch', 0)
                
                print(f"检查点加载成功，最佳准确率: {self.best_test_acc:.4f}")
                print(f"已训练epoch数: {len(self.train_loss_history)}")
                return True
            except Exception as e:
                print(f"加载检查点时出错: {e}")
                print("将从头开始训练")
                return False
        else:
            print(f"检查点不存在: {checkpoint_path}")
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
            labels = labels.squeeze(1).to(self.device)
            
            # 前向传播
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
                samples_per_sec = (batch_idx + 1) * self.batch_size / elapsed if elapsed > 0 else 0
                
                print(f"  批次 {batch_idx:4d}/{len(self.train_loader)}: "
                      f"损失={loss.item():.4f}, 准确率={batch_acc:.4f}, "
                      f"速度={samples_per_sec:.1f}样本/秒")
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        epoch_time = time.time() - start_time
        print(f"  Epoch {epoch}训练完成: 损失={avg_loss:.4f}, 准确率={avg_acc:.4f}, 时间={epoch_time:.2f}秒")
        
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
        print("=== 开始扩展训练 ===")
        
        # 设置数据和模型
        self.setup_data()
        self.setup_model()
        
        # 尝试加载现有模型
        if self.start_from_checkpoint and os.path.exists(self.start_from_checkpoint):
            self.load_checkpoint(self.start_from_checkpoint)
        else:
            print("从头开始训练")
        
        # 早停计数器
        patience_counter = 0
        
        # 起始epoch
        start_epoch = len(self.train_loss_history) + 1
        total_epochs = start_epoch - 1 + self.num_epochs
        
        # 训练循环
        for epoch in range(start_epoch, total_epochs + 1):
            print(f"\nEpoch {epoch}/{total_epochs}:")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            
            # 评估
            test_loss, test_acc = self.evaluate(self.test_loader)
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_acc)
            
            print(f"  Epoch {epoch}测试结果: 损失={test_loss:.4f}, 准确率={test_acc:.4f}")
            
            # 更新学习率
            self.scheduler.step(test_acc)
            
            # 保存最佳模型
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_epoch = epoch
                patience_counter = 0
                self.save_model(epoch, f"epoch_{epoch}_acc_{test_acc:.4f}")
                print(f"  [OK] 新的最佳模型已保存! 准确率: {test_acc:.4f}")
            else:
                patience_counter += 1
                print(f"  [WAIT] 早停计数器: {patience_counter}/{self.patience}")
            
            # 检查早停
            if patience_counter >= self.patience:
                print(f"\n[WARN] 早停触发! 连续{self.patience}个epoch没有改善")
                print(f"最佳准确率: {self.best_test_acc:.4f} (epoch {self.best_epoch})")
                break
        
        # 最终保存
        self.save_model(epoch, "final_model")
        
        print(f"\n训练完成!")
        print(f"最终准确率: {test_acc:.4f}")
        print(f"最佳准确率: {self.best_test_acc:.4f} (epoch {self.best_epoch})")
        
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
            'best_epoch': self.best_epoch,
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
    
    def save_best_model(self, path: str):
        """保存最佳模型（别名）"""
        self.save_model(self.best_epoch, "best_model")
        # 也可以复制文件到指定路径
        import shutil
        shutil.copy2(self.model_save_path, path)
        print(f"最佳模型已复制到: {path}")
    
    def evaluate_on_test_set(self) -> float:
        """在测试集上评估模型"""
        if self.test_loader is None:
            self.setup_data()
        
        if self.model is None:
            if not self.load_checkpoint(self.model_save_path):
                raise ValueError("没有可评估的模型")
        
        print("\n=== 在测试集上评估 ===")
        test_loss, test_acc = self.evaluate(self.test_loader)
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        
        return test_acc


def main():
    """主函数"""
    print("扩展训练PyTorch TextCNN模型")
    print("=" * 60)
    print("目标: 达到或超过94.78%的准确率")
    print("当前最佳: 93.37% (textcnn_quick.pth)")
    print("差距: -1.41%")
    print("=" * 60)
    
    # 创建训练器
    trainer = ExtendedTextCNNTrainer(
        batch_size=64,
        max_seq_len=11,
        learning_rate=0.0005,  # 降低学习率进行微调
        num_epochs=30,         # 最多30个epoch
        patience=8,            # 早停耐心值
        num_workers=4,
        use_amp=False,         # CPU上禁用混合精度
        model_save_path="data/textcnn_extended.pth",
        vocab_path="data/vocab.pkl",
        embedding_matrix_path="data/embedding_matrix.npy",
        cat_mapping_path="data/cat_id_mapping.csv",
        start_from_checkpoint="data/textcnn_quick.pth"  # 从快速训练结果继续
    )
    
    # 开始训练
    best_acc = trainer.train()
    
    # 在测试集上评估
    test_acc = trainer.evaluate_on_test_set()
    
    print(f"\n训练完成!")
    print(f"最佳准确率: {best_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"目标准确率: 0.9478 (94.78%)")
    print(f"差距: {test_acc - 0.9478:+.4f}")
    
    # 测试单个预测
    print("\n=== 测试单个预测 ===")
    test_text = "99990017 尊贵 年货 礼盒 盒 9999"
    
    try:
        # 加载词汇表
        vocab = Vocabulary()
        vocab.load("data/vocab.pkl")
        
        # 创建模型（简化版本）
        model = create_textcnn_from_vocab(
            vocab_size=vocab.get_vocab_size(),
            num_classes=246,
            embedding_matrix_path="data/embedding_matrix.npy",
            trainable_embeddings=False
        )
        
        checkpoint = torch.load("data/textcnn_extended.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 预处理文本
        sequence = vocab.text_to_sequence(test_text, 11)
        text_tensor = torch.LongTensor(sequence).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(text_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        pred_id = predicted.item()
        
        # 获取类别名称
        cat_mapping = pd.read_csv("data/cat_id_mapping.csv", encoding='utf-8-sig')
        cat_name = cat_mapping.loc[
            cat_mapping['cat_id'] == pred_id, 'ClassName'
        ].values[0] if pred_id in cat_mapping['cat_id'].values else "未知"
        
        print(f"输入文本: {test_text}")
        print(f"预测类别ID: {pred_id}")
        print(f"预测类别名称: {cat_name}")
        
        # 获取Top-3预测
        probs = probabilities.squeeze().cpu().numpy()
        top3_indices = np.argsort(probs)[-3:][::-1]
        for idx in top3_indices:
            cat_name = cat_mapping.loc[
                cat_mapping['cat_id'] == idx, 'ClassName'
            ].values[0] if idx in cat_mapping['cat_id'].values else "未知"
            print(f"  - 类别 {idx} ({cat_name}): {probs[idx]*100:.2f}%")
    except Exception as e:
        print(f"预测时出错: {e}")
    
    return trainer


if __name__ == "__main__":
    main()