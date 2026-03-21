'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 16:00:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-14 16:00:00
FilePath: \myproject\text_dataset.py
Description: PyTorch文本分类数据集实现

Copyright (c) 2026 by papersnake, All Rights Reserved. 
'''

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import Tuple, List, Optional
from embedding_manager import Vocabulary


class TextClassificationDataset(Dataset):
    """
    文本分类数据集
    """
    
    def __init__(self, 
                 data_path: str = "data/products_processed.csv",
                 vocab_path: str = "data/vocab.pkl",
                 cat_mapping_path: str = "data/cat_id_mapping.csv",
                 max_seq_len: int = 11,
                 mode: str = 'train',
                 test_size: float = 0.25,
                 random_state: int = 0):
        """
        初始化数据集
        
        Parameters:
        data_path: 处理后的数据文件路径
        vocab_path: 词汇表文件路径
        cat_mapping_path: 类别映射文件路径
        max_seq_len: 最大序列长度
        mode: 数据集模式 ('train', 'test', 'all')
        test_size: 测试集比例（仅在mode为'train'或'test'时使用）
        random_state: 随机种子
        """
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.cat_mapping_path = cat_mapping_path
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.test_size = test_size
        self.random_state = random_state
        
        # 加载数据
        self.data = pd.read_csv(data_path, encoding='utf-8-sig')
        self.cat_mapping = pd.read_csv(cat_mapping_path, encoding='utf-8-sig')
        
        # 加载词汇表
        self.vocab = Vocabulary()
        self.vocab.load(vocab_path)
        
        # 准备数据
        self.texts, self.labels, self.indices = self._prepare_data()
        
        print(f"数据集初始化完成: {mode}")
        print(f"样本数量: {len(self.texts)}")
        print(f"类别数量: {len(self.cat_mapping)}")
        print(f"最大序列长度: {max_seq_len}")
    
    def _prepare_data(self) -> Tuple[List[str], List[int], List[int]]:
        """
        准备数据，根据模式划分训练集和测试集
        
        Returns:
        tuple: (文本列表, 标签列表, 索引列表)
        """
        # 确保数据包含必要的列
        required_cols = ['preprocessed', 'cat_id']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"数据缺少必要列: {col}")
        
        # 划分训练集和测试集
        if self.mode in ['train', 'test']:
            from sklearn.model_selection import train_test_split
            
            X = self.data['preprocessed'].values
            y = self.data['cat_id'].values
            indices = self.data.index.values
            
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, indices,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
            
            if self.mode == 'train':
                return X_train.tolist(), y_train.tolist(), idx_train.tolist()
            else:
                return X_test.tolist(), y_test.tolist(), idx_test.tolist()
        else:
            # 使用所有数据
            X = self.data['preprocessed'].values.tolist()
            y = self.data['cat_id'].values.tolist()
            indices = self.data.index.values.tolist()
            return X, y, indices
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Parameters:
        idx: 样本索引
        
        Returns:
        tuple: (文本序列张量, 标签张量)
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本转换为序列
        sequence = self.vocab.text_to_sequence(text, self.max_seq_len)
        
        # 转换为张量
        text_tensor = torch.LongTensor(sequence)
        label_tensor = torch.LongTensor([label])
        
        return text_tensor, label_tensor
    
    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重（用于处理类别不平衡）
        
        Returns:
        torch.Tensor: 类别权重
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # 获取所有类别
        classes = np.arange(len(self.cat_mapping))
        
        # 计算类别权重
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=self.labels
        )
        
        return torch.FloatTensor(weights)
    
    def get_num_classes(self) -> int:
        """获取类别数量"""
        return len(self.cat_mapping)
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.vocab.get_vocab_size()
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.vocab.get_embedding_dim()
    
    def show_sample(self, idx: int = 0) -> None:
        """
        显示样本示例
        
        Parameters:
        idx: 样本索引
        """
        text = self.texts[idx]
        label = self.labels[idx]
        sequence = self.vocab.text_to_sequence(text, self.max_seq_len)
        
        print(f"样本索引: {idx}")
        print(f"原始文本: {text}")
        print(f"序列长度: {len(text.split())}")
        print(f"截断/填充后序列: {sequence}")
        print(f"序列长度: {len(sequence)}")
        print(f"标签: {label}")
        
        # 获取类别名称
        cat_name = self.cat_mapping.loc[
            self.cat_mapping['cat_id'] == label, 'ClassName'
        ].values
        if len(cat_name) > 0:
            print(f"类别名称: {cat_name[0]}")


def create_data_loaders(batch_size: int = 32,
                        max_seq_len: int = 11,
                        num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器
    
    Parameters:
    batch_size: 批大小
    max_seq_len: 最大序列长度
    num_workers: 数据加载工作线程数（CPU训练建议设为0）
    
    Returns:
    tuple: (训练数据加载器, 测试数据加载器)
    """
    # 创建训练数据集
    train_dataset = TextClassificationDataset(
        max_seq_len=max_seq_len,
        mode='train',
        test_size=0.25,
        random_state=0
    )
    
    # 创建测试数据集
    test_dataset = TextClassificationDataset(
        max_seq_len=max_seq_len,
        mode='test',
        test_size=0.25,
        random_state=0
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # CPU训练无需锁页内存
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"训练数据加载器: {len(train_loader)}个批次")
    print(f"测试数据加载器: {len(test_loader)}个批次")
    
    return train_loader, test_loader


def test_dataset():
    """测试数据集功能"""
    print("=== 测试数据集 ===")
    
    # 创建数据集
    dataset = TextClassificationDataset(
        max_seq_len=11,
        mode='all'
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {dataset.get_num_classes()}")
    print(f"词汇表大小: {dataset.get_vocab_size()}")
    print(f"嵌入维度: {dataset.get_embedding_dim()}")
    
    # 显示样本示例
    dataset.show_sample(0)
    dataset.show_sample(100)
    
    # 测试数据加载
    sample_idx = 0
    text_tensor, label_tensor = dataset[sample_idx]
    print(f"\n样本张量形状:")
    print(f"文本张量: {text_tensor.shape}")
    print(f"标签张量: {label_tensor.shape}")
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(batch_size=4)
    
    # 测试批处理
    print("\n=== 测试批处理 ===")
    for batch_idx, (texts, labels) in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"  文本张量形状: {texts.shape}")
        print(f"  标签张量形状: {labels.shape}")
        print(f"  文本张量示例: {texts[0]}")
        print(f"  标签张量示例: {labels[0]}")
        
        if batch_idx >= 2:  # 只显示前3个批次
            break
    
    return dataset, train_loader, test_loader


if __name__ == "__main__":
    test_dataset()