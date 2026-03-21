'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 16:10:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 14:36:46
FilePath: \myproject\text_cnn.py
Description: TextCNN模型架构定义

Copyright (c) 2026 by papersnake, All Rights Reserved. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class TextCNN(nn.Module):
    """
    TextCNN模型用于文本分类
    参考：Kim Y. (2014) Convolutional Neural Networks for Sentence Classification
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 num_classes: int = 246,
                 filter_sizes: List[int] = [3, 4, 5],
                 num_filters: int = 100,
                 dropout_rate: float = 0.5,
                 embedding_matrix: Optional[np.ndarray] = None,
                 trainable_embeddings: bool = True):
        """
        初始化TextCNN模型

        Parameters:
        vocab_size: 词汇表大小
        embedding_dim: 嵌入维度
        num_classes: 类别数量
        filter_sizes: 卷积核尺寸列表
        num_filters: 每个尺寸的过滤器数量
        dropout_rate: Dropout概率
        embedding_matrix: 预训练嵌入矩阵（可选）
        trainable_embeddings: 嵌入层是否可训练
        """
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        # 1. 嵌入层
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0  # 假设pad_idx=0
        )

        # 如果提供了预训练嵌入矩阵，则加载
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(embedding_matrix))

        # 设置嵌入层是否可训练
        self.embedding.weight.requires_grad = trainable_embeddings

        # 2. 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,  # 文本输入通道数为1
                out_channels=num_filters,
                kernel_size=(fs, embedding_dim)
            )
            for fs in filter_sizes
        ])

        # 3. Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 4. 全连接层
        # 卷积后的特征维度：len(filter_sizes) * num_filters
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

        # 初始化权重
        self._init_weights()

        print(f"TextCNN模型初始化完成:")
        print(f"  - 词汇表大小: {vocab_size}")
        print(f"  - 嵌入维度: {embedding_dim}")
        print(f"  - 类别数量: {num_classes}")
        print(f"  - 卷积核尺寸: {filter_sizes}")
        print(f"  - 每个尺寸过滤器数: {num_filters}")
        print(f"  - Dropout率: {dropout_rate}")
        print(f"  - 嵌入层可训练: {trainable_embeddings}")

    def _init_weights(self):
        """初始化模型权重"""
        # 嵌入层已初始化（可能使用预训练权重）

        # 初始化卷积层权重
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

        # 初始化全连接层权重
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Parameters:
        x: 输入张量，形状为(batch_size, seq_len)

        Returns:
        torch.Tensor: 输出logits，形状为(batch_size, num_classes)
        """
        batch_size = x.size(0)

        # 1. 嵌入层
        # x: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # 2. 添加通道维度
        # embedded: (batch_size, seq_len, embedding_dim) -> (batch_size, 1, seq_len, embedding_dim)
        embedded = embedded.unsqueeze(1)

        # 3. 卷积层和池化层
        conv_outputs = []
        for conv in self.convs:
            # 卷积: (batch_size, 1, seq_len, embedding_dim) -> (batch_size, num_filters, seq_len-fs+1, 1)
            conv_out = F.relu(conv(embedded))

            # 最大池化: (batch_size, num_filters, seq_len-fs+1, 1) -> (batch_size, num_filters, 1, 1)
            pooled = F.max_pool2d(conv_out, kernel_size=(conv_out.size(2), 1))

            # 去除多余维度: (batch_size, num_filters, 1, 1) -> (batch_size, num_filters)
            conv_outputs.append(pooled.squeeze(3).squeeze(2))

        # 4. 拼接所有卷积层的输出
        # cat_output: (batch_size, len(filter_sizes) * num_filters)
        cat_output = torch.cat(conv_outputs, dim=1)

        # 5. Dropout
        cat_output = self.dropout(cat_output)

        # 6. 全连接层
        logits = self.fc(cat_output)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别

        Parameters:
        x: 输入张量，形状为(batch_size, seq_len)

        Returns:
        torch.Tensor: 预测的类别索引，形状为(batch_size,)
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别概率

        Parameters:
        x: 输入张量，形状为(batch_size, seq_len)

        Returns:
        torch.Tensor: 类别概率，形状为(batch_size, num_classes)
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

    def get_embedding_layer(self) -> nn.Embedding:
        """获取嵌入层"""
        return self.embedding

    def get_num_parameters(self) -> int:
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """获取可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_textcnn_from_vocab(vocab_size: int,
                              num_classes: int,
                              embedding_matrix_path: str = "data/embedding_matrix.npy",
                              trainable_embeddings: bool = True) -> TextCNN:
    """
    从词汇表和嵌入矩阵创建TextCNN模型

    Parameters:
    vocab_size: 词汇表大小
    num_classes: 类别数量
    embedding_matrix_path: 嵌入矩阵文件路径
    trainable_embeddings: 嵌入层是否可训练

    Returns:
    TextCNN: TextCNN模型实例
    """
    # 加载嵌入矩阵
    embedding_matrix = None
    if embedding_matrix_path and embedding_matrix_path.endswith('.npy'):
        try:
            embedding_matrix = np.load(embedding_matrix_path)
            print(f"嵌入矩阵已加载，形状: {embedding_matrix.shape}")

            # 检查嵌入矩阵维度
            embedding_dim = embedding_matrix.shape[1]

            # 创建模型
            model = TextCNN(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                filter_sizes=[3, 4, 5],
                num_filters=100,
                dropout_rate=0.5,
                embedding_matrix=embedding_matrix,
                trainable_embeddings=trainable_embeddings
            )

            return model

        except Exception as e:
            print(f"加载嵌入矩阵失败: {e}")
            print("将使用随机初始化嵌入")

    # 如果没有嵌入矩阵或加载失败，使用随机初始化
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=300,
        num_classes=num_classes,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        dropout_rate=0.5,
        embedding_matrix=None,
        trainable_embeddings=trainable_embeddings
    )

    return model


def test_textcnn():
    """测试TextCNN模型"""
    print("=== 测试TextCNN模型 ===")

    # 模拟参数
    vocab_size = 7466
    num_classes = 246
    batch_size = 4
    seq_len = 11

    # 创建模型
    model = create_textcnn_from_vocab(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_matrix_path="data/embedding_matrix.npy",
        trainable_embeddings=True
    )

    print(f"\n模型参数量: {model.get_num_parameters():,}")
    print(f"可训练参数量: {model.get_trainable_parameters():,}")

    # 测试前向传播
    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\n输入张量形状: {input_tensor.shape}")

    # 前向传播
    logits = model(input_tensor)
    print(f"输出logits形状: {logits.shape}")

    # 预测
    predictions = model.predict(input_tensor)
    print(f"预测类别形状: {predictions.shape}")
    print(f"预测类别: {predictions}")

    # 概率预测
    probabilities = model.predict_proba(input_tensor)
    print(f"概率预测形状: {probabilities.shape}")
    print(f"概率总和: {probabilities.sum(dim=1)}")

    # 测试嵌入层
    embedding_layer = model.get_embedding_layer()
    print(f"\n嵌入层权重形状: {embedding_layer.weight.shape}")
    print(f"嵌入层是否可训练: {embedding_layer.weight.requires_grad}")

    return model


if __name__ == "__main__":
    test_textcnn()
