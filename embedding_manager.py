'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 15:45:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-21 12:05:17
FilePath: \\myproject\\embedding_manager.py
Description: 词汇表构建与预训练词向量加载管理器

Copyright (c) 2026 by papersnake, All Rights Reserved.
'''

import numpy as np
import pickle
from collections import Counter
from typing import List, Optional
import torch
import torch.nn as nn

# 可选导入gensim的KeyedVectors（用于预训练词向量）
try:
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    KeyedVectors = None  # 类型占位符


class Vocabulary:
    """
    词汇表类，用于构建词汇表和词索引映射
    """

    def __init__(self, min_freq: int = 5):
        """
        初始化词汇表

        Parameters:
        min_freq: 词的最小出现频次
        """
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.embedding_dim = 300  # 默认嵌入维度

        # 特殊标记
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.pad_idx = 0
        self.unk_idx = 1

        # 初始化特殊标记
        self.word2idx[self.pad_token] = self.pad_idx
        self.word2idx[self.unk_token] = self.unk_idx
        self.idx2word[self.pad_idx] = self.pad_token
        self.idx2word[self.unk_idx] = self.unk_token

        self.vocab_size = 2  # 初始大小（包含特殊标记）

    def build_from_texts(self, texts: List[str]) -> None:
        """
        从文本列表构建词汇表

        Parameters:
        texts: 预处理后的文本列表（已分词，空格分隔）
        """
        # 统计词频
        word_freq = Counter()
        for text in texts:
            if isinstance(text, str):
                words = text.split()
                word_freq.update(words)

        # 过滤低频词
        filtered_words = {word: freq for word, freq in word_freq.items()
                          if freq >= self.min_freq}

        # 构建词汇表
        for word in sorted(filtered_words.keys()):
            if word not in self.word2idx:
                idx = self.vocab_size
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.vocab_size += 1

        # 保存词频
        self.word_freq = Counter(filtered_words)

        print(f"词汇表构建完成，总词数: {self.vocab_size}")
        print(f"过滤后词数（频次>={self.min_freq}）: {len(filtered_words)}")

    def text_to_sequence(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """
        将文本转换为索引序列

        Parameters:
        text: 预处理后的文本（已分词，空格分隔）
        max_len: 最大序列长度，超出部分截断，不足部分填充

        Returns:
        List[int]: 索引序列
        """
        words = text.split() if isinstance(text, str) else []

        # 转换为索引
        indices = [self.word2idx.get(word, self.unk_idx) for word in words]

        # 处理序列长度
        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices = indices + [self.pad_idx] * (max_len - len(indices))

        return indices

    def sequences_to_texts(self, sequences: List[List[int]]) -> List[str]:
        """
        将索引序列转换回文本

        Parameters:
        sequences: 索引序列列表

        Returns:
        List[str]: 文本列表
        """
        texts = []
        for seq in sequences:
            words = [self.idx2word.get(idx, self.unk_token) for idx in seq
                     if idx != self.pad_idx and idx in self.idx2word]
            texts.append(' '.join(words))
        return texts

    def save(self, path: str) -> None:
        """
        保存词汇表到文件

        Parameters:
        path: 保存路径
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'min_freq': self.min_freq,
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'pad_idx': self.pad_idx,
            'unk_idx': self.unk_idx
        }

        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)

        print(f"词汇表已保存到: {path}")

    def load(self, path: str) -> None:
        """
        从文件加载词汇表

        Parameters:
        path: 加载路径
        """
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)

        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.word_freq = Counter(vocab_data['word_freq'])
        self.min_freq = vocab_data['min_freq']
        self.embedding_dim = vocab_data['embedding_dim']
        self.vocab_size = vocab_data['vocab_size']
        self.pad_token = vocab_data['pad_token']
        self.unk_token = vocab_data['unk_token']
        self.pad_idx = vocab_data['pad_idx']
        self.unk_idx = vocab_data['unk_idx']

        print(f"词汇表已加载，总词数: {self.vocab_size}")

    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.vocab_size

    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.embedding_dim


class EmbeddingManager:
    """
    嵌入管理器，用于加载预训练词向量并创建嵌入矩阵
    """

    def __init__(self, vocab: Vocabulary, embedding_dim: int = 300):
        """
        初始化嵌入管理器

        Parameters:
        vocab: Vocabulary对象
        embedding_dim: 嵌入维度
        """
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embedding_matrix = None
        self.pretrained_vectors = None

    def load_pretrained_vectors(self, vector_path: str,
                                binary: bool = True) -> None:
        """
        加载预训练词向量（支持Word2Vec、FastText等格式）

        Parameters:
        vector_path: 词向量文件路径
        binary: 是否为二进制格式
        """
        print(f"加载预训练词向量: {vector_path}")

        if not GENSIM_AVAILABLE:
            print("警告: gensim库未安装，无法加载预训练词向量")
            print("将使用随机初始化嵌入")
            self.pretrained_vectors = None
            return

        try:
            # 使用gensim加载词向量
            self.pretrained_vectors = KeyedVectors.load_word2vec_format(
                vector_path, binary=binary
            )
            print(f"词向量加载成功，词汇量: {len(self.pretrained_vectors)}")
            print(f"词向量维度: {self.pretrained_vectors.vector_size}")

            # 更新嵌入维度
            self.embedding_dim = self.pretrained_vectors.vector_size
            self.vocab.embedding_dim = self.embedding_dim

        except Exception as e:
            print(f"加载预训练词向量失败: {e}")
            print("将使用随机初始化嵌入")
            self.pretrained_vectors = None

    def build_embedding_matrix(self) -> np.ndarray:
        """
        构建嵌入矩阵

        Returns:
        np.ndarray: 嵌入矩阵，形状为(vocab_size, embedding_dim)
        """
        vocab_size = self.vocab.get_vocab_size()
        embedding_matrix = np.random.normal(
            scale=0.1,
            size=(vocab_size, self.embedding_dim)
        )

        # 特殊标记使用零向量
        embedding_matrix[self.vocab.pad_idx] = np.zeros(self.embedding_dim)

        if self.pretrained_vectors is not None:
            # 使用预训练词向量初始化
            print("使用预训练词向量初始化嵌入矩阵...")
            loaded_count = 0

            for word, idx in self.vocab.word2idx.items():
                if word in self.pretrained_vectors:
                    embedding_matrix[idx] = self.pretrained_vectors[word]
                    loaded_count += 1
                elif word.lower() in self.pretrained_vectors:
                    embedding_matrix[idx] = self.pretrained_vectors[word.lower()]
                    loaded_count += 1

            print(f"预训练词向量覆盖: {loaded_count}/{vocab_size} 个词")
            print(f"覆盖率: {loaded_count/vocab_size*100:.2f}%")
        else:
            print("使用随机初始化嵌入矩阵")

        self.embedding_matrix = embedding_matrix
        return embedding_matrix

    def get_embedding_matrix(self) -> np.ndarray:
        """获取嵌入矩阵"""
        if self.embedding_matrix is None:
            self.build_embedding_matrix()
        return self.embedding_matrix

    def create_embedding_layer(self, trainable: bool = True) -> nn.Embedding:
        """
        创建PyTorch嵌入层

        Parameters:
        trainable: 是否可训练

        Returns:
        nn.Embedding: PyTorch嵌入层
        """
        if self.embedding_matrix is None:
            self.build_embedding_matrix()

        vocab_size = self.vocab.get_vocab_size()
        embedding_dim = self.embedding_dim

        # 创建嵌入层
        embedding_layer = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=self.vocab.pad_idx)

        # 设置权重
        embedding_layer.weight.data.copy_(
            torch.from_numpy(self.embedding_matrix))

        # 设置是否可训练
        embedding_layer.weight.requires_grad = trainable

        print(f"嵌入层创建完成，大小: {vocab_size}x{embedding_dim}")
        print(f"嵌入层可训练: {trainable}")

        return embedding_layer

    def save_embedding_matrix(self, path: str) -> None:
        """
        保存嵌入矩阵

        Parameters:
        path: 保存路径
        """
        if self.embedding_matrix is not None:
            np.save(path, self.embedding_matrix)
            print(f"嵌入矩阵已保存到: {path}")

    def load_embedding_matrix(self, path: str) -> None:
        """
        加载嵌入矩阵

        Parameters:
        path: 加载路径
        """
        self.embedding_matrix = np.load(path)
        print(f"嵌入矩阵已加载，形状: {self.embedding_matrix.shape}")


def test_vocabulary():
    """测试词汇表功能"""
    # 创建示例文本
    texts = [
        "苹果 手机 新品 发布",
        "华为 手机 新品 发布",
        "小米 手机 新品 发布",
        "苹果 电脑 新品 发布"
    ]

    # 构建词汇表
    vocab = Vocabulary(min_freq=1)
    vocab.build_from_texts(texts)

    print(f"词汇表大小: {vocab.get_vocab_size()}")
    print(f"词到索引映射: {list(vocab.word2idx.items())[:10]}")

    # 测试文本转换
    test_text = "苹果 手机 新品"
    sequence = vocab.text_to_sequence(test_text)
    print(f"文本 '{test_text}' -> 序列: {sequence}")

    # 测试序列转文本
    decoded_text = vocab.sequences_to_texts([sequence])[0]
    print(f"序列 {sequence} -> 文本: '{decoded_text}'")

    return vocab


def test_embedding_manager():
    """测试嵌入管理器功能"""
    vocab = test_vocabulary()

    # 创建嵌入管理器
    embedding_manager = EmbeddingManager(vocab, embedding_dim=300)

    # 构建嵌入矩阵（随机初始化）
    embedding_matrix = embedding_manager.build_embedding_matrix()
    print(f"嵌入矩阵形状: {embedding_matrix.shape}")

    # 创建PyTorch嵌入层
    embedding_layer = embedding_manager.create_embedding_layer(trainable=True)
    print(f"嵌入层: {embedding_layer}")

    # 测试前向传播
    test_sequence = torch.LongTensor(
        [[vocab.word2idx['苹果'], vocab.word2idx['手机']]])
    embedded = embedding_layer(test_sequence)
    print(f"输入序列: {test_sequence}")
    print(f"嵌入后形状: {embedded.shape}")

    return embedding_manager


if __name__ == "__main__":
    print("=== 测试词汇表 ===")
    vocab = test_vocabulary()

    print("\n=== 测试嵌入管理器 ===")
    embedding_manager = test_embedding_manager()
