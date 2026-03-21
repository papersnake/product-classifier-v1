'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 15:50:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-14 15:50:00
FilePath: \\myproject\\build_vocab.py
Description: 构建词汇表并保存

Copyright (c) 2026 by papersnake, All Rights Reserved. 
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import DataPreprocessor
from embedding_manager import Vocabulary, EmbeddingManager
import pickle
import numpy as np


def build_vocabulary_from_data():
    """
    从处理后的数据构建词汇表
    """
    print("=== 构建词汇表 ===")
    
    # 1. 加载和处理数据
    preprocessor = DataPreprocessor()
    products = preprocessor.load_and_filter_data()
    cat_id_df = preprocessor.create_category_mapping()
    products_processed = preprocessor.apply_preprocessing()
    
    # 2. 准备训练数据（仅使用训练集构建词汇表）
    X_train, X_test, y_train, y_test, idx_train, idx_test = preprocessor.prepare_training_data()
    
    print(f"训练集文本数量: {len(X_train)}")
    print(f"测试集文本数量: {len(X_test)}")
    
    # 3. 构建词汇表
    vocab = Vocabulary(min_freq=5)  # 与TF-IDF的min_df=5保持一致
    vocab.build_from_texts(X_train.tolist())
    
    # 4. 显示词汇表信息
    print(f"\n词汇表信息:")
    print(f"- 总词数（含特殊标记）: {vocab.get_vocab_size()}")
    print(f"- 实际词数: {vocab.get_vocab_size() - 2}")
    print(f"- 嵌入维度: {vocab.get_embedding_dim()}")
    
    # 5. 保存词汇表
    vocab_path = "data/vocab.pkl"
    vocab.save(vocab_path)
    
    # 6. 创建嵌入管理器（暂时使用随机初始化）
    embedding_manager = EmbeddingManager(vocab, embedding_dim=300)
    
    # 7. 构建嵌入矩阵
    embedding_matrix = embedding_manager.build_embedding_matrix()
    
    # 8. 保存嵌入矩阵
    embedding_path = "data/embedding_matrix.npy"
    embedding_manager.save_embedding_matrix(embedding_path)
    
    # 9. 保存类别映射（如果不存在）
    if not os.path.exists("data/cat_id_mapping.csv"):
        preprocessor.save_category_mapping("data/cat_id_mapping.csv")
    
    # 10. 测试词汇表功能
    print("\n=== 词汇表测试 ===")
    test_texts = [
        "苹果 手机 新品 发布",
        "华为 手机 新品 发布"
    ]
    
    for text in test_texts:
        sequence = vocab.text_to_sequence(text, max_len=20)
        print(f"文本: {text[:30]}...")
        print(f"序列: {sequence}")
        print(f"序列长度: {len(sequence)}")
        print()
    
    # 11. 统计序列长度
    print("=== 序列长度统计 ===")
    train_sequences = [vocab.text_to_sequence(text) for text in X_train[:1000]]
    seq_lengths = [len(seq) for seq in train_sequences]
    
    print(f"平均序列长度: {np.mean(seq_lengths):.2f}")
    print(f"最大序列长度: {np.max(seq_lengths)}")
    print(f"最小序列长度: {np.min(seq_lengths)}")
    print(f"95%分位数: {np.percentile(seq_lengths, 95):.2f}")
    
    # 建议的最大序列长度（覆盖95%的样本）
    suggested_max_len = int(np.percentile(seq_lengths, 95))
    print(f"建议的最大序列长度: {suggested_max_len}")
    
    # 保存序列长度统计
    with open("data/seq_length_stats.txt", "w") as f:
        f.write(f"平均序列长度: {np.mean(seq_lengths):.2f}\n")
        f.write(f"最大序列长度: {np.max(seq_lengths)}\n")
        f.write(f"最小序列长度: {np.min(seq_lengths)}\n")
        f.write(f"95%分位数: {np.percentile(seq_lengths, 95):.2f}\n")
        f.write(f"建议的最大序列长度: {suggested_max_len}\n")
    
    print(f"\n词汇表已保存到: {vocab_path}")
    print(f"嵌入矩阵已保存到: {embedding_path}")
    print(f"序列长度统计已保存到: data/seq_length_stats.txt")
    
    return vocab, embedding_manager, suggested_max_len


def load_pretrained_vectors(vocab_path: str = "data/vocab.pkl", 
                           vector_path: str = None):
    """
    加载预训练词向量并更新嵌入矩阵
    
    Parameters:
    vocab_path: 词汇表路径
    vector_path: 预训练词向量文件路径
    """
    if vector_path is None or not os.path.exists(vector_path):
        print(f"预训练词向量文件不存在: {vector_path}")
        print("请下载中文预训练词向量文件，例如：")
        print("1. 腾讯AI Lab中文词向量: https://ai.tencent.com/ailab/nlp/zh/embedding.html")
        print("2. 中文维基百科Word2Vec: https://github.com/Embedding/Chinese-Word-Vectors")
        print("3. FastText中文词向量: https://fasttext.cc/docs/en/crawl-vectors.html")
        return None
    
    # 加载词汇表
    vocab = Vocabulary()
    vocab.load(vocab_path)
    
    # 创建嵌入管理器并加载预训练词向量
    embedding_manager = EmbeddingManager(vocab)
    embedding_manager.load_pretrained_vectors(vector_path, binary=True)
    
    # 构建嵌入矩阵
    embedding_matrix = embedding_manager.build_embedding_matrix()
    
    # 保存新的嵌入矩阵
    new_embedding_path = "data/embedding_matrix_pretrained.npy"
    embedding_manager.save_embedding_matrix(new_embedding_path)
    
    print(f"预训练嵌入矩阵已保存到: {new_embedding_path}")
    
    return embedding_manager


if __name__ == "__main__":
    # 构建词汇表（随机初始化）
    vocab, embedding_manager, max_len = build_vocabulary_from_data()
    
    print(f"\n=== 下一步 ===")
    print("1. 词汇表和嵌入矩阵已构建完成")
    print("2. 建议的最大序列长度: ", max_len)
    print("3. 如需使用预训练词向量，请运行:")
    print("   python build_vocab.py --pretrained <词向量文件路径>")
    print("4. 接下来可以开始训练PyTorch模型")