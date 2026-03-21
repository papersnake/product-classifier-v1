'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-21 11:00:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-21 11:00:00
FilePath: \myproject\train_fasttext_embeddings.py
Description: 使用 Gensim FastText 在产品数据上训练词向量

Copyright (c) 2026 by papersnake, All Rights Reserved. 
'''

import pandas as pd
import numpy as np
from gensim.models import FastText
import os
import pickle

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products_processed.csv")
FASTTEXT_MODEL_DIR = os.path.join(DATA_DIR, "fasttext_model")
FASTTEXT_VECTORS = os.path.join(DATA_DIR, "fasttext_vectors.txt")


def load_product_data():
    """加载产品数据"""
    print("加载产品数据...")
    df = pd.read_csv(PRODUCTS_CSV, encoding='utf-8-sig')
    texts = df['preprocessed'].dropna().astype(str).tolist()
    sentences = [text.split() for text in texts]
    print(f"加载 {len(sentences)} 条样本")
    return sentences


def train_fasttext(sentences, vector_size=100, window=5, min_count=1, epochs=10):
    """训练 FastText 模型"""
    print(f"\n训练 FastText 模型...")
    print(f"参数: vector_size={vector_size}, window={window}, min_count={min_count}, epochs={epochs}")
    
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,
        epochs=epochs
    )
    
    return model


def save_model(model, save_dir):
    """保存 FastText 模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "fasttext.model")
    model.save(model_path)
    print(f"FastText 模型已保存: {model_path}")
    
    return model_path


def export_to_word2vec_format(model, output_path):
    """导出为 word2vec 格式（供 embedding_manager.py 使用）"""
    model.wv.save_word2vec_format(output_path, binary=False)
    print(f"词向量已导出: {output_path}")
    
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"词向量数量: {len(lines) - 1}")  # 减去头部行


def analyze_model(model):
    """分析模型"""
    print("\n=== FastText 模型分析 ===")
    print(f"词表大小: {len(model.wv)}")
    print(f"向量维度: {model.wv.vector_size}")
    
    # 词频统计
    all_words = []
    for words in model.wv.index_to_key[:100]:
        all_words.append(words)
    
    print(f"\nTop 20 词: {all_words[:20]}")
    
    # 测试 OOV 处理
    print("\n=== OOV 测试 ===")
    test_words = ["新产品", "未知词XYZ", "旺旺"]
    for word in test_words:
        if word in model.wv:
            print(f"  '{word}': 已登录词, 向量范数: {np.linalg.norm(model.wv[word]):.4f}")
        else:
            print(f"  '{word}': OOV词, 使用子词信息生成向量")


def test_similarity(model):
    """测试词语相似度"""
    print("\n=== 相似度测试 ===")
    test_words = ["牛奶", "酸奶", "裤子", "内裤"]
    
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            print(f"\n与 '{word}' 最相似的词:")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.4f}")
        else:
            print(f"\n'{word}' 不在词表中")


def build_vocab_from_fasttext(model, output_path):
    """构建与 FastText 兼容的词表"""
    word2idx = {}
    idx2word = {}
    
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    idx2word[0] = '<PAD>'
    idx2word[1] = '<UNK>'
    
    for idx, word in enumerate(model.wv.index_to_key, start=2):
        word2idx[word] = idx
        idx2word[idx] = word
    
    vocab_data = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': len(word2idx),
        'embedding_dim': model.wv.vector_size,
        'pad_token': '<PAD>',
        'unk_token': '<UNK>',
        'pad_idx': 0,
        'unk_idx': 1
    }
    
    vocab_path = os.path.join(DATA_DIR, "fasttext_vocab.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    
    print(f"词表已保存: {vocab_path}")
    print(f"词表大小: {len(word2idx)}")
    
    return vocab_data


def main():
    print("=" * 60)
    print("FastText 词向量训练")
    print("=" * 60)
    
    # 1. 加载数据
    sentences = load_product_data()
    
    # 2. 训练模型
    model = train_fasttext(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        epochs=10
    )
    
    # 3. 保存模型
    save_model(model, FASTTEXT_MODEL_DIR)
    
    # 4. 导出词向量
    export_to_word2vec_format(model, FASTTEXT_VECTORS)
    
    # 5. 分析模型
    analyze_model(model)
    
    # 6. 相似度测试
    test_similarity(model)
    
    # 7. 构建词表
    build_vocab_from_fasttext(model, os.path.join(DATA_DIR, "fasttext_vocab.pkl"))
    
    print("\n" + "=" * 60)
    print("FastText 训练完成!")
    print(f"词向量文件: {FASTTEXT_VECTORS}")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    main()
