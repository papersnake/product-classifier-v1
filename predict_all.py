"""
预测 products_processed.csv 中的所有商品，找出预测与实际不同的商品
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FASTTEXT_VECTORS = os.path.join(DATA_DIR, "fasttext_vectors.txt")
FASTTEXT_VOCAB = os.path.join(DATA_DIR, "fasttext_vocab.pkl")
CAT_MAPPING_PATH = os.path.join(DATA_DIR, "cat_id_mapping.csv")
MODEL_PATH = os.path.join(DATA_DIR, "textcnn_fasttext.pth")
DATA_PATH = os.path.join(DATA_DIR, "products_processed.csv")


def load_fasttext_vocab():
    with open(FASTTEXT_VOCAB, 'rb') as f:
        return pickle.load(f)


def load_fasttext_vectors():
    vectors = {}
    embedding_dim = None
    with open(FASTTEXT_VECTORS, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split()
        embedding_dim = int(header[1])
        for line in f:
            parts = line.strip().split()
            if len(parts) >= embedding_dim + 1:
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vectors[word] = vector
    return vectors, embedding_dim


def build_embedding_matrix(vocab_data, vectors, embedding_dim):
    vocab_size = vocab_data['vocab_size']
    word2idx = vocab_data['word2idx']
    
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim)).astype(np.float32)
    embedding_matrix[vocab_data['pad_idx']] = np.zeros(embedding_dim)
    
    loaded_count = 0
    for word, idx in word2idx.items():
        if word in vectors:
            embedding_matrix[idx] = vectors[word]
            loaded_count += 1
    
    print(f"词向量覆盖率: {loaded_count}/{vocab_size} ({loaded_count/vocab_size*100:.2f}%)")
    return embedding_matrix


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, embedding_matrix=None,
                 filter_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim))
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool2d(conv_out, (conv_out.size(2), 1))
            conv_outputs.append(pooled.squeeze(3).squeeze(2))
        
        cat = torch.cat(conv_outputs, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)


def predict_all(model, data_path, vocab_data, max_seq_len=11, batch_size=128):
    """对所有数据进行预测"""
    model.eval()
    
    word2idx = vocab_data['word2idx']
    unk_idx = vocab_data['unk_idx']
    pad_idx = vocab_data['pad_idx']
    
    predictions = []
    actuals = []
    texts = []
    
    df = pd.read_csv(data_path, encoding='utf-8-sig', low_memory=False)
    actuals = df['cat_id'].values.tolist()
    texts = df['preprocessed'].tolist()
    
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            
            batch_indices = []
            for text in batch_texts:
                words = str(text).split()
                indices = [word2idx.get(w, unk_idx) for w in words]
                
                if len(indices) > max_seq_len:
                    indices = indices[:max_seq_len]
                else:
                    indices = indices + [pad_idx] * (max_seq_len - len(indices))
                
                batch_indices.append(indices)
            
            batch_tensor = torch.LongTensor(batch_indices)
            outputs = model(batch_tensor)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.tolist())
    
    return actuals, predictions


def main():
    print("=" * 60)
    print("TextCNN (FastText) 预测结果对比")
    print("=" * 60)
    
    vocab_data = load_fasttext_vocab()
    vectors, embedding_dim = load_fasttext_vectors()
    embedding_matrix = build_embedding_matrix(vocab_data, vectors, embedding_dim)
    
    vocab_size = vocab_data['vocab_size']
    cat_mapping = pd.read_csv(CAT_MAPPING_PATH, encoding='utf-8-sig')
    num_classes = len(cat_mapping)
    
    cat_id_to_name = dict(zip(cat_mapping['cat_id'], cat_mapping['ClassName']))
    
    print(f"\n模型配置:")
    print(f"  词表大小: {vocab_size}")
    print(f"  嵌入维度: {embedding_dim}")
    print(f"  类别数: {num_classes}")
    
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        embedding_matrix=embedding_matrix,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        dropout=0.5
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    print(f"\n模型已加载: {MODEL_PATH}")
    
    actuals, predictions = predict_all(model, DATA_PATH, vocab_data)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
    df['predicted_cat_id'] = predictions
    df['actual_cat_name'] = df['cat_id'].map(cat_id_to_name)
    df['predicted_cat_name'] = df['predicted_cat_id'].map(cat_id_to_name)
    
    mismatched = df[df['cat_id'] != df['predicted_cat_id']].copy()
    mismatched['actual_cat_name'] = mismatched['cat_id'].map(cat_id_to_name)
    mismatched['predicted_cat_name'] = mismatched['predicted_cat_id'].map(cat_id_to_name)
    
    total = len(df)
    errors = len(mismatched)
    accuracy = (total - errors) / total * 100
    
    print(f"\n预测结果统计:")
    print(f"  总样本数: {total}")
    print(f"  错误预测: {errors}")
    print(f"  准确率: {accuracy:.2f}%")
    
    output_cols = ['ItemCode', 'ItemName', 'itemclsname', 'cat_id', 
                   'predicted_cat_id', 'actual_cat_name', 'predicted_cat_name']
    mismatched_output = mismatched[output_cols].copy()
    mismatched_output.columns = ['ItemCode', 'ItemName', 'CategoryName', 
                                  'ActualCatId', 'PredictedCatId', 
                                  'ActualCatName', 'PredictedCatName']
    
    output_path = os.path.join(DATA_DIR, "mismatched_predictions.csv")
    mismatched_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n错误预测已保存到: {output_path}")
    
    print("\n前10条错误预测示例:")
    print("-" * 80)
    for _, row in mismatched_output.head(10).iterrows():
        print(f"商品: {row['ItemName'][:30]}...")
        print(f"  实际: {row['ActualCatName']} (ID: {row['ActualCatId']})")
        print(f"  预测: {row['PredictedCatName']} (ID: {row['PredictedCatId']})")
        print()


if __name__ == "__main__":
    main()
