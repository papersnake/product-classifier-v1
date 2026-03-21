'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-21 11:30:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-21 12:14:44
FilePath: \\myproject\\train_textcnn_fasttext.py
Description: 使用 FastText 词向量训练 TextCNN

Copyright (c) 2026 by papersnake, All Rights Reserved.
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FASTTEXT_VECTORS = os.path.join(DATA_DIR, "fasttext_vectors.txt")
FASTTEXT_VOCAB = os.path.join(DATA_DIR, "fasttext_vocab.pkl")
CAT_MAPPING_PATH = os.path.join(DATA_DIR, "cat_id_mapping.csv")


def load_fasttext_vocab():
    """加载 FastText 词表"""
    print("加载 FastText 词表...")
    with open(FASTTEXT_VOCAB, 'rb') as f:
        vocab_data = pickle.load(f)
    print(f"词表大小: {vocab_data['vocab_size']}")
    return vocab_data


def load_fasttext_vectors():
    """加载 FastText 词向量"""
    print("加载 FastText 词向量...")
    vectors = {}
    with open(FASTTEXT_VECTORS, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split()
        vocab_size, embedding_dim = int(header[0]), int(header[1])
        print(f"词向量: {vocab_size} 词, 维度 {embedding_dim}")

        for line in f:
            parts = line.strip().split()
            if len(parts) >= embedding_dim + 1:
                word = parts[0]
                vector = np.array([float(x)
                                  for x in parts[1:]], dtype=np.float32)
                vectors[word] = vector

    print(f"已加载 {len(vectors)} 个词向量")
    return vectors, embedding_dim


def build_embedding_matrix(vocab_data, vectors, embedding_dim):
    """构建嵌入矩阵"""
    vocab_size = vocab_data['vocab_size']
    word2idx = vocab_data['word2idx']

    embedding_matrix = np.random.normal(
        scale=0.1,
        size=(vocab_size, embedding_dim)
    ).astype(np.float32)

    embedding_matrix[vocab_data['pad_idx']] = np.zeros(embedding_dim)

    loaded_count = 0
    for word, idx in word2idx.items():
        if word in vectors:
            embedding_matrix[idx] = vectors[word]
            loaded_count += 1

    print(
        f"词向量覆盖率: {loaded_count}/{vocab_size} ({loaded_count/vocab_size*100:.2f}%)")
    return embedding_matrix


class ProductDataset(Dataset):
    """产品分类数据集"""

    def __init__(self, data_path, vocab_data, max_seq_len=11, mode='train', test_size=0.25):
        self.data = pd.read_csv(data_path, encoding='utf-8-sig')
        self.vocab_data = vocab_data
        self.max_seq_len = max_seq_len
        self.word2idx = vocab_data['word2idx']
        self.unk_idx = vocab_data['unk_idx']
        self.pad_idx = vocab_data['pad_idx']

        from sklearn.model_selection import train_test_split

        if mode in ['train', 'test']:
            X = self.data['preprocessed'].values
            y = self.data['cat_id'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, stratify=y
            )

            if mode == 'train':
                self.texts = X_train.tolist()
                self.labels = y_train.tolist()
            else:
                self.texts = X_test.tolist()
                self.labels = y_test.tolist()
        else:
            self.texts = self.data['preprocessed'].values.tolist()
            self.labels = self.data['cat_id'].values.tolist()

        print(f"数据集模式: {mode}, 样本数: {len(self.texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        words = text.split()
        indices = [self.word2idx.get(w, self.unk_idx) for w in words]

        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            indices = indices + [self.pad_idx] * \
                (self.max_seq_len - len(indices))

        return torch.LongTensor(indices), torch.LongTensor([label])


class TextCNN(nn.Module):
    """TextCNN 模型"""

    def __init__(self, vocab_size, embedding_dim, num_classes, embedding_matrix=None,
                 filter_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

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


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for texts, labels in loader:
        labels = labels.squeeze(1)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * texts.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += texts.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for texts, labels in loader:
            labels = labels.squeeze(1)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * texts.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += texts.size(0)

    return total_loss / total, correct / total


def main():
    print("=" * 60)
    print("FastText + TextCNN 训练")
    print("=" * 60)

    device = torch.device('cpu')

    vocab_data = load_fasttext_vocab()
    vectors, embedding_dim = load_fasttext_vectors()
    embedding_matrix = build_embedding_matrix(
        vocab_data, vectors, embedding_dim)

    vocab_size = vocab_data['vocab_size']
    cat_mapping = pd.read_csv(CAT_MAPPING_PATH, encoding='utf-8-sig')
    num_classes = len(cat_mapping)

    print("\n模型配置:")
    print(f"  词表大小: {vocab_size}")
    print(f"  嵌入维度: {embedding_dim}")
    print(f"  类别数: {num_classes}")

    train_dataset = ProductDataset(
        os.path.join(DATA_DIR, "products_processed.csv"),
        vocab_data, mode='train'
    )
    test_dataset = ProductDataset(
        os.path.join(DATA_DIR, "products_processed.csv"),
        vocab_data, mode='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        embedding_matrix=embedding_matrix,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        dropout=0.5
    )
    model.to(device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_acc = 0
    patience, patience_counter = 5, 0
    num_epochs = 15

    print("\n开始训练...")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        scheduler.step(test_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch:2d} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Test: loss={test_loss:.4f} acc={test_acc:.4f} | "
              f"{elapsed:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(
                DATA_DIR, "textcnn_fasttext.pth"))
            print(f"  [保存] 最佳准确率: {best_acc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("\n早停触发")
            break

    print("-" * 60)
    print("\n训练完成!")
    print(f"最佳测试准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"模型已保存: {os.path.join(DATA_DIR, 'textcnn_fasttext.pth')}")

    return best_acc


if __name__ == "__main__":
    main()
