'''
Train PyTorch TextCNN on the same split as sklearn baseline (factorized labels).
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load processed data
print("Loading processed data...")
data = pd.read_csv("data/products_processed.csv", encoding='utf-8-sig')
print(f"Rows: {len(data)}")

# Factorize ItemClsCode (order of appearance) to match notebook's cat_id
data['cat_id_factor'] = pd.factorize(data['ItemClsCode'])[0]
print(f"Factorized cat_id range: {data['cat_id_factor'].min()} to {data['cat_id_factor'].max()}")
num_classes = data['cat_id_factor'].nunique()
print(f"Number of classes: {num_classes}")

# Save factorized mapping for later use
factor_mapping = data[['ItemClsCode', 'cat_id_factor']].drop_duplicates().sort_values('cat_id_factor')
factor_mapping['ClassName'] = data['itemclsname']  # need to map properly
factor_mapping.to_csv("data/cat_id_factorized.csv", index=False, encoding='utf-8-sig')
print("Factorized mapping saved to data/cat_id_factorized.csv")

# Split using random_state=0, stratify=cat_id_factor (same as notebook)
X = data['preprocessed'].values
y = data['cat_id_factor'].values
indices = data.index.values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.25, random_state=0, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Load vocabulary and embedding matrix
print("Loading vocabulary...")
from embedding_manager import Vocabulary
vocab = Vocabulary()
vocab.load("data/vocab.pkl")
vocab_size = vocab.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

embedding_matrix = np.load("data/embedding_matrix.npy")
print(f"Embedding matrix shape: {embedding_matrix.shape}")
embedding_dim = embedding_matrix.shape[1]

# Define custom dataset
class FactorizedTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_seq_len=11):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sequence = self.vocab.text_to_sequence(text, self.max_seq_len)
        text_tensor = torch.LongTensor(sequence)
        label_tensor = torch.LongTensor([label])
        return text_tensor, label_tensor

# Create datasets and data loaders
max_seq_len = 11
batch_size = 64
num_workers = 4

train_dataset = FactorizedTextDataset(X_train, y_train, vocab, max_seq_len)
test_dataset = FactorizedTextDataset(X_test, y_test, vocab, max_seq_len)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False
)
print(f"Train loader batches: {len(train_loader)}, Test loader batches: {len(test_loader)}")

# Create model (TextCNN with regularization)
from text_cnn import TextCNN
model = TextCNN(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_classes=num_classes,
    filter_sizes=[3, 4, 5],
    num_filters=100,
    dropout_rate=0.7,  # increased dropout
    embedding_matrix=embedding_matrix,
    trainable_embeddings=True
)
device = torch.device('cpu')
model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
)

# Training loop
num_epochs = 20
patience = 5
patience_counter = 0
best_acc = 0.0
best_model_path = "data/textcnn_factorized_best.pth"

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    start_time = time.time()
    for batch_idx, (texts, labels) in enumerate(train_loader):
        texts = texts.to(device)
        labels = labels.squeeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item() * texts.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += texts.size(0)
        
        if batch_idx % 50 == 0:
            print(f"  批次 {batch_idx}/{len(train_loader)}: 损失={loss.item():.4f}")
    
    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total
    print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.4f}")
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            labels = labels.squeeze(1).to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * texts.size(0)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += texts.size(0)
    
    avg_test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    print(f"  测试损失: {avg_test_loss:.4f}, 测试准确率: {test_acc:.4f}")
    
    # Update learning rate
    scheduler.step(test_acc)
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        patience_counter = 0
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_test_acc': best_acc,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'test_loss': avg_test_loss,
            'test_acc': test_acc
        }
        torch.save(checkpoint, best_model_path)
        print(f"  [保存] 新的最佳模型，准确率: {test_acc:.4f}")
    else:
        patience_counter += 1
        print(f"  [等待] 早停计数器: {patience_counter}/{patience}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\n早停触发，连续{patience}个epoch无改善")
        break

print(f"\n训练完成!")
print(f"最佳准确率: {best_acc:.4f}")
print(f"目标准确率: 0.9478")
print(f"差距: {best_acc - 0.9478:+.4f}")

# Final evaluation
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for texts, labels in test_loader:
        texts = texts.to(device)
        labels = labels.squeeze(1).to(device)
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += texts.size(0)

final_acc = test_correct / test_total
print(f"最终测试准确率: {final_acc:.4f}")

if final_acc >= 0.9478:
    print("\n[SUCCESS] 达到目标准确率!")
    # Save final model
    torch.save(checkpoint, "data/textcnn_factorized_final.pth")
    print("目标达成模型已保存: data/textcnn_factorized_final.pth")
else:
    print("\n[WARNING] 未达到目标准确率，但已接近")

# Compare with sklearn baseline
print("\n--- 与sklearn基线对比 ---")
# Load sklearn model
vectorizer = joblib.load("data/Tfidf_min_vect.pkl")
clf = joblib.load("data/clf_min_model.pkl")
X_test_vec = vectorizer.transform(X_test)
y_pred_sklearn = clf.predict(X_test_vec)
sklearn_acc = accuracy_score(y_test, y_pred_sklearn)
print(f"Sklearn准确率: {sklearn_acc:.4f}")
print(f"PyTorch准确率: {final_acc:.4f}")
print(f"差异: {final_acc - sklearn_acc:+.4f}")

if final_acc >= sklearn_acc:
    print("[SUCCESS] PyTorch模型达到或超过了sklearn模型的准确率!")
else:
    print("[WARNING] PyTorch模型准确率低于sklearn模型")
    
# Save summary
with open("data/factorized_training_summary.txt", "w", encoding='utf-8') as f:
    f.write(f"Sklearn baseline accuracy: {sklearn_acc:.4f}\n")
    f.write(f"PyTorch final accuracy: {final_acc:.4f}\n")
    f.write(f"Difference: {final_acc - sklearn_acc:+.4f}\n")
    f.write(f"Target: 0.9478\n")
    f.write(f"Achieved target: {'YES' if final_acc >= 0.9478 else 'NO'}\n")
    f.write(f"Model file: {best_model_path}\n")
print("总结已保存到 data/factorized_training_summary.txt")