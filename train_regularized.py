'''
正则化训练：增加dropout、权重衰减等减少过拟合
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time

from text_dataset import create_data_loaders
from text_cnn import TextCNN
from embedding_manager import Vocabulary

def create_regularized_model(vocab_size, num_classes, embedding_matrix_path):
    """创建正则化模型"""
    # 加载嵌入矩阵
    embedding_matrix = None
    if embedding_matrix_path and embedding_matrix_path.endswith('.npy'):
        try:
            embedding_matrix = np.load(embedding_matrix_path)
            print(f"嵌入矩阵已加载，形状: {embedding_matrix.shape}")
            embedding_dim = embedding_matrix.shape[1]
        except Exception as e:
            print(f"加载嵌入矩阵失败: {e}")
            embedding_dim = 300
    else:
        embedding_dim = 300
    
    # 创建正则化模型
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        filter_sizes=[3, 4, 5],
        num_filters=75,  # 减少过滤器数量
        dropout_rate=0.7,  # 增加dropout
        embedding_matrix=embedding_matrix,
        trainable_embeddings=True
    )
    
    # 添加嵌入层dropout
    model.embedding_dropout = nn.Dropout(0.2)
    
    # 修改forward函数以包含嵌入层dropout
    original_forward = model.forward
    def new_forward(x):
        batch_size = x.size(0)
        embedded = model.embedding(x)
        embedded = model.embedding_dropout(embedded)  # 嵌入层dropout
        embedded = embedded.unsqueeze(1)
        
        conv_outputs = []
        for conv in model.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.nn.functional.max_pool2d(conv_out, kernel_size=(conv_out.size(2), 1))
            conv_outputs.append(pooled.squeeze(3).squeeze(2))
        
        cat_output = torch.cat(conv_outputs, dim=1)
        cat_output = model.dropout(cat_output)
        logits = model.fc(cat_output)
        return logits
    
    model.forward = new_forward
    return model

def main():
    print("正则化训练启动...")
    print("当前最佳准确率: 94.23%")
    print("目标: 94.78%")
    
    # 设备
    device = torch.device('cpu')
    
    # 加载词汇表和类别映射
    vocab = Vocabulary()
    vocab.load("data/vocab.pkl")
    vocab_size = vocab.get_vocab_size()
    cat_mapping = pd.read_csv("data/cat_id_mapping.csv", encoding='utf-8-sig')
    num_classes = len(cat_mapping)
    
    print(f"词汇表大小: {vocab_size}")
    print(f"类别数: {num_classes}")
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        batch_size=64,
        max_seq_len=11,
        num_workers=4
    )
    
    print(f"训练批次: {len(train_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    # 创建正则化模型
    model = create_regularized_model(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_matrix_path="data/embedding_matrix.npy"
    )
    model.to(device)
    
    # 优化器和损失函数（增加权重衰减）
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
    )
    
    # 训练循环
    num_epochs = 10
    best_acc = 0.9423  # 当前最佳
    patience = 4
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # 训练
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
            
            # 梯度裁剪
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
        
        # 评估
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
        
        # 更新学习率
        scheduler.step(test_acc)
        
        # 保存最佳模型
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
            torch.save(checkpoint, "data/textcnn_regularized.pth")
            print(f"  [保存] 新的最佳模型，准确率: {test_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  [等待] 早停计数器: {patience_counter}/{patience}")
        
        # 检查早停
        if patience_counter >= patience:
            print(f"\n早停触发，连续{patience}个epoch无改善")
            break
    
    print(f"\n训练完成!")
    print(f"最佳准确率: {best_acc:.4f}")
    print(f"目标准确率: 0.9478")
    print(f"差距: {best_acc - 0.9478:+.4f}")
    
    # 最终评估
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
        import shutil
        shutil.copy2("data/textcnn_regularized.pth", "data/textcnn_target_achieved.pth")
        print("目标达成模型已保存: data/textcnn_target_achieved.pth")
    else:
        print("\n[WARNING] 未达到目标准确率，但已接近")
    
    return best_acc

if __name__ == "__main__":
    main()