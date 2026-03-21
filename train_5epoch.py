
'''
训练5个额外epoch以提升准确率
'''

from embedding_manager import Vocabulary
from text_cnn import create_textcnn_from_vocab
from text_dataset import create_data_loaders
import time
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("继续训练5个额外epoch...")
    print("当前准确率: 93.37%")
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

    # 创建模型
    model = create_textcnn_from_vocab(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_matrix_path="data/embedding_matrix.npy",
        trainable_embeddings=True
    )
    model.to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    # 加载现有模型
    checkpoint_path = "data/textcnn_quick.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"已加载检查点: {checkpoint_path}")
        if 'best_test_acc' in checkpoint:
            print(f"检查点准确率: {checkpoint['best_test_acc']:.4f}")
    else:
        print("警告: 检查点不存在，从头开始训练")

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # 训练循环
    num_epochs = 5
    best_acc = 0.9337  # 当前最佳准确率
    patience = 3
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
            optimizer.step()

            train_loss += loss.item() * texts.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += texts.size(0)

            if batch_idx % 50 == 0:
                print(
                    f"  批次 {batch_idx}/{len(train_loader)}: 损失={loss.item():.4f}")

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
            torch.save(checkpoint, "data/textcnn_improved.pth")
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

    return best_acc


if __name__ == "__main__":
    main()
