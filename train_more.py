'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 21:00:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-14 21:00:00
FilePath: \\myproject\\train_more.py
Description: 继续训练PyTorch TextCNN模型以达到目标准确率

Copyright (c) 2026 by papersnake, All Rights Reserved. 
'''

import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_textcnn_optimized import OptimizedTextCNNTrainer

def main():
    """主函数：继续训练模型20个额外epoch"""
    print("继续训练PyTorch TextCNN模型（20个额外epoch）...")
    print("=" * 60)
    print(f"当前准确率: 93.37%")
    print(f"目标准确率: 94.78%")
    print(f"差距: -1.41%")
    print("=" * 60)
    
    # 创建优化的训练器
    trainer = OptimizedTextCNNTrainer(
        batch_size=64,
        max_seq_len=11,
        learning_rate=0.0005,  # 降低学习率进行微调
        num_epochs=20,  # 训练20个额外epoch
        patience=8,     # 早停耐心值
        num_workers=4,
        use_amp=False,
        model_save_path='data/textcnn_more_trained.pth'
    )
    
    # 设置数据和模型
    trainer.setup_data()
    trainer.setup_model()
    
    # 加载现有最终模型
    existing_checkpoint = 'data/textcnn_final.pth'
    if os.path.exists(existing_checkpoint):
        print(f"加载现有模型: {existing_checkpoint}")
        checkpoint = torch.load(existing_checkpoint, map_location=trainer.device)
        
        # 恢复模型状态
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器状态（如果存在）
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("优化器状态已恢复")
        else:
            print("警告: 检查点中没有优化器状态，将使用新优化器")
        
        # 恢复训练历史
        trainer.train_loss_history = checkpoint.get('train_loss_history', [])
        trainer.train_acc_history = checkpoint.get('train_acc_history', [])
        trainer.test_loss_history = checkpoint.get('test_loss_history', [])
        trainer.test_acc_history = checkpoint.get('test_acc_history', [])
        
        # 恢复最佳准确率
        trainer.best_test_acc = max(trainer.test_acc_history) if trainer.test_acc_history else 0.0
        
        print(f"现有模型加载成功，最佳准确率: {trainer.best_test_acc:.4f}")
        print(f"已训练epoch数: {len(trainer.train_loss_history)}")
        
        # 调整起始epoch
        start_epoch = len(trainer.train_loss_history)
        if start_epoch > 0:
            print(f"将从第{start_epoch+1}个epoch继续训练")
    else:
        print(f"警告: 现有模型不存在 {existing_checkpoint}")
        print("将从头开始训练")
    
    # 开始训练
    trainer.train()
    
    print("\n" + "=" * 60)
    print("继续训练完成!")
    print("=" * 60)
    
    # 评估模型
    test_acc = trainer.evaluate_on_test_set()
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"目标准确率: 0.9478 (94.78%)")
    print(f"差距: {test_acc - 0.9478:+.4f}")
    
    if test_acc >= 0.9478:
        print("[SUCCESS] 达到目标准确率!")
        # 保存为新的最终模型
        trainer.save_best_model('data/textcnn_final_improved.pth')
        print("改进的最终模型已保存: data/textcnn_final_improved.pth")
    else:
        print("[WARNING] 未达到目标准确率，可能需要更多训练")
        # 仍然保存最佳模型
        trainer.save_best_model('data/textcnn_best_so_far.pth')
        print("当前最佳模型已保存: data/textcnn_best_so_far.pth")
    
    # 绘制训练历史
    trainer.plot_training_history('data/training_history_more.png')
    print("训练历史图已保存: data/training_history_more.png")
    
    return trainer

if __name__ == "__main__":
    main()