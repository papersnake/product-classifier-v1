'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 20:45:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-14 20:45:00
FilePath: \\myproject\\create_final_checkpoint.py
Description: 创建最终模型检查点并清理临时文件

Copyright (c) 2026 by papersnake, All Rights Reserved. 
'''

import os
import shutil
import pandas as pd
import torch

def create_final_checkpoint():
    print("创建最终模型检查点...")
    print("=" * 60)
    
    # 最佳模型文件（当前最佳准确率）
    best_model_path = "data/textcnn_test.pth"  # 93.37%准确率
    final_model_path = "data/textcnn_final.pth"
    
    if not os.path.exists(best_model_path):
        print(f"错误: 最佳模型文件不存在: {best_model_path}")
        return False
    
    # 复制最佳模型为最终模型
    try:
        # 加载检查点以验证
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        # 添加额外信息
        checkpoint['final_model'] = True
        checkpoint['description'] = "PyTorch TextCNN最终模型 - 产品分类"
        checkpoint['accuracy_target'] = 0.9478
        checkpoint['current_accuracy'] = checkpoint.get('test_acc', 0.0)
        
        # 保存为最终模型
        torch.save(checkpoint, final_model_path)
        
        print(f"最终模型已创建: {final_model_path}")
        print(f"模型准确率: {checkpoint.get('test_acc', 0.0):.4f}")
        print(f"目标准确率: 0.9478")
        print(f"差距: {checkpoint.get('test_acc', 0.0) - 0.9478:+.4f}")
        
        # 创建模型信息文件
        model_info = {
            'model_name': ['textcnn_final'],
            'file_path': [final_model_path],
            'accuracy': [checkpoint.get('test_acc', 0.0)],
            'target_accuracy': [0.9478],
            'difference': [checkpoint.get('test_acc', 0.0) - 0.9478],
            'vocab_size': [checkpoint.get('vocab_size', 0)],
            'num_classes': [checkpoint.get('num_classes', 0)],
            'max_seq_len': [checkpoint.get('max_seq_len', 11)],
            'epochs_trained': [checkpoint.get('epoch', 0)],
            'creation_date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        
        model_info_df = pd.DataFrame(model_info)
        model_info_df.to_csv('data/model_info.csv', index=False, encoding='utf-8-sig')
        print("模型信息已保存到: data/model_info.csv")
        
        return True
        
    except Exception as e:
        print(f"创建最终模型检查点时出错: {e}")
        return False

def clean_temporary_files():
    print("\n清理临时文件...")
    print("=" * 60)
    
    # 要保留的文件列表
    essential_files = [
        'data/products.csv',
        'data/products_processed.csv',
        'data/cat_id_mapping.csv',
        'data/cat_small_id_df.csv',
        'data/vocab.pkl',
        'data/embedding_matrix.npy',
        'data/Tfidf_min_vect.pkl',
        'data/clf_min_model.pkl',
        'data/textcnn_final.pth',  # 最终模型
        'data/model_info.csv',
        'data/seq_length_stats.txt',
        'data/productnames.dict'
    ]
    
    # 临时文件模式
    temp_patterns = [
        'data/textcnn_*.pth',  # 除了最终模型
        'data/*.log',
        'data/*_accuracy.csv',
        'data/temp_*.py',
        'quick_train.log'
    ]
    
    # 实际文件列表
    all_files = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # 收集要删除的文件
    files_to_delete = []
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        
        # 检查是否是必需文件
        is_essential = False
        for essential in essential_files:
            if file_path.endswith(essential.replace('data/', '')):
                is_essential = True
                break
        
        # 如果是最终模型，保留
        if file_path == 'data/textcnn_final.pth':
            is_essential = True
        
        if not is_essential:
            # 检查是否匹配临时模式
            for pattern in temp_patterns:
                pattern_name = pattern.replace('data/', '').replace('*', '')
                if pattern_name in file_name or file_name.startswith('temp_'):
                    files_to_delete.append(file_path)
                    break
    
    # 删除临时文件
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"已删除: {file_path}")
                deleted_count += 1
        except Exception as e:
            print(f"删除失败 {file_path}: {e}")
    
    print(f"\n已删除 {deleted_count} 个临时文件")
    
    # 清理项目根目录的临时Python文件
    root_temp_files = ['temp_*.py', 'train_*.py', 'eval_*.py', 'test_*.py']
    current_dir = os.getcwd()
    for file_name in os.listdir(current_dir):
        if file_name.endswith('.py') and (
            file_name.startswith('temp_') or 
            file_name.startswith('train_') and file_name != 'train_textcnn_optimized.py' or
            file_name.startswith('eval_') or
            file_name.startswith('test_') and file_name not in ['test_pytorch_integration.py', 'testpipelin.py', 'testonnx.py']
        ):
            try:
                file_path = os.path.join(current_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"已删除: {file_name}")
                    deleted_count += 1
            except Exception as e:
                print(f"删除失败 {file_name}: {e}")
    
    return deleted_count

def generate_final_report():
    print("\n生成最终报告...")
    print("=" * 60)
    
    report = []
    report.append("=" * 60)
    report.append("PyTorch TextCNN模型部署 - 最终报告")
    report.append("=" * 60)
    report.append("")
    
    # 模型信息
    if os.path.exists('data/model_info.csv'):
        model_info = pd.read_csv('data/model_info.csv')
        for _, row in model_info.iterrows():
            report.append(f"模型名称: {row['model_name']}")
            report.append(f"文件路径: {row['file_path']}")
            report.append(f"准确率: {row['accuracy']:.4f}")
            report.append(f"目标准确率: {row['target_accuracy']:.4f}")
            report.append(f"差距: {row['difference']:+.4f}")
            report.append(f"词汇表大小: {row['vocab_size']}")
            report.append(f"类别数量: {row['num_classes']}")
            report.append(f"最大序列长度: {row['max_seq_len']}")
            report.append(f"训练epoch数: {row['epochs_trained']}")
            report.append(f"创建时间: {row['creation_date']}")
            report.append("")
    
    # 文件统计
    data_files = [f for f in os.listdir('data') if os.path.isfile(os.path.join('data', f))]
    model_files = [f for f in data_files if f.endswith('.pth') or f.endswith('.pkl') or f.endswith('.onnx')]
    csv_files = [f for f in data_files if f.endswith('.csv')]
    
    report.append(f"数据目录文件统计:")
    report.append(f"  总文件数: {len(data_files)}")
    report.append(f"  模型文件: {len(model_files)}")
    report.append(f"  CSV文件: {len(csv_files)}")
    report.append(f"  其他文件: {len(data_files) - len(model_files) - len(csv_files)}")
    report.append("")
    
    # 项目状态
    report.append("项目状态总结:")
    report.append("  ✓ PyTorch TextCNN模型架构实现完成")
    report.append("  ✓ 统一预处理模块创建完成")
    report.append("  ✓ 优化训练框架实现完成（CPU优化）")
    report.append("  ✓ 词汇表和嵌入矩阵构建完成")
    report.append("  ✓ 现有模型准确率: 93.37% (2个epoch)")
    report.append("  ✓ 目标准确率: 94.78% (sklearn基准)")
    report.append("  ✓ Excel插件集成框架完成")
    report.append("  ⚠️  需进一步训练以达到目标准确率")
    report.append("  ⚠️  sklearn评估存在预处理不一致问题")
    report.append("")
    
    # 下一步建议
    report.append("下一步建议:")
    report.append("  1. 继续训练PyTorch模型至30-50个epoch")
    report.append("  2. 调整超参数（学习率、批大小、dropout等）")
    report.append("  3. 尝试预训练中文词向量提升性能")
    report.append("  4. 修复sklearn评估的预处理一致性")
    report.append("  5. 完整测试Excel插件集成")
    report.append("")
    
    report.append("=" * 60)
    
    # 保存报告
    report_text = "\n".join(report)
    with open('data/final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n最终报告已保存到: data/final_report.txt")
    
    return report_text

def main():
    print("创建最终检查点并清理临时文件")
    print("=" * 60)
    
    # 创建最终模型检查点
    if not create_final_checkpoint():
        print("创建最终检查点失败")
        return
    
    # 清理临时文件
    deleted_count = clean_temporary_files()
    
    # 生成最终报告
    generate_final_report()
    
    print("\n" + "=" * 60)
    print("完成!")
    print(f"已删除 {deleted_count} 个临时文件")
    print("最终模型: data/textcnn_final.pth")
    print("模型信息: data/model_info.csv")
    print("最终报告: data/final_report.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()