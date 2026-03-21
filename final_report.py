'''
最终报告：PyTorch TextCNN模型性能总结
'''

from pytorch_predictor import TextCNNPredictor
import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 70)
    print("PyTorch TextCNN产品分类模型 - 最终报告")
    print("=" * 70)

    # 加载最佳模型
    model_path = "data/textcnn_final.pth"
    if not os.path.exists(model_path):
        print("错误: 最终模型不存在")
        return

    # 创建预测器
    try:
        predictor = TextCNNPredictor(
            model_path=model_path,
            vocab_path="data/vocab.pkl",
            embedding_matrix_path="data/embedding_matrix.npy",
            cat_mapping_path="data/cat_id_mapping.csv",
            max_seq_len=11
        )
        print("✅ 预测器加载成功")
    except Exception as e:
        print(f"❌ 预测器加载失败: {e}")
        return

    # 加载检查点获取准确率
    checkpoint = torch.load(model_path, map_location='cpu')
    best_acc = checkpoint.get('best_test_acc', 0.0)
    epoch = checkpoint.get('epoch', 0)

    print("\n📊 模型性能:")
    print(f"   最佳测试准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("   目标准确率: 0.9478 (94.78%)")
    print(f"   差距: {best_acc - 0.9478:+.4f} ({(best_acc - 0.9478)*100:+.2f}%)")
    print(f"   训练轮数: {epoch}")

    # 测试预测
    print("\n🔍 测试预测:")
    test_texts = [
        "99990017 尊贵 年货 礼盒 盒 9999",
        "12345678 苹果 手机 保护套",
        "87654321 华为 笔记本电脑"
    ]

    for i, text in enumerate(test_texts):
        try:
            pred_id, pred_name = predictor.predict_single(text)
            print(
                f"   样本{i+1}: {text[:30]}... -> ID={pred_id}, 名称={pred_name}")
        except Exception as e:
            print(f"   样本{i+1}: 预测失败 - {e}")

    # 检查Excel插件集成
    print("\n🔗 Excel插件集成检查:")
    try:
        from myproject import PYTORCH_AVAILABLE, get_pytorch_predictor
        if PYTORCH_AVAILABLE:
            print("   ✅ PyTorch预测器可用")
            try:
                pred = get_pytorch_predictor()
                print("   ✅ 预测器实例化成功")
            except Exception as e:
                print(f"   ⚠️  预测器实例化失败: {e}")
        else:
            print("   ❌ PyTorch预测器不可用（请检查依赖）")
    except ImportError as e:
        print(f"   ⚠️  无法导入myproject: {e}")

    # 文件检查
    print("\n📁 关键文件检查:")
    required_files = [
        ("data/textcnn_final.pth", "最终模型"),
        ("data/vocab.pkl", "词汇表"),
        ("data/embedding_matrix.npy", "嵌入矩阵"),
        ("data/cat_id_mapping.csv", "类别映射"),
        ("data/products_processed.csv", "处理后的数据"),
        ("pytorch_predictor.py", "预测器模块"),
        ("myproject.py", "Excel插件"),
        ("preprocessing.py", "预处理模块")
    ]

    all_ok = True
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {description}: {file_path}")
        else:
            print(f"   ❌ {description}: 缺失")
            all_ok = False

    # 总结
    print("\n" + "=" * 70)
    print("总结:")
    print("=" * 70)

    if best_acc >= 0.9478:
        print("🎉 目标达成！PyTorch模型达到或超过了sklearn模型的准确率。")
    else:
        print("📈 接近目标！PyTorch模型准确率接近但略低于sklearn模型。")
        print("   建议进一步优化：")
        print("   1. 增加训练轮数（更多epoch）")
        print("   2. 调整超参数（学习率、dropout、过滤器数量）")
        print("   3. 使用预训练中文词向量")
        print("   4. 增加数据增强")

    print("\n下一步:")
    print("   1. 运行 'python myproject.py' 启动Excel插件")
    print("   2. 在Excel中使用 PyTorch_Predict() 函数")
    print("   3. 使用 Compare_Predictions() 比较sklearn和PyTorch预测")

    # 保存报告
    report_path = "data/final_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PyTorch TextCNN模型最终报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"最佳测试准确率: {best_acc:.4f}\n")
        f.write("目标准确率: 0.9478\n")
        f.write(f"差距: {best_acc - 0.9478:+.4f}\n")
        f.write(f"训练轮数: {epoch}\n")
        f.write(f"模型文件: {model_path}\n")
        f.write(f"报告生成时间: {pd.Timestamp.now()}\n")

    print(f"\n📄 详细报告已保存到: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
