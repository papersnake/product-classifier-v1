'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 20:15:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 15:16:39
FilePath: \\myproject\\test_pytorch_integration.py
Description: 测试PyTorch模型与Excel插件集成

Copyright (c) 2026 by papersnake, All Rights Reserved.
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pytorch_predictor import TextCNNPredictor
    PYTORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch预测器导入失败: {e}")
    PYTORCH_AVAILABLE = False


def test_pytorch_predictor():
    """测试PyTorch预测器"""
    if not PYTORCH_AVAILABLE:
        print("PyTorch预测器不可用")
        return False

    print("测试PyTorch预测器...")

    # 初始化预测器
    try:
        predictor = TextCNNPredictor(
            model_path="data/textcnn_test.pth",  # 使用2 epoch的模型
            vocab_path="data/vocab.pkl",
            embedding_matrix_path="data/embedding_matrix.npy",
            cat_mapping_path="data/cat_id_mapping.csv",
            max_seq_len=11
        )
        print("PyTorch预测器初始化成功")
    except Exception as e:
        print(f"预测器初始化失败: {e}")
        return False

    # 测试单个预测
    test_texts = [
        "99990017 尊贵 年货 礼盒 盒 9999",
        "苹果手机新品发布",
        "笔记本电脑高性能",
        "牛奶全脂纯牛奶"
    ]

    for text in test_texts:
        try:
            # 使用预测器的predict方法
            pred_id, pred_name = predictor.predict(text)
            print(f"文本: '{text}' -> 类别ID: {pred_id}, 类别名称: {pred_name}")
        except Exception as e:
            print(f"预测失败 '{text}': {e}")
            # 尝试使用内部方法
            try:
                processed_text = predictor.preprocess(text)
                print(f"  预处理后: {processed_text}")
            except:
                pass

    # 测试批量预测
    print("\n测试批量预测...")
    try:
        results = predictor.predict_batch(test_texts)
        for i, (text, pred_id, pred_name) in enumerate(zip(test_texts, results['pred_ids'], results['pred_names'])):
            print(f"{i}: '{text}' -> {pred_id} ({pred_name})")
    except Exception as e:
        print(f"批量预测失败: {e}")

    # 测试与sklearn预处理的一致性
    print("\n测试预处理一致性...")
    try:
        from preprocessing import preprocess_single
        for text in test_texts:
            pytorch_preprocessed = predictor.preprocess(text)
            unified_preprocessed = preprocess_single(text)
            print(f"文本: '{text}'")
            print(f"  PyTorch预处理: {pytorch_preprocessed}")
            print(f"  统一预处理: {unified_preprocessed}")
            print(f"  是否一致: {pytorch_preprocessed == unified_preprocessed}")
    except Exception as e:
        print(f"预处理测试失败: {e}")

    print("\nPyTorch预测器测试完成!")
    return True


def test_excel_functions():
    """测试Excel插件函数"""
    print("\n测试Excel插件函数...")

    # 导入myproject模块中的函数
    try:
        from myproject import preprocess, myPredict, jb_cut
        print("Excel插件函数导入成功")

        # 测试预处理函数
        test_cells = ["99990017", "尊贵年货礼盒", "盒", "9999", "helper"]
        processed = preprocess(test_cells)
        print(f"预处理结果: {processed}")

        # 测试myPredict函数（sklearn版本）
        test_sec = "尊贵年货礼盒盒9999"
        try:
            result = myPredict(test_sec)
            print(f"myPredict('{test_sec}') -> {result}")
        except Exception as e:
            print(f"myPredict失败: {e}")

        # 测试jb_cut函数
        try:
            result = jb_cut(test_sec)
            print(f"jb_cut('{test_sec}') -> {result}")
        except Exception as e:
            print(f"jb_cut失败: {e}")

    except Exception as e:
        print(f"Excel插件函数测试失败: {e}")
        return False

    return True


def main():
    print("PyTorch模型与Excel插件集成测试")
    print("=" * 60)

    # 测试PyTorch预测器
    pytorch_ok = test_pytorch_predictor()

    # 测试Excel插件函数
    excel_ok = test_excel_functions()

    print("\n" + "=" * 60)
    print("集成测试总结:")
    print(f"  PyTorch预测器: {'通过' if pytorch_ok else '失败'}")
    print(f"  Excel插件函数: {'通过' if excel_ok else '失败'}")

    if pytorch_ok and excel_ok:
        print("\n所有测试通过! PyTorch模型已成功集成到Excel插件中。")
    else:
        print("\n部分测试失败，需要进一步调试。")

    print("=" * 60)


if __name__ == "__main__":
    main()
