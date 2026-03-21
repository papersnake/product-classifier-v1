"""
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-20 10:10:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 10:10:00
FilePath: myproject/test_onnx_textcnn.py
Description: Test ONNX TextCNN model inference

Copyright (c) 2026 by papersnake, All Rights Reserved.
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
import os
import sys
from embedding_manager import Vocabulary
import preprocessing

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ONNXTextCNNPredictor:
    """
    ONNX TextCNN预测器，用于模型推理
    """

    def __init__(self,
                 onnx_model_path: str = "data/textcnn.onnx",
                 vocab_path: str = "data/vocab.pkl",
                 cat_mapping_path: str = "data/cat_id_mapping.csv",
                 max_seq_len: int = 11):
        """
        初始化ONNX预测器

        Parameters:
        onnx_model_path: ONNX模型路径
        vocab_path: 词汇表路径
        cat_mapping_path: 类别映射路径
        max_seq_len: 最大序列长度
        """
        self.onnx_model_path = onnx_model_path
        self.vocab_path = vocab_path
        self.cat_mapping_path = cat_mapping_path
        self.max_seq_len = max_seq_len

        self._load_resources()
        print("ONNX TextCNN预测器初始化完成")
        print(f"模型: {onnx_model_path}")
        print(f"词汇表大小: {self.vocab.get_vocab_size()}")
        print(f"类别数量: {self.num_classes}")
        print(f"最大序列长度: {max_seq_len}")

    def _load_resources(self):
        """加载所有必要的资源"""
        # 1. 加载词汇表
        self.vocab = Vocabulary()
        self.vocab.load(self.vocab_path)

        # 2. 加载类别映射
        self.cat_mapping = pd.read_csv(
            self.cat_mapping_path, encoding='utf-8-sig', dtype={'cat_id': 'Int64', 'ItemClsCode': str})
        self.num_classes = len(self.cat_mapping)

        # 3. 加载ONNX模型
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {self.onnx_model_path}")

        self.session = ort.InferenceSession(
            self.onnx_model_path, providers=['CPUExecutionProvider'])
        print("ONNX模型加载成功")

    def preprocess(self, text: str) -> str:
        """
        预处理文本

        Parameters:
        text: 原始文本或产品信息

        Returns:
        str: 预处理后的文本
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        if " " in text.strip():
            return text.strip()

        try:
            parts = text.split()
            if len(parts) >= 4:
                cells = {
                    'ItemCode': parts[0] if len(parts) > 0 else "",
                    'ItemName': parts[1] if len(parts) > 1 else "",
                    'unit': parts[2] if len(parts) > 2 else "",
                    'MainSupcode': parts[3] if len(parts) > 3 else ""
                }
            else:
                cells = {
                    'ItemCode': "",
                    'ItemName': text,
                    'unit': "",
                    'MainSupcode': ""
                }

            return preprocessing.preprocess_single(cells)
        except Exception as e:
            print(f"预处理失败: {e}")
            import jieba
            words = jieba.cut(text)
            return " ".join(words)

    def predict_single(self, text: str, return_prob: bool = False):
        """
        预测单个文本

        Parameters:
        text: 输入文本
        return_prob: 是否返回概率分布

        Returns:
        tuple: (预测类别ID, 预测类别名称, [可选]概率分布)
        """
        sequence = self.vocab.text_to_sequence(text, self.max_seq_len)
        input_tensor = np.array(sequence, dtype=np.int64).reshape(1, -1)

        outputs = self.session.run(None, {'input_ids': input_tensor})
        output = outputs[0][0]

        probabilities = self._softmax(output)
        predicted_class = int(np.argmax(probabilities))

        pred_name = self._get_category_name(predicted_class)

        if return_prob:
            return predicted_class, pred_name, probabilities
        else:
            return predicted_class, pred_name

    def predict_batch(self, texts: list, return_prob: bool = False):
        """
        批量预测

        Parameters:
        texts: 文本列表
        return_prob: 是否返回概率分布

        Returns:
        list: 预测结果列表
        """
        if len(texts) == 0:
            return []

        # 逐个预测（因为ONNX模型可能是固定batch size的）
        results = []
        for text in texts:
            pred_id, pred_name, probs = self.predict_single(
                text, return_prob=True)
            if return_prob:
                results.append((pred_id, pred_name, probs))
            else:
                results.append((pred_id, pred_name))

        return results

    def _softmax(self, x):
        """计算softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _get_category_name(self, cat_id: int) -> str:
        """获取类别名称"""
        try:
            cat_name = self.cat_mapping.loc[
                self.cat_mapping['cat_id'] == cat_id, 'ClassName'
            ].values[0] if cat_id in self.cat_mapping['cat_id'].values else f"类别_{cat_id}"
        except Exception:
            cat_name = f"类别_{cat_id}"
        return cat_name

    def get_itemcls_code(self, cat_id: int) -> str:
        """获取ItemClsCode"""
        try:
            result = self.cat_mapping.loc[
                self.cat_mapping['cat_id'] == cat_id, 'ItemClsCode'
            ].values
            return result[0] if len(result) > 0 else str(cat_id)
        except Exception:
            return str(cat_id)

    def evaluate_accuracy(self, texts: list, labels: list) -> float:
        """
        评估准确率

        Parameters:
        texts: 文本列表
        labels: 真实标签列表

        Returns:
        float: 准确率
        """
        if len(texts) != len(labels):
            raise ValueError("文本和标签数量不匹配")

        predictions = self.predict_batch(texts, return_prob=False)
        pred_ids = [p[0] for p in predictions]

        correct = sum(1 for pred, true in zip(
            pred_ids, labels) if pred == true)
        accuracy = correct / len(labels)

        return accuracy


def test_onnx_predictor():
    """测试ONNX预测器"""
    print("=== 测试ONNX TextCNN预测器 ===")

    onnx_model_path = "data/textcnn.onnx"

    if not os.path.exists(onnx_model_path):
        print(f"ONNX模型不存在: {onnx_model_path}")
        print("请先运行 convert_textcnn_to_onnx.py 生成ONNX模型")
        return None

    try:
        predictor = ONNXTextCNNPredictor(
            onnx_model_path=onnx_model_path,
            vocab_path="data/vocab.pkl",
            cat_mapping_path="data/cat_id_mapping.csv",
            max_seq_len=11
        )

        print("\n1. 单个预测测试:")
        test_text = "99990017 尊贵 年货 礼盒 盒 9999"
        print(f"   输入文本: {test_text}")

        pred_id, pred_name = predictor.predict_single(test_text)
        itemcls_code = predictor.get_itemcls_code(pred_id)
        print(
            f"   预测结果: ID={pred_id}, 名称={pred_name}, ItemClsCode={itemcls_code}")

        print("\n2. Top-3预测测试:")
        pred_id, pred_name, probs = predictor.predict_single(
            test_text, return_prob=True)
        top3_indices = np.argsort(probs)[-3:][::-1]
        for i, idx in enumerate(top3_indices):
            cat_name = predictor._get_category_name(idx)
            print(
                f"   Top-{i+1}: ID={idx}, 名称={cat_name}, 概率={probs[idx]:.4f}")

        print("\n3. 批量预测测试:")
        test_texts = [
            "99990017 尊贵 年货 礼盒 盒 9999",
            "99990020 俏 粽娘 粽子 袋 1071",
            "6972310 壮狼 冰丝 男士 精品 内裤 z5924 条 2019"
        ]

        batch_results = predictor.predict_batch(test_texts)
        for i, (text, (pred_id, pred_name)) in enumerate(zip(test_texts, batch_results)):
            print(
                f"   样本{i+1}: {text[:30]}... -> ID={pred_id}, 名称={pred_name}")

        print("\n4. DataFrame预测测试:")
        test_data = pd.DataFrame({
            'ItemCode': ['69345678', '69654321'],
            'ItemName': ['恒源通羊毛女袜子', '宣利苹果汁'],
            'unit': ['双', '瓶'],
            'MainSupcode': ['S001', 'S002']
        })

        processed_texts = preprocessing.preprocess_batch(test_data)
        print("   预处理后的文本:")
        for i, text in enumerate(processed_texts):
            print(f"     {i}: {text}")

        df_results = predictor.predict_batch(processed_texts.tolist())
        print("   预测结果:")
        for i, (pred_id, pred_name) in enumerate(df_results):
            print(f"     {i}: ID={pred_id}, 名称={pred_name}")

        print("\nONNX预测器测试成功!")
        return predictor

    except Exception as e:
        print(f"ONNX预测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_pytorch_and_onnx():
    """比较PyTorch和ONNX模型的预测结果"""
    print("=== 比较PyTorch和ONNX模型 ===")

    import torch
    from text_cnn import create_textcnn_from_vocab
    from embedding_manager import Vocabulary

    onnx_model_path = "data/textcnn.onnx"

    if not os.path.exists(onnx_model_path):
        print(f"ONNX模型不存在: {onnx_model_path}")
        return

    try:
        # 加载ONNX预测器
        onnx_predictor = ONNXTextCNNPredictor(
            onnx_model_path=onnx_model_path,
            vocab_path="data/vocab.pkl",
            cat_mapping_path="data/cat_id_mapping.csv"
        )
        print("ONNX预测器加载成功")

        # 加载PyTorch模型
        print("加载PyTorch模型...")
        vocab = Vocabulary()
        vocab.load("data/vocab.pkl")
        vocab_size = vocab.get_vocab_size()

        import pandas as pd
        cat_mapping = pd.read_csv(
            "data/cat_id_mapping.csv", encoding='utf-8-sig')
        num_classes = len(cat_mapping)

        device = torch.device('cpu')
        pytorch_model = create_textcnn_from_vocab(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_matrix_path="data/embedding_matrix.npy",
            trainable_embeddings=False
        )

        checkpoint = torch.load("data/textcnn_final.pth", map_location=device)
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        pytorch_model.to(device)
        pytorch_model.eval()
        print("PyTorch模型加载成功")

        # 测试文本
        test_texts = [
            "99990017 尊贵 年货 礼盒 盒 9999",
            "12345678 果为 蓝莓 果汁 饮料",
            "87654321 莫来 女士 文胸",
            "55556666 奔富 洛神 葡萄酒 干红",
            "11112222 美的 电饭锅 家用"
        ]

        print("\n预测结果对比:")
        print("-" * 80)
        print(f"{'文本':<30} {'PyTorch':<15} {'ONNX':<15} {'一致'}")
        print("-" * 80)

        all_match = True
        for text in test_texts:
            # PyTorch预测
            sequence = vocab.text_to_sequence(text, 11)
            input_tensor = torch.LongTensor([sequence]).to(device)
            with torch.no_grad():
                output = pytorch_model(input_tensor)
                probs = torch.softmax(output, dim=1).squeeze().numpy()
                pytorch_pred = int(np.argmax(probs))

            # ONNX预测
            onnx_pred, _, _ = onnx_predictor.predict_single(
                text, return_prob=True)

            match = "YES" if pytorch_pred == onnx_pred else "NO"
            if pytorch_pred != onnx_pred:
                all_match = False

            print(f"{text[:28]:<30} {pytorch_pred:<15} {onnx_pred:<15} {match}")

        print("-" * 80)
        print(f"总体结果: {'所有预测一致' if all_match else '存在不一致'}")

    except Exception as e:
        print(f"比较测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("1. 测试ONNX预测器:")
    predictor = test_onnx_predictor()

    if predictor:
        print("\n" + "=" * 60)
        print("2. 比较PyTorch和ONNX模型:")
        compare_pytorch_and_onnx()
