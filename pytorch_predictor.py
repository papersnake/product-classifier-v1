r'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 18:00:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 14:39:50
FilePath: \myproject\pytorch_predictor.py
Description: PyTorch TextCNN预测器,用于Excel插件集成

Copyright (c) 2026 by papersnake, All Rights Reserved.
'''

import preprocessing
from embedding_manager import Vocabulary
from text_cnn import create_textcnn_from_vocab
import torch
import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
import os

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TextCNNPredictor:
    """
    PyTorch TextCNN预测器,用于Excel插件集成
    """

    def __init__(self,
                 model_path: str = "data/textcnn_test.pth",
                 vocab_path: str = "data/vocab.pkl",
                 embedding_matrix_path: str = "data/embedding_matrix.npy",
                 cat_mapping_path: str = "data/cat_id_mapping.csv",
                 max_seq_len: int = 11):
        """
        初始化预测器

        Parameters:
        model_path: PyTorch模型路径
        vocab_path: 词汇表路径
        embedding_matrix_path: 嵌入矩阵路径
        cat_mapping_path: 类别映射路径
        max_seq_len: 最大序列长度
        """
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.embedding_matrix_path = embedding_matrix_path
        self.cat_mapping_path = cat_mapping_path
        self.max_seq_len = max_seq_len

        # 设备设置
        self.device = torch.device('cpu')

        # 加载资源
        self.vocab = None
        self.model = None
        self.cat_mapping = None
        self.num_classes = 0

        # 初始化
        self._load_resources()

        print("TextCNN预测器初始化完成")
        print(f"模型: {model_path}")
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
            self.cat_mapping_path, encoding='utf-8-sig')
        self.num_classes = len(self.cat_mapping)

        # 3. 创建模型
        vocab_size = self.vocab.get_vocab_size()
        self.model = create_textcnn_from_vocab(
            vocab_size=vocab_size,
            num_classes=self.num_classes,
            embedding_matrix_path=self.embedding_matrix_path,
            trainable_embeddings=False  # 推理时不训练嵌入层
        )

        # 4. 加载模型权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            print("模型加载成功")
            if 'best_test_acc' in checkpoint:
                print(f"模型准确率: {checkpoint['best_test_acc']:.4f}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

    def preprocess(self, text: str) -> str:
        """
        预处理文本（统一预处理）

        Parameters:
        text: 原始文本或产品信息

        Returns:
        str: 预处理后的文本（空格分隔的分词结果）
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 如果文本已经看起来像分词结果（包含空格），直接返回
        if " " in text.strip():
            # 可能已经是分词结果，但需要确保格式正确
            return text.strip()

        # 否则，假设是原始产品信息，需要预处理
        # 创建模拟的DataFrame行进行预处理
        try:
            # 尝试解析文本：假设格式为"ItemCode ItemName unit MainSupcode"或类似
            parts = text.split()
            if len(parts) >= 4:
                # 可能是完整的产品信息
                cells = {
                    'ItemCode': parts[0] if len(parts) > 0 else "",
                    'ItemName': parts[1] if len(parts) > 1 else "",
                    'unit': parts[2] if len(parts) > 2 else "",
                    'MainSupcode': parts[3] if len(parts) > 3 else ""
                }
            else:
                # 否则，假设是产品名称
                cells = {
                    'ItemCode': "",
                    'ItemName': text,
                    'unit': "",
                    'MainSupcode': ""
                }

            return preprocessing.preprocess_single(cells)
        except Exception as e:
            print(f"预处理失败: {e}")
            # 回退：使用jieba分词
            import jieba
            words = jieba.cut(text)
            return " ".join(words)

    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        预处理文本并转换为模型输入

        Parameters:
        text: 原始文本

        Returns:
        torch.Tensor: 模型输入张量
        """
        # 使用统一的预处理模块
        # 注意：这里假设text已经是预处理后的格式（空格分隔的分词结果）
        # 如果text是原始产品信息，需要先调用preprocessing.preprocess_single

        sequence = self.vocab.text_to_sequence(text, self.max_seq_len)
        text_tensor = torch.LongTensor(sequence).unsqueeze(0)  # 添加批次维度
        return text_tensor

    def predict_single(self, text: str, return_prob: bool = False):
        """
        预测单个文本

        Parameters:
        text: 输入文本
        return_prob: 是否返回概率分布

        Returns:
        tuple: (预测类别ID, 预测类别名称, [可选]概率分布)
        """
        # 预处理
        text_tensor = self.preprocess_text(text)
        text_tensor = text_tensor.to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(text_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        pred_id = predicted.item()

        # 获取类别名称
        pred_name = self._get_category_name(pred_id)

        if return_prob:
            probs = probabilities.squeeze().cpu().numpy()
            return pred_id, pred_name, probs
        else:
            return pred_id, pred_name

    def predict_batch(self, texts: List[str], return_prob: bool = False):
        """
        批量预测

        Parameters:
        texts: 文本列表
        return_prob: 是否返回概率分布

        Returns:
        list: 预测结果列表
        """
        batch_size = len(texts)
        if batch_size == 0:
            return []

        # 预处理所有文本
        sequences = []
        for text in texts:
            sequence = self.vocab.text_to_sequence(text, self.max_seq_len)
            sequences.append(sequence)

        # 创建批次张量
        text_tensor = torch.LongTensor(sequences)
        text_tensor = text_tensor.to(self.device)

        # 批量预测
        with torch.no_grad():
            outputs = self.model(text_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        results = []
        for i in range(batch_size):
            pred_id = predicted[i].item()
            pred_name = self._get_category_name(pred_id)

            if return_prob:
                probs = probabilities[i].cpu().numpy()
                results.append((pred_id, pred_name, probs))
            else:
                results.append((pred_id, pred_name))

        return results

    def predict_from_dataframe(self,
                               data: pd.DataFrame,
                               columns: List[str] = [
                                   'ItemCode', 'ItemName', 'unit', 'MainSupcode'],
                               return_prob: bool = False) -> pd.DataFrame:
        """
        从DataFrame预测

        Parameters:
        data: 包含产品数据的DataFrame
        columns: 需要预处理的列名
        return_prob: 是否返回概率分布

        Returns:
        pandas.DataFrame: 包含预测结果的DataFrame
        """
        # 批量预处理
        processed_texts = preprocessing.preprocess_batch(data, columns)

        # 批量预测
        predictions = self.predict_batch(processed_texts.tolist(), return_prob)

        # 创建结果DataFrame
        result_data = data.copy()

        if return_prob:
            pred_ids = [p[0] for p in predictions]
            pred_names = [p[1] for p in predictions]
            prob_arrays = [p[2] for p in predictions]

            result_data['pred_cat_id'] = pred_ids
            result_data['pred_class_name'] = pred_names
            result_data['probabilities'] = prob_arrays

            # 添加Top-3预测
            for i in range(len(result_data)):
                probs = prob_arrays[i]
                top3_indices = np.argsort(probs)[-3:][::-1]

                for j, idx in enumerate(top3_indices[:3]):
                    cat_name = self._get_category_name(idx)
                    result_data.at[i, f'top{j+1}_cat_id'] = idx
                    result_data.at[i, f'top{j+1}_class_name'] = cat_name
                    result_data.at[i, f'top{j+1}_prob'] = probs[idx]
        else:
            pred_ids = [p[0] for p in predictions]
            pred_names = [p[1] for p in predictions]

            result_data['pred_cat_id'] = pred_ids
            result_data['pred_class_name'] = pred_names

        return result_data

    def predict(self, text: str, return_prob: bool = False):
        """
        预测单个文本（predict_single的别名）

        Parameters:
        text: 输入文本
        return_prob: 是否返回概率分布

        Returns:
        tuple: (预测类别ID, 预测类别名称, [可选]概率分布)
        """
        return self.predict_single(text, return_prob)

    def _get_category_name(self, cat_id: int) -> str:
        """获取类别名称"""
        try:
            cat_name = self.cat_mapping.loc[
                self.cat_mapping['cat_id'] == cat_id, 'ClassName'
            ].values[0] if cat_id in self.cat_mapping['cat_id'].values else f"类别_{cat_id}"
        except Exception:
            cat_name = f"类别_{cat_id}"
        return cat_name

    def get_top_n_predictions(self, text: str, n: int = 3) -> List[Tuple[int, str, float]]:
        """
        获取Top-N预测结果

        Parameters:
        text: 输入文本
        n: 返回的Top-N数量

        Returns:
        list: [(类别ID, 类别名称, 概率), ...]
        """
        pred_id, pred_name, probs = self.predict_single(text, return_prob=True)

        # 获取Top-N
        top_n_indices = np.argsort(probs)[-n:][::-1]
        top_n_results = []

        for idx in top_n_indices:
            cat_name = self._get_category_name(idx)
            prob = probs[idx]
            top_n_results.append((idx, cat_name, prob))

        return top_n_results

    def evaluate_accuracy(self,
                          texts: List[str],
                          labels: List[int]) -> float:
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


def test_predictor():
    """测试预测器功能"""
    print("=== 测试PyTorch预测器 ===")

    try:
        # 创建预测器
        predictor = TextCNNPredictor(
            model_path="data/textcnn_test.pth",
            vocab_path="data/vocab.pkl",
            embedding_matrix_path="data/embedding_matrix.npy",
            cat_mapping_path="data/cat_id_mapping.csv",
            max_seq_len=11
        )

        # 测试单个预测
        test_text = "99990017 尊贵 年货 礼盒 盒 9999"
        print("\n1. 单个预测测试:")
        print(f"   输入文本: {test_text}")

        pred_id, pred_name = predictor.predict_single(test_text)
        print(f"   预测结果: ID={pred_id}, 名称={pred_name}")

        # 测试Top-3预测
        print("\n2. Top-3预测测试:")
        top3 = predictor.get_top_n_predictions(test_text, n=3)
        for i, (cat_id, cat_name, prob) in enumerate(top3):
            print(f"   Top-{i+1}: ID={cat_id}, 名称={cat_name}, 概率={prob:.4f}")

        # 测试批量预测
        print("\n3. 批量预测测试:")
        test_texts = [
            "99990017 尊贵 年货 礼盒 盒 9999",
            "12345678 苹果 手机 产品 测试",
            "87654321 华为 手机 新品 发布"
        ]

        batch_results = predictor.predict_batch(test_texts)
        for i, (text, (pred_id, pred_name)) in enumerate(zip(test_texts, batch_results)):
            print(
                f"   样本{i+1}: {text[:30]}... -> ID={pred_id}, 名称={pred_name}")

        # 测试DataFrame预测
        print("\n4. DataFrame预测测试:")
        test_data = pd.DataFrame({
            'ItemCode': ['12345678', '87654321'],
            'ItemName': ['苹果手机', '华为手机'],
            'unit': ['个', '部'],
            'MainSupcode': ['S001', 'S002']
        })

        df_results = predictor.predict_from_dataframe(test_data)
        print("   预测结果DataFrame:")
        print(df_results[['ItemCode', 'ItemName',
              'pred_cat_id', 'pred_class_name']])

        print("\n✅ 预测器测试成功!")
        return predictor

    except Exception as e:
        print(f"❌ 预测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_predictor()
