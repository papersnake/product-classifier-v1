'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 16:40:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-04-12 19:21:04
FilePath: \\myproject\\evaluate_model.py
Description: 评估和对比PyTorch TextCNN与现有sklearn模型性能

Copyright (c) 2026 by papersnake, All Rights Reserved.
'''

from embedding_manager import Vocabulary
from text_cnn import create_textcnn_from_vocab
from text_dataset import TextClassificationDataset
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def configure_matplotlib_chinese_font() -> None:
    """配置 Matplotlib 以支持中文字体显示。"""
    chinese_fonts = [
        "SimHei",
        "Microsoft YaHei",
        "STHeiti",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            mpl.rcParams['font.sans-serif'] = [font_name] + mpl.rcParams['font.sans-serif']
            mpl.rcParams['axes.unicode_minus'] = False
            return

    mpl.rcParams['axes.unicode_minus'] = False
    print("警告: 未检测到常见中文字体，中文标签可能无法正常显示。")


configure_matplotlib_chinese_font()


class ModelEvaluator:
    """
    模型评估器，用于对比PyTorch TextCNN和sklearn模型性能
    """

    def __init__(self,
                 vocab_path: str = "data/vocab.pkl",
                 embedding_matrix_path: str = "data/embedding_matrix.npy",
                 cat_mapping_path: str = "data/cat_id_mapping.csv",
                 pytorch_model_path: str = "data/textcnn_final.pth",
                 sklearn_vectorizer_path: str = "data/Tfidf_min_vect.pkl",
                 sklearn_model_path: str = "data/clf_min_model.pkl"):
        """
        初始化评估器

        Parameters:
        vocab_path: 词汇表路径
        embedding_matrix_path: 嵌入矩阵路径
        cat_mapping_path: 类别映射路径
        pytorch_model_path: PyTorch模型路径
        sklearn_vectorizer_path: sklearn向量器路径
        sklearn_model_path: sklearn模型路径
        """
        self.vocab_path = vocab_path
        self.embedding_matrix_path = embedding_matrix_path
        self.cat_mapping_path = cat_mapping_path
        self.pytorch_model_path = pytorch_model_path
        self.sklearn_vectorizer_path = sklearn_vectorizer_path
        self.sklearn_model_path = sklearn_model_path

        # 设备设置
        self.device = torch.device('cpu')

        # 加载类别映射
        self.cat_mapping = pd.read_csv(cat_mapping_path, encoding='utf-8-sig')
        self.num_classes = len(self.cat_mapping)

        # 加载测试数据集
        self.test_dataset = TextClassificationDataset(
            max_seq_len=11,
            mode='test',
            test_size=0.25,
            random_state=0
        )

        # 模型
        self.pytorch_model = None
        self.sklearn_vectorizer = None
        self.sklearn_model = None

        print("模型评估器初始化完成")
        print(f"测试集大小: {len(self.test_dataset)}")
        print(f"类别数量: {self.num_classes}")

    def load_pytorch_model(self):
        """加载PyTorch模型"""
        print("=== 加载PyTorch模型 ===")

        # 创建模型
        vocab = Vocabulary()
        vocab.load(self.vocab_path)
        vocab_size = vocab.get_vocab_size()

        self.pytorch_model = create_textcnn_from_vocab(
            vocab_size=vocab_size,
            num_classes=self.num_classes,
            embedding_matrix_path=self.embedding_matrix_path,
            trainable_embeddings=False  # 评估时不训练
        )

        # 加载模型权重
        if os.path.exists(self.pytorch_model_path):
            checkpoint = torch.load(
                self.pytorch_model_path, map_location=self.device)
            self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            self.pytorch_model.to(self.device)
            self.pytorch_model.eval()

            print("PyTorch模型加载成功")
            print(f"模型准确率 (保存时): {checkpoint.get('test_acc', '未知')}")
        else:
            print(f"警告: PyTorch模型文件不存在: {self.pytorch_model_path}")
            print("请先训练模型或指定正确的路径")

    def load_sklearn_model(self):
        """加载sklearn模型"""
        print("=== 加载sklearn模型 ===")

        if os.path.exists(self.sklearn_vectorizer_path) and os.path.exists(self.sklearn_model_path):
            self.sklearn_vectorizer = joblib.load(self.sklearn_vectorizer_path)
            self.sklearn_model = joblib.load(self.sklearn_model_path)
            print("sklearn模型加载成功")
            print(
                f"向量器特征数: {self.sklearn_vectorizer.get_feature_names_out().shape[0]}")
            print(f"分类器: {type(self.sklearn_model).__name__}")
        else:
            print("警告: sklearn模型文件不存在")
            print(f"向量器路径: {self.sklearn_vectorizer_path}")
            print(f"模型路径: {self.sklearn_model_path}")

    def evaluate_pytorch(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        评估PyTorch模型

        Returns:
        tuple: (真实标签, 预测标签, 准确率)
        """
        if self.pytorch_model is None:
            self.load_pytorch_model()

        print("\n=== 评估PyTorch模型 ===")

        self.pytorch_model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for i in range(len(self.test_dataset)):
                text_tensor, label_tensor = self.test_dataset[i]

                # 添加批次维度
                text_tensor = text_tensor.unsqueeze(0).to(self.device)

                # 预测
                outputs = self.pytorch_model(text_tensor)
                _, predicted = torch.max(outputs, 1)

                all_predictions.append(predicted.item())
                all_labels.append(label_tensor.item())

                # 进度显示
                if i % 1000 == 0:
                    print(f"  处理进度: {i}/{len(self.test_dataset)}")

        # 转换为numpy数组
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)

        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)

        print(f"PyTorch模型准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(y_true, y_pred, target_names=[
              str(i) for i in range(self.num_classes)]))

        return y_true, y_pred, accuracy

    def evaluate_sklearn(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        评估sklearn模型

        Returns:
        tuple: (真实标签, 预测标签, 准确率)
        """
        if self.sklearn_vectorizer is None or self.sklearn_model is None:
            self.load_sklearn_model()

        print("\n=== 评估sklearn模型 ===")

        # 准备数据
        labels = self.test_dataset.labels

        # 使用测试集预处理后的文本（与训练时相同）
        # sklearn模型是在preprocessed列上训练的，所以直接使用预处理后的文本
        sklearn_texts = self.test_dataset.texts

        # 使用sklearn向量器转换文本
        X_test = self.sklearn_vectorizer.transform(sklearn_texts)
        y_true = np.array(labels)

        # 预测
        y_pred = self.sklearn_model.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)

        print(f"sklearn模型准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(y_true, y_pred, target_names=[
              str(i) for i in range(self.num_classes)]))

        return y_true, y_pred, accuracy

    def compare_models(self):
        """对比两个模型的性能"""
        print("\n" + "=" * 60)
        print("模型性能对比")
        print("=" * 60)

        # 评估PyTorch模型
        y_true_pytorch, y_pred_pytorch, acc_pytorch = self.evaluate_pytorch()

        # 评估sklearn模型
        y_true_sklearn, y_pred_sklearn, acc_sklearn = self.evaluate_sklearn()

        # 确保使用相同的测试集
        # 由于两个模型使用相同的测试数据集，y_true应该相同
        if len(y_true_pytorch) != len(y_true_sklearn):
            print("警告: 测试集大小不一致")

        # 性能对比
        print("\n" + "=" * 60)
        print("性能对比总结")
        print("=" * 60)
        print(f"{'模型':<20} {'准确率':<10} {'相对性能':<15}")
        print(f"{'-'*20} {'-'*10} {'-'*15}")
        print(f"{'PyTorch TextCNN':<20} {acc_pytorch:.4f}     {'-'}")
        print(
            f"{'sklearn LinearSVC':<20} {acc_sklearn:.4f}     {f'{acc_pytorch-acc_sklearn:+.4f}'}")

        if acc_pytorch > acc_sklearn:
            improvement = (acc_pytorch - acc_sklearn) / acc_sklearn * 100
            print(f"\n[SUCCESS] PyTorch模型优于sklearn模型: +{improvement:.2f}%")
        elif acc_pytorch < acc_sklearn:
            gap = (acc_sklearn - acc_pytorch) / acc_sklearn * 100
            print(f"\n[WARNING] PyTorch模型低于sklearn模型: -{gap:.2f}%")
        else:
            print("\n[INFO] 两个模型性能相同")

        # 生成混淆矩阵对比
        self.plot_confusion_matrices(
            y_true_pytorch, y_pred_pytorch, y_pred_sklearn)

        # 分析差异
        self.analyze_differences(
            y_true_pytorch, y_pred_pytorch, y_pred_sklearn)

    def plot_confusion_matrices(self, y_true: np.ndarray,
                                y_pred_pytorch: np.ndarray,
                                y_pred_sklearn: np.ndarray):
        """绘制混淆矩阵对比图"""
        try:
            print("\n生成混淆矩阵对比图...")

            # 计算混淆矩阵
            cm_pytorch = confusion_matrix(y_true, y_pred_pytorch)
            cm_sklearn = confusion_matrix(y_true, y_pred_sklearn)

            # 创建子图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # PyTorch模型混淆矩阵
            sns.heatmap(cm_pytorch, annot=False, fmt='d', cmap='Blues',
                        ax=axes[0], cbar_kws={'shrink': 0.8})
            axes[0].set_title(
                f'PyTorch TextCNN (准确率: {accuracy_score(y_true, y_pred_pytorch):.4f})')
            axes[0].set_xlabel('预测标签')
            axes[0].set_ylabel('真实标签')

            # sklearn模型混淆矩阵
            sns.heatmap(cm_sklearn, annot=False, fmt='d', cmap='Reds',
                        ax=axes[1], cbar_kws={'shrink': 0.8})
            axes[1].set_title(
                f'sklearn LinearSVC (准确率: {accuracy_score(y_true, y_pred_sklearn):.4f})')
            axes[1].set_xlabel('预测标签')
            axes[1].set_ylabel('真实标签')

            # 差异矩阵
            diff_matrix = cm_pytorch - cm_sklearn
            sns.heatmap(diff_matrix, annot=False, fmt='d', cmap='coolwarm',
                        center=0, ax=axes[2], cbar_kws={'shrink': 0.8})
            axes[2].set_title('差异矩阵 (PyTorch - sklearn)')
            axes[2].set_xlabel('预测标签')
            axes[2].set_ylabel('真实标签')

            plt.tight_layout()
            plt.savefig('data/model_comparison.png',
                        dpi=150, bbox_inches='tight')
            print("混淆矩阵对比图已保存到: data/model_comparison.png")

        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")

    def analyze_differences(self, y_true: np.ndarray,
                            y_pred_pytorch: np.ndarray,
                            y_pred_sklearn: np.ndarray):
        """分析两个模型预测的差异"""
        print("\n" + "=" * 60)
        print("预测差异分析")
        print("=" * 60)

        # 找出预测不一致的样本
        disagreement_mask = y_pred_pytorch != y_pred_sklearn
        disagreement_indices = np.where(disagreement_mask)[0]

        print(f"预测不一致的样本数: {len(disagreement_indices)}")
        print(f"不一致比例: {len(disagreement_indices)/len(y_true)*100:.2f}%")

        if len(disagreement_indices) > 0:
            # 分析每个模型在这些样本上的准确率
            pytorch_correct = y_pred_pytorch[disagreement_indices] == y_true[disagreement_indices]
            sklearn_correct = y_pred_sklearn[disagreement_indices] == y_true[disagreement_indices]

            pytorch_win = np.sum(pytorch_correct & ~sklearn_correct)
            sklearn_win = np.sum(sklearn_correct & ~pytorch_correct)
            both_wrong = np.sum(~pytorch_correct & ~sklearn_correct)
            both_correct = np.sum(pytorch_correct & sklearn_correct)  # 应该为0

            print("\n在不一致的样本中:")
            print(f"  PyTorch正确而sklearn错误: {pytorch_win} 个")
            print(f"  sklearn正确而PyTorch错误: {sklearn_win} 个")
            print(f"  两个都错误: {both_wrong} 个")
            print(f"  两个都正确: {both_correct} 个 (理论上应为0)")

            # 显示一些不一致的示例
            print("\n前5个不一致的样本示例:")
            for i in range(min(5, len(disagreement_indices))):
                idx = disagreement_indices[i]
                text = self.test_dataset.texts[idx]
                true_label = y_true[idx]
                pred_pytorch = y_pred_pytorch[idx]
                pred_sklearn = y_pred_sklearn[idx]

                # 获取类别名称
                true_name = self.cat_mapping.loc[
                    self.cat_mapping['cat_id'] == true_label, 'ClassName'
                ].values[0] if true_label in self.cat_mapping['cat_id'].values else "未知"

                pytorch_name = self.cat_mapping.loc[
                    self.cat_mapping['cat_id'] == pred_pytorch, 'ClassName'
                ].values[0] if pred_pytorch in self.cat_mapping['cat_id'].values else "未知"

                sklearn_name = self.cat_mapping.loc[
                    self.cat_mapping['cat_id'] == pred_sklearn, 'ClassName'
                ].values[0] if pred_sklearn in self.cat_mapping['cat_id'].values else "未知"

                print(f"\n示例 {i+1}:")
                print(f"  文本: {text[:50]}...")
                print(f"  真实类别: {true_label} ({true_name})")
                print(f"  PyTorch预测: {pred_pytorch} ({pytorch_name})")
                print(f"  sklearn预测: {pred_sklearn} ({sklearn_name})")

        # 分析每个类别的性能差异
        print("\n" + "=" * 60)
        print("类别级别的性能分析")
        print("=" * 60)

        # 计算每个类别的准确率
        class_acc_pytorch = []
        class_acc_sklearn = []

        for class_id in range(self.num_classes):
            mask = y_true == class_id
            if np.sum(mask) > 0:
                acc_pytorch = np.mean(y_pred_pytorch[mask] == class_id)
                acc_sklearn = np.mean(y_pred_sklearn[mask] == class_id)
                class_acc_pytorch.append(acc_pytorch)
                class_acc_sklearn.append(acc_sklearn)

        # 找出PyTorch表现更好的类别
        pytorch_better = np.sum(
            np.array(class_acc_pytorch) > np.array(class_acc_sklearn))
        sklearn_better = np.sum(
            np.array(class_acc_sklearn) > np.array(class_acc_pytorch))
        equal = np.sum(np.array(class_acc_pytorch) ==
                       np.array(class_acc_sklearn))

        print(f"PyTorch表现更好的类别数: {pytorch_better}")
        print(f"sklearn表现更好的类别数: {sklearn_better}")
        print(f"表现相同的类别数: {equal}")

        # 找出差异最大的类别
        if len(class_acc_pytorch) > 0:
            diff = np.array(class_acc_pytorch) - np.array(class_acc_sklearn)
            max_improvement_idx = np.argmax(diff)
            max_decline_idx = np.argmin(diff)

            print(f"\nPyTorch提升最大的类别: {max_improvement_idx}")
            print(
                f"  PyTorch准确率: {class_acc_pytorch[max_improvement_idx]:.4f}")
            print(
                f"  sklearn准确率: {class_acc_sklearn[max_improvement_idx]:.4f}")
            print(f"  提升: {diff[max_improvement_idx]:+.4f}")

            print(f"\nPyTorch下降最大的类别: {max_decline_idx}")
            print(f"  PyTorch准确率: {class_acc_pytorch[max_decline_idx]:.4f}")
            print(f"  sklearn准确率: {class_acc_sklearn[max_decline_idx]:.4f}")
            print(f"  下降: {diff[max_decline_idx]:+.4f}")


def main():
    """主函数"""
    print("模型性能评估与对比")
    print("=" * 60)

    # 创建评估器
    evaluator = ModelEvaluator(
        vocab_path="data/vocab.pkl",
        embedding_matrix_path="data/embedding_matrix.npy",
        cat_mapping_path="data/cat_id_mapping.csv",
        pytorch_model_path="data/textcnn_final.pth",  # 使用最终模型
        sklearn_vectorizer_path="data/Tfidf_min_vect.pkl",
        sklearn_model_path="data/clf_min_model.pkl"
    )

    # 对比模型性能
    evaluator.compare_models()

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
