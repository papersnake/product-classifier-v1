"""
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-20 10:00:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 10:00:00
FilePath: myproject/convert_textcnn_to_onnx.py
Description: 将PyTorch TextCNN模型转换为ONNX格式

Copyright (c) 2026 by papersnake, All Rights Reserved.
"""

import pandas as pd
import torch
from embedding_manager import Vocabulary
from text_cnn import create_textcnn_from_vocab
import warnings
import os
import sys

# 重定向stdout/stderr来捕获编码问题


class UnicodeFilter:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        try:
            self.stream.write(data)
        except UnicodeEncodeError:
            encoded = data.encode('gbk', errors='replace').decode('gbk')
            self.stream.write(encoded)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = UnicodeFilter(sys.stdout)
sys.stderr = UnicodeFilter(sys.stderr)

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def convert_textcnn_to_onnx(
    pytorch_model_path: str = "data/textcnn_final.pth",
    vocab_path: str = "data/vocab.pkl",
    embedding_matrix_path: str = "data/embedding_matrix.npy",
    cat_mapping_path: str = "data/cat_id_mapping.csv",
    onnx_model_path: str = "data/textcnn.onnx",
    max_seq_len: int = 11
):
    """
    将PyTorch TextCNN模型转换为ONNX格式

    Parameters:
    pytorch_model_path: PyTorch模型路径
    vocab_path: 词汇表路径
    embedding_matrix_path: 嵌入矩阵路径
    cat_mapping_path: 类别映射路径
    onnx_model_path: ONNX模型输出路径
    max_seq_len: 最大序列长度
    """
    print("=== Convert PyTorch TextCNN to ONNX ===")

    # 加载词汇表
    print("1. Loading vocabulary...")
    vocab = Vocabulary()
    vocab.load(vocab_path)
    vocab_size = vocab.get_vocab_size()
    print(f"   Vocabulary size: {vocab_size}")

    # 加载类别映射
    print("2. Loading category mapping...")
    cat_mapping = pd.read_csv(cat_mapping_path, encoding='utf-8-sig')
    num_classes = len(cat_mapping)
    print(f"   Number of classes: {num_classes}")

    # 加载PyTorch模型
    print("3. Loading PyTorch model...")
    device = torch.device('cpu')

    model = create_textcnn_from_vocab(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_matrix_path=embedding_matrix_path,
        trainable_embeddings=False
    )

    checkpoint = torch.load(
        pytorch_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(
        f"   Model loaded, accuracy: {checkpoint.get('best_test_acc', 'N/A')}")

    # 创建示例输入
    print("4. Creating sample input...")
    example_input = torch.randint(
        0, vocab_size, (1, max_seq_len), dtype=torch.long)
    print(f"   Input shape: {example_input.shape}")

    # 导出ONNX
    print("5. Exporting ONNX model...")

    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

    torch.onnx.export(
        model,
        example_input,
        onnx_model_path,
        input_names=['input_ids'],
        output_names=['output'],
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
    )

    print(f"   ONNX model saved to: {onnx_model_path}")

    # 验证ONNX模型
    print("6. Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("   ONNX model verified successfully!")

    # 获取模型信息
    print("\n=== Conversion Complete ===")
    print(f"PyTorch model: {pytorch_model_path}")
    print(f"ONNX model: {onnx_model_path}")
    print(
        f"Model size: {os.path.getsize(onnx_model_path) / (1024*1024):.2f} MB")

    return onnx_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert PyTorch TextCNN to ONNX')
    parser.add_argument('--pytorch_model', type=str, default='data/textcnn_final.pth',
                        help='PyTorch model path')
    parser.add_argument('--vocab', type=str, default='data/vocab.pkl',
                        help='Vocabulary path')
    parser.add_argument('--embedding', type=str, default='data/embedding_matrix.npy',
                        help='Embedding matrix path')
    parser.add_argument('--cat_mapping', type=str, default='data/cat_id_mapping.csv',
                        help='Category mapping path')
    parser.add_argument('--onnx_model', type=str, default='data/textcnn.onnx',
                        help='ONNX model output path')
    parser.add_argument('--max_seq_len', type=int, default=11,
                        help='Max sequence length')

    args = parser.parse_args()

    convert_textcnn_to_onnx(
        pytorch_model_path=args.pytorch_model,
        vocab_path=args.vocab,
        embedding_matrix_path=args.embedding,
        cat_mapping_path=args.cat_mapping,
        onnx_model_path=args.onnx_model,
        max_seq_len=args.max_seq_len
    )
