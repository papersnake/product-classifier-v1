'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-21 12:00:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-21 18:52:32
FilePath: \\myproject\\test_fasttext_predict.py
Description: FastText + TextCNN 模型预测测试

Copyright (c) 2026 by papersnake, All Rights Reserved.
'''

import pandas as pd
import torch
import torch.nn as nn
import pickle
import os
import jieba as jb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(DATA_DIR, "textcnn_fasttext.pth")
FASTTEXT_VOCAB = os.path.join(DATA_DIR, "fasttext_vocab.pkl")
CAT_MAPPING_PATH = os.path.join(DATA_DIR, "cat_id_mapping.csv")
DICT_PATH = os.path.join(DATA_DIR, "productnames.dict")


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes,
                 filter_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool2d(conv_out, (conv_out.size(2), 1))
            conv_outputs.append(pooled.squeeze(3).squeeze(2))
        cat = torch.cat(conv_outputs, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)


class FastTextPredictor:
    """FastText + TextCNN 预测器"""

    def __init__(self):
        self.device = torch.device('cpu')
        self.model = None
        self.vocab_data = None
        self.cat_mapping = None
        self.word2idx = None
        self.unk_idx = None
        self.pad_idx = None
        self.max_seq_len = 11

        self._load()

    def _load(self):
        """加载模型和配置"""
        print("加载 FastText + TextCNN 模型...")

        with open(FASTTEXT_VOCAB, 'rb') as f:
            self.vocab_data = pickle.load(f)

        self.cat_mapping = pd.read_csv(CAT_MAPPING_PATH, encoding='utf-8-sig')
        self.word2idx = self.vocab_data['word2idx']
        self.unk_idx = self.vocab_data['unk_idx']
        self.pad_idx = self.vocab_data['pad_idx']

        vocab_size = self.vocab_data['vocab_size']
        embedding_dim = self.vocab_data['embedding_dim']
        num_classes = len(self.cat_mapping)

        self.model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes
        )

        self.model.load_state_dict(torch.load(
            MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        jb.load_userdict(DICT_PATH)

        print("模型加载完成!")
        print(f"  词表大小: {vocab_size}")
        print(f"  嵌入维度: {embedding_dim}")
        print(f"  类别数: {num_classes}")

    def preprocess(self, item_code, item_name, unit, main_supcode):
        """预处理产品数据"""
        remove_chars = '`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?' + \
            '·！@#￥%……&*（）——【】、『』|；''："",。《》？'
        trans_table = dict([(ord(c), None) for c in remove_chars])

        parts = []
        if item_code:
            parts.append(str(item_code)[:8])
        if item_name:
            name_clean = str(item_name).translate(trans_table)
            words = jb.cut(name_clean)
            parts.append(' '.join(words))
        if unit:
            parts.append(str(unit).translate(trans_table))
        if main_supcode:
            parts.append(str(main_supcode))

        text = ' '.join(parts)

        indices = [self.word2idx.get(w, self.unk_idx) for w in text.split()]

        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            indices = indices + [self.pad_idx] * \
                (self.max_seq_len - len(indices))

        return torch.LongTensor([indices])

    def predict_single(self, item_code, item_name, unit, main_supcode):
        """预测单个产品"""
        tensor = self.preprocess(item_code, item_name, unit, main_supcode)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        cat_name = self.cat_mapping.loc[
            self.cat_mapping['cat_id'] == pred_idx, 'ClassName'
        ].values

        class_name = cat_name[0] if len(cat_name) > 0 else "未知"

        return {
            'pred_id': pred_idx,
            'class_name': class_name,
            'confidence': confidence
        }

    def predict_batch(self, items):
        """批量预测
        items: list of dict, each dict contains 'ItemCode', 'ItemName', 'unit', 'MainSupcode'
        """
        results = []
        for item in items:
            result = self.predict_single(
                item.get('ItemCode'),
                item.get('ItemName'),
                item.get('unit'),
                item.get('MainSupcode')
            )
            results.append({
                'ItemName': item.get('ItemName', ''),
                'pred_id': result['pred_id'],
                'class_name': result['class_name'],
                'confidence': result['confidence']
            })
        return pd.DataFrame(results)


def test_single_prediction():
    """测试单个预测"""
    print("\n" + "=" * 60)
    print("测试单个预测")
    print("=" * 60)

    predictor = FastTextPredictor()

    test_cases = [
        {
            'ItemCode': '01234567',
            'ItemName': '蒙牛特仑苏有机纯牛奶250ml',
            'unit': '盒',
            'MainSupcode': '1001'
        },
        {
            'ItemCode': '03070123',
            'ItemName': '男士纯棉内裤套装',
            'unit': '套',
            'MainSupcode': '2005'
        },
        {
            'ItemCode': '08010001',
            'ItemName': '奥利奥夹心饼干巧克力味',
            'unit': '包',
            'MainSupcode': '1038'
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"\n测试样本 {i+1}:")
        print(f"  产品名: {case['ItemName']}")
        result = predictor.predict_single(
            case['ItemCode'], case['ItemName'], case['unit'], case['MainSupcode']
        )
        print(f"  预测类别ID: {result['pred_id']}")
        print(f"  预测类别名: {result['class_name']}")
        print(
            f"  置信度: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")


def test_batch_prediction():
    """测试批量预测"""
    print("\n" + "=" * 60)
    print("测试批量预测")
    print("=" * 60)

    predictor = FastTextPredictor()

    batch_items = [
        {'ItemCode': '01234567', 'ItemName': '蒙牛特仑苏有机纯牛奶250ml',
            'unit': '盒', 'MainSupcode': '1001'},
        {'ItemCode': '03070123', 'ItemName': '男士纯棉内裤套装',
            'unit': '套', 'MainSupcode': '2005'},
        {'ItemCode': '08010001', 'ItemName': '奥利奥夹心饼干巧克力味',
            'unit': '包', 'MainSupcode': '1038'},
        {'ItemCode': '07020001', 'ItemName': '鲜活大闸蟹母蟹3两',
            'unit': '只', 'MainSupcode': '3013'},
        {'ItemCode': '08060001', 'ItemName': '盼盼法式软面包草莓味',
            'unit': '袋', 'MainSupcode': '1035'},
    ]

    results = predictor.predict_batch(batch_items)
    print("\n批量预测结果:")
    print(results.to_string(index=False))


def test_on_validation_data():
    """在验证集上测试"""
    print("\n" + "=" * 60)
    print("在测试集上评估")
    print("=" * 60)

    predictor = FastTextPredictor()

    data_path = os.path.join(DATA_DIR, "products_processed.csv")
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(
        df, test_size=0.25, random_state=0, stratify=df['cat_id'])

    test_df = test_df.head(100)

    correct = 0
    results = []

    for idx, row in test_df.iterrows():
        result = predictor.predict_single(
            row.get('ItemCode', ''),
            row.get('ItemName', ''),
            row.get('unit', ''),
            row.get('MainSupcode', '')
        )

        is_correct = result['pred_id'] == row['cat_id']
        if is_correct:
            correct += 1

        results.append({
            'ItemName': row['ItemName'][:20],
            'true_class': row['itemclsname'],
            'pred_class': result['class_name'],
            'correct': is_correct,
            'confidence': result['confidence']
        })

    accuracy = correct / len(test_df)
    print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"测试样本数: {len(test_df)}")
    print(f"正确数: {correct}")

    results_df = pd.DataFrame(results)
    wrong_df = results_df[~results_df['correct']].head(10)
    if len(wrong_df) > 0:
        print("\n错误样例 (前10个):")
        print(wrong_df.to_string(index=False))


def main():
    print("=" * 60)
    print("FastText + TextCNN 模型预测测试")
    print("=" * 60)

    test_single_prediction()
    test_batch_prediction()
    test_on_validation_data()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
