'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 17:00:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 15:16:03
FilePath: \\myproject\\preprocessing.py
Description: 统一文本预处理模块，确保训练和推理时预处理一致性

Copyright (c) 2026 by papersnake, All Rights Reserved.
'''

import pandas as pd
import jieba as jb
from typing import Union, Dict, List
import os

# 全局标点符号转换表
remove_chars = '`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?' + \
    '·！@#￥%……&*（）——【】、『』|；‘’：“”，。《》？'
trans_table = dict([(ord(c), None) for c in remove_chars])

# 加载jieba自定义词典（在模块导入时加载一次）
dict_path = os.path.join(os.path.dirname(
    __file__), "data", "productnames.dict")
jb.load_userdict(dict_path)


def preprocess_single(cells: Union[pd.Series, Dict[str, str]]) -> str:
    """
    预处理单个样本（与data_preprocessor.py保持一致）

    Parameters:
    cells: 包含ItemCode, ItemName, unit, MainSupcode的pandas Series或字典

    Returns:
    str: 预处理后的文本（空格分隔的分词结果）
    """
    # 确保cells是pandas Series（如果是字典则转换）
    if isinstance(cells, dict):
        cells = pd.Series(cells)

    # 创建副本以避免修改原始数据
    cells = cells.copy()

    # 处理ItemCode：只取前8位
    if pd.notna(cells.get('ItemCode')):
        cells['ItemCode'] = str(cells['ItemCode'])[0:8]

    # 移除标点符号，NaN转换为空字符串
    cleaned = []
    for cell in cells.values:
        if pd.notna(cell):
            cleaned.append(str(cell).translate(trans_table))
        else:
            cleaned.append('')

    # 中文分词
    segmented = [u' '.join(jb.cut(cell)) for cell in cleaned]
    return u' '.join(segmented)


def preprocess_batch(data: pd.DataFrame,
                     columns: List[str] = ['ItemCode', 'ItemName', 'unit', 'MainSupcode']) -> pd.Series:
    """
    批量预处理DataFrame数据

    Parameters:
    data: 包含产品数据的DataFrame
    columns: 需要预处理的列名列表

    Returns:
    pandas.Series: 预处理后的文本序列
    """
    # 确保数据包含必要的列
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必要列: {missing_cols}")

    # 应用预处理到每一行
    return data[columns].apply(preprocess_single, axis=1)


def preprocess_text_for_excel(cells: pd.Series) -> str:
    """
    Excel插件专用的预处理函数（向后兼容）
    注意：此函数假设输入为pandas Series，且包含ItemCode, ItemName, unit, MainSupcode列
    """
    return preprocess_single(cells)


def test_preprocessing():
    """测试预处理功能"""
    print("=== 测试预处理模块 ===")

    # 测试单个样本
    sample = pd.Series({
        'ItemCode': '1234567890',
        'ItemName': '测试商品名称',
        'unit': '个',
        'MainSupcode': '供应商001'
    })

    result = preprocess_single(sample)
    print(f"单个样本预处理结果: {result}")

    # 测试批量处理
    data = pd.DataFrame({
        'ItemCode': ['12345678', '87654321'],
        'ItemName': ['商品A', '商品B'],
        'unit': ['个', '箱'],
        'MainSupcode': ['S001', 'S002']
    })

    results = preprocess_batch(data)
    print("\n批量预处理结果:")
    for i, text in enumerate(results):
        print(f"  样本{i}: {text}")

    # 测试NaN处理
    sample_with_nan = pd.Series({
        'ItemCode': None,
        'ItemName': '测试商品',
        'unit': None,
        'MainSupcode': 'S003'
    })

    result_nan = preprocess_single(sample_with_nan)
    print(f"\n包含NaN的样本预处理结果: {result_nan}")

    print("\n预处理测试完成!")


if __name__ == "__main__":
    test_preprocessing()
