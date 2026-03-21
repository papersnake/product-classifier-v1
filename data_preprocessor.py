'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 15:30:00
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 16:29:55
FilePath: \\myproject\\data_preprocessor.py
Description: 数据预处理模块 - 从小类.ipynb提取的预处理逻辑

Copyright (c) 2026 by papersnake, All Rights Reserved.
'''

import pandas as pd
import jieba as jb
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import preprocessing


class DataPreprocessor:
    """
    数据预处理类，从小类.ipynb提取的完整预处理流程
    """

    def __init__(self, data_path: str = "data/products.csv",
                 dict_path: Optional[str] = None):
        """
        初始化预处理器

        Parameters:
        data_path: 产品数据CSV文件路径
        dict_path: jieba自定义词典路径
        """
        import os

        self.data_path = data_path

        # 设置默认词典路径
        if dict_path is None:
            # 使用相对于当前模块的路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.dict_path = os.path.join(
                base_dir, "data", "productnames.dict")
        else:
            self.dict_path = dict_path

        self.products = None
        self.cat_id_df = None
        self.cat_to_id = None
        self.id_to_cat = None

        # 初始化标点符号转换表
        self.remove_chars = '`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?' + \
            '·！@#￥%……&*（）——【】、『』|；‘’："，《》。？'
        self.trans_table = dict([(ord(c), None) for c in self.remove_chars])

        # 加载jieba自定义词典
        jb.load_userdict(self.dict_path)

    def load_and_filter_data(self) -> pd.DataFrame:
        """
        加载数据并应用过滤规则：
        1. 排除ItemClsCode以"23%"开头的类别
        2. 排除生鲜类别（ItemClsCode以"07"开头）
        3. 排除样本数少于10的类别

        Returns:
        pandas.DataFrame: 过滤后的产品数据
        """
        # 加载数据
        self.products = pd.read_csv(self.data_path, dtype={
            'ItemClsCode': str,
            'itemclsname': str,
            'ItemCode': str,
            'ItemName': str,
            'unit': str,
            'MainSupcode': str
        })

        print(f"原始数据行数: {len(self.products)}")

        # 1. 去掉ItemClsCode的空格
        self.products['ItemClsCode'] = self.products['ItemClsCode'].str.strip()
        # 2. 排除生鲜类别（ItemClsCode以"07"开头）
        self.products = self.products[~self.products['ItemClsCode'].str.startswith(
            '07')]
        print(f"排除生鲜类别后行数: {len(self.products)}")

        # 3. 排除样本数少于10的类别
        cat_counts = self.products['ItemClsCode'].value_counts()
        valid_cats = cat_counts[cat_counts >= 10].index
        self.products = self.products[self.products['ItemClsCode'].isin(
            valid_cats)]
        print(f"排除小样本类别后行数: {len(self.products)}")
        print(f"剩余类别数: {len(self.products['ItemClsCode'].unique())}")

        return self.products

    def create_category_mapping(self) -> pd.DataFrame:
        """
        创建类别ID映射

        Returns:
        pandas.DataFrame: 包含ItemClsCode, cat_id, ClassName的映射表
        """
        if self.products is None:
            raise ValueError("请先加载数据（调用load_and_filter_data）")

        # 获取唯一的类别
        unique_cats = self.products[[
            'ItemClsCode', 'itemclsname']].drop_duplicates()
        # unique_cats = unique_cats.sort_values('ItemClsCode')

        # 去除空格
        unique_cats['ItemClsCode'] = unique_cats['ItemClsCode'].str.strip()
        # 分配cat_id
        unique_cats['cat_id'] = range(len(unique_cats))

        # 保存映射表
        self.cat_id_df = unique_cats[[
            'ItemClsCode', 'cat_id', 'itemclsname']].copy()
        self.cat_id_df.columns = ['ItemClsCode', 'cat_id', 'ClassName']

        # 创建双向映射字典
        self.cat_to_id = dict(self.cat_id_df[['ItemClsCode', 'cat_id']].values)
        self.id_to_cat = dict(self.cat_id_df[['cat_id', 'ItemClsCode']].values)

        # 将cat_id添加到products数据中
        self.products['cat_id'] = self.products['ItemClsCode'].map(
            self.cat_to_id)

        print(f"类别映射创建完成，共{len(self.cat_id_df)}个类别")
        return self.cat_id_df

    def preprocess_text(self, cells: pd.Series) -> str:
        """
        文本预处理函数（使用统一的预处理模块）

        Parameters:
        cells: 包含ItemCode, ItemName, unit, MainSupcode的pandas Series

        Returns:
        str: 预处理后的文本
        """
        return preprocessing.preprocess_single(cells)

    def apply_preprocessing(self) -> pd.DataFrame:
        """
        应用文本预处理到所有数据

        Returns:
        pandas.DataFrame: 包含preprocessed列的数据
        """
        if self.products is None:
            raise ValueError("请先加载数据（调用load_and_filter_data）")

        # 应用预处理
        self.products.loc[:, 'preprocessed'] = self.products[
            ['ItemCode', 'ItemName', 'unit', 'MainSupcode']
        ].apply(self.preprocess_text, axis=1)

        print(f"文本预处理完成，共{len(self.products)}条数据")
        return self.products

    def prepare_training_data(self, test_size: float = 0.25, random_state: int = 0) -> Tuple:
        """
        准备训练和测试数据

        Parameters:
        test_size: 测试集比例
        random_state: 随机种子

        Returns:
        tuple: (X_train, X_test, y_train, y_test, indices_train, indices_test)
        """
        if self.products is None or 'preprocessed' not in self.products.columns:
            raise ValueError("请先完成数据预处理")

        X = self.products['preprocessed'].values
        y = self.products['cat_id'].values

        # 使用分层抽样划分训练集和测试集
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X, y, self.products.index,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")

        return X_train, X_test, y_train, y_test, indices_train, indices_test

    def save_category_mapping(self, path: str = "data/cat_id_mapping.csv"):
        """
        保存类别映射表

        Parameters:
        path: 保存路径
        """
        if self.cat_id_df is not None:
            self.cat_id_df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"类别映射表已保存到: {path}")

    def save_processed_data(self, path: str = "data/products_processed.csv"):
        """
        保存处理后的数据

        Parameters:
        path: 保存路径
        """
        if self.products is not None:
            self.products.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"处理后的数据已保存到: {path}")


def test_preprocessor():
    """测试预处理器功能"""
    preprocessor = DataPreprocessor()

    # 1. 加载和过滤数据
    products = preprocessor.load_and_filter_data()
    print(f"过滤后数据形状: {products.shape}")

    # 2. 创建类别映射
    cat_id_df = preprocessor.create_category_mapping()
    print(f"类别映射表形状: {cat_id_df.shape}")
    print(cat_id_df.head())

    # 3. 应用文本预处理
    products_processed = preprocessor.apply_preprocessing()
    print(f"预处理后数据形状: {products_processed.shape}")
    print(products_processed[['ItemName', 'preprocessed']].head())

    # 4. 准备训练数据
    X_train, X_test, y_train, y_test, idx_train, idx_test = preprocessor.prepare_training_data()

    # 5. 保存结果
    preprocessor.save_category_mapping()
    preprocessor.save_processed_data()

    return preprocessor


if __name__ == "__main__":
    test_preprocessor()
