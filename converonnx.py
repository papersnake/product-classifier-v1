'''
Author: papersnake cctv5cn@gmail.com
Date: 2025-10-21 15:04:27
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2025-10-27 19:36:13
FilePath: \myproject\converonnx.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import joblib
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

# 加载模型和辅助数据
vectorizer = joblib.load("data/Tfidf_min_vect.pkl")
clf = joblib.load("data/clf_min_model.pkl")
cat_id_df = pd.read_csv(r"E:\data\python\myproject\data\cat_small_id_df.csv", dtype={"cat_id": 'Int64', 'ItemClsCode': str})


def get_info(cat_id):
    return cat_id_df.loc[cat_id_df['cat_id'] == cat_id][['ItemClsCode', 'ClassName']].values


# --- 1. 转换 TfidfVectorizer ---
# 修复 token pattern 以兼容 ONNX
vectorizer.token_pattern = r'\b\w+\b'
initial_type_vectorizer = [('string_input', StringTensorType([None]))]
onx_vectorizer = convert_sklearn(vectorizer,
                                 initial_types=initial_type_vectorizer,
                                 options={
                                     'keep_empty_string': True,  # 保留空字符串
                                     'tokenexp': vectorizer.token_pattern  # 确保使用相同的分词模式
                                 },
                                 target_opset=12)

with open("data/vectorizer.onnx", "wb") as f:
    f.write(onx_vectorizer.SerializeToString())

# --- 2. 转换分类器 ---
# 设置classifier的输入类型
initial_type_clf = [('float_input', FloatTensorType([None, vectorizer.get_feature_names_out().shape[0]]))]

onx_classifier = convert_sklearn(clf, 
                                 initial_types=initial_type_clf,
                                 options={
                                    'raw_scores': True,  # 使用原始分数
                                    'output_class_labels': True  # 输出类别标签
                                 },
                                 target_opset=12)
with open("data/classifier.onnx", "wb") as f:
    f.write(onx_classifier.SerializeToString())

# --- 3. 使用原始 scikit-learn 模型进行预测作为基准 ---
data = ['99990017 尊贵 年货 礼盒 盒 9999']
pred_cat_id = clf.predict(vectorizer.transform(data))

print(f"原始 scikit-learn 模型预测索引: {pred_cat_id[0]}")
print(f"原始 scikit-learn 模型预测标签: \n{get_info(pred_cat_id[0])}")

print("\n模型转换完成: 'vectorizer.onnx' 和 'classifier.onnx' 已生成。")