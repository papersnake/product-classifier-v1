'''
Author: papersnake cctv5cn@gmail.com
Date: 2025-10-22 19:23:45
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2025-11-02 15:45:22
FilePath: \myproject\compare_models.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

# 加载原始模型
model_pipeline = joblib.load('data/model_pipeline.pkl')


# 加载ONNX模型
onnx_model_pipeline = ort.InferenceSession("data/pipeline.onnx")


# 加载类别信息用于展示
cat_id_df = pd.read_csv(r"E:\data\python\myproject\data\cat_small_id_df.csv", dtype={"cat_id": 'Int64', 'ItemClsCode': str})

# 准备测试数据
test_data = ['99990017 尊贵 年货 礼盒 盒 9999']


print("=== 1. Scikit-learn 模型预测过程 ===")


# 预测
prediction_pkl = model_pipeline.predict(test_data)
print(f"\n预测结果: {prediction_pkl[0]}")

print("\n=== 2. ONNX 模型预测过程 ===")
# 添加模型信息调试
print("Pipeline 模型信息:")
print(f"输入: {onnx_model_pipeline.get_inputs()}")
print(f"输出: {onnx_model_pipeline.get_outputs()}")

# 获取输入输出名称
input_name = onnx_model_pipeline.get_inputs()[0].name
output_name = onnx_model_pipeline.get_outputs()[0].name

# 准备输入数据
test_data_2d = np.array(test_data).reshape(-1, 1)  # 转换为2D数组
# 直接使用pipeline进行预测
prediction_onnx = onnx_model_pipeline.run([output_name], {input_name: test_data_2d})[0]
print(f"\n预测结果: {prediction_onnx[0]}")

print("\n=== 3. 结果对比 ===")



# 获取类别信息
def get_category_info(cat_id):
    info = cat_id_df.loc[cat_id_df['cat_id'] == cat_id][['ItemClsCode', 'ClassName']].values
    return f"ID: {cat_id}, Info: {info[0] if len(info) > 0 else '未找到'}"


print("\n类别信息:")
print(f"Scikit-learn: {get_category_info(prediction_pkl[0])}")
print(f"ONNX: {get_category_info(prediction_onnx[0])}")