'''
Author: papersnake cctv5cn@gmail.com
Date: 2025-10-31 20:26:37
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2025-10-31 20:28:33
FilePath: \myproject\testpipelin.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

# 加载和使用 pipeline
model = joblib.load('data/model_pipeline.pkl')

# 直接预测（不需要分别调用 vectorizer 和 classifier）
test_data = ['99990017 尊贵 年货 礼盒 盒 9999']
prediction = model.predict(test_data)

print(f"Pipeline 预测结果: {prediction[0]}")

# 转换为 ONNX
initial_type = [('string_input', StringTensorType([None, 1]))]
onx_pipeline = convert_sklearn(
    model,
    initial_types=initial_type,
    options={
        'keep_empty_string': True,
        'raw_scores': True,
        'output_class_labels': True
    },
    target_opset=12
)

# 保存 ONNX 模型
with open("data/pipeline.onnx", "wb") as f:
    f.write(onx_pipeline.SerializeToString())