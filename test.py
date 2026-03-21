'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-13 13:28:22
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 20:31:25
FilePath: \\myproject\\test.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved.
'''
import onnx
import os
model = onnx.load("data/textcnn.onnx")

if model.graph.initializer:
    print(f"内嵌权重数量: {len(model.graph.initializer)}")
else:
    print("没有内嵌权重")
# 检查是否引用了外部数据
if hasattr(model, "external_data") and model.external_data:
    print("外部数据文件:", model.external_data[0].location)
else:
    print("未使用外部数据")
    
for init in model.graph.initializer:
    data_size = len(init.raw_data)
    print(f"{init.name}: shape {list(init.dims)}, raw_data size = {data_size} bytes")
    
# 检查是否有任何 initializer 引用了外部数据
has_external = False
for init in model.graph.initializer:
    if init.external_data:
        print(f"Initializer '{init.name}' 引用了外部数据:")
        for ext in init.external_data:
            print(f"  - 文件: {ext.key} = {ext.value}")
        has_external = True

if not has_external:
    print("没有 initializer 引用外部数据，所有权重已内嵌。")

# 此外，有些模型可能使用整体外部数据存储，可通过检查文件大小判断

size = os.path.getsize("data/textcnn.onnx")
print(f"ONNX 文件大小: {size / 1024:.2f} KB")

path = "data/textcnn.onnx"
print(f"文件大小: {os.path.getsize(path)} bytes")

model = onnx.load(path)
print(f"模型序列化后字节数: {model.ByteSize()}")  # 应该与文件大小接近

total_raw = 0
for init in model.graph.initializer:
    total_raw += len(init.raw_data)
print(f"所有 initializer raw_data 总和: {total_raw} bytes")

for f in os.listdir("data/"):
    if ".data" in f:
        print(f, os.path.getsize(os.path.join("data", f)))