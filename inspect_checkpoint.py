'''
Author: papersnake cctv5cn@gmail.com
Date: 2026-03-14 16:39:32
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: 2026-03-20 15:15:08
FilePath: \\myproject\\inspect_checkpoint.py
Description: 检查PyTorch模型检查点文件内容,验证训练结果和模型状态

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved.
'''
import torch

checkpoint_path = "data/textcnn_final.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print("Keys:", checkpoint.keys())
for key, value in checkpoint.items():
    if isinstance(value, (int, float, str)):
        print(f"{key}: {value}")
    elif isinstance(value, dict):
        print(f"{key}: dict with keys {list(value.keys())}")
    else:
        print(f"{key}: type {type(value)}")

if 'best_test_acc' in checkpoint:
    print(f"Best test accuracy: {checkpoint['best_test_acc']}")
if 'train_acc' in checkpoint:
    print(f"Train accuracy: {checkpoint['train_acc']}")
if 'test_acc' in checkpoint:
    print(f"Test accuracy: {checkpoint['test_acc']}")
if 'epoch' in checkpoint:
    print(f"Epoch: {checkpoint['epoch']}")
