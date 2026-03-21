# 产品分类与价格计算系统

基于机器学习的产品自动分类和价格调整的 Excel 插件系统。

## 功能特性

### 产品分类
- **多模型支持**: sklearn (TF-IDF + 分类器) 和 PyTorch TextCNN
- **ONNX 部署**: 支持 ONNX 格式模型导出和部署
- **Excel 集成**: 通过 xlwings 实现 Excel 函数调用

### 价格计算
- `AutoSalePrice(原价)`: 自动计算促销价格
- 智能小数处理

## 项目结构

```
├── myproject.py           # Excel 插件主程序
├── price_calculator.py    # 价格计算模块
├── preprocessing.py       # 文本预处理
├── pytorch_predictor.py   # PyTorch 预测器
├── text_cnn.py            # TextCNN 模型架构
├── data_preprocessor.py   # 数据预处理管道
├── evaluate_model.py      # 模型评估
├── train_*.py             # 各类训练脚本
└── data/                  # 模型和数据文件
```

## 环境配置

```bash
pip install scikit-learn jieba xlwings joblib onnxruntime pandas numpy skl2onnx torch
```

PyTorch CPU 版本:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 使用方法

### 运行 Excel 插件
```bash
python myproject.py
```

### 价格计算
```python
from price_calculator import AutoSalePrice
print(AutoSalePrice(25.7))  # 输出: 25.7 的促销价
```

## 模型性能

| 模型 | 准确率 |
|------|--------|
| sklearn | ~94.77% |
| PyTorch TextCNN | ~94.23% |

## 技术栈

- **机器学习**: scikit-learn, PyTorch
- **文本处理**: jieba (中文分词)
- **Excel 集成**: xlwings
- **模型格式**: ONNX, joblib

## License

Copyright (c) 2025 by papersnake. All Rights Reserved.
