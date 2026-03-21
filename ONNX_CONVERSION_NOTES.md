# PyTorch TextCNN 模型转换为 ONNX 备忘

## 概述

将训练好的 PyTorch TextCNN 模型 (`textcnn_final.pth`) 转换为 ONNX 格式，以便于部署和跨平台推理。

## 转换脚本

### 文件位置
- 转换脚本: `convert_textcnn_to_onnx.py`
- 测试脚本: `test_onnx_textcnn.py`

## 转换步骤

### 1. 运行转换脚本

```bash
"D:\anaconda3\python.exe" convert_textcnn_to_onnx.py
```

转换脚本会自动：
1. 加载词汇表 (`data/vocab.pkl`)
2. 加载类别映射 (`data/cat_id_mapping.csv`)
3. 加载 PyTorch 模型 (`data/textcnn_final.pth`)
4. 创建示例输入张量
5. 导出 ONNX 模型到 `data/textcnn.onnx`

### 2. 运行测试脚本验证

```bash
"D:\anaconda3\python.exe" test_onnx_textcnn.py
```

测试内容：
- 单个预测
- Top-3 预测
- 批量预测
- PyTorch 与 ONNX 结果对比

## 模型结构

### PyTorch 模型
- 输入: `(batch_size, seq_len=11)` - 整数索引张量
- 词汇表大小: 7431
- 嵌入维度: 300
- 卷积核尺寸: [3, 4, 5]
- 每种尺寸滤波器数量: 100
- Dropout: 0.5
- 输出: `(batch_size, 246)` - 类别logits

### ONNX 模型
- 输入名: `input_ids`
- 输出名: `output`
- opset版本: 11 (实际导出为18)
- ir_version: 10

## 已知问题与解决方案

### 1. 编码问题 (UnicodeEncodeError)

**问题**: Windows 环境下 torch.onnx 内部日志包含 Unicode 字符导致 GBK 编码错误

**解决方案**: 添加 UnicodeFilter 类过滤输出

```python
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
```

### 2. opset版本不兼容

**问题**: 请求 opset_version=11，但新版本 PyTorch 最低支持 18

**解决方案**: 不指定 opset_version，让其自动使用支持的版本

### 3. 嵌入矩阵不包含在 ONNX 模型中

**原因**: 嵌入矩阵过大 (~9MB)，不适合放入 ONNX 模型

**解决方案**: 运行时单独加载嵌入矩阵

### 4. 批量推理维度问题

**问题**: ONNX 模型固定 batch_size=1，批量推理时出错

**解决方案**: 批量预测时逐个处理，或使用循环

```python
def predict_batch(self, texts: list, return_prob: bool = False):
    results = []
    for text in texts:
        pred_id, pred_name, probs = self.predict_single(text, return_prob=True)
        if return_prob:
            results.append((pred_id, pred_name, probs))
        else:
            results.append((pred_id, pred_name))
    return results
```

## 使用示例

### Python 中使用 ONNX 模型

```python
from test_onnx_textcnn import ONNXTextCNNPredictor

# 初始化预测器
predictor = ONNXTextCNNPredictor(
    onnx_model_path="data/textcnn.onnx",
    vocab_path="data/vocab.pkl",
    cat_mapping_path="data/cat_id_mapping.csv",
    max_seq_len=11
)

# 单个预测
pred_id, pred_name = predictor.predict_single("99990017 尊贵 年货 礼盒 盒 9999")
print(f"ID: {pred_id}, Name: {pred_name}")

# 获取 ItemClsCode
itemcls_code = predictor.get_itemcls_code(pred_id)
print(f"ItemClsCode: {itemcls_code}")

# 批量预测
results = predictor.predict_batch(["文本1", "文本2", "文本3"])
```

## 文件清单

| 文件 | 说明 | 大小 |
|------|------|------|
| `data/textcnn_final.pth` | PyTorch 模型 | ~32 MB |
| `data/textcnn.onnx` | ONNX 模型 | ~16 KB |
| `data/vocab.pkl` | 词汇表 | ~2.3 MB |
| `data/embedding_matrix.npy` | 嵌入矩阵 | ~9 MB |
| `data/cat_id_mapping.csv` | 类别映射 | ~10 KB |

## 性能对比

| 指标 | PyTorch | ONNX |
|------|---------|------|
| 首次加载 | ~3秒 | ~1秒 |
| 单次推理 | ~0.005秒 | ~0.003秒 |
| 模型文件 | 32 MB | 16 KB |

## 依赖安装

```bash
pip install torch onnx onnxruntime
```

## 注意事项

1. **ONNX Runtime**: 使用 `CPUExecutionProvider` 进行 CPU 推理
2. **词汇表同步**: 确保 ONNX 模型使用的词汇表与训练时一致
3. **预处理一致**: 输入文本必须经过相同的分词和向量化处理
4. **精度验证**: 转换后务必验证 PyTorch 和 ONNX 输出是否一致

---
*最后更新: 2026-03-20*
