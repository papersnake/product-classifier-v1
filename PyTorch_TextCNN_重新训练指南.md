# PyTorch TextCNN 模型重新训练指南

## 📋 文档概述

本文档详细说明当新增商品分类或添加新商品后，如何重新训练 PyTorch TextCNN 模型的完整步骤。训练完成后，模型将自动集成到 Excel 插件中，确保 `PyTorch_col_Predict` 和 `col_Predict` 函数返回一致的结果。

**适用场景**：

- 新增商品分类（新增 `ItemClsCode`）
- 添加新的商品数据
- 模型性能需要优化
- 类别映射需要更新

**预期结果**：

- PyTorch 模型准确率达到或超过 sklearn 基线（当前：94.77%）
- Excel 插件中的两个预测函数返回一致结果
- 模型文件更新，支持新分类

---

## 🛠️ 环境准备

### 1. 激活 conda 环境

```bash
conda activate base
```

### 2. 检查依赖包

确保以下包已安装：

```bash
pip list | grep -E "(torch|pandas|numpy|jieba|scikit-learn|joblib)"
```

### 3. 安装缺失依赖

```bash
# PyTorch CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 其他依赖
pip install pandas numpy jieba scikit-learn joblib matplotlib seaborn
```

### 4. 验证环境

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import sklearn; print(f'sklearn版本: {sklearn.__version__}')"
```

---

## 📊 数据准备

### 1. 更新原始商品数据

编辑 `data/products.csv` 文件，添加新商品或新分类。确保包含以下列：

| 列名            | 类型     | 说明     | 示例           |
| ------------- | ------ | ------ | ------------ |
| `ItemClsCode` | string | 商品分类代码 | `"0126"`     |
| `itemclsname` | string | 分类名称   | `"低温液态奶"`    |
| `ItemCode`    | string | 商品代码   | `"99990017"` |
| `ItemName`    | string | 商品名称   | `"尊贵年货礼盒"`   |
| `unit`        | string | 单位     | `"盒"`        |
| `MainSupcode` | string | 供应商代码  | `"9999"`     |

**数据要求**：

- 每个分类至少需要 10 个样本
- 排除 `ItemClsCode` 以 `"23"` 开头的类别（系统保留）
- 排除 `ItemClsCode` 以 `"07"` 开头的类别（生鲜类）

### 2. 更新类别映射文件（关键步骤）

**重要**：必须确保两个类别映射文件保持一致：

| 文件                         | 用途           | 格式                             |
| -------------------------- | ------------ | ------------------------------ |
| `data/cat_id_mapping.csv`  | PyTorch 模型使用 | `ItemClsCode,cat_id,ClassName` |
| `data/cat_small_id_df.csv` | sklearn 模型使用 | `ItemClsCode,cat_id,ClassName` |

**检查一致性**：

```bash
python -c "
import pandas as pd
df1 = pd.read_csv('data/cat_id_mapping.csv', encoding='utf-8-sig')
df2 = pd.read_csv('data/cat_small_id_df.csv', encoding='utf-8-sig')
print('cat_id_mapping 类别数:', len(df1))
print('cat_small_id_df 类别数:', len(df2))
print('cat_id 差异:', set(df1['cat_id']) - set(df2['cat_id']))
print('ItemClsCode 差异:', set(df1['ItemClsCode'].str.strip()) - set(df2['ItemClsCode'].str.strip()))
"
```

**如果发现不一致**：

1. 备份现有文件
2. 运行数据预处理脚本重新生成映射文件
3. 手动检查并同步两个文件

---

## 🔧 数据预处理流程

### 步骤 1: 重新处理商品数据

运行数据预处理脚本，生成统一格式的训练数据：

```bash
python data_preprocessor.py
```

**此脚本将执行**：

1. 加载 `data/products.csv`
2. 过滤无效类别（23开头、07开头、样本数<10）
3. 创建新的类别映射（分配 `cat_id`）
4. 应用文本预处理（清洗、分词）
5. 生成 `data/products_processed.csv`
6. 保存类别映射到 `data/cat_id_mapping.csv`

**输出文件**：

- `data/products_processed.csv` - 预处理后的完整数据
- `data/cat_id_mapping.csv` - 新的类别映射（PyTorch用）

### 步骤 2: 同步 sklearn 类别映射

```bash
# 复制映射文件，确保两个模型使用相同的映射
cp data/cat_id_mapping.csv data/cat_small_id_df.csv
```

**或者手动更新**：

```python
import pandas as pd
df = pd.read_csv('data/cat_id_mapping.csv', encoding='utf-8-sig')
df.to_csv('data/cat_small_id_df.csv', index=False, encoding='utf-8')
```

### 步骤 3: 重建词汇表和嵌入矩阵

```bash
python build_vocab.py
```

**此脚本将执行**：

1. 从训练数据构建词汇表（最小词频=5）
2. 创建随机初始化的嵌入矩阵（维度=300）
3. 保存词汇表到 `data/vocab.pkl`
4. 保存嵌入矩阵到 `data/embedding_matrix.npy`
5. 分析序列长度并建议最大长度

**输出文件**：

- `data/vocab.pkl` - 词汇表文件
- `data/embedding_matrix.npy` - 词嵌入矩阵
- `data/seq_length_stats.txt` - 序列长度统计

**建议的最大序列长度**：

- 当前建议：11个词
- 覆盖95%的样本

---

## 🚀 模型训练

### 选项 A: 使用优化训练器（推荐）

```bash
python train_textcnn_optimized.py
```

**训练参数**（可在脚本中修改）：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 批大小，增大可提高CPU利用率 |
| `num_epochs` | 50 | 训练轮数 |
| `learning_rate` | 0.001 | 初始学习率 |
| `patience` | 5 | 早停耐心值 |
| `num_workers` | 4 | 数据加载工作线程数 |
| `use_amp` | True | 混合精度训练（CPU上禁用） |

**训练过程监控**：

- 每50个批次显示损失和准确率
- 每个epoch显示训练和测试结果
- 自动保存最佳模型
- 早停机制防止过拟合

**输出模型**：

- `data/textcnn_optimized.pth` - 训练完成的模型

### 选项 B: 使用其他训练脚本

```bash
# 更多训练轮数（80轮）
python train_extended.py

# 最终推进训练（目标94.78%准确率）
python train_final_push.py

# 快速训练（5轮，用于测试）
python train_5epoch.py
```

### 训练过程示例输出

```
=== 开始优化训练 ===
Epoch 1/50:
  批次    0/189: 损失=4.5123, 准确率=0.1406, 速度=128.5样本/秒
  Epoch 1训练完成: 损失=3.8765, 准确率=0.2345, 时间=32.45秒
  Epoch 1测试结果: 损失=3.6543, 准确率=0.4567
  [OK] 新的最佳模型已保存! 准确率: 0.4567
...
Epoch 25/50:
  批次  100/189: 损失=0.2345, 准确率=0.9123, 速度=145.2样本/秒
  Epoch 25训练完成: 损失=0.1897, 准确率=0.9345, 时间=28.76秒
  Epoch 25测试结果: 损失=0.2012, 准确率=0.9278
  [WAIT] 早停计数器: 3/5
```

### 训练中断恢复

如果训练中断，可以恢复到最后保存的检查点：

```bash
# 检查可用的检查点
ls -la data/textcnn_*.pth

# 重命名最佳模型
cp data/textcnn_epoch_25_acc_0.9278.pth data/textcnn_optimized.pth
```

---

## 📈 模型评估

### 1. 评估模型性能

```bash
python evaluate_model.py
```

**评估内容**：

1. 加载训练完成的 PyTorch 模型
2. 加载现有的 sklearn 模型
3. 在相同测试集上比较准确率
4. 生成分类报告和混淆矩阵

**预期输出**：

```
=== 模型性能对比 ===
PyTorch TextCNN 准确率: 94.23%
sklearn 基线准确率: 94.77%
差距: -0.54%
```

### 2. 详细评估报告

```bash
# 生成详细评估报告
python evaluate_model.py --detailed
```

**输出文件**：

- `data/classification_report_pytorch.txt` - PyTorch分类报告
- `data/classification_report_sklearn.txt` - sklearn分类报告
- `data/confusion_matrix.png` - 混淆矩阵可视化

### 3. 模型对比测试

```bash
python compare_models.py
```

**测试内容**：

- 随机选择100个样本进行预测
- 对比两个模型的预测结果
- 统计一致率和不一致样本

---

## 🔄 集成到 Excel 插件

### 步骤 1: 更新模型文件路径

检查 `myproject.py` 中的路径设置：

**关键路径**（第31-39行）：

```python
# sklearn模型路径
skmode_path = os.path.join(os.path.dirname(__file__), "data", "Tfidf_min_vect.pkl")
classifier_path = os.path.join(os.path.dirname(__file__), "data", "clf_min_model.pkl")

# 类别映射路径（必须一致！）
cat_id_df_path = os.path.join(os.path.dirname(__file__), "data", "cat_small_id_df.csv")
```

**PyTorch模型路径**（在 `get_pytorch_predictor` 函数中）：

```python
# pytorch_predictor.py 第32-35行
model_path = "data/textcnn_optimized.pth"
vocab_path = "data/vocab.pkl"
embedding_matrix_path = "data/embedding_matrix.npy"
cat_mapping_path = "data/cat_id_mapping.csv"
```

### 步骤 2: 测试 Excel 函数

```bash
# 测试 PyTorch 预测器
python test_pytorch_integration.py

# 测试 Excel 函数
python test_excel_functions.py
```

**测试内容**：

- 单个预测：`PyTorch_Predict`
- 批量预测：`PyTorch_col_Predict`
- 比较预测：`Compare_Predictions`

### 步骤 3: 启动 Excel 插件

```bash
python myproject.py
```

**在 Excel 中测试**：

1. 打开 Excel
2. 加载 `myproject.xlam` 插件（如果已创建）
3. 在单元格中输入公式：

**单个预测**：

```
=PyTorch_Predict("99990017 尊贵年货礼盒 盒 9999")
```

**批量预测**：

```
=PyTorch_col_Predict(A2:E100)
```

**比较预测**：

```
=Compare_Predictions("测试商品名称")
```

---

## 🧪 验证步骤

### 1. 一致性验证

确保 PyTorch 和 sklearn 预测结果一致：

```bash
python compare_models.py --validate
```

**验证指标**：

- 预测一致率（应 > 95%）
- 不一致样本分析
- 错误类型分类

### 2. 准确率验证

```bash
# 检查训练日志
python -c "
import torch
checkpoint = torch.load('data/textcnn_optimized.pth', map_location='cpu')
print('最佳测试准确率:', checkpoint.get('best_test_acc', 'N/A'))
print('训练轮数:', checkpoint.get('epoch', 'N/A'))
print('训练损失历史:', len(checkpoint.get('train_loss_history', [])))
print('测试损失历史:', len(checkpoint.get('test_loss_history', [])))
"
```

### 3. 端到端测试

```bash
# 完整测试流程
python final_summary.py
```

**测试内容**：

1. 数据预处理验证
2. 模型加载测试
3. 预测功能测试
4. Excel集成测试

---

## ⚠️ 注意事项

### 关键检查点

1. **类别映射一致性**：确保 `cat_id_mapping.csv` 和 `cat_small_id_df.csv` 的 `cat_id` 完全一致
2. **词汇表更新**：新增商品可能引入新词汇，必须重新构建词汇表
3. **序列长度**：最大序列长度应为11（覆盖95%样本）
4. **文件编码**：所有 CSV 文件使用 `utf-8-sig` 编码

### 常见问题解决

#### 问题1: 预测结果不一致

```bash
# 调试预测不一致
python -c "
from myproject import Compare_Predictions
result = Compare_Predictions('测试商品')
print('比较结果:', result)
"

# 检查类别映射
python -c "
import pandas as pd
df1 = pd.read_csv('data/cat_id_mapping.csv')
df2 = pd.read_csv('data/cat_small_id_df.csv')
print('cat_id_mapping 前5行:')
print(df1.head())
print('\ncat_small_id_df 前5行:')
print(df2.head())
"
```

#### 问题2: 模型加载失败

```bash
# 检查模型文件
python inspect_checkpoint.py

# 测试模型预测
python test_pytorch_integration.py
```

#### 问题3: Excel 插件不工作

```bash
# 测试路径
python test_path_fix.py

# 重新安装依赖
pip install xlwings --upgrade
```

### 性能优化建议

1. **CPU 利用率**：增大 `batch_size`（64-128）
2. **数据加载**：增加 `num_workers`（2-4个线程）
3. **内存管理**：监控内存使用，避免过大批处理
4. **模型大小**：TextCNN 参数量约 2.5M，适合 CPU 推理

---

## 📁 文件更新清单

重新训练后以下文件会被更新：

### 数据文件

```
data/
├── products_processed.csv      # 预处理后的商品数据
├── vocab.pkl                  # 词汇表文件
├── embedding_matrix.npy       # 嵌入矩阵
├── cat_id_mapping.csv         # PyTorch 类别映射（必须！）
├── cat_small_id_df.csv        # sklearn 类别映射（必须一致！）
├── seq_length_stats.txt       # 序列长度统计
└── model_info.csv             # 模型信息记录
```

### 模型文件

```
data/
├── textcnn_optimized.pth      # PyTorch 模型权重（主模型）
├── textcnn_epoch_*_acc_*.pth  # 各个epoch的检查点
├── textcnn_final.pth          # 最终模型（备份）
├── Tfidf_min_vect.pkl         # sklearn TF-IDF 向量器
└── clf_min_model.pkl          # sklearn 分类器
```

### 日志文件

```
data/
├── training_log.txt           # 训练日志
├── evaluation_report.txt      # 评估报告
└── comparison_results.csv     # 模型对比结果
```

---

## 🚨 紧急恢复

### 1. 训练中断恢复

```bash
# 查找最新的检查点
ls -t data/textcnn_*.pth | head -5

# 恢复到最后可用的检查点
cp data/textcnn_epoch_20_acc_0.9234.pth data/textcnn_optimized.pth

# 重新启动训练（从指定epoch继续）
python train_textcnn_optimized.py --resume data/textcnn_optimized.pth
```

### 2. 模型回滚

```bash
# 备份当前模型
cp data/textcnn_optimized.pth data/textcnn_optimized_backup_$(date +%Y%m%d_%H%M%S).pth

# 回滚到上一版本
cp data/textcnn_final.pth data/textcnn_optimized.pth
```

### 3. 数据损坏恢复

```bash
# 重新生成所有预处理数据
python data_preprocessor.py --force
python build_vocab.py --force
```

---

## 📞 技术支持

### 1. 快速诊断

```bash
# 运行完整诊断
python final_summary.py --diagnose
```

### 2. 获取系统信息

```bash
# 显示环境信息
python -c "
import sys, torch, sklearn, pandas as pd
print('Python版本:', sys.version[:20])
print('PyTorch版本:', torch.__version__)
print('sklearn版本:', sklearn.__version__)
print('pandas版本:', pd.__version__)
print('工作目录:', os.getcwd())
"
```

### 3. 验证步骤检查表

- [ ] 数据预处理完成（`products_processed.csv` 存在）
- [ ] 词汇表构建完成（`vocab.pkl` 存在）
- [ ] 类别映射一致（两个CSV文件的 `cat_id` 相同）
- [ ] 模型训练完成（`textcnn_optimized.pth` 存在）
- [ ] 评估通过（准确率 > 94%）
- [ ] Excel 插件测试通过

### 4. 联系支持

如果遇到无法解决的问题：

1. 保存完整的训练日志
2. 备份所有数据文件
3. 记录错误信息和截图
4. 联系开发团队：papersnake cctv5cn@gmail.com

---

## 🔄 完整工作流程总结

### 第一阶段：数据准备（10分钟）

1. 更新 `data/products.csv` 添加新数据
2. 运行 `python data_preprocessor.py`
3. 同步类别映射文件

### 第二阶段：词汇表构建（5分钟）

1. 运行 `python build_vocab.py`
2. 检查序列长度统计

### 第三阶段：模型训练（30-60分钟）

1. 运行 `python train_textcnn_optimized.py`
2. 监控训练过程，等待早停
3. 保存最佳模型

### 第四阶段：评估验证（10分钟）

1. 运行 `python evaluate_model.py`
2. 运行 `python compare_models.py`
3. 检查准确率和一致性

### 第五阶段：集成部署（5分钟）

1. 更新模型路径（如果需要）
2. 测试 Excel 函数
3. 启动 Excel 插件

**总时间**：约60-90分钟

---

## 📝 更新记录

| 版本  | 日期         | 更新内容          | 作者         |
| --- | ---------- | ------------- | ---------- |
| 1.0 | 2026-03-14 | 初始版本，完整重新训练指南 | papersnake |
| 1.1 | 2026-03-14 | 增加紧急恢复和问题解决章节 | papersnake |

**文档状态**: ✅ 完成  
**最后验证**: 2026-03-14  
**适用版本**: PyTorch TextCNN v1.0  
**兼容性**: Python 3.8+, PyTorch 1.12+, Excel 2016+

---

> **重要提示**：重新训练前请务必备份现有模型和数据文件。训练过程中不要关闭命令行窗口，直到看到"训练完成"提示。