import sys
import os

import pandas as pd
import numpy as np
import jieba as jb
import joblib
from sklearn.metrics import accuracy_score, classification_report
from text_dataset import TextClassificationDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载数据
print("加载数据...")
raw_data = pd.read_csv("data/products_processed.csv", encoding='utf-8-sig')
print(f"数据行数: {len(raw_data)}")

# 加载sklearn模型
print("加载sklearn模型...")
vectorizer = joblib.load("data/Tfidf_min_vect.pkl")
clf = joblib.load("data/clf_min_model.pkl")
print(f"特征数: {len(vectorizer.get_feature_names_out())}")

# 加载类别映射
# cat_mapping = pd.read_csv("data/cat_id_mapping.csv", encoding='utf-8-sig')
cat_mapping = pd.read_csv("data/cat_small_id_df.csv", encoding='utf-8-sig')
print(f"类别数: {len(cat_mapping)}")

# 加载测试集索引（与PyTorch测试集相同）

test_dataset = TextClassificationDataset(mode='test')
print(f"测试集大小: {len(test_dataset)}")

# 准备sklearn格式的文本
print("预处理文本...")
sklearn_texts = []
for idx in test_dataset.indices:
    item_name = raw_data.loc[idx, 'ItemName']
    if not isinstance(item_name, str):
        item_name = str(item_name) if not pd.isna(item_name) else ""
    words = jb.cut(item_name)
    format_sec = " ".join(words)
    sklearn_texts.append(format_sec)

# 转换和预测
print("转换和预测...")
X_test = vectorizer.transform(sklearn_texts)
y_true = np.array(test_dataset.labels)
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print(f"\nsklearn模型准确率: {accuracy:.4f}")
print("目标准确率: 0.9478 (94.78%)")
print(f"差距: {accuracy - 0.9478:+.4f}")

# 分类报告
print("\n分类报告:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(len(cat_mapping))]))

# 保存结果
with open("data/sklearn_accuracy.txt", "w", encoding='utf-8') as f:
    f.write(f"准确率: {accuracy:.4f}\n")
    f.write("目标: 0.9478\n")
    f.write(f"差距: {accuracy - 0.9478:+.4f}\n")

print("结果已保存到 data/sklearn_accuracy.txt")
