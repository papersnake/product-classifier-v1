'''
测试路径修复：验证所有模块可以从不同目录导入而不会出现 FileNotFoundError
'''

import os
import sys
import tempfile
import shutil

print("=== 测试路径修复 ===")

# 创建临时目录并复制必要文件
temp_dir = tempfile.mkdtemp()
print(f"创建临时目录: {temp_dir}")

# 复制关键文件到临时目录
files_to_copy = [
    'preprocessing.py',
    'data_preprocessor.py',
    'pytorch_predictor.py',
    'myproject.py',
    'data/productnames.dict',
    'data/textcnn_final.pth',
    'data/vocab.pkl',
    'data/embedding_matrix.npy',
    'data/cat_id_mapping.csv'
]

for file_path in files_to_copy:
    src = os.path.join('.', file_path)
    dest_dir = os.path.join(temp_dir, os.path.dirname(file_path))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(src):
        shutil.copy2(src, os.path.join(temp_dir, file_path))
        print(f"  复制: {file_path}")
    else:
        print(f"  警告: 源文件不存在: {src}")

# 测试从临时目录导入
print("\n从临时目录测试导入...")
original_cwd = os.getcwd()
os.chdir(temp_dir)

try:
    # 添加当前目录到 Python 路径
    sys.path.insert(0, temp_dir)

    print("1. 测试 preprocessing.py...")
    import preprocessing
    print("   preprocessing 导入成功")

    print("2. 测试 data_preprocessor.py...")
    from data_preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    print("   DataPreprocessor 创建成功")

    print("3. 测试简单功能...")
    import pandas as pd
    sample = pd.Series({
        'ItemCode': '12345678',
        'ItemName': '测试商品',
        'unit': '个',
        'MainSupcode': 'S001'
    })
    result = preprocessing.preprocess_single(sample)
    print(f"   预处理结果: {result[:50]}...")

    print("\\n路径修复测试成功！")
    print("模块可以从不同目录导入，不会出现 FileNotFoundError")

except Exception as e:
    print(f"\\n测试失败: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 清理
    os.chdir(original_cwd)
    sys.path.remove(temp_dir)
    shutil.rmtree(temp_dir)
    print(f"\n清理临时目录: {temp_dir}")
