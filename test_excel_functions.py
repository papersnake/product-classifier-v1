import sys
import os
# Simulate Excel cell inputs as pandas Series
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


print("Testing Excel plugin functions...")
print("=" * 60)

# Import functions from myproject
from myproject import PyTorch_Predict, Compare_Predictions

# Test 1: PyTorch prediction
print("\n1. Testing PyTorch_Predict function:")
# Simulate Excel cell inputs (ItemCode, ItemName, unit, MainSupcode)
test_input = pd.Series({
    'ItemCode': '99990017',
    'ItemName': '尊贵年货礼盒',
    'unit': '盒',
    'MainSupcode': '9999'
})

try:
    result = PyTorch_Predict(test_input)
    print(f"   Input: {test_input.to_dict()}")
    print(f"   Result: {result}")
    print("   [OK] PyTorch_Predict works")
except Exception as e:
    print(f"   [ERROR] PyTorch_Predict failed: {e}")

# Test 2: Compare predictions
print("\n2. Testing Compare_Predictions function:")
try:
    result = Compare_Predictions(test_input)
    print(f"   Input: {test_input.to_dict()}")
    print(f"   Result: {result}")
    print("   [OK] Compare_Predictions works")
except Exception as e:
    print(f"   [ERROR] Compare_Predictions failed: {e}")

# Test 3: Test multiple inputs
print("\n3. Testing multiple inputs:")
test_inputs = [
    pd.Series({'ItemCode': '69113340', 'ItemName': '双鹿极能碱性电池5#4粒', 'unit': '个', 'MainSupcode': '2010'}),
    pd.Series({'ItemCode': '87654321', 'ItemName': '华为笔记本电脑', 'unit': '台', 'MainSupcode': '2002'}),
    pd.Series({'ItemCode': '55556666', 'ItemName': '测试商品', 'unit': '件', 'MainSupcode': '3003'}),
]

for i, test_input in enumerate(test_inputs, 1):
    try:
        pytorch_result = PyTorch_Predict(test_input)
        print(f"   Input {i}: {test_input['ItemName']}")
        print(f"     PyTorch: {pytorch_result}")
    except Exception as e:
        print(f"   Input {i} failed: {e}")

print("\n" + "=" * 60)
print("Excel plugin functions test completed.")
