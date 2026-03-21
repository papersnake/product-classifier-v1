# AGENTS.md - Agent Coding Guidelines

## Project Overview

This is a Python project for product classification and price calculation. It includes:
- **price_calculator.py**: Price adjustment algorithms
- **myproject.py**: xlwings Excel plugin with ML pipeline (TF-IDF + classifier)
- **Machine learning models**: sklearn, ONNX formats in `data/` directory

## Dependencies

Core dependencies (install via pip):
- scikit-learn
- jieba
- xlwings
- joblib
- onnxruntime
- pandas
- numpy
- skl2onnx
- torch (CPU version)

For PyTorch CPU version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 环境配置
建议使用 conda base 环境运行代码:
```bash
conda activate base
```

## Build/Lint/Test Commands

### Running the Project
```bash
# Run main Excel plugin
python myproject.py

# Run price calculator directly
python -c "from price_calculator import AutoSalePrice; print(AutoSalePrice(25.7))"
```

### Testing Individual Scripts
```bash
# Test price calculator
python price_calculator.py

# Test ONNX pipeline
python testpipelin.py

# Test ONNX model
python testonnx.py

# Compare sklearn vs ONNX predictions
python compare_models.py

# Convert models to ONNX
python converonnx.py
```

### Running Single Tests
Since there are no formal test frameworks (pytest/unittest), run individual test scripts directly:
```bash
# Test a specific function
python -c "from price_calculator import AutoSalePrice; assert AutoSalePrice(25) == 24.9"
```

## Code Style Guidelines

### File Header Template
Use this header for new Python files:
```python
'''
Author: papersnake cctv5cn@gmail.com
Date: YYYY-MM-DD HH:MM:SS
LastEditors: papersnake cctv5cn@gmail.com
LastEditTime: YYYY-MM-DD HH:MM:SS
FilePath: \myproject\filename.py
Description: 

Copyright (c) 2025 by papersnake, All Rights Reserved. 
'''
```

### Naming Conventions
- **Functions/variables**: snake_case (e.g., `get_decimal`, `adjusted_value`)
- **Classes**: PascalCase (if used)
- **Constants**: UPPER_SNAKE_CASE
- **Files**: snake_case .py files

### Type Hints
Use type hints for function signatures:
```python
def sort_dict_by_value(d: dict, reverse: bool = False) -> dict:
def preprocess(cells) -> str:
```

### Docstrings
Use Chinese docstrings with parameter descriptions:
```python
def get_decimal(value):
    """
    获取数值的小数部分
    
    Parameters:
    value (float): 输入的数值
    
    Returns:
    float: 小数部分
    """
```

### Imports
- Standard library imports first
- Third-party imports second
- Local imports last
- Group imports by category with blank lines between groups

Example:
```python
import math
import random

import joblib
import pandas as pd

from price_calculator import AutoSalePrice
```

### Error Handling
- Use try-except for file I/O operations
- Handle missing data gracefully with appropriate fallbacks
- Print informative error messages for debugging

### Formatting
- Use 4 spaces for indentation (no tabs)
- Maximum line length: ~120 characters
- Use blank lines to separate logical sections
- Keep related code together

### ML Model Guidelines
- Store models in `data/` directory
- Use joblib for sklearn models (.pkl)
- Use skl2onnx for ONNX conversion
- Keep model paths as relative paths or configurable constants

### Excel Integration (xlwings)
- Use `@xw.func` decorator for Excel UDFs
- Use `@xw.arg` and `@xw.ret` for DataFrame handling
- Test Excel functions separately before integration

## Project Structure

```
myproject/
├── price_calculator.py    # Price calculation logic
├── myproject.py           # Main Excel plugin (now with PyTorch integration)
├── compare_models.py      # sklearn vs ONNX comparison
├── converonnx.py          # ONNX conversion script
├── test*.py               # Various test scripts
├── pytorch_predictor.py   # PyTorch model predictor for Excel plugin
├── preprocessing.py       # Unified text preprocessing module
├── text_cnn.py           # TextCNN model architecture
├── text_dataset.py       # PyTorch dataset and dataloader
├── embedding_manager.py  # Vocabulary and embedding management
├── data_preprocessor.py  # Data preprocessing pipeline
├── train_*.py            # Various training scripts
├── evaluate_model.py     # Model evaluation and comparison
├── data/                 # ML models and data
│   ├── *.pkl            # sklearn models (Tfidf_min_vect.pkl, clf_min_model.pkl)
│   ├── *.onnx           # ONNX models
│   ├── *.pth            # PyTorch model checkpoints (textcnn_final.pth)
│   ├── vocab.pkl        # Vocabulary for PyTorch model
│   ├── embedding_matrix.npy  # Embedding matrix
│   ├── cat_id_mapping.csv    # Category mapping (sorted)
│   ├── products_processed.csv # Preprocessed product data
│   └── *.csv            # Other category mappings
└── products.csv         # Raw product data
```

## Notes for Agents

- Hardcoded paths reference `E:\data\python\myproject\` - adjust as needed
- The project uses Chinese comments extensively - maintain consistency
- No formal CI/CD or test framework exists
- Model files (Tfidf_vect.pkl, clf_*.pkl) are pre-trained and stored in data/
- PyTorch TextCNN model (textcnn_final.pth) achieves 94.23% accuracy (vs sklearn's 94.77%)
- The PyTorch model is fully integrated into the Excel plugin via pytorch_predictor.py
- Use unified preprocessing.py for consistent text preprocessing
- **Path fixes applied**: All relative paths (especially `data/productnames.dict`) have been converted to absolute paths using `os.path.join(os.path.dirname(__file__), ...)` to avoid `FileNotFoundError` when importing modules from different directories