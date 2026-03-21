'''
Simple evaluation of sklearn baseline with factorized labels.
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load processed data (already filtered)
print("Loading processed data...")
data = pd.read_csv("data/products_processed.csv", encoding='utf-8-sig')
print(f"Rows: {len(data)}")

# Factorize ItemClsCode (order of appearance) to match notebook's cat_id
data['cat_id_factor'] = pd.factorize(data['ItemClsCode'])[0]
print(f"Factorized cat_id range: {data['cat_id_factor'].min()} to {data['cat_id_factor'].max()}")
print(f"Unique categories: {data['cat_id_factor'].nunique()}")

# Split using random_state=0, stratify=cat_id_factor
X = data['preprocessed'].values
y = data['cat_id_factor'].values
indices = data.index.values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.25, random_state=0, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Load sklearn model
print("Loading sklearn model...")
vectorizer = joblib.load("data/Tfidf_min_vect.pkl")
clf = joblib.load("data/clf_min_model.pkl")
print(f"Feature count: {len(vectorizer.get_feature_names_out())}")

# Ensure no NaN in X_test
nan_mask = pd.isna(X_test)
if nan_mask.any():
    print(f"Warning: {nan_mask.sum()} NaN in X_test, removing.")
    X_test = X_test[~nan_mask]
    y_test = y_test[~nan_mask]
    idx_test = idx_test[~nan_mask]

# Transform and predict
X_test_vec = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_vec)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nSklearn model accuracy (correct mapping): {accuracy:.4f}")
print(f"Target accuracy: 0.9478")
print(f"Difference: {accuracy - 0.9478:+.4f}")

# Save result
with open("data/sklearn_baseline_accuracy_fixed.txt", "w", encoding='utf-8') as f:
    f.write(f"准确率: {accuracy:.4f}\n")
    f.write(f"目标: 0.9478\n")
    f.write(f"差距: {accuracy - 0.9478:+.4f}\n")

print("\n--- Evaluating PyTorch model on same test set ---")
# Load PyTorch predictor
from pytorch_predictor import TextCNNPredictor
predictor = TextCNNPredictor(
    model_path="data/textcnn_final.pth",
    vocab_path="data/vocab.pkl",
    embedding_matrix_path="data/embedding_matrix.npy",
    cat_mapping_path="data/cat_id_mapping.csv",
    max_seq_len=11
)

# Predict each test sample (using preprocessed text)
y_pred_pytorch = []
for i, text in enumerate(X_test):
    pred_id, pred_name = predictor.predict_single(text)
    y_pred_pytorch.append(pred_id)
    if i % 1000 == 0:
        print(f"  Processed {i}/{len(X_test)}")

# Convert y_test from factorized cat_id to our cat_id mapping
# Load our cat_id mapping
cat_mapping = pd.read_csv("data/cat_id_mapping.csv", encoding='utf-8-sig')
code_to_our_cat = dict(cat_mapping[['ItemClsCode', 'cat_id']].values)
# Get ItemClsCode for each test sample
test_codes = data.loc[idx_test, 'ItemClsCode']
y_true_our = test_codes.map(code_to_our_cat).values

accuracy_pytorch = accuracy_score(y_true_our, y_pred_pytorch)
print(f"\nPyTorch model accuracy (our mapping) on same test set: {accuracy_pytorch:.4f}")
print(f"Difference from sklearn: {accuracy_pytorch - accuracy:+.4f}")

# Also compute PyTorch accuracy using factorized labels? Need mapping from our cat_id to factorized cat_id
# Create mapping from ItemClsCode to factorized cat_id
code_to_factor = dict(data[['ItemClsCode', 'cat_id_factor']].values)
# Convert PyTorch predictions (our cat_id) to factorized cat_id via ItemClsCode?
# We need to map predicted cat_id back to ItemClsCode, then to factorized cat_id.
# Create reverse mapping our cat_id -> ItemClsCode
our_cat_to_code = dict(cat_mapping[['cat_id', 'ItemClsCode']].values)
# For each predicted cat_id, get ItemClsCode, then factorized cat_id
y_pred_factor = []
for pred in y_pred_pytorch:
    code = our_cat_to_code.get(pred)
    if code is None:
        # fallback: use 0
        y_pred_factor.append(0)
    else:
        y_pred_factor.append(code_to_factor[code])
        
accuracy_pytorch_factor = accuracy_score(y_test, y_pred_factor)
print(f"PyTorch model accuracy (factorized mapping) on same test set: {accuracy_pytorch_factor:.4f}")
print(f"Difference from sklearn (factorized): {accuracy_pytorch_factor - accuracy:+.4f}")

print("\nSummary:")
print(f"Sklearn baseline: {accuracy:.4f}")
print(f"PyTorch (our mapping): {accuracy_pytorch:.4f}")
print(f"PyTorch (factorized): {accuracy_pytorch_factor:.4f}")
print(f"Target: 0.9478")
if accuracy_pytorch_factor >= 0.9478:
    print("SUCCESS: PyTorch meets or exceeds target!")
else:
    print("WARNING: PyTorch below target.")