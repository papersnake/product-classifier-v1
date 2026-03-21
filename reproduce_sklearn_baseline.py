import sys
import os


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Load raw data
print("Loading raw data...")
raw_data = pd.read_csv("data/products.csv", dtype={
    'ItemClsCode': str,
    'itemclsname': str,
    'ItemCode': str,
    'ItemName': str,
    'unit': str,
    'MainSupcode': str
})
print(f"Original rows: {len(raw_data)}")

# Apply filtering as in notebook
# 1. Exclude ItemClsCode starting with '23'
raw_data = raw_data[~raw_data['ItemClsCode'].str.startswith('23')]
print(f"After excluding 23%: {len(raw_data)}")

# 2. Exclude fresh categories (starting with '07')
raw_data = raw_data[~raw_data['ItemClsCode'].str.startswith('07')]
print(f"After excluding 07: {len(raw_data)}")

# 3. Exclude categories with less than 10 samples
cat_counts = raw_data['ItemClsCode'].value_counts()
valid_cats = cat_counts[cat_counts >= 10].index
raw_data = raw_data[raw_data['ItemClsCode'].isin(valid_cats)]
print(f"After excluding small categories: {len(raw_data)}")
print(f"Remaining categories: {len(raw_data['ItemClsCode'].unique())}")

# Factorize ItemClsCode (order of appearance) to get cat_id
raw_data['cat_id_factor'] = pd.factorize(raw_data['ItemClsCode'])[0]
print(f"Factorized cat_id range: {raw_data['cat_id_factor'].min()} to {raw_data['cat_id_factor'].max()}")

# Load preprocessed column (from products_processed.csv) - we need to match indices
# Since products_processed.csv has same rows but different order? Let's load processed data
processed_data = pd.read_csv("data/products_processed.csv", encoding='utf-8-sig')
print(f"Processed data rows: {len(processed_data)}")
# Assume same order as raw_data after filtering (should be)
# We'll merge by ItemClsCode, ItemCode, ItemName, etc. For simplicity, assume indices match.
# We'll just use preprocessed column from processed_data (since we have same filtering)
raw_data['preprocessed'] = processed_data['preprocessed']

# Split using random_state=0, stratify=cat_id_factor
X = raw_data['preprocessed'].values
y = raw_data['cat_id_factor'].values
indices = raw_data.index.values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.25, random_state=0, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Load sklearn model
print("Loading sklearn model...")
vectorizer = joblib.load("data/Tfidf_min_vect.pkl")
clf = joblib.load("data/clf_min_model.pkl")
print(f"Feature count: {len(vectorizer.get_feature_names_out())}")

# Transform test texts
X_test_vec = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_vec)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nSklearn model accuracy (correct mapping): {accuracy:.4f}")
print("Target accuracy: 0.9478")
print(f"Difference: {accuracy - 0.9478:+.4f}")

# Classification report
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(len(np.unique(y)))]))

# Save results
with open("data/sklearn_baseline_accuracy.txt", "w", encoding='utf-8') as f:
    f.write(f"准确率: {accuracy:.4f}\n")
    f.write("目标: 0.9478\n")
    f.write(f"差距: {accuracy - 0.9478:+.4f}\n")

print("Results saved to data/sklearn_baseline_accuracy.txt")

# Now evaluate PyTorch model on same test set
print("\n--- Evaluating PyTorch model on same test set ---")
# We need to map cat_id_factor to our cat_id mapping (sorted)
# Load our cat_id mapping
cat_mapping = pd.read_csv("data/cat_id_mapping.csv", encoding='utf-8-sig')
# Create mapping from ItemClsCode to our cat_id
code_to_our_cat = dict(cat_mapping[['ItemClsCode', 'cat_id']].values)

# Map factorized cat_id to our cat_id via ItemClsCode
# For each test sample, get its ItemClsCode
test_data = raw_data.loc[idx_test]
test_data['our_cat_id'] = test_data['ItemClsCode'].map(code_to_our_cat)
# Some categories may not be in mapping? Should be all.
print(f"Missing mapping count: {test_data['our_cat_id'].isna().sum()}")

# Load PyTorch predictor
from pytorch_predictor import TextCNNPredictor
predictor = TextCNNPredictor(
    model_path="data/textcnn_final.pth",
    vocab_path="data/vocab.pkl",
    embedding_matrix_path="data/embedding_matrix.npy",
    cat_mapping_path="data/cat_id_mapping.csv",
    max_seq_len=11
)

# Predict each test sample
y_pred_pytorch = []
for text in X_test:
    pred_id, pred_name = predictor.predict_single(text)
    y_pred_pytorch.append(pred_id)

y_true_our = test_data['our_cat_id'].values
accuracy_pytorch = accuracy_score(y_true_our, y_pred_pytorch)
print(f"PyTorch model accuracy (our mapping) on same test set: {accuracy_pytorch:.4f}")
print(f"Difference from sklearn: {accuracy_pytorch - accuracy:+.4f}")
