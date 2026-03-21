'''
Debug sklearn model accuracy
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Load data
print("Loading data...")
raw_data = pd.read_csv("data/products_processed.csv", encoding='utf-8-sig')
print(f"Data shape: {raw_data.shape}")

# Load sklearn model
print("Loading sklearn model...")
vectorizer = joblib.load("data/Tfidf_min_vect.pkl")
clf = joblib.load("data/clf_min_model.pkl")
print(f"Feature count: {len(vectorizer.get_feature_names_out())}")

# Load category mapping
cat_mapping = pd.read_csv("data/cat_id_mapping.csv", encoding='utf-8-sig')
print(f"Class count: {len(cat_mapping)}")

# Get the preprocessed column
preprocessed = raw_data['preprocessed'].values
labels = raw_data['cat_id'].values

# Split using same random state as notebook (random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    preprocessed, labels, raw_data.index, random_state=0, stratify=labels
)
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# Transform using vectorizer (already fitted on entire dataset)
print("Transforming training data...")
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Predict on training data (should be high accuracy)
print("Predicting on training data...")
y_train_pred = clf.predict(X_train_vec)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Training accuracy: {train_acc:.4f}")

# Predict on test data
print("Predicting on test data...")
y_test_pred = clf.predict(X_test_vec)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {test_acc:.4f}")

# Check a few samples
print("\nSample predictions (first 5 test samples):")
for i in range(min(5, len(X_test))):
    print(f"  Sample {i}:")
    print(f"    Text: {X_test[i][:50]}...")
    print(f"    True label: {y_test[i]}")
    print(f"    Pred label: {y_test_pred[i]}")
    print(f"    Correct: {y_test[i] == y_test_pred[i]}")