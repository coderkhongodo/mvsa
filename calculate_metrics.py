import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix
)

# Load the predictions file
df = pd.read_csv('/root/mvsa_multimodal/results/mm_late/bernice-vit-concat_task3_seed30_itc0.1itm0.1_preds.csv')

# Extract true labels and predictions
y_true = df['label'].values
y_pred = df['prediction'].values

print("Dataset Info:")
print(f"Total samples: {len(df)}")
print(f"Classes: {sorted(df['label'].unique())}")
print(f"Class distribution:")
print(df['label'].value_counts().sort_index())

print("\n" + "="*50)
print("METRICS CALCULATION")
print("="*50)

# Calculate metrics
f1_weighted = f1_score(y_true, y_pred, average='weighted')
f1_macro = f1_score(y_true, y_pred, average='macro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
recall_macro = recall_score(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)

print(f"F1 Weighted: {f1_weighted:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"Precision Weighted: {precision_weighted:.4f}")
print(f"Precision Macro: {precision_macro:.4f}")
print(f"Recall Weighted: {recall_weighted:.4f}")
print(f"Recall Macro: {recall_macro:.4f}")
print(f"Accuracy: {accuracy:.4f}")

print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=['Neutral', 'Positive', 'Negative']))

print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print("Predicted →")
print("Actual ↓")
print("          0    1    2")
for i in range(3):
    print(f"Class {i}: {cm[i]}")

print("\n" + "="*50)
print("PER-CLASS METRICS")
print("="*50)
# Per-class metrics
for i, class_name in enumerate(['Neutral', 'Positive', 'Negative']):
    f1_class = f1_score(y_true, y_pred, average=None)[i]
    precision_class = precision_score(y_true, y_pred, average=None)[i]
    recall_class = recall_score(y_true, y_pred, average=None)[i]
    print(f"{class_name} (Class {i}):")
    print(f"  F1: {f1_class:.4f}")
    print(f"  Precision: {precision_class:.4f}")
    print(f"  Recall: {recall_class:.4f}")
    print()
