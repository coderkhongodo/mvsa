import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv('/root/mvsa_multimodal/MVSA_MULTIMODAL/results/img_only/MVSA_img_20__vit_task3_seed30_preds.csv')
y_true = df['label'].values
y_pred = df['prediction'].values

print("Dataset Info:")
print(f"Total samples: {len(df)}")
print(f"Class distribution:")
print(df['label'].value_counts().sort_index())

print("\n" + "="*50)
print("METRICS")
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
print("CLASSIFICATION REPORT")
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
