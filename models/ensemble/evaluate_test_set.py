"""
Comprehensive Test Set Evaluation for Ensemble Model
======================================================

Evaluates the production ensemble model on the held-out test set.
Generates detailed metrics, visualizations, and error analysis.
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = "../../data/shark_test_data.csv"
PRODUCTION_DIR = "./production"
BASE_MODELS_DIR = "./models"
OUTPUT_DIR = "./test_results"

# Meta-learner type
META_LEARNER_TYPE = "xgb"  # or "nn"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("="*80)
print("ENSEMBLE MODEL - TEST SET EVALUATION")
print("="*80)


# ============================================================================
# LOAD BASE MODEL PREDICTION FUNCTIONS
# ============================================================================

from predict_rf import get_rf_predictions
from predict_rb import get_rb_predictions
from predict_stats import get_stats_predictions
from predict_resnet import get_resnet_predictions
from predict_cnn import get_cnn_predictions


# ============================================================================
# LOAD TEST DATA
# ============================================================================

print("\n[1/6] Loading test data...")
test_df = pd.read_csv(DATA_PATH)
species_col = "Species"

# Extract features and labels
X_test_df = test_df.drop(columns=[species_col])  # Keep as DataFrame for predictions
X_test = X_test_df.values  # Numpy array for meta-learner
y_test = test_df[species_col].values

print(f"  Test samples: {len(X_test)}")
print(f"  Unique species: {len(np.unique(y_test))}")


# ============================================================================
# GET BASE MODEL PREDICTIONS
# ============================================================================

print("\n[2/6] Generating base model predictions...")

# Get predictions from each base model
print("  - Random Forest...")
rf_proba = get_rf_predictions(X_test_df)
if rf_proba is None:
    raise RuntimeError("Random Forest predictions failed")

print("  - Rule-Based...")
rb_proba = get_rb_predictions(X_test_df)
if rb_proba is None:
    raise RuntimeError("Rule-Based predictions failed")

print("  - Statistical...")
stats_proba = get_stats_predictions(X_test_df)
if stats_proba is None:
    raise RuntimeError("Statistical predictions failed")

# Load metadata to get num_classes
with open(f"{PRODUCTION_DIR}/model_metadata.json", 'r') as f:
    metadata = json.load(f)
    num_classes = metadata['n_classes']

print("  - ResNet...")
resnet_proba = get_resnet_predictions(X_test_df, num_classes)
if resnet_proba is None:
    raise RuntimeError("ResNet predictions failed")

print("  - CNN (EfficientNet)...")
cnn_proba = get_cnn_predictions(X_test_df, num_classes)
if cnn_proba is None:
    raise RuntimeError("CNN predictions failed")

# Stack predictions
X_stacked = np.hstack([rf_proba, rb_proba, stats_proba, resnet_proba, cnn_proba])
print(f"\n  Stacked features shape: {X_stacked.shape}")


# ============================================================================
# LOAD META-LEARNER
# ============================================================================

print("\n[3/6] Loading ensemble meta-learner...")

if META_LEARNER_TYPE == "xgb":
    meta_path = f"{PRODUCTION_DIR}/final_xgb_meta.pkl"
    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)

    meta_model = meta_data['model']

    # Use species classes from metadata instead of label encoder
    species_classes = np.array(metadata['species_classes'])

    print(f"  Model: XGBoost")
    print(f"  Config: {meta_data['config']}")

    # Make predictions
    import xgboost as xgb
    dtest = xgb.DMatrix(X_stacked)
    y_pred_proba = meta_model.predict(dtest)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = species_classes[y_pred_encoded]

    # Encode test labels manually
    test_label_encoder = LabelEncoder()
    test_label_encoder.fit(species_classes)
    y_test_encoded = test_label_encoder.transform(y_test)

elif META_LEARNER_TYPE == "nn":
    meta_path = f"{PRODUCTION_DIR}/final_nn_meta.pth"
    checkpoint = torch.load(meta_path, map_location=DEVICE)

    # Rebuild model
    model = nn.Sequential(
        nn.Linear(checkpoint['input_dim'], 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, checkpoint['num_classes'])
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Use species classes from metadata
    species_classes = np.array(metadata['species_classes'])

    print(f"  Model: Neural Network")
    print(f"  Config: {checkpoint['config']}")

    # Make predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_stacked).to(DEVICE)
        outputs = model(X_tensor)
        y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        y_pred = species_classes[y_pred_encoded]

    # Encode test labels
    test_label_encoder = LabelEncoder()
    test_label_encoder.fit(species_classes)
    y_test_encoded = test_label_encoder.transform(y_test)

else:
    raise ValueError(f"Unknown meta-learner type: {META_LEARNER_TYPE}")


# ============================================================================
# CALCULATE METRICS
# ============================================================================

print("\n[4/6] Calculating metrics...")

# Overall accuracy (use encoded labels)
test_acc = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"\n  TEST SET ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Per-class metrics (use string labels for interpretability)
species_list = species_classes
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=species_list, zero_division=0
)

# Create dataframe
metrics_df = pd.DataFrame({
    'Species': species_list,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
}).sort_values('F1-Score', ascending=False)

print(f"\n  Macro-averaged metrics:")
print(f"    Precision: {precision.mean():.4f}")
print(f"    Recall: {recall.mean():.4f}")
print(f"    F1-Score: {f1.mean():.4f}")

# Misclassifications
misclassified_idx = np.where(y_test_encoded != y_pred_encoded)[0]
print(f"\n  Misclassifications: {len(misclassified_idx)} / {len(y_test)} ({100*len(misclassified_idx)/len(y_test):.2f}%)")


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

print("\n[5/6] Performing error analysis...")

misclassified_data = []
for idx in misclassified_idx:
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    confidence = y_pred_proba[idx].max()

    # Get probability for true class
    true_class_idx = y_test_encoded[idx]
    true_proba = y_pred_proba[idx][true_class_idx]

    misclassified_data.append({
        'sample_idx': idx,
        'true_species': true_label,
        'predicted_species': pred_label,
        'confidence': confidence,
        'true_proba': true_proba
    })

if len(misclassified_data) > 0:
    error_df = pd.DataFrame(misclassified_data).sort_values('confidence', ascending=False)
    print("\nMisclassified samples:")
    print(error_df[['true_species', 'predicted_species', 'confidence']].to_string(index=False))
else:
    error_df = pd.DataFrame(columns=['sample_idx', 'true_species', 'predicted_species', 'confidence', 'true_proba'])
    print("\nNO MISCLASSIFICATIONS! Perfect 100% accuracy on test set!")


# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[6/6] Creating visualizations...")

# 1. Confusion Matrix
print("  - Confusion matrix...", end=" ")
cm = confusion_matrix(y_test, y_pred, labels=species_list)

plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=False, cmap='Blues', square=True,
            xticklabels=species_list, yticklabels=species_list,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Species', fontsize=12, fontweight='bold')
plt.ylabel('True Species', fontsize=12, fontweight='bold')
plt.title(f'Ensemble Model - Test Set Confusion Matrix\nAccuracy: {test_acc*100:.2f}%',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK")

# 2. Per-class performance
print("  - Per-class performance...", end=" ")
fig, axes = plt.subplots(1, 3, figsize=(20, 30))

all_sp = metrics_df.sort_values('F1-Score', ascending=True)

# Precision
ax = axes[0]
colors = plt.cm.RdYlGn(all_sp['Precision'].values)
ax.barh(all_sp['Species'], all_sp['Precision'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Precision', fontsize=11, fontweight='bold')
ax.set_title(f'Precision by Species\n(Test Set)', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1.05])
ax.grid(axis='x', alpha=0.3)

# Recall
ax = axes[1]
colors = plt.cm.RdYlGn(all_sp['Recall'].values)
ax.barh(all_sp['Species'], all_sp['Recall'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax.set_title(f'Recall by Species\n(Test Set)', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1.05])
ax.grid(axis='x', alpha=0.3)

# F1-Score
ax = axes[2]
colors = plt.cm.RdYlGn(all_sp['F1-Score'].values)
ax.barh(all_sp['Species'], all_sp['F1-Score'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
ax.set_title(f'F1-Score by Species\n(Test Set)', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1.05])
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/per_class_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK")

# 3. Confidence distribution
print("  - Confidence distribution...", end=" ")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

max_confidence = y_pred_proba.max(axis=1)
correct = (y_test_encoded == y_pred_encoded).astype(int)

# Histogram
ax = axes[0]
ax.hist(max_confidence[correct==1], bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
ax.hist(max_confidence[correct==0], bins=30, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
ax.set_xlabel('Max Predicted Probability', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Calibration curve
ax = axes[1]
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_accs = []
bin_counts = []

for i in range(len(bins)-1):
    mask = (max_confidence >= bins[i]) & (max_confidence < bins[i+1])
    acc = correct[mask].mean() if mask.sum() > 0 else 0
    bin_accs.append(acc)
    bin_counts.append(mask.sum())

ax.plot(bin_centers, bin_accs, 'o-', linewidth=2.5, markersize=10,
        color='steelblue', label='Observed', markeredgecolor='black', markeredgewidth=1)
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, label='Perfect Calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
ax.set_ylabel('Empirical Accuracy', fontsize=11, fontweight='bold')
ax.set_title('Calibration Curve', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confidence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK")

# 4. Top and bottom performers
print("  - Top/bottom performers...", end=" ")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 10
top_10 = metrics_df.head(10).sort_values('F1-Score', ascending=True)
colors_top = plt.cm.Greens(np.linspace(0.5, 0.9, len(top_10)))
ax = axes[0]
bars = ax.barh(top_10['Species'], top_10['F1-Score'], color=colors_top, edgecolor='black', linewidth=1)
ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Best Performing Species', fontsize=13, fontweight='bold')
ax.set_xlim([0, 1.05])
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_10['F1-Score'])):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Bottom 10
bottom_10 = metrics_df.tail(10).sort_values('F1-Score', ascending=True)
colors_bottom = plt.cm.Reds(np.linspace(0.5, 0.9, len(bottom_10)))
ax = axes[1]
bars = ax.barh(bottom_10['Species'], bottom_10['F1-Score'], color=colors_bottom, edgecolor='black', linewidth=1)
ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Bottom 10 Performing Species', fontsize=13, fontweight='bold')
ax.set_xlim([0, 1.05])
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, (bar, val) in enumerate(zip(bars, bottom_10['F1-Score'])):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/top_bottom_performers.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK")


# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\nSaving results...")

# Save metrics
metrics_df.to_csv(f'{OUTPUT_DIR}/per_class_metrics.csv', index=False)
print(f"  - {OUTPUT_DIR}/per_class_metrics.csv")

# Save error analysis
error_df.to_csv(f'{OUTPUT_DIR}/misclassified_samples.csv', index=False)
print(f"  - {OUTPUT_DIR}/misclassified_samples.csv")

# Save summary
summary = {
    'meta_learner_type': META_LEARNER_TYPE,
    'test_samples': int(len(X_test)),
    'num_species': int(len(species_list)),
    'accuracy': float(test_acc),
    'macro_precision': float(precision.mean()),
    'macro_recall': float(recall.mean()),
    'macro_f1': float(f1.mean()),
    'misclassifications': int(len(misclassified_idx)),
    'error_rate': float(len(misclassified_idx) / len(y_test)),
    'base_models': ['rf', 'rb', 'stats', 'resnet', 'cnn'],
    'stacked_features': int(X_stacked.shape[1])
}

with open(f'{OUTPUT_DIR}/test_results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  - {OUTPUT_DIR}/test_results_summary.json")

print("\n" + "="*80)
print("TEST SET EVALUATION COMPLETE")
print("="*80)
print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%")
print(f"Results saved to: {OUTPUT_DIR}/")
print("="*80)
