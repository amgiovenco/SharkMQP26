"""
Generate two separate weighted confusion matrices:
1. Real data only (blue) - cm_weighted_real_only_impact.png
2. Real + synthetic data (green) - cm_weighted_real_synthetic_impact.png

This script loads predictions from the comparison_results.json and creates individual PNG files.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Setup
RESULTS_DIR = Path("./results")
METRICS_FILE = RESULTS_DIR / "comparison_results.json"
PREDICTIONS_FILE = RESULTS_DIR / "predictions.json"

print("=" * 80)
print("GENERATING WEIGHTED CONFUSION MATRICES")
print("=" * 80)

# Load results
print("\nLoading data from JSON files...")
with open(METRICS_FILE, 'r') as f:
    results = json.load(f)

with open(PREDICTIONS_FILE, 'r') as f:
    predictions_data = json.load(f)

# Extract predictions and labels
y_true_normal = np.array(predictions_data['normal_data']['y_true'])
y_pred_normal = np.array(predictions_data['normal_data']['y_pred'])
y_true_synthetic = np.array(predictions_data['synthetic_data']['y_true'])
y_pred_synthetic = np.array(predictions_data['synthetic_data']['y_pred'])

# Get class mapping and names
class_mapping = predictions_data['class_mapping']
class_names = sorted(class_mapping.items(), key=lambda x: int(x[0]))
class_names = [name for _, name in class_names]

num_classes = len(class_names)

print(f"Classes found: {num_classes}")
print(f"Class names: {class_names}")

# Compute confusion matrices
print("\nComputing confusion matrices...")
cm_normal = confusion_matrix(y_true_normal, y_pred_normal, labels=np.arange(num_classes))
cm_synthetic = confusion_matrix(y_true_synthetic, y_pred_synthetic, labels=np.arange(num_classes))

# Normalize by row (true label) to show percentages
cm_normal_weighted = cm_normal.astype('float') / cm_normal.sum(axis=1)[:, np.newaxis]
cm_synthetic_weighted = cm_synthetic.astype('float') / cm_synthetic.sum(axis=1)[:, np.newaxis]

print(f"Normal data confusion matrix shape: {cm_normal.shape}")
print(f"Synthetic data confusion matrix shape: {cm_synthetic.shape}")

# Determine figure size based on number of classes
if num_classes <= 10:
    figsize = (12, 10)
    label_size = 10
elif num_classes <= 20:
    figsize = (16, 14)
    label_size = 9
else:
    figsize = (20, 18)
    label_size = 8

# ============================================================================
# 1. REAL DATA ONLY (BLUE)
# ============================================================================
print("\nGenerating Real Data Only weighted confusion matrix (Blue)...")
fig, ax = plt.subplots(figsize=figsize)

sns.heatmap(cm_normal_weighted, cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            cbar_kws={'label': 'Normalized Proportion'},
            annot=False, fmt='.2f')

ax.set_title('Real Data Only - Weighted Confusion Matrix\n(5-Fold CV, Seed=8, ExtraTreesClassifier)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xticklabels(class_names, rotation=90, ha='right', fontsize=label_size)
ax.set_yticklabels(class_names, rotation=0, fontsize=label_size)

plt.tight_layout()
output_path_normal = RESULTS_DIR / 'cm_weighted_real_only_impact.png'
plt.savefig(output_path_normal, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {output_path_normal}")

# ============================================================================
# 2. REAL + SYNTHETIC DATA (GREEN)
# ============================================================================
print("Generating Real + Synthetic Data weighted confusion matrix (Green)...")
fig, ax = plt.subplots(figsize=figsize)

sns.heatmap(cm_synthetic_weighted, cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            cbar_kws={'label': 'Normalized Proportion'},
            annot=False, fmt='.2f')

ax.set_title('Real + Synthetic Data - Weighted Confusion Matrix\n(5-Fold CV, Seed=8, ExtraTreesClassifier)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xticklabels(class_names, rotation=90, ha='right', fontsize=label_size)
ax.set_yticklabels(class_names, rotation=0, fontsize=label_size)

plt.tight_layout()
output_path_synthetic = RESULTS_DIR / 'cm_weighted_real_synthetic_impact.png'
plt.savefig(output_path_synthetic, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {output_path_synthetic}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nReal Data Only:")
print(f"  Mean Accuracy: {results['normal_data']['mean_accuracy']:.4f}")
print(f"  Mean F1:       {results['normal_data']['mean_f1']:.4f}")

print(f"\nReal + Synthetic Data:")
print(f"  Mean Accuracy: {results['synthetic_data']['mean_accuracy']:.4f}")
print(f"  Mean F1:       {results['synthetic_data']['mean_f1']:.4f}")

print(f"\nImprovement:")
print(f"  Accuracy Δ:    {results['comparison']['accuracy_improvement']:+.4f}")
print(f"  F1 Δ:          {results['comparison']['f1_improvement']:+.4f}")

print("\n" + "=" * 80)
print("OUTPUT FILES:")
print("=" * 80)
print(f"  {output_path_normal.name} (Blue - Real Data Only)")
print(f"  {output_path_synthetic.name} (Green - Real + Synthetic Data)")
print("\n✓ COMPLETE!")
