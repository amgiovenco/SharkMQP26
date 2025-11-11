"""
Generate Confusion Matrices from JSON Results
==============================================
Reads predictions from cv_augmentation_comparison.json
Creates both unweighted (raw counts) and weighted (normalized) confusion matrices
No annotations - only colors shown
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Load JSON results
results_dir = Path("./results")
json_path = results_dir / "cv_augmentation_comparison.json"

print(f"Loading results from {json_path}...")
with open(json_path, 'r') as f:
    results = json.load(f)

class_labels = results["metadata"]["class_labels"]
print(f"Class labels: {class_labels}")

# Extract predictions
baseline_y_true = results["baseline"]["predictions"]["y_true"]
baseline_y_pred = results["baseline"]["predictions"]["y_pred"]

augmented_y_true = results["augmented"]["predictions"]["y_true"]
augmented_y_pred = results["augmented"]["predictions"]["y_pred"]

print(f"Baseline: {len(baseline_y_true)} predictions")
print(f"Augmented: {len(augmented_y_true)} predictions")

# ============== Generate Confusion Matrices ==============

# Overall confusion matrices
cm_baseline = confusion_matrix(baseline_y_true, baseline_y_pred, labels=class_labels)
cm_augmented = confusion_matrix(augmented_y_true, augmented_y_pred, labels=class_labels)

# Normalize (weighted)
cm_baseline_norm = cm_baseline.astype('float') / cm_baseline.sum(axis=1)[:, np.newaxis]
cm_augmented_norm = cm_augmented.astype('float') / cm_augmented.sum(axis=1)[:, np.newaxis]

# ============== Plot 1: Unweighted vs Weighted (Side by Side) ==============

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Baseline unweighted
sns.heatmap(cm_baseline, cmap='Blues', ax=axes[0, 0], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0)
axes[0, 0].set_title('Baseline: Unweighted (Raw Counts)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Predicted', fontsize=11)
axes[0, 0].set_ylabel('True', fontsize=11)
axes[0, 0].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[0, 0].set_yticklabels(class_labels, rotation=0, fontsize=8)

# Baseline weighted
sns.heatmap(cm_baseline_norm, cmap='Blues', ax=axes[0, 1], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1)
axes[0, 1].set_title('Baseline: Weighted (Normalized)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Predicted', fontsize=11)
axes[0, 1].set_ylabel('True', fontsize=11)
axes[0, 1].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[0, 1].set_yticklabels(class_labels, rotation=0, fontsize=8)

# Augmented unweighted
sns.heatmap(cm_augmented, cmap='Oranges', ax=axes[1, 0], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0)
axes[1, 0].set_title('Augmented: Unweighted (Raw Counts)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Predicted', fontsize=11)
axes[1, 0].set_ylabel('True', fontsize=11)
axes[1, 0].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[1, 0].set_yticklabels(class_labels, rotation=0, fontsize=8)

# Augmented weighted
sns.heatmap(cm_augmented_norm, cmap='Oranges', ax=axes[1, 1], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1)
axes[1, 1].set_title('Augmented: Weighted (Normalized)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Predicted', fontsize=11)
axes[1, 1].set_ylabel('True', fontsize=11)
axes[1, 1].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[1, 1].set_yticklabels(class_labels, rotation=0, fontsize=8)

plt.suptitle('Confusion Matrices: Baseline vs Augmented (No Annotations)',
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(results_dir / "05_confusion_matrices_unweighted_weighted.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved 05_confusion_matrices_unweighted_weighted.png")
plt.close()

# ============== Plot 2: Baseline Unweighted vs Weighted ==============

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_baseline, cmap='Blues', ax=axes[0], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0)
axes[0].set_title('Baseline: Unweighted (Raw Counts)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=11)
axes[0].set_ylabel('True', fontsize=11)
axes[0].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(class_labels, rotation=0, fontsize=8)

sns.heatmap(cm_baseline_norm, cmap='Blues', ax=axes[1], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1)
axes[1].set_title('Baseline: Weighted (Normalized)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('True', fontsize=11)
axes[1].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(class_labels, rotation=0, fontsize=8)

plt.suptitle('Baseline: Unweighted vs Weighted (Color Only)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(results_dir / "06_baseline_confusion_matrices.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved 06_baseline_confusion_matrices.png")
plt.close()

# ============== Plot 3: Augmented Unweighted vs Weighted ==============

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_augmented, cmap='Oranges', ax=axes[0], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0)
axes[0].set_title('Augmented: Unweighted (Raw Counts)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=11)
axes[0].set_ylabel('True', fontsize=11)
axes[0].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(class_labels, rotation=0, fontsize=8)

sns.heatmap(cm_augmented_norm, cmap='Oranges', ax=axes[1], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1)
axes[1].set_title('Augmented: Weighted (Normalized)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('True', fontsize=11)
axes[1].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(class_labels, rotation=0, fontsize=8)

plt.suptitle('Augmented: Unweighted vs Weighted (Color Only)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(results_dir / "07_augmented_confusion_matrices.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved 07_augmented_confusion_matrices.png")
plt.close()

# ============== Plot 4: Side-by-side comparison (Weighted only) ==============

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_baseline_norm, cmap='Blues', ax=axes[0], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1)
axes[0].set_title('Baseline: Weighted (Normalized)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=11)
axes[0].set_ylabel('True', fontsize=11)
axes[0].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(class_labels, rotation=0, fontsize=8)

sns.heatmap(cm_augmented_norm, cmap='Oranges', ax=axes[1], cbar=True, annot=False,
            xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1)
axes[1].set_title('Augmented: Weighted (Normalized)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('True', fontsize=11)
axes[1].set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(class_labels, rotation=0, fontsize=8)

plt.suptitle('Comparison: Baseline vs Augmented (Weighted/Normalized)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(results_dir / "08_comparison_weighted_only.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved 08_comparison_weighted_only.png")
plt.close()

print("\n" + "="*70)
print("CONFUSION MATRICES GENERATED!")
print("="*70)
print(f"\nSaved to {results_dir}/:")
print(f"  - 05_confusion_matrices_unweighted_weighted.png (2x2 grid)")
print(f"  - 06_baseline_confusion_matrices.png (unweighted + weighted)")
print(f"  - 07_augmented_confusion_matrices.png (unweighted + weighted)")
print(f"  - 08_comparison_weighted_only.png (side-by-side weighted)")
print("\nAll matrices show colors only (no annotations/numbers)")
print("Unweighted = raw count values")
print("Weighted = normalized by row (per-class)")
print("="*70)
