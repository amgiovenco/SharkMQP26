"""
Compare 5-fold cross-validation performance on:
- Normal data (real only)
- Real + synthetic data (synthetic only in training, real only in validation/test)
Both using seed 8 with identical model parameters.
Generates comparison metrics and visualizations in /results folder.
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from scipy.integrate import simpson
import warnings
warnings.filterwarnings('ignore')

# Configuration
REAL_DATA_PATH = "../../data/shark_dataset.csv"
SYNTHETIC_DATA_PATH = "../../data/synthetic_only.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_SPLITS = 5
RESULTS_DIR = Path("./results")

# Create results directory
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("SYNTHETIC DATA COMPARISON: 5-Fold Cross-Validation (ExtraTrees)")
print("=" * 80)

# Load data
print("\n[1/4] Loading data...")
real_data = pd.read_csv(REAL_DATA_PATH)
synthetic_data = pd.read_csv(SYNTHETIC_DATA_PATH)

X_real_raw = real_data.drop(columns=[SPECIES_COL])
y_real = real_data[SPECIES_COL]
X_synthetic_raw = synthetic_data.drop(columns=[SPECIES_COL])
y_synthetic = synthetic_data[SPECIES_COL]

print(f" Real data shape: {X_real_raw.shape}")
print(f" Synthetic data shape: {X_synthetic_raw.shape}")

# Drop species with <2 samples in real data
counts = y_real.value_counts()
valid_classes = counts[counts >= 2].index
mask_real = y_real.isin(valid_classes)
X_real_raw = X_real_raw[mask_real].reset_index(drop=True)
y_real = y_real[mask_real].to_numpy()

# Filter synthetic data to only include valid classes from real data
mask_synthetic = y_synthetic.isin(valid_classes)
X_synthetic_raw = X_synthetic_raw[mask_synthetic].reset_index(drop=True)
y_synthetic = y_synthetic[mask_synthetic].to_numpy()

print(f" After filtering: Real {X_real_raw.shape}, Synthetic {X_synthetic_raw.shape}")
print(f" Classes: {len(np.unique(y_real))}")

# Feature engineering (matches train_cv_model.py exactly)
def feature_engineering(df):
    features = pd.DataFrame()
    temps = df.columns.astype(float)
    features['max'] = df.max(axis=1)
    features['min'] = df.min(axis=1)
    features['mean'] = df.mean(axis=1)
    features['std'] = df.std(axis=1)
    features['auc'] = df.apply(lambda row: simpson(row, temps), axis=1)
    features['centroid'] = df.apply(lambda row: np.sum(row*temps)/np.sum(row), axis=1)
    features['temp_peak'] = df.apply(lambda row: temps[np.argmax(row)], axis=1)
    features['fwhm'] = df.apply(lambda row: np.sum(row > 0.5*row.max()), axis=1)
    features['rise_time'] = df.apply(lambda row: np.argmax(row), axis=1)
    features['decay_time'] = df.apply(lambda row: len(row) - np.argmax(row[::-1]), axis=1)
    features['auc_left'] = df.apply(lambda row: simpson(row[:np.argmax(row)+1], temps[:np.argmax(row)+1]), axis=1)
    features['auc_right'] = df.apply(lambda row: simpson(row[np.argmax(row):], temps[np.argmax(row):]), axis=1)
    features['asymmetry'] = features['auc_left'] / (features['auc_right'] + 1e-8)
    return features

def enhanced_features(features):
    """Add interaction features (matches train_cv_model.py exactly)"""
    enhanced = features.copy()
    enhanced['fwhm_rise_ratio'] = features['fwhm'] / (features['rise_time'] + 1e-8)
    enhanced['peak_temp_std'] = features['temp_peak'] * features['std']
    enhanced['asymmetry_fwhm'] = features['asymmetry'] * features['fwhm']
    enhanced['rise_decay_ratio'] = features['rise_time'] / (features['decay_time'] + 1e-8)
    return enhanced

print("\n[2/4] Extracting features...")
X_real_features = feature_engineering(X_real_raw)
X_real_features = enhanced_features(X_real_features)
X_real = X_real_features.to_numpy(float)

X_synthetic_features = feature_engineering(X_synthetic_raw)
X_synthetic_features = enhanced_features(X_synthetic_features)
X_synthetic = X_synthetic_features.to_numpy(float)

print(f" Feature shape: Real {X_real.shape}, Synthetic {X_synthetic.shape}")

# Updated model: ExtraTreesClassifier with specified parameters
best_params = {
    'n_estimators': 900,
    'max_depth': 40,
    'min_samples_split': 4,
    'min_samples_leaf': 2,
    'max_features': 0.5,
    'class_weight': 'balanced_subsample',  # Note: specific to ExtraTrees
    'random_state': RANDOM_STATE,
    'n_jobs': 1
}

print("\n[3/4] Running 5-fold cross-validation...")
print(f" Model: ExtraTreesClassifier")
print(f" Parameters: {best_params}")

# StratifiedKFold for both scenarios
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Results containers
results_normal = {
    'fold_results': [],
    'accuracies': [], 'f1_scores': [], 'precisions': [], 'recalls': [],
    'all_y_true': [], 'all_y_pred': []
}
results_synthetic = {
    'fold_results': [],
    'accuracies': [], 'f1_scores': [], 'precisions': [], 'recalls': [],
    'all_y_true': [], 'all_y_pred': []
}

# Get class names for later use
unique_classes = np.unique(y_real)
class_names = {i: str(c) for i, c in enumerate(unique_classes)}

print("\n Fold Progress:")
print(" " + "-" * 76)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real, y_real), 1):
    # SCENARIO 1: Normal data only
    X_train_normal = X_real[train_idx]
    X_test_normal = X_real[test_idx]
    y_train_normal = y_real[train_idx]
    y_test_normal = y_real[test_idx]

    model_normal = ExtraTreesClassifier(**best_params)
    model_normal.fit(X_train_normal, y_train_normal)
    y_pred_normal = model_normal.predict(X_test_normal)

    acc_normal = accuracy_score(y_test_normal, y_pred_normal)
    f1_normal = f1_score(y_test_normal, y_pred_normal, average='macro', zero_division=0)
    prec_normal = precision_score(y_test_normal, y_pred_normal, average='macro', zero_division=0)
    rec_normal = recall_score(y_test_normal, y_pred_normal, average='macro', zero_division=0)

    results_normal['accuracies'].append(acc_normal)
    results_normal['f1_scores'].append(f1_normal)
    results_normal['precisions'].append(prec_normal)
    results_normal['recalls'].append(rec_normal)
    results_normal['all_y_true'].extend(y_test_normal)
    results_normal['all_y_pred'].extend(y_pred_normal)
    results_normal['fold_results'].append({
        'fold': fold_idx,
        'accuracy': float(acc_normal),
        'f1': float(f1_normal),
        'precision': float(prec_normal),
        'recall': float(rec_normal),
        'test_size': len(y_test_normal),
        'train_size': len(y_train_normal)
    })

    # SCENARIO 2: Real + Synthetic (synthetic only in training)
    X_train_synthetic = np.vstack([X_real[train_idx], X_synthetic])
    y_train_synthetic = np.hstack([y_real[train_idx], y_synthetic])
    X_test_synthetic = X_real[test_idx]
    y_test_synthetic = y_real[test_idx]

    model_synthetic = ExtraTreesClassifier(**best_params)
    model_synthetic.fit(X_train_synthetic, y_train_synthetic)
    y_pred_synthetic = model_synthetic.predict(X_test_synthetic)

    acc_synthetic = accuracy_score(y_test_synthetic, y_pred_synthetic)
    f1_synthetic = f1_score(y_test_synthetic, y_pred_synthetic, average='macro', zero_division=0)
    prec_synthetic = precision_score(y_test_synthetic, y_pred_synthetic, average='macro', zero_division=0)
    rec_synthetic = recall_score(y_test_synthetic, y_pred_synthetic, average='macro', zero_division=0)

    results_synthetic['accuracies'].append(acc_synthetic)
    results_synthetic['f1_scores'].append(f1_synthetic)
    results_synthetic['precisions'].append(prec_synthetic)
    results_synthetic['recalls'].append(rec_synthetic)
    results_synthetic['all_y_true'].extend(y_test_synthetic)
    results_synthetic['all_y_pred'].extend(y_pred_synthetic)
    results_synthetic['fold_results'].append({
        'fold': fold_idx,
        'accuracy': float(acc_synthetic),
        'f1': float(f1_synthetic),
        'precision': float(prec_synthetic),
        'recall': float(rec_synthetic),
        'test_size': len(y_test_synthetic),
        'train_size': len(y_train_synthetic)
    })

    print(f" Fold {fold_idx}/5 | Normal: Acc={acc_normal:.4f} F1={f1_normal:.4f} | "
          f"Synthetic: Acc={acc_synthetic:.4f} F1={f1_synthetic:.4f}")
    print(" " + "-" * 76)

# Convert to numpy arrays
results_normal['all_y_true'] = np.array(results_normal['all_y_true'])
results_normal['all_y_pred'] = np.array(results_normal['all_y_pred'])
results_synthetic['all_y_true'] = np.array(results_synthetic['all_y_true'])
results_synthetic['all_y_pred'] = np.array(results_synthetic['all_y_pred'])

# Summary statistics
summary_normal = {
    'scenario': 'Normal Data Only',
    'mean_accuracy': float(np.mean(results_normal['accuracies'])),
    'std_accuracy': float(np.std(results_normal['accuracies'])),
    'mean_f1': float(np.mean(results_normal['f1_scores'])),
    'std_f1': float(np.std(results_normal['f1_scores'])),
    'mean_precision': float(np.mean(results_normal['precisions'])),
    'mean_recall': float(np.mean(results_normal['recalls'])),
    'fold_results': results_normal['fold_results']
}

summary_synthetic = {
    'scenario': 'Real + Synthetic Data',
    'mean_accuracy': float(np.mean(results_synthetic['accuracies'])),
    'std_accuracy': float(np.std(results_synthetic['accuracies'])),
    'mean_f1': float(np.mean(results_synthetic['f1_scores'])),
    'std_f1': float(np.std(results_synthetic['f1_scores'])),
    'mean_precision': float(np.mean(results_synthetic['precisions'])),
    'mean_recall': float(np.mean(results_synthetic['recalls'])),
    'fold_results': results_synthetic['fold_results']
}

# Comparison
comparison = {
    'accuracy_improvement': float(summary_synthetic['mean_accuracy'] - summary_normal['mean_accuracy']),
    'f1_improvement': float(summary_synthetic['mean_f1'] - summary_normal['mean_f1']),
    'precision_improvement': float(summary_synthetic['mean_precision'] - summary_normal['mean_precision']),
    'recall_improvement': float(summary_synthetic['mean_recall'] - summary_normal['mean_recall'])
}

# Comprehensive results
comprehensive_results = {
    'metadata': {
        'seed': RANDOM_STATE,
        'n_splits': N_SPLITS,
        'model_type': 'ExtraTreesClassifier',  # Updated model name
        'model_parameters': best_params,
        'real_data_shape': [int(X_real.shape[0]), int(X_real.shape[1])],
        'synthetic_data_shape': [int(X_synthetic.shape[0]), int(X_synthetic.shape[1])],
        'total_classes': int(len(unique_classes))
    },
    'normal_data': summary_normal,
    'synthetic_data': summary_synthetic,
    'comparison': comparison
}

# Save JSON results
print("\n[4/4] Saving results...")
results_file = RESULTS_DIR / "comparison_results.json"
with open(results_file, 'w') as f:
    json.dump(comprehensive_results, f, indent=2)
print(f" Saved: {results_file}")

# Save predictions for confusion matrix
predictions_data = {
    'normal_data': {
        'y_true': results_normal['all_y_true'].tolist(),
        'y_pred': results_normal['all_y_pred'].tolist()
    },
    'synthetic_data': {
        'y_true': results_synthetic['all_y_true'].tolist(),
        'y_pred': results_synthetic['all_y_pred'].tolist()
    },
    'class_mapping': class_names
}
predictions_file = RESULTS_DIR / "predictions.json"
with open(predictions_file, 'w') as f:
    json.dump(predictions_data, f, indent=2)
print(f" Saved predictions to: {predictions_file}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# 1. Metrics by Fold
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['accuracies', 'f1_scores', 'precisions', 'recalls']
metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
axes_flat = axes.flatten()

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes_flat[idx]
    folds = list(range(1, N_SPLITS + 1))
    ax.plot(folds, results_normal[metric], 'o-', label='Normal Data', linewidth=2, markersize=8)
    ax.plot(folds, results_synthetic[metric], 's-', label='Real + Synthetic', linewidth=2, markersize=8)
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel(name, fontsize=11)
    ax.set_title(f'{name} by Fold', fontsize=12, fontweight='bold')
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(RESULTS_DIR / "metrics_by_fold.png", dpi=300, bbox_inches='tight')
print(f" Saved: {RESULTS_DIR / 'metrics_by_fold.png'}")
plt.close()

# 2. Mean Metrics Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
metrics_list = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
normal_means = [summary_normal['mean_accuracy'], summary_normal['mean_f1'],
                summary_normal['mean_precision'], summary_normal['mean_recall']]
synthetic_means = [summary_synthetic['mean_accuracy'], summary_synthetic['mean_f1'],
                   summary_synthetic['mean_precision'], summary_synthetic['mean_recall']]

x = np.arange(len(metrics_list))
width = 0.35
bars1 = ax.bar(x - width/2, normal_means, width, label='Normal Data', alpha=0.8)
bars2 = ax.bar(x + width/2, synthetic_means, width, label='Real + Synthetic', alpha=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Mean Metrics Comparison (ExtraTrees)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_list, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "mean_metrics_comparison.png", dpi=300, bbox_inches='tight')
print(f" Saved: {RESULTS_DIR / 'mean_metrics_comparison.png'}")
plt.close()

# 3. Improvement Overview
fig, ax = plt.subplots(figsize=(12, 6))
improvements = [
    comparison['accuracy_improvement'],
    comparison['f1_improvement'],
    comparison['precision_improvement'],
    comparison['recall_improvement']
]
metric_names_imp = ['Accuracy\nImprovement', 'F1 Score\nImprovement',
                    'Precision\nImprovement', 'Recall\nImprovement']
colors = ['green' if x >= 0 else 'red' for x in improvements]
bars = ax.bar(metric_names_imp, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Improvement', fontsize=12)
ax.set_title('Performance Improvement: Real + Synthetic vs Normal (ExtraTrees)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, improvements):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.4f}',
            ha='center', va='bottom' if val >= 0 else 'top', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "improvement_overview.png", dpi=300, bbox_inches='tight')
print(f" Saved: {RESULTS_DIR / 'improvement_overview.png'}")
plt.close()

# 4. Box Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics_to_plot = [
    ('accuracies', 'Accuracy'), ('f1_scores', 'F1 Score'),
    ('precisions', 'Precision'), ('recalls', 'Recall')
]
axes_flat = axes.flatten()

for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
    ax = axes_flat[idx]
    data_to_plot = [results_normal[metric_key], results_synthetic[metric_key]]
    bp = ax.boxplot(data_to_plot, labels=['Normal Data', 'Real + Synthetic'],
                    patch_artist=True, widths=0.6)
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(RESULTS_DIR / "metrics_distribution.png", dpi=300, bbox_inches='tight')
print(f" Saved: {RESULTS_DIR / 'metrics_distribution.png'}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY RESULTS (ExtraTreesClassifier)")
print("=" * 80)
print(f"\nNORMAL DATA (Real Only):")
print(f" Mean Accuracy: {summary_normal['mean_accuracy']:.4f} ± {summary_normal['std_accuracy']:.4f}")
print(f" Mean F1 Score: {summary_normal['mean_f1']:.4f} ± {summary_normal['std_f1']:.4f}")
print(f" Mean Precision: {summary_normal['mean_precision']:.4f}")
print(f" Mean Recall: {summary_normal['mean_recall']:.4f}")

print(f"\nREAL + SYNTHETIC DATA:")
print(f" Mean Accuracy: {summary_synthetic['mean_accuracy']:.4f} ± {summary_synthetic['std_accuracy']:.4f}")
print(f" Mean F1 Score: {summary_synthetic['mean_f1']:.4f} ± {summary_synthetic['std_f1']:.4f}")
print(f" Mean Precision: {summary_synthetic['mean_precision']:.4f}")
print(f" Mean Recall: {summary_synthetic['mean_recall']:.4f}")

print(f"\nIMPROVEMENT (Real + Synthetic vs Normal):")
print(f" Accuracy: {comparison['accuracy_improvement']:+.4f}")
print(f" F1 Score: {comparison['f1_improvement']:+.4f}")
print(f" Precision: {comparison['precision_improvement']:+.4f}")
print(f" Recall: {comparison['recall_improvement']:+.4f}")

print(f"\nDATA INFO:")
print(f" Real data samples: {X_real.shape[0]}")
print(f" Synthetic data samples: {X_synthetic.shape[0]}")
print(f" Total features: {X_real.shape[1]}")
print(f" Number of classes: {len(unique_classes)}")
print(f" Random seed: {RANDOM_STATE}")

print(f"\nOUTPUT FILES:")
print(f" Results JSON: {results_file}")
print(f" Predictions JSON: {predictions_file}")
print(f" Metrics by fold: {RESULTS_DIR / 'metrics_by_fold.png'}")
print(f" Mean metrics: {RESULTS_DIR / 'mean_metrics_comparison.png'}")
print(f" Improvement: {RESULTS_DIR / 'improvement_overview.png'}")
print(f" Distribution: {RESULTS_DIR / 'metrics_distribution.png'}")
print(f" Confusion matrix (Normal): {RESULTS_DIR / 'cm_normal_aggregated.png'}")
print(f" Confusion matrix (Synthetic): {RESULTS_DIR / 'cm_synthetic_aggregated.png'}")

# ============================================================================
# PER-FOLD TEST ACCURACIES TABLE
# ============================================================================
print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION TEST ACCURACIES")
print("=" * 80)
print(f"{'Fold':<6} {'Normal Data':<18} {'Real + Synthetic':<20} {'Improvement'}")
print("-" * 80)

improvements = []
for i in range(5):
    fold_normal = summary_normal['fold_results'][i]['accuracy']
    fold_synth = summary_synthetic['fold_results'][i]['accuracy']
    imp = fold_synth - fold_normal
    improvements.append(imp)
    print(f"{i+1:<6} {fold_normal:<18.4f} {fold_synth:<20.4f} {imp:+.4f}")

mean_normal = np.mean([f['accuracy'] for f in summary_normal['fold_results']])
std_normal = np.std([f['accuracy'] for f in summary_normal['fold_results']])
mean_synth = np.mean([f['accuracy'] for f in summary_synthetic['fold_results']])
std_synth = np.std([f['accuracy'] for f in summary_synthetic['fold_results']])
mean_imp = np.mean(improvements)

print("-" * 80)
print(f"{'MEAN':<6} {mean_normal:.4f} ± {std_normal:.4f}{'':<6} {mean_synth:.4f} ± {std_synth:.4f}{'':<8} {mean_imp:+.4f}")
print("=" * 80)

# ============================================================================
# AGGREGATED CONFUSION MATRICES
# ============================================================================
print("\n[Generating] Aggregated Confusion Matrices...")

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generate aggregated confusion matrix from all test folds combined."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'}, cbar=True)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {RESULTS_DIR / filename}")

# Normal Data (All Test Folds Combined)
y_true_normal = np.array(results_normal['all_y_true'])
y_pred_normal = np.array(results_normal['all_y_pred'])
plot_confusion_matrix(y_true_normal, y_pred_normal,
                      "Confusion Matrix - Normal Data (All Test Folds Combined)",
                      "cm_normal_aggregated.png")

# Real + Synthetic (All Test Folds Combined)
y_true_synth = np.array(results_synthetic['all_y_true'])
y_pred_synth = np.array(results_synthetic['all_y_pred'])
plot_confusion_matrix(y_true_synth, y_pred_synth,
                      "Confusion Matrix - Real + Synthetic (All Test Folds Combined)",
                      "cm_synthetic_aggregated.png")

print("\n" + "=" * 80)
print("EXTRA TREES COMPARISON COMPLETE!")
print("=" * 80)