"""
Comprehensive comparison of 5-fold CV with and without synthetic data augmentation.
Compares: Real data only vs Real + Synthetic (synthetic only in training sets).
Seed: 8, Exactly matching existing model training setup.
"""
import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
REAL_DATA_PATH = "../../data/shark_dataset.csv"
SYNTHETIC_DATA_PATH = "../../data/synthetic_only.csv"
RESULTS_DIR = Path("./results")
RANDOM_STATE = 8
N_SPLITS = 5

# Best model parameters from optimization (ExtraTreesClassifier)
MODEL_PARAMS = {
    'n_estimators': 790,
    'min_samples_leaf': 1,
    'max_depth': 15,
    'max_features': None,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# =============================================================================
# FEATURE ENGINEERING (Copied from existing code)
# =============================================================================
def _curve_features(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Engineer ~14 features from a single curve y(t)."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    k = max(1, int(0.05 * len(y)))
    baseline = np.median(y[:k])
    yb = y - baseline
    yb = np.clip(yb, 0.0, None)

    idx_max = int(np.argmax(yb))
    ymax = float(yb[idx_max])
    tmax = float(t[idx_max])

    auc = float(np.trapezoid(yb, t))
    centroid = float(np.trapezoid(yb * t, t) / (auc + 1e-12))

    half = 0.5 * ymax
    above = np.where(yb >= half)[0]
    fwhm = float(t[above[-1]] - t[above[0]]) if above.size > 0 else 0.0

    def cross(level):
        idx = np.where(yb >= level)[0]
        return (int(idx[0]) if idx.size else idx_max), (int(idx[-1]) if idx.size else idx_max)

    lo, hi = 0.1 * ymax, 0.9 * ymax
    lo_i1, hi_i1 = cross(lo)[0], cross(hi)[0]
    lo_i2, hi_i2 = cross(lo)[1], cross(hi)[1]
    rise_time = float(t[max(hi_i1, lo_i1)] - t[min(hi_i1, lo_i1)]) if ymax > 0 else 0.0
    decay_time = float(t[max(lo_i2, hi_i2)] - t[min(lo_i2, hi_i2)]) if ymax > 0 else 0.0

    auc_left = float(np.trapezoid(yb[:idx_max + 1], t[:idx_max + 1]))
    auc_right = float(np.trapezoid(yb[idx_max:], t[idx_max:]))
    asymmetry = float((auc_right - auc_left) / (auc + 1e-12))

    mean_val = float(np.mean(y))
    std_val = float(np.std(y))
    max_val = float(np.max(y))
    min_val = float(np.min(y))

    return np.array([
        ymax, tmax, auc, centroid, fwhm, rise_time, decay_time,
        auc_left, auc_right, asymmetry, mean_val, std_val, max_val, min_val
    ], dtype=float)


def engineer_features(X_raw: pd.DataFrame):
    """Convert full sequence to compact 14-dim curve features."""
    t = X_raw.columns.astype(float).to_numpy()
    M = X_raw.to_numpy(float)
    F = np.vstack([_curve_features(M[i, :], t) for i in range(M.shape[0])])
    names = [
        "ymax", "tmax", "auc", "centroid", "fwhm", "rise", "decay",
        "auc_left", "auc_right", "asym", "mean", "std", "max", "min"
    ]
    return pd.DataFrame(F, columns=names), names


def load_and_prepare_data(csv_path, species_col='Species'):
    """Load data and extract features."""
    df = pd.read_csv(csv_path)
    X_raw = df.drop(columns=[species_col])
    y = df[species_col].astype(str)
    Xf, feature_names = engineer_features(X_raw)
    return Xf.to_numpy(float), y, feature_names


# =============================================================================
# METRICS COLLECTION
# =============================================================================
def compute_comprehensive_metrics(y_true, y_pred, fold_idx, scenario_name, class_names, n_classes):
    """Compute all relevant metrics for a fold."""
    # Get all class labels (0 to num_classes-1)
    labels = np.arange(n_classes)

    return {
        'fold': fold_idx,
        'scenario': scenario_name,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'f1_micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'precision_micro': float(precision_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'recall_micro': float(recall_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        'n_samples': int(len(y_true)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names,
                                                       output_dict=True, zero_division=0, labels=labels)
    }


# =============================================================================
# CROSS-VALIDATION FUNCTIONS
# =============================================================================
def run_cv_real_only(X_real, y_real, le, class_names):
    """5-fold CV using only real data."""
    print("\n" + "="*70)
    print("SCENARIO 1: 5-FOLD CV WITH REAL DATA ONLY")
    print("="*70)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    all_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real, y_real), 1):
        print(f"\nFold {fold_idx}/{N_SPLITS}:")

        X_train, X_test = X_real[train_idx], X_real[test_idx]
        y_train, y_test = y_real[train_idx], y_real[test_idx]

        # Train model with exact same setup as existing code
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = ExtraTreesClassifier(**MODEL_PARAMS)
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)

        # Metrics
        metrics = compute_comprehensive_metrics(y_test, y_pred, fold_idx,
                                                'real_only', class_names, len(class_names))
        fold_results.append(metrics)
        all_predictions.append((y_test, y_pred))

        print(f"  Train size: {len(X_train)} (100% real)")
        print(f"  Test size:  {len(X_test)}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  F1 Macro:   {metrics['f1_macro']:.4f}")

    return fold_results, all_predictions


def run_cv_with_synthetic(X_real, y_real, X_synth, y_synth, le, class_names):
    """5-fold CV using real + synthetic (synthetic only in training)."""
    print("\n" + "="*70)
    print("SCENARIO 2: 5-FOLD CV WITH REAL + SYNTHETIC DATA")
    print("(Synthetic data only added to training sets)")
    print("="*70)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    all_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real, y_real), 1):
        print(f"\nFold {fold_idx}/{N_SPLITS}:")

        # Real data split
        X_train_real, X_test = X_real[train_idx], X_real[test_idx]
        y_train_real, y_test = y_real[train_idx], y_real[test_idx]

        # Augment training set with ALL synthetic data
        X_train = np.vstack([X_train_real, X_synth])
        y_train = np.concatenate([y_train_real, y_synth])

        # Train model with exact same setup
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = ExtraTreesClassifier(**MODEL_PARAMS)
        clf.fit(X_train_scaled, y_train)

        # Predict (only on real test data)
        y_pred = clf.predict(X_test_scaled)

        # Metrics
        metrics = compute_comprehensive_metrics(y_test, y_pred, fold_idx,
                                                'real_synthetic', class_names, len(class_names))
        fold_results.append(metrics)
        all_predictions.append((y_test, y_pred))

        real_pct = len(X_train_real) / len(X_train) * 100
        synth_pct = len(X_synth) / len(X_train) * 100
        print(f"  Train size: {len(X_train)} ({len(X_train_real)} real + {len(X_synth)} synthetic)")
        print(f"              ({real_pct:.1f}% real, {synth_pct:.1f}% synthetic)")
        print(f"  Test size:  {len(X_test)} (100% real)")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  F1 Macro:   {metrics['f1_macro']:.4f}")

    return fold_results, all_predictions


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_confusion_matrices(real_results, synth_results, class_names, save_dir):
    """Plot unweighted and weighted (normalized) confusion matrices for both scenarios (separate PNGs)."""
    # Aggregate confusion matrices across all folds
    cm_real = np.sum([np.array(r['confusion_matrix']) for r in real_results], axis=0)
    cm_synth = np.sum([np.array(r['confusion_matrix']) for r in synth_results], axis=0)

    # Normalize by row (divide by sum of each row)
    cm_real_normalized = cm_real.astype('float') / cm_real.sum(axis=1, keepdims=True)
    cm_synth_normalized = cm_synth.astype('float') / cm_synth.sum(axis=1, keepdims=True)

    # Figure 1: Unweighted CM - Real Data Only
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_real, annot=False, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix (Unweighted): Real Data Only', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Species', fontsize=12)
    ax.set_xlabel('Predicted Species', fontsize=12)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / 'cm_unweighted_real_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cm_unweighted_real_only.png")

    # Figure 2: Unweighted CM - Real + Synthetic
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_synth, annot=False, cmap='Greens', ax=ax,
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix (Unweighted): Real + Synthetic Data', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Species', fontsize=12)
    ax.set_xlabel('Predicted Species', fontsize=12)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / 'cm_unweighted_real_synthetic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cm_unweighted_real_synthetic.png")

    # Figure 3: Weighted CM - Real Data Only
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_real_normalized, annot=False, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Normalized'}, vmin=0, vmax=1)
    ax.set_title('Confusion Matrix (Weighted/Normalized): Real Data Only', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Species', fontsize=12)
    ax.set_xlabel('Predicted Species', fontsize=12)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / 'cm_weighted_real_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cm_weighted_real_only.png")

    # Figure 4: Weighted CM - Real + Synthetic
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_synth_normalized, annot=False, cmap='Greens', ax=ax,
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Normalized'}, vmin=0, vmax=1)
    ax.set_title('Confusion Matrix (Weighted/Normalized): Real + Synthetic Data', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Species', fontsize=12)
    ax.set_xlabel('Predicted Species', fontsize=12)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / 'cm_weighted_real_synthetic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cm_weighted_real_synthetic.png")


def plot_metric_comparison(real_results, synth_results, save_dir):
    """Plot comparison of key metrics across folds."""
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    metric_labels = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision (Macro)', 'Recall (Macro)']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        real_vals = [r[metric] for r in real_results]
        synth_vals = [r[metric] for r in synth_results]

        x = np.arange(1, N_SPLITS + 1)
        width = 0.35

        axes[idx].bar(x - width/2, real_vals, width, label='Real Only', color='#3498db', alpha=0.8)
        axes[idx].bar(x + width/2, synth_vals, width, label='Real + Synthetic', color='#2ecc71', alpha=0.8)

        axes[idx].set_xlabel('Fold', fontsize=11)
        axes[idx].set_ylabel(label, fontsize=11)
        axes[idx].set_title(f'{label} by Fold', fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim([min(min(real_vals), min(synth_vals)) - 0.05,
                            max(max(real_vals), max(synth_vals)) + 0.05])

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_by_fold_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_by_fold_comparison.png")


def plot_boxplot_comparison(real_results, synth_results, save_dir):
    """Box plot comparison of metrics."""
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    metric_labels = ['Accuracy', 'F1\n(Macro)', 'F1\n(Weighted)', 'Precision\n(Macro)', 'Recall\n(Macro)']

    real_data = [[r[m] for r in real_results] for m in metrics]
    synth_data = [[r[m] for r in synth_results] for m in metrics]

    fig, ax = plt.subplots(figsize=(14, 6))

    positions_real = np.arange(len(metrics)) * 2
    positions_synth = positions_real + 0.8

    bp1 = ax.boxplot(real_data, positions=positions_real, widths=0.6,
                     patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor='#3498db', alpha=0.7),
                     medianprops=dict(color='darkblue', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    bp2 = ax.boxplot(synth_data, positions=positions_synth, widths=0.6,
                     patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                     medianprops=dict(color='darkgreen', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    ax.set_xticks((positions_real + positions_synth) / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metric Distribution Across 5 Folds', fontsize=14, fontweight='bold')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Real Only', 'Real + Synthetic'],
              loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 1.0])

    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_boxplot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_boxplot_comparison.png")


def plot_per_class_f1_comparison(real_results, synth_results, class_names, save_dir):
    """Plot per-class F1 scores comparison."""
    # Average per-class F1 across folds
    real_class_f1 = {}
    synth_class_f1 = {}

    for cls in class_names:
        real_f1_vals = [r['classification_report'][cls]['f1-score'] for r in real_results]
        synth_f1_vals = [r['classification_report'][cls]['f1-score'] for r in synth_results]
        real_class_f1[cls] = np.mean(real_f1_vals)
        synth_class_f1[cls] = np.mean(synth_f1_vals)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.35

    ax.bar(x - width/2, [real_class_f1[cls] for cls in class_names], width,
           label='Real Only', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, [synth_class_f1[cls] for cls in class_names], width,
           label='Real + Synthetic', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Species Class', fontsize=12)
    ax.set_ylabel('Average F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score Comparison (Averaged Across Folds)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_class_f1_comparison.png")


def plot_summary_table(real_results, synth_results, save_dir):
    """Create a summary comparison table."""
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']

    summary_data = []
    for metric in metrics:
        real_vals = [r[metric] for r in real_results]
        synth_vals = [r[metric] for r in synth_results]

        summary_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Real Only (Mean)': f"{np.mean(real_vals):.4f}",
            'Real Only (Std)': f"{np.std(real_vals):.4f}",
            'Real + Synthetic (Mean)': f"{np.mean(synth_vals):.4f}",
            'Real + Synthetic (Std)': f"{np.std(synth_vals):.4f}",
            'Improvement': f"{(np.mean(synth_vals) - np.mean(real_vals)):.4f}"
        })

    df = pd.DataFrame(summary_data)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colColours=['#f0f0f0']*len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code improvements
    for i in range(len(df)):
        improvement = float(df.iloc[i]['Improvement'])
        color = '#d4edda' if improvement > 0 else '#f8d7da' if improvement < 0 else '#fff3cd'
        table[(i+1, 5)].set_facecolor(color)

    plt.title('Summary Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_table.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("="*70)
    print("SYNTHETIC DATA AUGMENTATION COMPARISON")
    print("5-Fold Cross-Validation (Seed=8)")
    print("="*70)

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    print(f"\nResults will be saved to: {RESULTS_DIR.absolute()}")

    # Load data
    print("\nLoading datasets...")
    X_real, y_real_str, feature_names = load_and_prepare_data(REAL_DATA_PATH)
    X_synth, y_synth_str, _ = load_and_prepare_data(SYNTHETIC_DATA_PATH)

    print(f"  Real data:      {X_real.shape[0]} samples, {X_real.shape[1]} features")
    print(f"  Synthetic data: {X_synth.shape[0]} samples, {X_synth.shape[1]} features")

    # Encode labels
    le = LabelEncoder()
    y_real = le.fit_transform(y_real_str)
    y_synth = le.transform(y_synth_str)
    class_names = le.classes_.tolist()

    print(f"  Classes: {class_names}")
    print(f"  Number of classes: {len(class_names)}")

    # Model info
    print(f"\nModel: ExtraTreesClassifier")
    print(f"  Parameters: {MODEL_PARAMS}")

    # Run both scenarios
    real_results, real_predictions = run_cv_real_only(X_real, y_real, le, class_names)
    synth_results, synth_predictions = run_cv_with_synthetic(X_real, y_real, X_synth, y_synth, le, class_names)

    # Compute summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']

    print("\nReal Data Only:")
    for metric in metrics_to_compare:
        vals = [r[metric] for r in real_results]
        print(f"  {metric:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\nReal + Synthetic Data:")
    for metric in metrics_to_compare:
        vals = [r[metric] for r in synth_results]
        print(f"  {metric:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\nImprovement (Real+Synth - Real):")
    for metric in metrics_to_compare:
        real_vals = [r[metric] for r in real_results]
        synth_vals = [r[metric] for r in synth_results]
        improvement = np.mean(synth_vals) - np.mean(real_vals)
        pct_improvement = (improvement / np.mean(real_vals)) * 100
        arrow = "UP" if improvement > 0 else "DOWN" if improvement < 0 else "FLAT"
        print(f"  {metric:20s}: {improvement:+.4f} ({pct_improvement:+.2f}%) {arrow}")

    # Save JSON results
    print("\nSaving results to JSON...")
    results_json = {
        'experiment_info': {
            'description': '5-fold CV comparison: Real only vs Real + Synthetic',
            'random_state': RANDOM_STATE,
            'n_splits': N_SPLITS,
            'real_data_path': REAL_DATA_PATH,
            'synthetic_data_path': SYNTHETIC_DATA_PATH,
            'real_data_size': int(X_real.shape[0]),
            'synthetic_data_size': int(X_synth.shape[0]),
            'n_features': int(X_real.shape[1]),
            'feature_names': feature_names,
            'classes': class_names,
            'n_classes': len(class_names)
        },
        'model_config': {
            'model_type': 'ExtraTreesClassifier',
            'parameters': MODEL_PARAMS
        },
        'results': {
            'real_only': {
                'fold_results': real_results,
                'summary': {
                    metric: {
                        'mean': float(np.mean([r[metric] for r in real_results])),
                        'std': float(np.std([r[metric] for r in real_results])),
                        'min': float(np.min([r[metric] for r in real_results])),
                        'max': float(np.max([r[metric] for r in real_results])),
                        'values': [float(r[metric]) for r in real_results]
                    }
                    for metric in metrics_to_compare
                }
            },
            'real_synthetic': {
                'fold_results': synth_results,
                'summary': {
                    metric: {
                        'mean': float(np.mean([r[metric] for r in synth_results])),
                        'std': float(np.std([r[metric] for r in synth_results])),
                        'min': float(np.min([r[metric] for r in synth_results])),
                        'max': float(np.max([r[metric] for r in synth_results])),
                        'values': [float(r[metric]) for r in synth_results]
                    }
                    for metric in metrics_to_compare
                }
            },
            'comparison': {
                metric: {
                    'improvement': float(np.mean([r[metric] for r in synth_results]) -
                                       np.mean([r[metric] for r in real_results])),
                    'improvement_percentage': float(
                        ((np.mean([r[metric] for r in synth_results]) -
                          np.mean([r[metric] for r in real_results])) /
                         np.mean([r[metric] for r in real_results])) * 100
                    )
                }
                for metric in metrics_to_compare
            }
        }
    }

    json_path = RESULTS_DIR / 'synthetic_augmentation_comparison.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✅ Saved: {json_path.name}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrices(real_results, synth_results, class_names, RESULTS_DIR)
    plot_metric_comparison(real_results, synth_results, RESULTS_DIR)
    plot_boxplot_comparison(real_results, synth_results, RESULTS_DIR)
    plot_per_class_f1_comparison(real_results, synth_results, class_names, RESULTS_DIR)
    plot_summary_table(real_results, synth_results, RESULTS_DIR)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {RESULTS_DIR.absolute()}")
    print(f"  - JSON data: synthetic_augmentation_comparison.json")
    print(f"  - Confusion matrices (4 files):")
    print(f"    * cm_unweighted_real_only.png")
    print(f"    * cm_unweighted_real_synthetic.png")
    print(f"    * cm_weighted_real_only.png")
    print(f"    * cm_weighted_real_synthetic.png")
    print(f"  - Metrics by fold: metrics_by_fold_comparison.png")
    print(f"  - Box plots: metrics_boxplot_comparison.png")
    print(f"  - Per-class F1: per_class_f1_comparison.png")
    print(f"  - Summary table: summary_table.png")
    print("\nTo regenerate confusion matrices from JSON: python visualize_from_json.py")
    print("\nDone! You can now review the comparison results.")


if __name__ == "__main__":
    main()
