"""
Compare 5-fold cross-validation with real data vs real+synthetic data (synthetic only in training).

This script:
1. Performs 5-fold CV with seed 8 on normal data
2. Performs 5-fold CV with seed 8 on real+synthetic data (synthetic ONLY in training sets)
3. Collects metrics and outputs JSON comparison
4. Generates confusion matrices and comparison charts
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
REAL_DATA_PATH = "../../data/shark_dataset.csv"
SYNTHETIC_DATA_PATH = "../../data/synthetic_only.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_SPLITS = 5

# Top 18 features (from feature_importance.csv)
TOP_18_FEATURES = [
    'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
    'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
    'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
    'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
    'slope_std', 'y_left_max'
]

# Best model parameters (from optimize_stats.py and train_cv_model.py)
BEST_PARAMS = {
    'n_estimators': 1700,
    'max_depth': None,
    'min_samples_split': 9,
    'min_samples_leaf': 1,
    'max_features': 0.7,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# ============================================================================
# FUNCTIONS
# ============================================================================

def preprocess_curve(x, y):
    """Preprocess a single curve (same as optimize_stats.py and train_cv_model.py)"""
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(1.5 / dx)) | 1)
    if win >= len(y):
        win = max(7, (len(y)//2)*2 - 1)
    y_smooth = savgol_filter(y, window_length=win, polyorder=3, mode="interp")

    q = np.quantile(y_smooth, 0.3)
    mask = y_smooth <= q

    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_smooth[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_baseline = y_smooth - baseline
    else:
        y_baseline = y_smooth - np.min(y_smooth)

    scale = np.quantile(y_baseline, 0.99)
    y_norm = y_baseline / scale if scale > 0 else y_baseline
    y_norm = np.maximum(y_norm, 0.0)
    return y_norm


def extract_features(x, y):
    """Extract all features from a curve"""
    feat = {}

    # Basic stats (7)
    feat["mean"] = float(np.mean(y))
    feat["std"] = float(np.std(y))
    feat["min"] = float(np.min(y))
    feat["max"] = float(np.max(y))
    feat["range"] = float(np.ptp(y))
    feat["skewness"] = float(pd.Series(y).skew())
    feat["kurtosis"] = float(pd.Series(y).kurtosis())

    # Derivatives (5)
    dy = np.gradient(y, x)
    feat["max_slope"] = float(np.max(np.abs(dy)))
    feat["mean_abs_slope"] = float(np.mean(np.abs(dy)))
    feat["slope_std"] = float(np.std(dy))
    d2y = np.gradient(dy, x)
    feat["max_curvature"] = float(np.max(np.abs(d2y)))
    feat["mean_abs_curvature"] = float(np.mean(np.abs(d2y)))

    # Peaks (4)
    peaks, props = find_peaks(y, prominence=0.1)
    if len(peaks) > 0:
        proms = props.get("prominences", [0])
        feat["n_peaks"] = float(len(peaks))
        feat["max_prominence"] = float(np.max(proms))
        feat["mean_prominence"] = float(np.mean(proms))
        feat["peak_max_x"] = float(x[peaks[np.argmax(proms)]])
    else:
        feat["n_peaks"] = 0.0
        feat["max_prominence"] = 0.0
        feat["mean_prominence"] = 0.0
        feat["peak_max_x"] = float(x[np.argmax(y)])

    # Regional stats (9)
    n = len(y)
    for region, start, end in [("left", 0, n//3), ("middle", n//3, 2*n//3), ("right", 2*n//3, n)]:
        feat[f"y_{region}_mean"] = float(np.mean(y[start:end]))
        feat[f"y_{region}_std"] = float(np.std(y[start:end]))
        feat[f"y_{region}_max"] = float(np.max(y[start:end]))

    # Quartiles (4)
    q = np.percentile(y, [25, 50, 75])
    feat["q25"] = float(q[0])
    feat["q50"] = float(q[1])
    feat["q75"] = float(q[2])
    feat["iqr"] = float(q[2] - q[0])

    # FFT (11)
    fft_vals = np.abs(fft(y - np.mean(y)))
    fft_power = fft_vals ** 2

    for i, idx in enumerate(np.argsort(fft_power)[-5:][::-1]):
        feat[f"fft_power_{i}"] = float(fft_power[idx])

    feat["fft_total_power"] = float(np.sum(fft_power))
    feat["fft_entropy"] = float(entropy(fft_power + 1e-10))

    return feat


def load_and_process_data(csv_path, description=""):
    """Load CSV and extract features"""
    print(f"Loading {description}...")
    df = pd.read_csv(csv_path)
    temp_cols = sorted([c for c in df.columns if c != SPECIES_COL], key=lambda c: float(c))
    x_axis = np.array([float(c) for c in temp_cols], dtype=float)

    print(f"  Loaded {len(df)} samples, {df[SPECIES_COL].nunique()} species")

    # Preprocess
    print(f"  Preprocessing curves...")
    x_proc = np.array([preprocess_curve(x_axis, df.iloc[i, 1:].values.astype(float)) for i in range(len(df))])

    # Extract features
    print(f"  Extracting features...")
    feat_list = []
    for i in range(len(df)):
        if i % 200 == 0:
            print(f"    {i}/{len(df)}")
        f = extract_features(x_axis, x_proc[i])
        f[SPECIES_COL] = df.iloc[i][SPECIES_COL]
        feat_list.append(f)

    feat_df = pd.DataFrame(feat_list).fillna(0.0)
    return feat_df


def select_features(feat_df):
    """Select top 18 features"""
    feat_df_selected = feat_df[TOP_18_FEATURES + [SPECIES_COL]].copy()
    X = feat_df_selected.drop(columns=[SPECIES_COL]).to_numpy(float)
    y = feat_df_selected[SPECIES_COL].astype(str).to_numpy()
    return X, y


def run_cv_experiment(X, y, scenario_name, synthetic_X=None, synthetic_y=None):
    """
    Run 5-fold stratified CV (80/20 train/test per fold).
    - Train: 80% real (+ synthetic if provided)
    - Test: 20% real
    Returns: summary, list of (y_true, y_pred) for test sets
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"  → 5-fold stratified CV (seed 8)")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Add synthetic data to training only
        if synthetic_X is not None:
            X_train = np.vstack([X_train, synthetic_X])
            y_train = np.concatenate([y_train, synthetic_y])
            data_note = f"Real: {len(train_idx)} | Synth: {len(synthetic_X)} | Train: {len(X_train)}"
        else:
            data_note = f"Real only: {len(X_train)}"

        # Train model
        model = ExtraTreesClassifier(**BEST_PARAMS)

        # Calibration (no prefit, use cv=3 for consistency with first script)
        calibrated_model = CalibratedClassifierCV(model, cv=3, method="isotonic")
        calibrated_model.fit(X_train, y_train)

        # Evaluate on VAL (fold's holdout)
        y_val_pred = calibrated_model.predict(X_test)
        y_val_proba = calibrated_model.predict_proba(X_test)

        val_acc = accuracy_score(y_test, y_val_pred)
        val_f1 = f1_score(y_test, y_val_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_val_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_val_pred, average='macro', zero_division=0)

        fold_results.append({
            "fold": fold,
            "seed": RANDOM_STATE,
            "val_accuracy": float(val_acc),
            "val_f1": float(val_f1),
            "val_precision": float(precision),
            "val_recall": float(recall),
            "train_size": len(X_train),
            "val_size": len(X_test),
            "data_note": data_note
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_val_pred)
        all_y_proba.append(y_val_proba)

        print(f"  Fold {fold}/5 | "
              f"Val: {val_acc:.4f} (F1: {val_f1:.4f}) | "
              f"{data_note}")

    # Aggregate VAL metrics
    val_accs = [r["val_accuracy"] for r in fold_results]
    val_f1s = [r["val_f1"] for r in fold_results]
    precisions = [r["val_precision"] for r in fold_results]
    recalls = [r["val_recall"] for r in fold_results]

    summary = {
        "scenario": scenario_name,
        "evaluation": "5-fold stratified CV",
        "n_folds": 5,
        "seed": RANDOM_STATE,
        "accuracy": {
            "mean": float(np.mean(val_accs)),
            "std": float(np.std(val_accs)),
            "min": float(np.min(val_accs)),
            "max": float(np.max(val_accs))
        },
        "precision": {
            "mean": float(np.mean(precisions)),
            "std": float(np.std(precisions))
        },
        "recall": {
            "mean": float(np.mean(recalls)),
            "std": float(np.std(recalls))
        },
        "f1": {
            "mean": float(np.mean(val_f1s)),
            "std": float(np.std(val_f1s))
        },
        "fold_results": fold_results
    }

    print(f"\n  VAL Mean Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"  VAL Mean F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    return summary, all_y_true, all_y_pred

def save_confusion_matrix_data(y_true, y_pred):
    """Save confusion matrix (both unweighted and weighted) as JSON for later visualization"""
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(set(list(y_true) + list(y_pred)))

    # Compute weighted (normalized by true class)
    cm_weighted = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    cm_weighted = np.nan_to_num(cm_weighted)

    # Convert to JSON-serializable format
    cm_dict = {
        "classes": classes,
        "matrix_unweighted": cm.tolist(),
        "matrix_weighted": cm_weighted.tolist(),
        "n_classes": len(classes)
    }
    return cm_dict


def plot_comparison_metrics(results_normal, results_synthetic):
    """Create comparison bar charts"""
    scenarios = [results_normal["scenario"], results_synthetic["scenario"]]
    accuracies = [results_normal["accuracy"]["mean"], results_synthetic["accuracy"]["mean"]]
    f1_scores_list = [results_normal["f1"]["mean"], results_synthetic["f1"]["mean"]]
    precisions = [results_normal["precision"]["mean"], results_synthetic["precision"]["mean"]]
    recalls = [results_normal["recall"]["mean"], results_synthetic["recall"]["mean"]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance (VAL): Real vs Real+Synthetic\n(5-fold Stratified CV)', fontsize=14, fontweight='bold')

    # Accuracy
    axes[0, 0].bar(scenarios, accuracies, color=['#3498db', '#2ecc71'], alpha=0.8)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    # F1 Score
    axes[0, 1].bar(scenarios, f1_scores_list, color=['#3498db', '#2ecc71'], alpha=0.8)
    axes[0, 1].set_ylabel('F1 Score (Macro)', fontsize=11)
    axes[0, 1].set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(f1_scores_list):
        axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    # Precision
    axes[1, 0].bar(scenarios, precisions, color=['#3498db', '#2ecc71'], alpha=0.8)
    axes[1, 0].set_ylabel('Precision (Macro)', fontsize=11)
    axes[1, 0].set_title('Precision Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(precisions):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    # Recall
    axes[1, 1].bar(scenarios, recalls, color=['#3498db', '#2ecc71'], alpha=0.8)
    axes[1, 1].set_ylabel('Recall (Macro)', fontsize=11)
    axes[1, 1].set_title('Recall Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate(recalls):
        axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    filepath = results_dir / "comparison_metrics.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: comparison_metrics.png")
    return filepath


def plot_fold_val_accuracy(results_normal, results_synthetic):
    """Compare validation accuracy across all folds"""
    folds = list(range(1, 6))
    acc_normal = [r["val_accuracy"] for r in results_normal["fold_results"]]
    acc_synthetic = [r["val_accuracy"] for r in results_synthetic["fold_results"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(folds))
    width = 0.35

    bars1 = ax.bar(x - width/2, acc_normal, width, label=results_normal["scenario"],
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, acc_synthetic, width, label=results_synthetic["scenario"],
                   color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_title('Validation Accuracy per Fold (5-fold CV)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in folds])
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = results_dir / "fold_accuracy_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: fold_accuracy_comparison.png")
    return filepath


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPARING: Real Data vs Real+Synthetic Data (Synthetic only in Training)")
    print("="*70)

    # Load and process real data
    real_feat_df = load_and_process_data(REAL_DATA_PATH, "real shark data")
    X_real, y_real = select_features(real_feat_df)

    # Load and process synthetic data
    synthetic_feat_df = load_and_process_data(SYNTHETIC_DATA_PATH, "synthetic shark data")
    X_synthetic, y_synthetic = select_features(synthetic_feat_df)

    print(f"\nData Summary:")
    print(f"  Real data:      {X_real.shape[0]} samples, {len(set(y_real))} classes")
    print(f"  Synthetic data: {X_synthetic.shape[0]} samples, {len(set(y_synthetic))} classes")

    # Scenario 1: Real data only
    results_real, y_true_real, y_pred_real = run_cv_experiment(
        X_real, y_real,
        scenario_name="Real Data Only (5-fold CV)"
    )

    # Scenario 2: Real + Synthetic (synthetic only in training)
    results_synthetic, y_true_synthetic, y_pred_synthetic = run_cv_experiment(
        X_real, y_real,
        scenario_name="Real + Synthetic in Training (5-fold CV)",
        synthetic_X=X_synthetic,
        synthetic_y=y_synthetic
    )

    # ====================================================================
    # CONFUSION MATRICES (saved for visualization script)
    # ====================================================================
    print(f"\n{'='*70}")
    print("Computing confusion matrices...")
    print(f"{'='*70}")

    cm_real = save_confusion_matrix_data(y_true_real, y_pred_real)
    cm_synthetic = save_confusion_matrix_data(y_true_synthetic, y_pred_synthetic)

    # ====================================================================
    # COMPARISON CHARTS
    # ====================================================================
    print(f"\n{'='*70}")
    print("Generating comparison visualizations...")
    print(f"{'='*70}")

    plot_comparison_metrics(results_real, results_synthetic)
    plot_fold_val_accuracy(results_real, results_synthetic)

    # ====================================================================
    # SAVE JSON RESULTS
    # ====================================================================
    print(f"\n{'='*70}")
    print("Saving results...")
    print(f"{'='*70}")

    # Improvement calculations
    improvement_accuracy = (
        results_synthetic["accuracy"]["mean"] - results_real["accuracy"]["mean"]
    )
    improvement_f1 = results_synthetic["f1"]["mean"] - results_real["f1"]["mean"]
    improvement_percent = (improvement_accuracy / results_real["accuracy"]["mean"]) * 100 if results_real["accuracy"]["mean"] > 0 else 0

    # Combined results
    combined_results = {
        "experiment": "Compare Real vs Real+Synthetic using 5-fold stratified CV (seed 8)",
        "evaluation_protocol": "5-fold CV: ~80% train (real + synth for scenario 2), ~20% test (real)",
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_state": RANDOM_STATE,
        "model_type": "ExtraTreesClassifier",
        "model_parameters": BEST_PARAMS,
        "n_splits": 5,
        "top_features_used": TOP_18_FEATURES,
        "data_summary": {
            "real_data_path": str(REAL_DATA_PATH),
            "synthetic_data_path": str(SYNTHETIC_DATA_PATH),
            "real_samples": int(X_real.shape[0]),
            "synthetic_samples": int(X_synthetic.shape[0]),
            "n_features": int(X_real.shape[1]),
            "n_classes": int(len(set(y_real)))
        },
        "scenario_1_real_only": results_real,
        "scenario_2_real_synthetic": results_synthetic,
        "confusion_matrices": {
            "real_only": cm_real,
            "real_synthetic": cm_synthetic
        },
        "improvements": {
            "accuracy_absolute": float(improvement_accuracy),
            "accuracy_percent": float(improvement_percent),
            "f1_absolute": float(improvement_f1),
            "note": "Positive values indicate synthetic data helped; negative values indicate it hindered"
        }
    }

    # Save combined results
    results_file = results_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print(f"  Saved: comparison_results.json")

    # ====================================================================
    # SUMMARY REPORT
    # ====================================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\nScenario 1: Real Data Only (5-fold CV)")
    print(f" VAL Accuracy: {results_real['accuracy']['mean']:.4f} ± {results_real['accuracy']['std']:.4f}")
    print(f" VAL F1 Score: {results_real['f1']['mean']:.4f} ± {results_real['f1']['std']:.4f}")

    print(f"\nScenario 2: Real + Synthetic Training (5-fold CV)")
    print(f" VAL Accuracy: {results_synthetic['accuracy']['mean']:.4f} ± {results_synthetic['accuracy']['std']:.4f}")
    print(f" VAL F1 Score: {results_synthetic['f1']['mean']:.4f} ± {results_synthetic['f1']['std']:.4f}")

    print(f"\nImprovement from Synthetic Data:")
    print(f"  Accuracy: {improvement_accuracy:+.4f} ({improvement_percent:+.2f}%)")
    print(f"  F1 Score: {improvement_f1:+.4f}")

    print(f"\nOutput Files:")
    print(f"  1. comparison_results.json - Complete metrics + confusion matrices in JSON format")
    print(f"  2. comparison_metrics.png - Bar charts comparing accuracy/F1/precision/recall")
    print(f"  3. fold_accuracy_comparison.png - Accuracy per fold comparison")
    print(f"\nNote: Run 'python visualize_confusion_matrices.py' to generate confusion matrix visualizations")

    print(f"\n{'='*70}")
    print("✓ Comparison script completed successfully!")
    print(f"{'='*70}\n")
