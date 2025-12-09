"""
Confidence Calibration Analysis for Statistics (ExtraTrees) Model

Tests Statistics model stability across different random seeds and analyzes:
1. Test accuracy consistency
2. Confidence calibration (are high confidences actually correct?)
3. Top-k prediction quality
4. Confidence when correct vs wrong

Run this to verify your Statistics model is production-ready for user-facing predictions.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Test these seeds for stability
SEEDS_TO_TEST = [8, 42, 123]

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
REAL_DATA_PATH = DATA_DIR / "shark_dataset.csv"
SYNTHETIC_DIR = DATA_DIR / "syntheticDataIndividual"

# Output
OUTPUT_DIR = SCRIPT_DIR / "confidence_analysis_stats_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data split (0/50/50 - all synthetic for training)
REAL_VAL_SPLIT = 0.5  # 50% validation
REAL_TEST_SPLIT = 0.5  # 50% test

# Use all synthetic data (50 per species)
MAX_SYN_PER_SPECIES = 50

# Statistics model parameters
STATS_N_ESTIMATORS = 1700
STATS_MAX_DEPTH = None
STATS_MIN_SAMPLES_SPLIT = 9
STATS_MIN_SAMPLES_LEAF = 1
STATS_MAX_FEATURES = 0.7
STATS_CLASS_WEIGHT = 'balanced'

# Top features
STATS_TOP_FEATURES = [
    'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
    'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
    'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
    'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
    'slope_std', 'y_left_max'
]

print("=" * 80)
print("CONFIDENCE CALIBRATION ANALYSIS - STATISTICS MODEL")
print("=" * 80)
print(f"Testing seeds: {SEEDS_TO_TEST}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 80)
print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_real_data() -> pd.DataFrame:
    """Load real shark dataset."""
    if not REAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Real data not found at {REAL_DATA_PATH}")
    return pd.read_csv(REAL_DATA_PATH)

def load_synthetic_data(species_list: List[str]) -> Dict[str, pd.DataFrame]:
    """Load synthetic data from individual species CSV files."""
    synthetic_data = {}

    if not SYNTHETIC_DIR.exists():
        print(f"Warning: Synthetic data directory not found at {SYNTHETIC_DIR}")
        return synthetic_data

    for species in species_list:
        filename = f"synthetic_{species.replace(' ', '_')}.csv"
        filepath = SYNTHETIC_DIR / filename

        if filepath.exists():
            df = pd.read_csv(filepath)
            synthetic_data[species] = df.reset_index(drop=True)

    return synthetic_data

def create_synthetic_training_set(synthetic_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all synthetic data (up to 50 per species)."""
    all_synthetic = []

    for species, df in synthetic_data.items():
        # Use up to MAX_SYN_PER_SPECIES samples per species
        n_samples = min(len(df), MAX_SYN_PER_SPECIES)
        sampled = df.iloc[:n_samples].copy()
        all_synthetic.append(sampled)

    synthetic_train = pd.concat(all_synthetic, ignore_index=True)
    return synthetic_train.reset_index(drop=True)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def preprocess_curve(x, y):
    """Preprocess curve with smoothing and baseline removal."""
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
    """Extract 36 features from fluorescence curve."""
    feat = {}
    feat["mean"] = float(np.mean(y))
    feat["std"] = float(np.std(y))
    feat["min"] = float(np.min(y))
    feat["max"] = float(np.max(y))
    feat["range"] = float(np.ptp(y))
    feat["skewness"] = float(pd.Series(y).skew())
    feat["kurtosis"] = float(pd.Series(y).kurtosis())
    dy = np.gradient(y, x)
    feat["max_slope"] = float(np.max(np.abs(dy)))
    feat["mean_abs_slope"] = float(np.mean(np.abs(dy)))
    feat["slope_std"] = float(np.std(dy))
    d2y = np.gradient(dy, x)
    feat["max_curvature"] = float(np.max(np.abs(d2y)))
    feat["mean_abs_curvature"] = float(np.mean(np.abs(d2y)))
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
    n = len(y)
    for region, start, end in [("left", 0, n//3), ("middle", n//3, 2*n//3), ("right", 2*n//3, n)]:
        feat[f"y_{region}_mean"] = float(np.mean(y[start:end]))
        feat[f"y_{region}_std"] = float(np.std(y[start:end]))
        feat[f"y_{region}_max"] = float(np.max(y[start:end]))
    q = np.percentile(y, [25, 50, 75])
    feat["q25"] = float(q[0])
    feat["q50"] = float(q[1])
    feat["q75"] = float(q[2])
    feat["iqr"] = float(q[2] - q[0])
    fft_vals = np.abs(fft(y - np.mean(y)))
    fft_power = fft_vals ** 2
    for i, idx in enumerate(np.argsort(fft_power)[-5:][::-1]):
        feat[f"fft_power_{i}"] = float(fft_power[idx])
    feat["fft_total_power"] = float(np.sum(fft_power))
    feat["fft_entropy"] = float(entropy(fft_power + 1e-10))
    return feat

def prepare_stats_data(df: pd.DataFrame):
    """Prepare feature matrix for statistics model."""
    temp_cols = sorted([c for c in df.columns if c != 'Species'], key=lambda c: float(c))
    x_axis = np.array([float(c) for c in temp_cols], dtype=float)

    # Preprocess curves
    x_proc = np.array([preprocess_curve(x_axis, df.iloc[i, 1:].values.astype(float)) for i in range(len(df))])

    # Extract features
    feat_list = []
    for i in range(len(df)):
        f = extract_features(x_axis, x_proc[i])
        f['Species'] = df.iloc[i]['Species']
        feat_list.append(f)

    feat_df = pd.DataFrame(feat_list).fillna(0.0)
    X = feat_df[STATS_TOP_FEATURES].to_numpy(float)
    y = df['Species'].astype(str).to_numpy()

    return X, y

# ============================================================================
# CONFIDENCE ANALYSIS FUNCTIONS
# ============================================================================

def evaluate_stats_with_confidence(model, X_test, y_test):
    """
    Evaluate Statistics model and return detailed confidence metrics.

    Returns:
        dict with:
        - accuracy: Overall accuracy
        - f1: Macro F1 score
        - all_probs: numpy array of all predicted probabilities
        - all_preds: numpy array of predicted classes
        - all_labels: numpy array of true labels
        - avg_confidence: Average max probability
        - confidence_when_correct: Avg confidence on correct predictions
        - confidence_when_wrong: Avg confidence on wrong predictions
        - top3_accuracy: Fraction where true label is in top 3
        - top5_accuracy: Fraction where true label is in top 5
        - confidence_gap: Avg gap between 1st and 2nd choice
    """
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    all_probs = model.predict_proba(X_test)

    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Convert string labels to indices for comparison
    label_encoder = LabelEncoder()
    label_encoder.fit(y_test)
    y_test_encoded = label_encoder.transform(y_test)
    y_pred_encoded = label_encoder.transform(y_pred)

    # Confidence metrics
    max_probs = np.max(all_probs, axis=1)
    avg_confidence = np.mean(max_probs)

    # Confidence when correct vs wrong
    correct_mask = (y_pred_encoded == y_test_encoded)
    conf_when_correct = np.mean(max_probs[correct_mask]) if correct_mask.any() else 0.0
    conf_when_wrong = np.mean(max_probs[~correct_mask]) if (~correct_mask).any() else 0.0

    # Top-k accuracy
    top3_correct = []
    top5_correct = []
    confidence_gaps = []

    for i, (probs, true_label) in enumerate(zip(all_probs, y_test_encoded)):
        # Get top-k indices (sorted descending)
        sorted_indices = np.argsort(probs)[::-1]
        top3 = sorted_indices[:3]
        top5 = sorted_indices[:5]

        top3_correct.append(true_label in top3)
        top5_correct.append(true_label in top5)

        # Confidence gap between 1st and 2nd choice
        if len(sorted_indices) >= 2:
            gap = probs[sorted_indices[0]] - probs[sorted_indices[1]]
            confidence_gaps.append(gap)

    top3_accuracy = np.mean(top3_correct)
    top5_accuracy = np.mean(top5_correct)
    avg_confidence_gap = np.mean(confidence_gaps) if confidence_gaps else 0.0

    return {
        'accuracy': accuracy,
        'f1': f1,
        'all_probs': all_probs,
        'all_preds': y_pred_encoded,
        'all_labels': y_test_encoded,
        'avg_confidence': avg_confidence,
        'confidence_when_correct': conf_when_correct,
        'confidence_when_wrong': conf_when_wrong,
        'num_correct': np.sum(correct_mask),
        'num_wrong': np.sum(~correct_mask),
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'confidence_gap': avg_confidence_gap
    }

def compute_calibration_metrics(probs, preds, labels, n_bins=10):
    """
    Compute calibration curve and Expected Calibration Error (ECE).

    Returns:
        dict with:
        - ece: Expected Calibration Error
        - bin_accuracies: Accuracy in each confidence bin
        - bin_confidences: Average confidence in each bin
        - bin_counts: Number of samples in each bin
    """
    max_probs = np.max(probs, axis=1)
    correct = (preds == labels)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(max_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    total_samples = len(max_probs)

    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = max_probs[mask].mean()
            bin_count = mask.sum()

            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)

            # ECE contribution
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    return {
        'ece': ece,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_and_evaluate_stats(train_data, test_data, seed):
    """Train Statistics model with given seed and evaluate with confidence metrics."""
    np.random.seed(seed)

    print(f"  Preparing features with seed {seed}...")
    X_train, y_train = prepare_stats_data(train_data)
    X_test, y_test = prepare_stats_data(test_data)

    print(f"  Training ExtraTrees model...")
    model = ExtraTreesClassifier(
        n_estimators=STATS_N_ESTIMATORS,
        max_depth=STATS_MAX_DEPTH,
        min_samples_split=STATS_MIN_SAMPLES_SPLIT,
        min_samples_leaf=STATS_MIN_SAMPLES_LEAF,
        max_features=STATS_MAX_FEATURES,
        class_weight=STATS_CLASS_WEIGHT,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate with confidence metrics
    print(f"  Evaluating on test set...")
    results = evaluate_stats_with_confidence(model, X_test, y_test)

    # Compute calibration
    calibration = compute_calibration_metrics(
        results['all_probs'],
        results['all_preds'],
        results['all_labels']
    )
    results['calibration'] = calibration

    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confidence_analysis(all_results, output_dir):
    """Generate comprehensive confidence analysis plots."""

    # Extract metrics across seeds
    seeds = list(all_results.keys())
    accuracies = [all_results[s]['accuracy'] for s in seeds]
    f1s = [all_results[s]['f1'] for s in seeds]
    avg_confs = [all_results[s]['avg_confidence'] for s in seeds]
    conf_correct = [all_results[s]['confidence_when_correct'] for s in seeds]
    conf_wrong = [all_results[s]['confidence_when_wrong'] for s in seeds]
    top3_accs = [all_results[s]['top3_accuracy'] for s in seeds]
    conf_gaps = [all_results[s]['confidence_gap'] for s in seeds]
    eces = [all_results[s]['calibration']['ece'] for s in seeds]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistics Model Confidence Calibration Analysis Across Seeds', fontsize=16, fontweight='bold')

    # Plot 1: Accuracy and F1
    ax = axes[0, 0]
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, f1s, width, label='Macro F1', alpha=0.8)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Score')
    ax.set_title('Accuracy and F1 Across Seeds')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.set_ylim([0.95, 1.0])
    ax.grid(True, alpha=0.3)

    # Plot 2: Confidence metrics
    ax = axes[0, 1]
    ax.bar(x, avg_confs, alpha=0.8, label='Avg Confidence')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Confidence')
    ax.set_title('Average Confidence')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim([0.9, 1.0])
    ax.grid(True, alpha=0.3)

    # Plot 3: Confidence when correct vs wrong
    ax = axes[0, 2]
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, conf_correct, width, label='When Correct', alpha=0.8, color='green')
    ax.bar(x + width/2, conf_wrong, width, label='When Wrong', alpha=0.8, color='red')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence: Correct vs Wrong Predictions')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3)

    # Plot 4: Top-k accuracy
    ax = axes[1, 0]
    ax.bar(x, top3_accs, alpha=0.8)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Top-3 Accuracy')
    ax.set_title('Top-3 Accuracy (True label in top 3 predictions)')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim([0.95, 1.0])
    ax.grid(True, alpha=0.3)

    # Plot 5: Confidence gap
    ax = axes[1, 1]
    ax.bar(x, conf_gaps, alpha=0.8)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Confidence Gap')
    ax.set_title('Avg Gap Between 1st and 2nd Choice')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.grid(True, alpha=0.3)

    # Plot 6: Expected Calibration Error
    ax = axes[1, 2]
    ax.bar(x, eces, alpha=0.8, color='orange')
    ax.set_xlabel('Seed')
    ax.set_ylabel('ECE')
    ax.set_title('Expected Calibration Error (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved confidence analysis plot to {output_dir / 'confidence_analysis_stats.png'}")

    # Calibration curve for first seed
    seed = seeds[0]
    cal = all_results[seed]['calibration']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(cal['bin_confidences'], cal['bin_accuracies'], 'o-',
            label=f'Statistics Model (ECE={cal["ece"]:.4f})', linewidth=2, markersize=8)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Statistics Model Calibration Curve (Seed {seed})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved calibration curve to {output_dir / 'calibration_curve_stats.png'}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nConfidence Calibration Analysis - Statistics Model")
    print("Testing ExtraTrees model with 0/50/50 split (all synthetic training) across multiple seeds")
    print(f"\nConfiguration:")
    print(f"  Model: ExtraTrees (Statistics with Feature Engineering)")
    print(f"  Seeds to test: {SEEDS_TO_TEST}")
    print(f"  Max synthetic per species: {MAX_SYN_PER_SPECIES}")

    # Load data
    print("\nLoading data...")
    real_data = load_real_data()
    print(f"  Loaded {len(real_data)} real samples")

    # Split real data 50/50 for validation and test (use seed 8 for splitting)
    print("\nSplitting real data 50/50...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=REAL_TEST_SPLIT, random_state=8)
    for val_idx, test_idx in sss.split(real_data, real_data['Species']):
        real_val = real_data.iloc[val_idx].reset_index(drop=True)
        real_test = real_data.iloc[test_idx].reset_index(drop=True)

    num_classes = len(real_data['Species'].unique())
    print(f"  Validation: {len(real_val)} samples")
    print(f"  Test: {len(real_test)} samples")
    print(f"  Number of classes: {num_classes}")

    # Load synthetic data
    print("\nLoading synthetic data...")
    synthetic_data = load_synthetic_data(real_data['Species'].unique().tolist())
    print(f"  Loaded synthetic data for {len(synthetic_data)} species")

    # Create synthetic training set
    print("\nCreating synthetic training set...")
    synthetic_train = create_synthetic_training_set(synthetic_data)
    print(f"  Synthetic training: {len(synthetic_train)} samples")

    # Test across multiple seeds
    all_results = {}

    print("\n" + "="*80)
    print("TRAINING AND EVALUATING ACROSS SEEDS")
    print("="*80)

    for seed in SEEDS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}")

        print(f"  Training data: {len(synthetic_train)} samples (all synthetic)")
        print(f"  Test data: {len(real_test)} samples (real)")

        # Train and evaluate
        results = train_and_evaluate_stats(synthetic_train, real_test, seed)
        all_results[seed] = results

        # Print results
        print(f"\n  Results for seed {seed}:")
        print(f"    Accuracy: {results['accuracy']:.4f}")
        print(f"    Macro F1: {results['f1']:.4f}")
        print(f"    Avg Confidence: {results['avg_confidence']:.4f}")
        print(f"    Confidence when correct: {results['confidence_when_correct']:.4f}")
        print(f"    Confidence when wrong: {results['confidence_when_wrong']:.4f}")
        print(f"    Num wrong: {results['num_wrong']}")
        print(f"    Top-3 Accuracy: {results['top3_accuracy']:.4f}")
        print(f"    Top-5 Accuracy: {results['top5_accuracy']:.4f}")
        print(f"    Confidence Gap (1st vs 2nd): {results['confidence_gap']:.4f}")
        print(f"    Expected Calibration Error: {results['calibration']['ece']:.4f}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY ACROSS SEEDS")
    print("="*80)

    accuracies = [all_results[s]['accuracy'] for s in SEEDS_TO_TEST]
    f1s = [all_results[s]['f1'] for s in SEEDS_TO_TEST]
    eces = [all_results[s]['calibration']['ece'] for s in SEEDS_TO_TEST]

    print(f"\nAccuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Macro F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"ECE:       {np.mean(eces):.4f} ± {np.std(eces):.4f}")

    # Check stability
    acc_range = max(accuracies) - min(accuracies)
    f1_range = max(f1s) - min(f1s)

    print(f"\nStability Check:")
    print(f"  Accuracy range: {acc_range:.4f} (max - min)")
    print(f"  F1 range: {f1_range:.4f} (max - min)")

    if acc_range < 0.02 and f1_range < 0.02:
        print(f"  ✓ STABLE: Results are consistent across seeds (<2% variation)")
    else:
        print(f"  ✗ UNSTABLE: Results vary significantly across seeds (>2% variation)")

    # Check calibration
    avg_ece = np.mean(eces)
    if avg_ece < 0.05:
        print(f"\nCalibration Check:")
        print(f"  ✓ WELL CALIBRATED: ECE = {avg_ece:.4f} < 0.05")
    else:
        print(f"\nCalibration Check:")
        print(f"  ⚠ POORLY CALIBRATED: ECE = {avg_ece:.4f} > 0.05")
        print(f"  Consider adding temperature scaling for better confidence estimates")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_confidence_analysis(all_results, OUTPUT_DIR)

    # Save detailed results
    summary = {
        'model': 'Statistics (ExtraTrees)',
        'seeds_tested': SEEDS_TO_TEST,
        'max_synthetic_per_species': MAX_SYN_PER_SPECIES,
        'summary_stats': {
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'ece_mean': float(np.mean(eces)),
            'ece_std': float(np.std(eces)),
            'accuracy_range': float(acc_range),
            'f1_range': float(f1_range)
        },
        'per_seed_results': {}
    }

    for seed in SEEDS_TO_TEST:
        r = all_results[seed]
        summary['per_seed_results'][seed] = {
            'accuracy': float(r['accuracy']),
            'f1': float(r['f1']),
            'avg_confidence': float(r['avg_confidence']),
            'confidence_when_correct': float(r['confidence_when_correct']),
            'confidence_when_wrong': float(r['confidence_when_wrong']),
            'num_wrong': int(r['num_wrong']),
            'top3_accuracy': float(r['top3_accuracy']),
            'top5_accuracy': float(r['top5_accuracy']),
            'confidence_gap': float(r['confidence_gap']),
            'ece': float(r['calibration']['ece'])
        }

    results_path = OUTPUT_DIR / 'confidence_analysis_stats_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nCheck {OUTPUT_DIR} for:")
    print(f"  - confidence_analysis_stats.png (comparison across seeds)")
    print(f"  - calibration_curve_stats.png (calibration curve)")
    print(f"  - confidence_analysis_stats_results.json (detailed results)")
