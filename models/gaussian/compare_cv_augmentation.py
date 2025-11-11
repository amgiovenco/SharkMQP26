"""
Comparison: 5-fold CV on Normal Data vs Real+Synthetic Data
============================================================
- Baseline: 5-fold CV on normal shark_dataset.csv
- Augmented: 5-fold CV with synthetic data ONLY in training sets
- Uses exact same feature extraction and model parameters
- Outputs JSON metrics and comparison charts to /results
"""
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Config
NORMAL_CSV_PATH = "../../data/shark_dataset.csv"
SYNTHETIC_CSV_PATH = "../../data/synthetic_only.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_SPLITS = 5
K_RANGE = (1, 6)
DECIMATE_STEP = 6

# Model parameters (exact same as train_cv_model.py)
BEST_PARAMS = {
    'n_estimators': 700,
    'max_depth': 20,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# ============== Feature Extraction (IDENTICAL to train_cv_model.py) ==============

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)

def gaussian_sum(x, *p):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(p), 3):
        amp, mu, sigma = p[i:i+3]
        y += gaussian(x, float(amp), float(mu), abs(float(sigma)))
    return y

def preprocess_curve(x, y):
    """Smooth + quadratic baseline removal + normalization."""
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(1.5 / dx)) | 1)
    if win >= len(y):
        win = max(7, (len(y)//2)*2 - 1)
    y_s = savgol_filter(y, window_length=max(7, win), polyorder=3, mode="interp")

    q = np.quantile(y_s, 0.3)
    mask = y_s <= q
    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_s[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_b = y_s - baseline
    else:
        y_b = y_s - np.min(y_s)

    scale = np.quantile(y_b, 0.99)
    if scale > 0:
        y_b = y_b / scale
    y_b = np.maximum(y_b, 0.0)
    return y_b

def decimate(x, y, step=8):
    return x[::step], y[::step]

def seed_peaks(x, y, k):
    spread = max(np.quantile(y, 0.90) - np.quantile(y, 0.10), 1e-6)
    prom = spread * 0.15
    peaks, props = find_peaks(y, prominence=prom, distance=max(1, len(x)//150))
    if len(peaks) == 0:
        peaks = np.argsort(y)[::-1][:k]
        prominences = y[peaks] - np.min(y)
    else:
        prominences = props["prominences"]

    if len(peaks) > 0:
        w_idx = peak_widths(y, peaks, rel_height=0.5)[0]
        w_c = w_idx * (x[1] - x[0])
    else:
        w_c = np.array([(x[-1]-x[0])/(3*k)]*k)

    min_w = max(0.2, 4*(x[1]-x[0]))
    max_w = (x[-1]-x[0]) / 3.0
    if np.ndim(w_c) == 0: w_c = np.array([w_c])
    w_c = np.clip(w_c, min_w, max_w)

    order = np.argsort(prominences)[::-1] if len(peaks) else np.arange(len(peaks))
    peaks = peaks[order][:k]
    w_c = w_c[order][:k]

    sort_lr = np.argsort(peaks)
    return peaks[sort_lr], w_c[sort_lr]

def fit_k(x, y, k):
    peaks, w_c = seed_peaks(x, y, k)
    p0, lo, hi = [], [], []
    y_max = max(np.max(y), 1e-6)
    for j, pk in enumerate(peaks):
        mu0 = float(x[pk])
        amp0 = float(max(y[pk], 1e-6))
        sigma0 = float(max(w_c[j] / (2*np.sqrt(2*np.log(2))), (x[1]-x[0])*2))
        p0 += [amp0, mu0, sigma0]
        lo += [0.0, mu0 - 3*sigma0, (x[1]-x[0])*1e-3]
        hi += [y_max*5 + 1e-6, mu0 + 3*sigma0, (x[-1]-x[0])]
    try:
        popt, _ = curve_fit(gaussian_sum, x, y, p0=p0, bounds=(lo, hi), maxfev=15000)
        return popt
    except:
        return None

def BIC(n_params, rss, n):
    return np.log(n)*n_params + n*np.log(rss/n + 1e-12)

def fit_best_K(x, y, K=(1, 5)):
    best = None
    for k in range(K[0], K[1]+1):
        try:
            popt = fit_k(x, y, k)
            if popt is None:
                continue
            yhat = gaussian_sum(x, *popt)
            rss = float(np.sum((y - yhat)**2))
            bic = float(BIC(3*k, rss, len(x)))
            if best is None or bic < best["bic"]:
                best = {"k": k, "popt": popt, "bic": bic, "rss": rss}
        except Exception:
            pass
    return best

def extract_enhanced_features(row_vals, X_axis):
    """Enhanced feature extraction with more curve statistics"""
    y0 = preprocess_curve(X_axis, np.asarray(row_vals, float))
    x, y = decimate(X_axis, y0, step=DECIMATE_STEP)
    best = fit_best_K(x, y, K=K_RANGE)

    if best is None:
        return {f"f{i}": 0.0 for i in range(30)}

    popt = best["popt"]

    # Original peak features (top 3 peaks)
    peaks = [{"amp": float(popt[i]), "mu": float(popt[i+1]), "sigma": abs(float(popt[i+2]))}
             for i in range(0, len(popt), 3)]
    peaks.sort(key=lambda d: d["amp"], reverse=True)

    feats = {}
    for i in range(3):  # Top 3 peaks
        if i < len(peaks):
            feats[f"peak{i+1}_mu"] = peaks[i]["mu"]
            feats[f"peak{i+1}_amp"] = peaks[i]["amp"]
            feats[f"peak{i+1}_sigma"] = peaks[i]["sigma"]
        else:
            feats[f"peak{i+1}_mu"] = 0.0
            feats[f"peak{i+1}_amp"] = 0.0
            feats[f"peak{i+1}_sigma"] = 0.0

    # Peak ratios and differences
    feats["delta_mu_12"] = feats["peak1_mu"] - feats["peak2_mu"]
    feats["delta_mu_23"] = feats["peak2_mu"] - feats["peak3_mu"]
    feats["amp_ratio_12"] = (feats["peak1_amp"]+1e-9)/(feats["peak2_amp"]+1e-9)
    feats["amp_ratio_23"] = (feats["peak2_amp"]+1e-9)/(feats["peak3_amp"]+1e-9)
    feats["sigma_ratio_12"] = (feats["peak1_sigma"]+1e-9)/(feats["peak2_sigma"]+1e-9)

    # Total features
    feats["total_amp"] = sum(p["amp"] for p in peaks)
    feats["total_area"] = sum(p["amp"] * p["sigma"] * np.sqrt(2*np.pi) for p in peaks)

    # Basic stats
    feats["mean"] = float(np.mean(y0))
    feats["std"] = float(np.std(y0))
    feats["max"] = float(np.max(y0))
    feats["min"] = float(np.min(y0))
    feats["range"] = feats["max"] - feats["min"]
    feats["auc"] = float(np.trapz(y0))

    return feats

# ============== Load Data ==============

print("Loading data...")
df_normal = pd.read_csv(NORMAL_CSV_PATH)
df_synthetic = pd.read_csv(SYNTHETIC_CSV_PATH)

X_raw_normal = df_normal.drop(columns=[SPECIES_COL])
y_normal = df_normal[SPECIES_COL].astype(str)

X_raw_synthetic = df_synthetic.drop(columns=[SPECIES_COL])
y_synthetic = df_synthetic[SPECIES_COL].astype(str)

# Filter species with <2 samples in normal data
counts = y_normal.value_counts()
valid_classes = counts[counts >= 2].index
mask_normal = y_normal.isin(valid_classes)
X_raw_normal = X_raw_normal[mask_normal].reset_index(drop=True)
y_normal = y_normal[mask_normal].to_numpy()

# Filter synthetic data to only include classes in normal data
mask_synthetic = y_synthetic.isin(valid_classes)
X_raw_synthetic = X_raw_synthetic[mask_synthetic].reset_index(drop=True)
y_synthetic = y_synthetic[mask_synthetic].to_numpy()

print(f"Normal data shape: {X_raw_normal.shape}, classes: {len(np.unique(y_normal))}")
print(f"Synthetic data shape: {X_raw_synthetic.shape}, classes: {len(np.unique(y_synthetic))}")

# ============== Extract Features ==============

print("Extracting Gaussian features...")
X_axis = X_raw_normal.columns.astype(float).values

# Extract features for normal data
print("  Extracting normal data features...")
feat_list_normal = []
for i in range(len(X_raw_normal)):
    if i % 100 == 0:
        print(f"    {i}/{len(X_raw_normal)}")
    feats = extract_enhanced_features(X_raw_normal.iloc[i].values, X_axis)
    feat_list_normal.append(feats)

feat_df_normal = pd.DataFrame(feat_list_normal).fillna(0.0)
X_normal = feat_df_normal.to_numpy(float)

# Extract features for synthetic data
print("  Extracting synthetic data features...")
feat_list_synthetic = []
for i in range(len(X_raw_synthetic)):
    if i % 100 == 0:
        print(f"    {i}/{len(X_raw_synthetic)}")
    feats = extract_enhanced_features(X_raw_synthetic.iloc[i].values, X_axis)
    feat_list_synthetic.append(feats)

feat_df_synthetic = pd.DataFrame(feat_list_synthetic).fillna(0.0)
X_synthetic = feat_df_synthetic.to_numpy(float)

print(f"Normal feature matrix: {X_normal.shape}")
print(f"Synthetic feature matrix: {X_synthetic.shape}")

# ============== Cross-Validation Comparison ==============

print("\n" + "="*70)
print("BASELINE: 5-Fold CV on Normal Data Only (seed=8)")
print("="*70)

# Normalize normal data
scaler_normal = StandardScaler()
X_normal_scaled = scaler_normal.fit_transform(X_normal)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

baseline_accuracies = []
baseline_f1_scores = []
baseline_precisions = []
baseline_recalls = []
baseline_fold_results = []
baseline_cms = []
baseline_predictions = {"y_true": [], "y_pred": []}

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_normal_scaled, y_normal), 1):
    X_train, X_test = X_normal_scaled[train_idx], X_normal_scaled[test_idx]
    y_train, y_test = y_normal[train_idx], y_normal[test_idx]

    model = ExtraTreesClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_normal))

    baseline_accuracies.append(acc)
    baseline_f1_scores.append(f1)
    baseline_precisions.append(prec)
    baseline_recalls.append(rec)
    baseline_cms.append(cm)
    baseline_predictions["y_true"].extend(y_test.tolist())
    baseline_predictions["y_pred"].extend(y_pred.tolist())

    baseline_fold_results.append({
        "fold": fold_idx,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "test_size": len(y_test),
        "train_size": len(y_train)
    })

    print(f"  Fold {fold_idx:2d}/5 | Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

print("\n" + "="*70)
print("AUGMENTED: 5-Fold CV with Real+Synthetic in Training (seed=8)")
print("="*70)

# Combine normal and synthetic for augmentation
X_combined = np.vstack([X_normal, X_synthetic])
y_combined = np.hstack([y_normal, y_synthetic])

# Normalize combined data
scaler_combined = StandardScaler()
X_combined_scaled = scaler_combined.fit_transform(X_combined)

# Create mapping: indices 0 to len(X_normal)-1 are real data
real_indices = set(range(len(X_normal)))

augmented_accuracies = []
augmented_f1_scores = []
augmented_precisions = []
augmented_recalls = []
augmented_fold_results = []
augmented_cms = []
augmented_predictions = {"y_true": [], "y_pred": []}

# Split on the REAL data indices only
skf_aug = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold_idx, (train_real_idx, test_real_idx) in enumerate(skf_aug.split(X_normal_scaled, y_normal), 1):
    # Train on: real data + all synthetic data
    X_train_real = X_combined_scaled[train_real_idx]
    y_train_real = y_combined[train_real_idx]

    # Add synthetic data to training
    X_train_synth = X_combined_scaled[len(X_normal):]  # Synthetic indices
    y_train_synth = y_combined[len(X_normal):]

    X_train_aug = np.vstack([X_train_real, X_train_synth])
    y_train_aug = np.hstack([y_train_real, y_train_synth])

    # Test on: real data only
    X_test = X_combined_scaled[test_real_idx]
    y_test = y_combined[test_real_idx]

    model = ExtraTreesClassifier(**BEST_PARAMS)
    model.fit(X_train_aug, y_train_aug)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_normal))

    augmented_accuracies.append(acc)
    augmented_f1_scores.append(f1)
    augmented_precisions.append(prec)
    augmented_recalls.append(rec)
    augmented_cms.append(cm)
    augmented_predictions["y_true"].extend(y_test.tolist())
    augmented_predictions["y_pred"].extend(y_pred.tolist())

    augmented_fold_results.append({
        "fold": fold_idx,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "test_size": len(y_test),
        "train_size": len(y_train_aug),
        "train_real": len(X_train_real),
        "train_synthetic": len(X_train_synth)
    })

    print(f"  Fold {fold_idx:2d}/5 | Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    print(f"           Train: {len(y_train_aug)} samples ({len(X_train_real)} real + {len(X_train_synth)} synthetic)")

# ============== Results Comparison ==============

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

print("\nBASELINE (Normal data only):")
print(f"  Mean Accuracy: {np.mean(baseline_accuracies):.4f} ± {np.std(baseline_accuracies):.4f}")
print(f"  Mean F1:       {np.mean(baseline_f1_scores):.4f} ± {np.std(baseline_f1_scores):.4f}")
print(f"  Mean Precision:{np.mean(baseline_precisions):.4f} ± {np.std(baseline_precisions):.4f}")
print(f"  Mean Recall:   {np.mean(baseline_recalls):.4f} ± {np.std(baseline_recalls):.4f}")

print("\nAUGMENTED (Normal + Synthetic in training):")
print(f"  Mean Accuracy: {np.mean(augmented_accuracies):.4f} ± {np.std(augmented_accuracies):.4f}")
print(f"  Mean F1:       {np.mean(augmented_f1_scores):.4f} ± {np.std(augmented_f1_scores):.4f}")
print(f"  Mean Precision:{np.mean(augmented_precisions):.4f} ± {np.std(augmented_precisions):.4f}")
print(f"  Mean Recall:   {np.mean(augmented_recalls):.4f} ± {np.std(augmented_recalls):.4f}")

acc_diff = np.mean(augmented_accuracies) - np.mean(baseline_accuracies)
print(f"\nImprovement: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")

# ============== Save Results to JSON ==============

print("\n" + "="*70)
print("Saving results...")
print("="*70)

results_dir = Path("./results")
results_dir.mkdir(exist_ok=True)

results = {
    "metadata": {
        "model": "ExtraTreesClassifier (Gaussian features)",
        "seed": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "model_params": {str(k): v for k, v in BEST_PARAMS.items() if k != 'n_jobs'},
        "class_labels": sorted(list(np.unique(y_normal)))
    },
    "baseline": {
        "description": "5-fold CV on normal shark_dataset.csv",
        "samples": {
            "total": len(X_normal),
            "train_per_fold": len(X_normal) * 4 // 5,
            "test_per_fold": len(X_normal) // 5
        },
        "metrics": {
            "mean_accuracy": float(np.mean(baseline_accuracies)),
            "std_accuracy": float(np.std(baseline_accuracies)),
            "mean_f1": float(np.mean(baseline_f1_scores)),
            "std_f1": float(np.std(baseline_f1_scores)),
            "mean_precision": float(np.mean(baseline_precisions)),
            "std_precision": float(np.std(baseline_precisions)),
            "mean_recall": float(np.mean(baseline_recalls)),
            "std_recall": float(np.std(baseline_recalls)),
            "min_accuracy": float(np.min(baseline_accuracies)),
            "max_accuracy": float(np.max(baseline_accuracies))
        },
        "fold_results": baseline_fold_results,
        "predictions": {
            "y_true": baseline_predictions["y_true"],
            "y_pred": baseline_predictions["y_pred"]
        }
    },
    "augmented": {
        "description": "5-fold CV with synthetic data ONLY in training sets",
        "samples": {
            "total_real": len(X_normal),
            "total_synthetic": len(X_synthetic),
            "test_real_per_fold": len(X_normal) // 5,
            "train_per_fold": f"~{(len(X_normal)*4//5)} real + {len(X_synthetic)} synthetic"
        },
        "metrics": {
            "mean_accuracy": float(np.mean(augmented_accuracies)),
            "std_accuracy": float(np.std(augmented_accuracies)),
            "mean_f1": float(np.mean(augmented_f1_scores)),
            "std_f1": float(np.std(augmented_f1_scores)),
            "mean_precision": float(np.mean(augmented_precisions)),
            "std_precision": float(np.std(augmented_precisions)),
            "mean_recall": float(np.mean(augmented_recalls)),
            "std_recall": float(np.std(augmented_recalls)),
            "min_accuracy": float(np.min(augmented_accuracies)),
            "max_accuracy": float(np.max(augmented_accuracies))
        },
        "fold_results": augmented_fold_results,
        "predictions": {
            "y_true": augmented_predictions["y_true"],
            "y_pred": augmented_predictions["y_pred"]
        }
    },
    "comparison": {
        "accuracy_improvement": float(acc_diff),
        "accuracy_improvement_percent": float(acc_diff * 100),
        "f1_improvement": float(np.mean(augmented_f1_scores) - np.mean(baseline_f1_scores)),
        "precision_improvement": float(np.mean(augmented_precisions) - np.mean(baseline_precisions)),
        "recall_improvement": float(np.mean(augmented_recalls) - np.mean(baseline_recalls))
    }
}

results_file = results_dir / "cv_augmentation_comparison.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"[OK] Results saved to {results_file}")

# ============== Generate Visualizations ==============

print("Generating visualizations...")

# 1. Accuracy comparison by fold
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

folds = list(range(1, N_SPLITS + 1))
axes[0].plot(folds, baseline_accuracies, 'o-', label='Baseline (Normal)', linewidth=2, markersize=8)
axes[0].plot(folds, augmented_accuracies, 's-', label='Augmented (Normal+Synthetic)', linewidth=2, markersize=8)
axes[0].set_xlabel('Fold', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Accuracy by Fold', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(folds)

# 2. Metrics comparison
metrics_names = ['Accuracy', 'F1', 'Precision', 'Recall']
baseline_means = [
    np.mean(baseline_accuracies),
    np.mean(baseline_f1_scores),
    np.mean(baseline_precisions),
    np.mean(baseline_recalls)
]
augmented_means = [
    np.mean(augmented_accuracies),
    np.mean(augmented_f1_scores),
    np.mean(augmented_precisions),
    np.mean(augmented_recalls)
]

x = np.arange(len(metrics_names))
width = 0.35

axes[1].bar(x - width/2, baseline_means, width, label='Baseline', alpha=0.8)
axes[1].bar(x + width/2, augmented_means, width, label='Augmented', alpha=0.8)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Metrics Comparison', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics_names)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_dir / "01_accuracy_metrics_comparison.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved 01_accuracy_metrics_comparison.png")
plt.close()

# 3. Box plot for fold-wise distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

data_to_plot = [
    [baseline_accuracies, augmented_accuracies],
    [baseline_f1_scores, augmented_f1_scores],
    [baseline_precisions, augmented_precisions],
    [baseline_recalls, augmented_recalls]
]

metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

for idx, (ax, data, metric) in enumerate(zip(axes.flatten(), data_to_plot, metric_names)):
    bp = ax.boxplot(data, labels=['Baseline', 'Augmented'], patch_artist=True)

    colors = ['lightblue', 'lightsalmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'{metric} Distribution Across Folds', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add individual points
    for i, (baseline_val, aug_val) in enumerate(zip(data[0], data[1])):
        ax.plot([1 + np.random.randn()*0.04], [baseline_val], 'o', color='darkblue', alpha=0.6, markersize=5)
        ax.plot([2 + np.random.randn()*0.04], [aug_val], 's', color='darkorange', alpha=0.6, markersize=5)

plt.suptitle('Fold-wise Metrics Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(results_dir / "02_metrics_distribution.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved 02_metrics_distribution.png")
plt.close()

# 4. Training set size impact
fig, ax = plt.subplots(figsize=(10, 6))

train_sizes = [fold["train_size"] for fold in augmented_fold_results]
train_real = [fold["train_real"] for fold in augmented_fold_results]
train_synthetic = [fold["train_synthetic"] for fold in augmented_fold_results]
folds = list(range(1, N_SPLITS + 1))

ax.bar(folds, train_real, label='Real Training Samples', color='steelblue', alpha=0.8)
ax.bar(folds, train_synthetic, bottom=train_real, label='Synthetic Training Samples', color='coral', alpha=0.8)

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Training Set Composition (Augmented)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(folds)

# Add total on top
for i, (fold, total) in enumerate(zip(folds, train_sizes)):
    ax.text(fold, total + 10, str(total), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(results_dir / "04_training_set_composition.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved 04_training_set_composition.png")
plt.close()

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nResults saved to: {results_dir}/")
print(f"  - cv_augmentation_comparison.json (metrics & predictions)")
print(f"  - 01_accuracy_metrics_comparison.png")
print(f"  - 02_metrics_distribution.png")
print(f"  - 03_training_set_composition.png")
print(f"\nConfusion matrices will be generated separately from JSON")
print("="*70)
