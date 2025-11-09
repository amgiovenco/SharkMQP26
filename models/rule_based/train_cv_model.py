"""
Cross-validation test for Rule-Based model (10-fold stratified, 90/10 split)
Sanity check to detect overfitting in train_final_model.py
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_SPLITS = 10

# Feature engineering (copied from train_final_model.py)
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

# Load data
print("Loading data...")
df = pd.read_csv(CSV_PATH)
X_raw = df.drop(columns=[SPECIES_COL])
y = df[SPECIES_COL].astype(str)

print("Extracting features...")
Xf, feature_names = engineer_features(X_raw)
X = Xf.to_numpy(float)

print(f"Data shape: {X.shape}")
print(f"Features: {feature_names}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Best parameters from train_final_model.py
best_params = {
    'n_estimators': 790,
    'min_samples_leaf': 1,
    'max_depth': 15,
    'max_features': None,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

print("\n" + "="*70)
print(f"CROSS-VALIDATION: Rule-Based Model (10-fold stratified, seed={RANDOM_STATE})")
print("="*70)
print(f"Best params: {best_params}")

# Create results directory
results_dir = Path("./cv_results")
results_dir.mkdir(exist_ok=True)
print(f"Saving fold models to: {results_dir}/")

# 10-fold stratified cross-validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_accuracies = []
fold_f1_scores = []
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Train on 90%
    clf = make_pipeline(
        StandardScaler(),
        ExtraTreesClassifier(**best_params)
    )
    clf.fit(X_train, y_train)

    # Test on 10%
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    fold_accuracies.append(acc)
    fold_f1_scores.append(f1)

    # Save fold model with accuracy in filename
    model_filename = f"fold_{fold_idx:02d}_acc_{acc:.4f}_f1_{f1:.4f}.pkl"
    joblib.dump(clf, results_dir / model_filename)

    fold_results.append({
        "fold": fold_idx,
        "accuracy": float(acc),
        "f1": float(f1),
        "test_size": len(y_test),
        "model_file": model_filename
    })

    print(f"  Fold {fold_idx:2d}/10 | Accuracy: {acc:.4f} | F1: {f1:.4f} | Test size: {len(y_test)} | Saved: {model_filename}")

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Mean F1:       {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
print(f"Min Accuracy:  {np.min(fold_accuracies):.4f}")
print(f"Max Accuracy:  {np.max(fold_accuracies):.4f}")
print(f"\nTraining data: 90% ({len(X) * 9 // 10} samples per fold)")
print(f"Test data:     10% ({len(X) // 10} samples per fold)")
print("\n[Note] train_final_model.py trains on 100% data, overfitting is expected.")
print("="*70)

# Save summary
summary = {
    "model": "Rule-Based",
    "seed": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "best_params": best_params,
    "mean_accuracy": float(np.mean(fold_accuracies)),
    "std_accuracy": float(np.std(fold_accuracies)),
    "mean_f1": float(np.mean(fold_f1_scores)),
    "std_f1": float(np.std(fold_f1_scores)),
    "min_accuracy": float(np.min(fold_accuracies)),
    "max_accuracy": float(np.max(fold_accuracies)),
    "fold_results": fold_results
}

summary_file = results_dir / "cv_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[OK] Summary saved to {summary_file}")
print(f"[OK] All fold models saved to {results_dir}/")
