"""
Train final rule-based model with known best parameters on 100% of data.
Best model: RandomForestClassifier with optimized hyperparameters
Best CV macro F1: 0.9292
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8

# Feature engineering (copied from optimize_model.py)
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

# Train final model with best parameters
print("\n" + "="*60)
print("Training final model on all data...")
print("="*60)

best_params = {
    'n_estimators': 800,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 0.8,
    'class_weight': 'balanced_subsample',
    'criterion': 'gini',
    'bootstrap': False,
    'ccp_alpha': 0.005,
    'warm_start': False,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

final_clf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(**best_params)
)

final_clf.fit(X, y_encoded)

bundle = {
    "model": final_clf,
    "label_encoder": le,
    "feature_names": feature_names,
    "model_type": "optimized_rb",
    "cv_macro_f1": 0.9292,
    "baseline_cv_macro_f1": 0.834749229442728,
    "improvement_percentage": 9.448757176775736,
    "params": {
        'model_type': 'rf',
        'rf_n_estimators': 800,
        'rf_max_depth': 20,
        'rf_min_samples_split': 5,
        'rf_min_samples_leaf': 1,
        'rf_max_features': 0.8,
        'rf_class_weight': 'balanced_subsample',
        'rf_criterion': 'gini',
        'rf_bootstrap': False,
        'rf_ccp_alpha': 0.005,
        'rf_warm_start': False
    }
}

os.makedirs("./models", exist_ok=True)
joblib.dump(bundle, "./models/rulebased_final.pkl")
print(f"Saved optimized model to ./models/rulebased_final.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Best Model: RandomForestClassifier (optimized_rb)")
print(f"Best CV Macro F1: 0.9292")
print(f"Baseline CV Macro F1: 0.8347")
print(f"Improvement: 9.45%")
print("\nModel saved to ./models/rulebased_final.pkl")
print("Done!")
