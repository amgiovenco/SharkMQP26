"""
Evaluate ExtraTrees classifier with best hyperparameters on an 80/20 train/test split.
Collects: Accuracy, Precision, Recall, F1-Score, Macro F1-Score, Avg. Confidence
No cross-validation.
"""
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================= CONFIG =============================
CSV_PATH = Path(__file__).parent.parent.parent / "data" / "shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TOP_FEATURES = [
    'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
    'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
    'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
    'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
    'slope_std', 'y_left_max'
]

BEST_PARAMS = {
    'n_estimators':      1400,
    'max_depth':         80,
    'min_samples_split': 3,
    'min_samples_leaf':  1,
    'max_features':      0.6,
    'class_weight':      'balanced_subsample',
    'bootstrap':         False,
    'ccp_alpha':         0.002,
    'random_state':      RANDOM_STATE,
    'n_jobs':            8,
}
# =================================================================


def preprocess_curve(x, y):
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(1.5 / dx)) | 1)
    if win >= len(y):
        win = max(7, (len(y) // 2) * 2 - 1)
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
    for region, start, end in [("left", 0, n // 3), ("middle", n // 3, 2 * n // 3), ("right", 2 * n // 3, n)]:
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


# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
temp_cols = sorted([c for c in df.columns if c != SPECIES_COL], key=lambda c: float(c))
x_axis = np.array([float(c) for c in temp_cols], dtype=float)
print(f"Loaded {len(df)} samples, {df[SPECIES_COL].nunique()} species")

# ─── Preprocess ───────────────────────────────────────────────────────────────
print("Preprocessing curves...")
x_proc = np.array([
    preprocess_curve(x_axis, df.iloc[i, 1:].values.astype(float))
    for i in range(len(df))
])

# ─── Feature extraction ───────────────────────────────────────────────────────
print("Extracting features...")
feat_list = []
for i in range(len(df)):
    if i % 100 == 0:
        print(f"  {i}/{len(df)}")
    f = extract_features(x_axis, x_proc[i])
    f[SPECIES_COL] = df.iloc[i][SPECIES_COL]
    feat_list.append(f)

feat_df = pd.DataFrame(feat_list).fillna(0.0)
X = feat_df[TOP_FEATURES].to_numpy(float)
y = df[SPECIES_COL].astype(str).to_numpy()
print(f"Features: {len(TOP_FEATURES)}")

# ─── 80/20 split (seed 8) ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
class_names = le.classes_
print(f"Classes ({len(class_names)}): {list(class_names)}")

# ─── Train ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING ExtraTrees (best params, 80/20 split, seed 8)")
print("="*60)

model = ExtraTreesClassifier(**BEST_PARAMS)
model.fit(X_train, y_train)
print("Training complete.")

# ─── Calibrate ────────────────────────────────────────────────────────────────
print("Calibrating probabilities (isotonic, cv=3)...")
calibrated = CalibratedClassifierCV(model, cv=3, method="isotonic")
calibrated.fit(X_train, y_train)
print("Calibration complete.")

# ─── Evaluate on test set ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATING ON TEST SET (20% holdout)")
print("="*60)

proba      = calibrated.predict_proba(X_test)
all_preds  = np.argmax(proba, axis=1)
all_probs  = proba.max(axis=1)

# Map encoded preds back to string labels for sklearn metrics
y_pred_labels = le.inverse_transform(all_preds)

accuracy    = 100. * accuracy_score(y_test, y_pred_labels)
precision_w = 100. * precision_score(y_test, y_pred_labels, average='weighted', zero_division=0)
recall_w    = 100. * recall_score(y_test, y_pred_labels, average='weighted', zero_division=0)
f1_w        = 100. * f1_score(y_test, y_pred_labels, average='weighted', zero_division=0)
macro_f1    = 100. * f1_score(y_test, y_pred_labels, average='macro', zero_division=0)
avg_conf    = 100. * float(all_probs.mean())

print(f"\n{'Metric':<25} {'Value':>10}")
print("-" * 36)
print(f"{'Model':<25} {'ExtraTrees':>10}")
print(f"{'Accuracy':<25} {accuracy:>9.2f}%")
print(f"{'Precision (weighted)':<25} {precision_w:>9.2f}%")
print(f"{'Recall (weighted)':<25} {recall_w:>9.2f}%")
print(f"{'F1-Score (weighted)':<25} {f1_w:>9.2f}%")
print(f"{'Macro F1-Score':<25} {macro_f1:>9.2f}%")
print(f"{'Avg. Confidence':<25} {avg_conf:>9.2f}%")

print("\nPer-class report:")
print(classification_report(y_test, y_pred_labels, zero_division=0))

# ─── Save results JSON ────────────────────────────────────────────────────────
results = {
    "model": "ExtraTrees",
    "split": {"train": "80%", "test": "20%", "random_state": RANDOM_STATE},
    "best_params": BEST_PARAMS,
    "metrics": {
        "accuracy":           round(accuracy,    4),
        "precision_weighted": round(precision_w, 4),
        "recall_weighted":    round(recall_w,    4),
        "f1_score_weighted":  round(f1_w,        4),
        "macro_f1_score":     round(macro_f1,    4),
        "avg_confidence":     round(avg_conf,    4),
    }
}

json_path = RESULTS_DIR / "eval_metrics_stats_results.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {json_path}")

# ─── Save bundle (same format as optimize_stats.py) ───────────────────────────
bundle = {
    "model":         calibrated,
    "feature_names": TOP_FEATURES,
    "label_encoder": le,
    "model_type":    "optimized_et",
    "cv_accuracy":   float(macro_f1),
    "test_accuracy": float(macro_f1),
    "params":        BEST_PARAMS,
}

bundle_path = RESULTS_DIR / "statistics_final.pkl"
with open(bundle_path, 'wb') as f:
    pickle.dump(bundle, f)
print(f"Saved model bundle to {bundle_path}")

print("\nDone!")
