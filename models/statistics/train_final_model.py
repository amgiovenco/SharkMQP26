"""
Train final Statistics model with known best parameters
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv(CSV_PATH)
temp_cols = sorted([c for c in df.columns if c != SPECIES_COL], key=lambda c: float(c))
x_axis = np.array([float(c) for c in temp_cols], dtype=float)

print(f"Loaded {len(df)} samples, {df[SPECIES_COL].nunique()} species")

# Preprocess
def preprocess_curve(x, y):
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

print("Preprocessing curves...")
x_proc = np.array([preprocess_curve(x_axis, df.iloc[i, 1:].values.astype(float)) for i in range(len(df))])

# Extract features
def extract_features(x, y):
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

    # Additional features
    feat["cv"] = feat["std"] / (feat["mean"] + 1e-10)  # coefficient of variation
    feat["peak_to_mean_ratio"] = feat["max"] / (feat["mean"] + 1e-10)

    return feat

print("Extracting features...")
feat_list = []
for i in range(len(df)):
    if i % 100 == 0:
        print(f"  {i}/{len(df)}")
    f = extract_features(x_axis, x_proc[i])
    f[SPECIES_COL] = df.iloc[i][SPECIES_COL]
    feat_list.append(f)

feat_df = pd.DataFrame(feat_list).fillna(0.0)

# Select only the 18 features used for prediction
top_18_features = [
    'peak_max_x', 'max_slope', 'y_middle_std', 'max', 'range', 'y_middle_max',
    'fft_power_1', 'fft_power_4', 'fft_power_0', 'fft_power_2', 'fft_entropy',
    'mean_abs_curvature', 'fft_power_3', 'y_middle_mean', 'y_right_max',
    'slope_std', 'mean_abs_slope', 'std'
]

feat_df_selected = feat_df[top_18_features + [SPECIES_COL]]
X = feat_df_selected.drop(columns=[SPECIES_COL]).to_numpy(float)
y = df[SPECIES_COL].astype(str).to_numpy()
feature_names = top_18_features

print(f"\nFeature matrix: {X.shape}")
print(f"Features: {feature_names}")

# Train final model with best parameters
print("\n" + "="*60)
print("Training final model on all data...")
print("="*60)

best_params = {
    'n_estimators': 1500,
    'max_depth': None,
    'min_samples_split': 6,
    'min_samples_leaf': 1,
    'max_features': 0.7,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': 1
}

final_model = RandomForestClassifier(**best_params)
final_model.fit(X, y)

# Apply calibration
calibrated_model = CalibratedClassifierCV(final_model, cv=3, method="isotonic")
calibrated_model.fit(X, y)

bundle = {
    "model": calibrated_model,
    "feature_names": feature_names,
    "model_type": "optimized_rf",
    "cv_accuracy": 0.9539,
    "params": best_params
}

joblib.dump(bundle, results_dir / "statistics_final.pkl")

print(f"Saved optimized model to {results_dir / 'statistics_final.pkl'}")

print("\n" + "="*60)
print("Done!")
print("="*60)
