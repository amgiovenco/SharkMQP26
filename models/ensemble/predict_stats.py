"""Statistics model prediction from local models directory."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.ensemble import ExtraTreesClassifier


def preprocess_curve_statistics(x, y):
    """Preprocess curve: smooth + baseline remove + normalize."""
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


def extract_statistics_features(x, y):
    """Extract all 36 features from curve."""
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

    # FFT features (11)
    fft_vals = np.abs(fft(y - np.mean(y)))
    fft_power = fft_vals ** 2  # IMPORTANT: Square to match training
    feat["fft_max"] = float(np.max(fft_vals))
    feat["fft_mean"] = float(np.mean(fft_vals))
    feat["fft_std"] = float(np.std(fft_vals))
    feat["fft_energy"] = float(np.sum(fft_power))

    # Top 5 FFT power values
    top_5_idx = np.argsort(fft_power)[-5:][::-1]
    for i in range(5):
        if i < len(top_5_idx):
            feat[f"fft_power_{i}"] = float(fft_power[top_5_idx[i]])
        else:
            feat[f"fft_power_{i}"] = 0.0

    freq_idx = np.argsort(fft_power)[-3:]
    for i, idx in enumerate(freq_idx):
        feat[f"fft_peak{i}"] = float(fft_power[idx])

    # Entropy and other
    norm_fft = fft_vals / (np.sum(fft_vals) + 1e-12)
    feat["fft_entropy"] = float(-np.sum(norm_fft * np.log(norm_fft + 1e-12)))
    feat["autocorr_lag1"] = float(np.corrcoef(y[:-1], y[1:])[0, 1])

    return feat


def get_stats_predictions(X_raw: pd.DataFrame, models_dir: str = "./models") -> np.ndarray:
    """Get statistics model predictions from local models directory."""
    print(" statistics...", end=" ", flush=True)
    try:
        model_path = Path(models_dir) / "STATISTICS_statistics_final.pkl"
        if not model_path.exists():
            print("[FAIL] STATISTICS_statistics_final.pkl not found in ./models/")
            return None

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        # Handle both dict wrapper and direct model
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            feature_names = data.get('feature_names', None)
        else:
            model = data
            feature_names = None

        temps = X_raw.columns.astype(float).values
        # Preprocess and extract all features
        features_list = []
        for idx, row in X_raw.iterrows():
            y_proc = preprocess_curve_statistics(temps, row.values)
            feat_dict = extract_statistics_features(temps, y_proc)
            features_list.append(feat_dict)

        X_eng = pd.DataFrame(features_list)

        # Filter to only the features used during training
        if feature_names is not None:
            X_eng = X_eng[feature_names]
        else:
            # Fallback to the known 18 features if metadata not available
            top_18 = [
                'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
                'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
                'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
                'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
                'slope_std', 'y_left_max'
            ]
            X_eng = X_eng[[col for col in top_18 if col in X_eng.columns]]

        # Make predictions using only the training features
        proba = model.predict_proba(X_eng)
        print(f"[OK] ({proba.shape})")
        return proba
    except Exception as e:
        print(f"[FAIL] error: {e}")
        return None
