"""statistics model prediction."""
import pickle
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft

def preprocess_curve_statistics(x, y):
    """preprocess curve exactly as in statistics notebook: smooth + baseline remove + normalize."""
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    # savgol smooth
    win = max(7, int(round(1.5 / dx)) | 1)
    if win >= len(y):
        win = max(7, (len(y)//2)*2 - 1)
    y_smooth = savgol_filter(y, window_length=win, polyorder=3, mode="interp")
    # baseline removal: fit quadratic through low points
    q = np.quantile(y_smooth, 0.3)
    mask = y_smooth <= q
    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_smooth[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_baseline = y_smooth - baseline
    else:
        y_baseline = y_smooth - np.min(y_smooth)
    # normalize to [0, 1] using 99th percentile
    scale = np.quantile(y_baseline, 0.99)
    y_norm = y_baseline / scale if scale > 0 else y_baseline
    y_norm = np.maximum(y_norm, 0.0)
    return y_norm

def extract_statistics_features(x, y):
    """extract all 36 features exactly as in statistics notebook."""
    feat = {}
    # basic stats (7)
    feat["mean"] = float(np.mean(y))
    feat["std"] = float(np.std(y))
    feat["min"] = float(np.min(y))
    feat["max"] = float(np.max(y))
    feat["range"] = float(np.ptp(y))
    feat["skewness"] = float(pd.Series(y).skew())
    feat["kurtosis"] = float(pd.Series(y).kurtosis())
    # derivatives (5)
    dy = np.gradient(y, x)
    feat["max_slope"] = float(np.max(np.abs(dy)))
    feat["mean_abs_slope"] = float(np.mean(np.abs(dy)))
    feat["slope_std"] = float(np.std(dy))
    d2y = np.gradient(dy, x)
    feat["max_curvature"] = float(np.max(np.abs(d2y)))
    feat["mean_abs_curvature"] = float(np.mean(np.abs(d2y)))
    # peaks (4)
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
    # regional stats (9)
    n = len(y)
    for region, start, end in [("left", 0, n//3), ("middle", n//3, 2*n//3), ("right", 2*n//3, n)]:
        feat[f"y_{region}_mean"] = float(np.mean(y[start:end]))
        feat[f"y_{region}_std"] = float(np.std(y[start:end]))
        feat[f"y_{region}_max"] = float(np.max(y[start:end]))
    # quartiles (4)
    q = np.percentile(y, [25, 50, 75])
    feat["q25"] = float(q[0])
    feat["q50"] = float(q[1])
    feat["q75"] = float(q[2])
    feat["iqr"] = float(q[2] - q[0])
    # frequency domain: FFT (11)
    fft_vals = np.abs(fft(y - np.mean(y)))
    fft_power = fft_vals ** 2
    for i, idx in enumerate(np.argsort(fft_power)[-5:][::-1]):
        feat[f"fft_power_{i}"] = float(fft_power[idx])
    feat["fft_total_power"] = float(np.sum(fft_power))
    feat["fft_entropy"] = float(entropy(fft_power + 1e-10))
    return feat

def get_stats_predictions(X_raw: pd.DataFrame, models_dir: str = "./results") -> np.ndarray:
    """get statistics model predictions."""
    try:
        # Load metadata to get the exact features used during training
        meta_path = Path(models_dir) / 'model_metadata.pkl'
        if not meta_path.exists():
            print(" statistics...[FAIL] model_metadata.pkl not found")
            return None
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        feature_names = meta['features']  # This is the list of top features used (e.g., top 17 or 18)

        # Load the model
        model_path = Path(models_dir) / 'trained_model.pkl'
        if not model_path.exists():
            print(" statistics...[FAIL] trained_model.pkl not found")
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        temps = X_raw.columns.astype(float).values
        # Preprocess and extract all 36 features
        features_list = []
        for idx, row in X_raw.iterrows():
            y_proc = preprocess_curve_statistics(temps, row.values)
            feat_dict = extract_statistics_features(temps, y_proc)
            features_list.append(feat_dict)
        X_eng_full = pd.DataFrame(features_list)
        
        # Select only the exact features used during training, in the same order
        if not all(f in X_eng_full.columns for f in feature_names):
            print(" statistics...[FAIL] Some required features missing in extracted data")
            return None
        X_eng = X_eng_full[feature_names]
        
        # Make predictions
        proba = model.predict_proba(X_eng)
        print(f" statistics...[OK] ({proba.shape}) using {len(feature_names)} features")
        return proba
    except Exception as e:
        print(f" statistics...[FAIL] error: {e}")
        return None
