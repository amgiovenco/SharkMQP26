# ensemble2.py
#
# Usage:
#   python ensemble2.py shark_training_data.csv shark_validation_data.csv shark_test_data.csv
#
# This version:
#   - Loads a pre-trained Gaussian model (joblib/pickle; supports bare estimator or dict bundle)
#   - Loads the pre-trained RF artifact (the 5-fold CV RF model artifact you saved)
#   - Computes features to match the RF artifact's expected feature names
#   - Tunes blend weight on VAL; evaluates on TEST

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, classification_report

# --- helpers for Gaussian features (unchanged API from your code) ---
from GaussianCurveCV import (
    build_features_with_axis,  # builds Gaussian features from raw numeric axis
    RANDOM_STATE
)

# =========================
# Paths: artifacts live next to this file
# =========================
BASE_DIR = Path(__file__).resolve().parent
GAUSS_MODEL_PATH = BASE_DIR / "gaussian_model.joblib"                   # or .pkl
RF_ARTIFACT_PATH = BASE_DIR / "ensemble_extratrees_calibrated.joblib"   # your RF 5-fold artifact

# =========================
# Generic loaders and unwrappers
# =========================
def joblib_or_pickle_load(path: Path):
    """Load .joblib or .pkl with a single function."""
    ext = path.suffix.lower()
    if ext in (".joblib", ".jl"):
        from joblib import load
        return load(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def unwrap_artifact(obj, *, label=""):
    """
    Accepts either a bare sklearn estimator or a dict bundle.
    Returns (estimator, classes or None, feature_names or None).
    For RF artifact, feature_names must be present.
    """
    est = obj
    classes = None
    feature_names = None

    if isinstance(obj, dict):
        # common keys used in bundles
        est = obj.get("estimator") or obj.get("model") or obj.get("clf") or obj.get("pipeline")
        classes = obj.get("classes")
        feature_names = obj.get("feature_names") or obj.get("features") or obj.get("feature_order")
        if est is None:
            raise ValueError(f"{label} artifact is a dict but no estimator found under keys "
                             f"['estimator','model','clf','pipeline']")

    # get classes from estimator if not provided
    if classes is None and hasattr(est, "classes_"):
        try:
            classes = list(est.classes_)
        except Exception:
            classes = None

    return est, classes, feature_names

# =========================
# Utility helpers
# =========================
def align_probas(p_mat, src_classes, tgt_classes):
    """Map probabilities from src class order to target class order, with renormalization."""
    idx_map = {c: i for i, c in enumerate(src_classes)}
    out = np.zeros((p_mat.shape[0], len(tgt_classes)), dtype=float)
    for j, c in enumerate(tgt_classes):
        i = idx_map.get(c, None)
        if i is not None:
            out[:, j] = p_mat[:, i]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s

def soft_blend(p1, p2, w):
    return w * p1 + (1.0 - w) * p2

# =========================
# RF feature pipeline (matches your 36-feature scheme used in artifact)
# =========================
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft

def preprocess_curve(x, y):
    """smooth + baseline remove + normalize (same style you used)"""
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(1.5 / dx)) | 1)  # odd
    if win >= len(y):
        win = max(7, (len(y)//2)*2 - 1)
    y_smooth = savgol_filter(y, window_length=win, polyorder=3, mode="interp")

    q = np.quantile(y_smooth, 0.3)
    mask = y_smooth <= q
    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_smooth[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_base = y_smooth - baseline
    else:
        y_base = y_smooth - np.min(y_smooth)

    scale = np.quantile(y_base, 0.99)
    y_norm = y_base / scale if scale > 0 else y_base
    y_norm = np.maximum(y_norm, 0.0)
    return y_norm

def extract_features(x, y):
    """36-ish feature block that includes names your RF artifact expects."""
    feat = {}

    # basic stats
    feat["mean"] = float(np.mean(y))
    feat["std"] = float(np.std(y))
    feat["min"] = float(np.min(y))
    feat["max"] = float(np.max(y))
    feat["range"] = float(np.ptp(y))
    feat["skewness"] = float(pd.Series(y).skew())
    feat["kurtosis"] = float(pd.Series(y).kurtosis())

    # derivatives
    dy = np.gradient(y, x)
    feat["max_slope"] = float(np.max(np.abs(dy)))
    feat["mean_abs_slope"] = float(np.mean(np.abs(dy)))
    feat["slope_std"] = float(np.std(dy))
    d2y = np.gradient(dy, x)
    feat["max_curvature"] = float(np.max(np.abs(d2y)))
    feat["mean_abs_curvature"] = float(np.mean(np.abs(d2y)))

    # peaks
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

    # regional stats
    n = len(y)
    thirds = [(0, n//3), (n//3, 2*n//3), (2*n//3, n)]
    labels = ["left", "middle", "right"]
    for (s, e), nm in zip(thirds, labels):
        seg = y[s:e] if e > s else y
        feat[f"y_{nm}_mean"] = float(np.mean(seg))
        feat[f"y_{nm}_std"] = float(np.std(seg))
        feat[f"y_{nm}_max"] = float(np.max(seg))

    # quartiles
    q = np.percentile(y, [25, 50, 75])
    feat["q25"] = float(q[0]); feat["q50"] = float(q[1]); feat["q75"] = float(q[2])
    feat["iqr"] = float(q[2] - q[0])

    # frequency domain (top 5 powers, total power, entropy)
    fft_vals = np.abs(fft(y - np.mean(y)))
    fft_power = fft_vals ** 2
    top_idx = np.argsort(fft_power)[-5:][::-1]
    for i, idx in enumerate(top_idx):
        feat[f"fft_power_{i}"] = float(fft_power[idx])
    feat["fft_total_power"] = float(np.sum(fft_power))
    feat["fft_entropy"] = float(entropy(fft_power + 1e-10))

    return feat

def build_rf_features(df, temp_cols, expected_feature_names):
    """Compute features for each row, return DataFrame with exactly expected_feature_names (fill missing with 0)."""
    x = np.array([float(c) for c in temp_cols], dtype=float)

    feats = []
    for i in range(len(df)):
        y_raw = df.iloc[i][temp_cols].to_numpy(float)
        y_proc = preprocess_curve(x, y_raw)
        f = extract_features(x, y_proc)
        feats.append(f)

    F = pd.DataFrame(feats).fillna(0.0)
    # Ensure all expected columns exist, in correct order
    for col in expected_feature_names:
        if col not in F.columns:
            F[col] = 0.0
    # Drop any extras and reorder
    F = F[expected_feature_names]
    return F

# =========================
# Main
# =========================
def main():
    # Require exactly 3 args (TRAIN/VAL/TEST)
    if len(sys.argv) != 4:
        prog = Path(sys.argv[0]).name
        print(f"Usage: python {prog} TRAIN.csv VAL.csv TEST.csv")
        sys.exit(1)

    train_csv, val_csv, test_csv = sys.argv[1:]

    # raw data
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)
    df_test  = pd.read_csv(test_csv)

    # Freeze numeric axis from TRAIN
    temp_cols = sorted([c for c in df_train.columns if c != "Species"], key=lambda c: float(c))

    # --- confirm artifacts exist
    print(f"Looking for Gaussian model at: {GAUSS_MODEL_PATH}")
    print(f"Looking for RF model at: {RF_ARTIFACT_PATH}")
    if not GAUSS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Gaussian model not found at {GAUSS_MODEL_PATH}")
    if not RF_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"RF artifact not found at {RF_ARTIFACT_PATH}")

    # --------------------------
    # Load & unwrap pre-trained models
    # --------------------------
    gauss_raw = joblib_or_pickle_load(GAUSS_MODEL_PATH)
    rf_raw    = joblib_or_pickle_load(RF_ARTIFACT_PATH)

    gaussian_model, classes_g_saved, _ = unwrap_artifact(gauss_raw, label="gaussian")
    rf_model,      classes_r_saved, rf_feat_names = unwrap_artifact(rf_raw,   label="rf")

    if rf_feat_names is None:
        raise ValueError("RF artifact is missing 'feature_names'. The ensemble needs this to build inputs.")

    # --------------------------
    # Build features for VAL/TEST
    # --------------------------
    # Gaussian features
    Xg_va, yg_va = build_features_with_axis(df_val,   temp_cols)
    Xg_te, yg_te = build_features_with_axis(df_test,  temp_cols)

    # RF features (match artifact expectations)
    Xr_va_df = build_rf_features(df_val,   temp_cols, rf_feat_names)
    Xr_te_df = build_rf_features(df_test,  temp_cols, rf_feat_names)

    # --------------------------
    # VAL predictions
    # --------------------------
    if hasattr(gaussian_model, "predict_proba"):
        p_g_val = gaussian_model.predict_proba(Xg_va)
    else:
        # As a fallback, allow decision_function — but probs are expected normally.
        p_g_val = gaussian_model.decision_function(Xg_va)
    classes_g = classes_g_saved or list(getattr(gaussian_model, "classes_", []))

    p_r_val = rf_model.predict_proba(Xr_va_df.values)
    classes_r = classes_r_saved or list(getattr(rf_model, "classes_", []))

    # --------------------------
    # Tune blend weight on VAL
    # --------------------------
    classes_all = sorted(set(classes_g).union(set(classes_r)))
    y_val_str = df_val["Species"].astype(str).values

    p_g_val_a = align_probas(p_g_val, classes_g, classes_all)
    p_r_val_a = align_probas(p_r_val, classes_r, classes_all)

    ws = np.linspace(0.0, 1.0, 51)
    best = {"w": None, "logloss": np.inf, "acc": -1.0}
    for w in ws:
        p_blend = soft_blend(p_g_val_a, p_r_val_a, w)
        ll = log_loss(y_val_str, p_blend, labels=classes_all)
        pred = np.array(classes_all)[np.argmax(p_blend, axis=1)]
        acc = accuracy_score(y_val_str, pred)
        if (ll < best["logloss"]) or (np.isclose(ll, best["logloss"]) and acc > best["acc"]):
            best = {"w": float(w), "logloss": float(ll), "acc": float(acc)}

    print(f"Best VAL blend: w={best['w']:.2f} | logloss={best['logloss']:.4f} | acc={best['acc']:.4f}")

    # --------------------------
    # TEST predictions
    # --------------------------
    if hasattr(gaussian_model, "predict_proba"):
        p_g_test = gaussian_model.predict_proba(Xg_te)
    else:
        p_g_test = gaussian_model.decision_function(Xg_te)
    classes_g_tv = classes_g  # same model, same class order

    p_r_test = rf_model.predict_proba(Xr_te_df.values)

    p_g_test_a = align_probas(p_g_test, classes_g_tv, classes_all)
    p_r_test_a = align_probas(p_r_test, classes_r,     classes_all)

    p_test = soft_blend(p_g_test_a, p_r_test_a, best["w"])
    y_pred = np.array(classes_all)[np.argmax(p_test, axis=1)]
    y_test = df_test["Species"].astype(str).values

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (ensemble on pre-trained bases): {test_acc:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=classes_all)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=classes_all, columns=classes_all))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=classes_all, zero_division=0))

if __name__ == "__main__":
    main()
