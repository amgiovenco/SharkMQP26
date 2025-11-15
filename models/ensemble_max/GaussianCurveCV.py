#!/usr/bin/env python3
# GaussianCurveCV.py — CLI version (CV on TRAIN only; final fit on TRAIN+VAL; TEST held-out)
# Usage:
#   python GaussianCurveCV.py shark_training_data.csv shark_validation_data.csv shark_test_data.csv

import warnings
warnings.filterwarnings("ignore")

from joblib import dump

import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
SPECIES_COL   = "Species"
K_RANGE       = (1, 6)       # try 1..6 Gaussians, pick by BIC
DECIMATE_STEP = 6            # downsample factor for curve fitting speed
CALIBRATE     = True         # isotonic calibration around RF (for FINAL model only now)
RANDOM_STATE  = 8            # keep this fixed as requested
MAX_CV_SPLITS = 4            # cap CV folds for stability (not strictly used here but kept for completeness)

# -------------------------
# Gaussian model helpers
# -------------------------
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)

def gaussian_sum(x, *p):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(p), 3):
        amp, mu, sigma = p[i:i+3]
        y += gaussian(x, float(amp), float(mu), abs(float(sigma)))
    return y

def preprocess_curve(x, y):
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    # Savitzky-Golay smoothing; window length odd and <= len(y)
    win = max(7, int(round(1.5 / dx)) | 1)
    if win >= len(y):
        win = max(7, (len(y)//2)*2 - 1)
    y_s = savgol_filter(y, window_length=max(7, win), polyorder=3, mode="interp")

    # Baseline via low-quantile quadratic fit
    q = np.quantile(y_s, 0.3)
    mask = y_s <= q
    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_s[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_b = y_s - baseline
    else:
        y_b = y_s - np.min(y_s)

    # Normalize to ~[0,1]
    scale = np.quantile(y_b, 0.99)
    if scale > 0:
        y_b = y_b / scale
    return np.maximum(y_b, 0.0)

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
        w_c = np.array([(x[-1]-x[0])/(3*max(k,1))]*max(k,1))

    min_w = max(0.2, 4*(x[1]-x[0]))
    max_w = (x[-1]-x[0]) / 3.0
    if np.ndim(w_c) == 0:
        w_c = np.array([w_c])
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
        lo += [0.0,   mu0 - 3*sigma0, (x[1]-x[0])*1e-3]
        hi += [y_max*5 + 1e-6, mu0 + 3*sigma0, (x[-1]-x[0])]
    popt, _ = curve_fit(gaussian_sum, x, y, p0=p0, bounds=(lo, hi), maxfev=15000)
    return popt

def BIC(n_params, rss, n):
    return np.log(n)*n_params + n*np.log(rss/n + 1e-12)

def fit_best_K(x, y, K=(1,5)):
    best = None
    for k in range(K[0], K[1]+1):
        try:
            popt = fit_k(x, y, k)
            yhat = gaussian_sum(x, *popt)
            rss = float(np.sum((y - yhat)**2))
            bic = float(BIC(3*k, rss, len(x)))
            if best is None or bic < best["bic"]:
                best = {"k": k, "popt": popt, "bic": bic}
        except Exception:
            # silently skip failed fits
            pass
    return best

def peaks_to_features(popt, k_keep=2):
    peaks = [{"amp": float(popt[i]), "mu": float(popt[i+1]), "sigma": abs(float(popt[i+2]))}
             for i in range(0, len(popt), 3)]
    peaks.sort(key=lambda d: d["amp"], reverse=True)
    feats = {}
    for i in range(k_keep):
        if i < len(peaks):
            feats[f"peak{i+1}_mu"] = peaks[i]["mu"]
            feats[f"peak{i+1}_amp"] = peaks[i]["amp"]
            feats[f"peak{i+1}_sigma"] = peaks[i]["sigma"]
        else:
            feats[f"peak{i+1}_mu"] = feats[f"peak{i+1}_amp"] = feats[f"peak{i+1}_sigma"] = 0.0
    feats["delta_mu_12"] = feats["peak1_mu"] - feats["peak2_mu"]
    feats["amp_ratio_12"] = (feats["peak1_amp"]+1e-9)/(feats["peak2_amp"]+1e-9) if feats["peak2_amp"]>0 else 1e9
    return feats

def extra_features_from_fit(x, y, popt):
    areas = []
    for i in range(0, len(popt), 3):
        amp, mu, sigma = float(popt[i]), float(popt[i+1]), abs(float(popt[i+2]))
        areas.append(amp * sigma * np.sqrt(2*np.pi))
    total_area = float(np.sum(areas)) if areas else 0.0

    # asymmetry around the main peak
    main_mu = float(popt[1]) if len(popt) >= 2 else x[np.argmax(y)]
    left  = y[(x >= main_mu-0.5) & (x <  main_mu)]
    right = y[(x >  main_mu)     & (x <= main_mu+0.5)]
    asym = (right.mean() - left.mean()) if (len(left)>3 and len(right)>3) else 0.0
    return {"total_area": total_area, "asym_0p5C": asym}

def extract_features_for_row(row_vals, X_axis, dec_step=DECIMATE_STEP):
    y0 = preprocess_curve(X_axis, np.asarray(row_vals, float))
    x, y = decimate(X_axis, y0, step=dec_step)
    best = fit_best_K(x, y, K=K_RANGE)
    if best is None:
        return {
            "peak1_mu":0,"peak1_amp":0,"peak1_sigma":0,
            "peak2_mu":0,"peak2_amp":0,"peak2_sigma":0,
            "delta_mu_12":0,"amp_ratio_12":0,"best_K":0,
            "total_area":0.0,"asym_0p5C":0.0,"total_amp":0.0
        }
    feats = peaks_to_features(best["popt"], k_keep=2)
    feats.update(extra_features_from_fit(x, y, best["popt"]))
    feats["best_K"] = best["k"]
    feats["total_amp"] = feats["peak1_amp"] + feats["peak2_amp"]
    return feats

def build_features_with_axis(df, temp_cols):
    """Build features using a fixed temperature axis (columns in the same order)."""
    missing = [c for c in temp_cols if c not in df.columns]
    if missing:
        raise ValueError(f"These expected feature columns are missing from a CSV: {missing[:8]}{'...' if len(missing)>8 else ''}")

    X_axis = np.array([float(c) for c in temp_cols], dtype=float)
    rows = []
    for i in range(len(df)):
        feats = extract_features_for_row(df.loc[i, temp_cols].values, X_axis)
        feats[SPECIES_COL] = df.loc[i, SPECIES_COL]
        rows.append(feats)
    feat_df = pd.DataFrame(rows).fillna(0.0)

    X = feat_df.drop(columns=[SPECIES_COL]).to_numpy(float)
    y = df[SPECIES_COL].astype(str).to_numpy()
    return X, y

# -------------------------
# Main (TRAIN-only CV, final fit on TRAIN+VAL)
# -------------------------
def main():
    # CLI
    parser = argparse.ArgumentParser(description="TRAIN-only CV, final fit on TRAIN+VAL, evaluate on TEST.")
    parser.add_argument("train_csv", help="Training CSV file")
    parser.add_argument("val_csv",   help="Validation CSV file")
    parser.add_argument("test_csv",  help="Test CSV file")
    args = parser.parse_args()

    # Load CSVs
    df_train = pd.read_csv(args.train_csv)
    df_val   = pd.read_csv(args.val_csv)
    df_test  = pd.read_csv(args.test_csv)

    # Check target column
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if SPECIES_COL not in d.columns:
            raise ValueError(f"[{name}] Missing required target column '{SPECIES_COL}'.")

    # Fix a single, shared temperature axis from TRAIN (assumed consistent across splits)
    temp_cols = sorted([c for c in df_train.columns if c != SPECIES_COL], key=lambda c: float(c))

    # Build features with a fixed axis for all splits
    X_train, y_train = build_features_with_axis(df_train, temp_cols)
    X_val,   y_val   = build_features_with_axis(df_val,   temp_cols)
    X_test,  y_test  = build_features_with_axis(df_test,  temp_cols)

    # ------------------------
    # 1) TRAIN-only CV accuracy (NO calibration here to avoid nested-CV issues)
    # ------------------------
    min_class_train = pd.Series(y_train).value_counts().min()
    cv_splits = min(3, min_class_train)  # safety: if min_class_train ever dropped below 3

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    base_rf_for_cv = RandomForestClassifier(
        n_estimators=800,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=1
    )

    cv_scores = cross_val_score(base_rf_for_cv, X_train, y_train, cv=cv, scoring="accuracy")
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    print(f"TRAIN-only CV (k={cv_splits}) accuracy: mean={cv_mean:.3f}, std={cv_std:.3f}")

    # ------------------------
    # 2) Final model: fit on TRAIN+VAL (calibrated if CALIBRATE=True)
    # ------------------------
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    base_rf_final = RandomForestClassifier(
        n_estimators=800,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=1
    )

    if CALIBRATE:
        # inner CV for calibration only, now on full TRAIN+VAL (no outer CV here)
        calib_cv_splits = min(3, pd.Series(y_tv).value_counts().min())
        clf_final = CalibratedClassifierCV(
            base_rf_final,
            cv=calib_cv_splits,
            method="isotonic"
        )
    else:
        clf_final = base_rf_final

    clf_final.fit(X_tv, y_tv)

    # Save artifact for ensemble use
    gaussian_artifacts = {
        "model": clf_final,                       # CalibratedClassifierCV or RF
        "temp_cols": temp_cols,                   # temperature axis to rebuild features
        "classes": clf_final.classes_.tolist(),   # string labels in model's class order
        "random_state": RANDOM_STATE,
        "config": {
            "K_RANGE": K_RANGE,
            "DECIMATE_STEP": DECIMATE_STEP,
            "CALIBRATE": CALIBRATE,
            "MAX_CV_SPLITS": MAX_CV_SPLITS
        }
    }
    dump(gaussian_artifacts, "gaussian_model.joblib")
    print("Saved gaussian_model.joblib")

    # Training (TRAIN+VAL) accuracy
    train_pred = clf_final.predict(X_tv)
    train_acc  = accuracy_score(y_tv, train_pred)

    # ------------------------
    # 3) Evaluate on held-out TEST
    # ------------------------
    test_pred = clf_final.predict(X_test)
    test_acc  = accuracy_score(y_test, test_pred)

    print(f"Training (train+val) accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    # Confusion matrix on TEST
    labels = np.unique(np.concatenate([y_tv, y_test]))
    cm = confusion_matrix(y_test, test_pred, labels=labels)

    fig_w = max(8, len(labels) * 0.35)
    fig_h = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, include_values=False)
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
