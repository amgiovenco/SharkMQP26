"""Gaussian curve model prediction."""

import pickle
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit


# =============================================================================
# GAUSSIAN FITTING FUNCTIONS (mirrored from GaussianCurve.ipynb)
# =============================================================================

def gaussian(x, amp, mu, sigma):
    """Single Gaussian function."""
    return amp * np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)


def gaussian_sum(x, *p):
    """Sum of multiple Gaussians."""
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(p), 3):
        amp, mu, sigma = p[i:i+3]
        y += gaussian(x, float(amp), float(mu), abs(float(sigma)))
    return y


def preprocess_curve(x, y):
    """Smooth + quadratic baseline removal + normalization."""
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(1.5 / dx)) | 1)      # ~1.5°C window, odd
    if win >= len(y):                            # safety
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
    """Downsample curve."""
    return x[::step], y[::step]


def seed_peaks(x, y, k):
    """Robust peak seeds: use prominence from spread, clamp widths."""
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
        w_c = w_idx * (x[1] - x[0])  # °C
    else:
        w_c = np.array([(x[-1]-x[0])/(3*k) ]*k)

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
    """Fit k Gaussians to curve."""
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
    """Bayesian Information Criterion."""
    return np.log(n)*n_params + n*np.log(rss/n + 1e-12)


def fit_best_K(x, y, K=(1, 5)):
    """Fit 1..K Gaussians and select best by BIC."""
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
            pass
    return best


def peaks_to_features(popt, k_keep=2):
    """Extract peak features from fit parameters."""
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
    """Extract additional features from Gaussian fit."""
    # area per peak ~ amp * sigma * sqrt(2π)
    areas = []
    for i in range(0, len(popt), 3):
        amp, mu, sigma = float(popt[i]), float(popt[i+1]), abs(float(popt[i+2]))
        areas.append(amp * sigma * np.sqrt(2*np.pi))
    total_area = float(np.sum(areas)) if areas else 0.0

    # asymmetry around main peak from the preprocessed y
    main_mu = float(popt[1]) if len(popt)>=2 else x[np.argmax(y)]
    left  = y[(x >= main_mu-0.5) & (x <  main_mu)]
    right = y[(x >  main_mu)     & (x <= main_mu+0.5)]
    asym = (right.mean() - left.mean()) if (len(left)>3 and len(right)>3) else 0.0
    return {"total_area": total_area, "asym_0p5C": asym}


def extract_features_for_row(row_vals, X_axis, K_RANGE=(1, 6), DECIMATE_STEP=6):
    """Extract Gaussian features for a single curve."""
    y0 = preprocess_curve(X_axis, np.asarray(row_vals, float))
    x, y = decimate(X_axis, y0, step=DECIMATE_STEP)
    best = fit_best_K(x, y, K=K_RANGE)
    if best is None:
        return {
            "peak1_mu":0,"peak1_amp":0,"peak1_sigma":0,
            "peak2_mu":0,"peak2_amp":0,"peak2_sigma":0,
            "delta_mu_12":0,"amp_ratio_12":0,"best_K":0,
            "total_area":0.0,"asym_0p5C":0.0,"total_amp":0.0
        }
    feats = peaks_to_features(best["popt"], k_keep=2)
    extras = extra_features_from_fit(x, y, best["popt"])
    feats.update(extras)
    feats["best_K"] = best["k"]
    feats["total_amp"] = feats["peak1_amp"] + feats["peak2_amp"]
    return feats


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def get_gaussian_predictions(X_raw: pd.DataFrame, models_dir: str = "./models") -> np.ndarray:
    """Get Gaussian curve model predictions from a single model file."""
    try:
        # Find the gaussian model file (expects exactly one gaussian*.pkl file)
        files = glob.glob(f"{models_dir}/gaussian*.pkl")
        # Filter out summary files
        files = [f for f in files if "summary" not in f.lower()]

        if not files:
            print("  gaussian...[FAIL] not found")
            return None

        if len(files) > 1:
            # If multiple files, use the most recent one
            model_path = max(files, key=lambda p: Path(p).stat().st_mtime)
        else:
            model_path = files[0]

        # Extract temperature columns (same as in GaussianCurve.ipynb)
        temp_cols = sorted([c for c in X_raw.columns], key=lambda c: float(c))
        X_axis = np.array([float(c) for c in temp_cols], dtype=float)

        # Extract features for all samples
        features_list = []
        for idx, row in X_raw.iterrows():
            feat_dict = extract_features_for_row(row.values, X_axis)
            features_list.append(feat_dict)

        X_eng = pd.DataFrame(features_list)

        # Load the single model
        # Try joblib first (used in training notebook), fall back to pickle
        try:
            import joblib
            bundle = joblib.load(model_path)
            print(f"    [debug] loaded with joblib")
        except:
            with open(model_path, 'rb') as f:
                bundle = pickle.load(f)
            print(f"    [debug] loaded with pickle")

        print(f"    [debug] bundle type: {type(bundle)}")

        # Handle case where bundle itself is the model (not a dict wrapper)
        if isinstance(bundle, dict):
            model = bundle["model"]
            feature_names = bundle.get("feature_names", None)
        else:
            # bundle is the model directly
            model = bundle
            feature_names = None
            print(f"    [debug] bundle is model directly, not a dict")

        # Debug: print what we got from the bundle
        if feature_names is not None:
            print(f"    [debug] feature_names from bundle: type={type(feature_names)}, len={len(feature_names)}")
            print(f"    [debug] first 3 names: {[str(f) for f in list(feature_names)[:3]]}")

        # If no feature names in bundle, use all columns from X_eng
        if feature_names is None:
            feature_names = X_eng.columns.tolist()
            print(f"    [debug] no feature_names in bundle, using all X_eng columns: {feature_names}")
        else:
            # Convert to list of native Python strings (handles numpy string arrays)
            feature_names = [str(f) for f in feature_names]

            # Filter to only columns that exist in X_eng
            X_eng_cols = X_eng.columns.tolist()
            existing_cols = [f for f in feature_names if f in X_eng_cols]
            if len(existing_cols) < len(feature_names):
                missing = [f for f in feature_names if f not in X_eng_cols]
                print(f"    [warning] missing columns: {missing}, using available columns")
            feature_names = existing_cols if existing_cols else X_eng_cols

        # Use only the feature columns that the model was trained on
        X_subset = X_eng[feature_names]
        proba = model.predict_proba(X_subset)

        print(f"  gaussian...[OK] ({proba.shape})")
        return proba

    except Exception as e:
        import traceback
        print(f"  gaussian...[FAIL] error: {e}")
        traceback.print_exc()
        print(f"    debug: feature_names in locals: {'feature_names' in locals()}")
        if 'feature_names' in locals():
            print(f"    debug: feature_names type: {type(feature_names)}")
        print(f"    debug: X_eng shape: {X_eng.shape if 'X_eng' in locals() else 'not set'}")
        print(f"    debug: X_eng columns: {list(X_eng.columns)[:5] if 'X_eng' in locals() else 'not set'}...")
        return None
