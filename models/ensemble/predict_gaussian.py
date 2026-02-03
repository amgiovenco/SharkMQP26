"""Gaussian curve model prediction from local models directory."""

import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import ExtraTreesClassifier



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


def preprocess_curve(x, y, params=None):
    """Smooth + baseline removal + normalization."""
    if params is None:
        params = {
            'savgol_win_temp': 1.5, 'polyorder': 3,
            'baseline_quantile': 0.3, 'scale_quantile': 0.99
        }
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(params['savgol_win_temp'] / dx)) | 1)
    if win >= len(y):
        win = max(7, (len(y)//2)*2 - 1)
    y_s = savgol_filter(y, window_length=win, polyorder=params['polyorder'], mode="interp")

    q = np.quantile(y_s, params['baseline_quantile'])
    mask = y_s <= q
    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_s[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_b = y_s - baseline
    else:
        y_b = y_s - np.min(y_s)

    scale = np.quantile(y_b, params['scale_quantile'])
    if scale > 0:
        y_b = y_b / scale
    y_b = np.maximum(y_b, 0.0)
    return y_b


def decimate(x, y, step=8):
    """Downsample curve."""
    return x[::step], y[::step]


def seed_peaks(x, y, k, params=None):
    """Robust peak seeds."""
    if params is None:
        params = {
            'prom_factor': 0.15, 'distance_divisor': 150, 'rel_height': 0.5,
            'min_w_temp': 0.2, 'min_w_factor': 4, 'max_w_divisor': 3.0
        }
    spread = max(np.quantile(y, 0.90) - np.quantile(y, 0.10), 1e-6)
    prom = spread * params['prom_factor']
    distance = max(1, len(x) // params['distance_divisor'])
    peaks, props = find_peaks(y, prominence=prom, distance=distance)
    if len(peaks) == 0:
        peaks = np.argsort(y)[::-1][:k]
        prominences = y[peaks] - np.min(y)
    else:
        prominences = props["prominences"]

    if len(peaks) > 0:
        w_idx = peak_widths(y, peaks, rel_height=params['rel_height'])[0]
        w_c = w_idx * (x[1] - x[0])
    else:
        w_c = np.array([(x[-1]-x[0])/(3*k) ]*k)

    dx = x[1] - x[0]
    min_w = max(params['min_w_temp'], params['min_w_factor']*dx)
    max_w = (x[-1]-x[0]) / params['max_w_divisor']
    if np.ndim(w_c) == 0:
        w_c = np.array([w_c])
    w_c = np.clip(w_c, min_w, max_w)

    order = np.argsort(prominences)[::-1] if len(peaks) else np.arange(len(peaks))
    peaks = peaks[order][:k]
    w_c = w_c[order][:k]

    sort_lr = np.argsort(peaks)
    return peaks[sort_lr], w_c[sort_lr]


def fit_k(x, y, k, params=None):
    """Fit k Gaussians."""
    if params is None:
        params = {'mu_bound_mult': 3, 'amp_hi_mult': 5.0}
    peaks, w_c = seed_peaks(x, y, k, params)
    p0, lo, hi = [], [], []
    y_max = max(np.max(y), 1e-6)
    for j, pk in enumerate(peaks):
        mu0 = float(x[pk])
        amp0 = float(max(y[pk], 1e-6))
        sigma0 = float(max(w_c[j] / (2*np.sqrt(2*np.log(2))), (x[1]-x[0])*2))
        p0 += [amp0, mu0, sigma0]
        lo += [0.0, mu0 - params['mu_bound_mult']*sigma0, (x[1]-x[0])*1e-3]
        hi += [y_max*params['amp_hi_mult'] + 1e-6, mu0 + params['mu_bound_mult']*sigma0, (x[-1]-x[0])]
    popt, _ = curve_fit(gaussian_sum, x, y, p0=p0, bounds=(lo, hi), maxfev=5000)
    return popt


def BIC(n_params, rss, n):
    """Bayesian Information Criterion."""
    return np.log(n)*n_params + n*np.log(rss/n + 1e-12)


def fit_best_K(x, y, params=None, K=(1, 5)):
    """Fit 1..K Gaussians and select best by BIC."""
    if params is None:
        params = {'k_max': 5}
    k_max = params.get('k_max', 5)
    best = None
    for k in range(1, k_max + 1):
        try:
            popt = fit_k(x, y, k, params)
            yhat = gaussian_sum(x, *popt)
            rss = float(np.sum((y - yhat)**2))
            bic = float(BIC(3*k, rss, len(x)))
            if best is None or bic < best["bic"]:
                best = {"k": k, "popt": popt, "bic": bic}
        except Exception:
            pass
    return best


def peaks_to_features(popt, k_keep=2):
    """Extract peak features."""
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
    """Extract additional features."""
    areas = []
    for i in range(0, len(popt), 3):
        amp, mu, sigma = float(popt[i]), float(popt[i+1]), abs(float(popt[i+2]))
        areas.append(amp * sigma * np.sqrt(2*np.pi))
    total_area = float(np.sum(areas)) if areas else 0.0

    main_mu = float(popt[1]) if len(popt)>=2 else x[np.argmax(y)]
    left  = y[(x >= main_mu-0.5) & (x <  main_mu)]
    right = y[(x >  main_mu)     & (x <= main_mu+0.5)]
    asym = (right.mean() - left.mean()) if (len(left)>3 and len(right)>3) else 0.0
    return {"total_area": total_area, "asym_0p5C": asym}


def extract_features_for_row(row_vals, X_axis, K_RANGE=(1, 5), DECIMATE_STEP=5):
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


def get_gaussian_predictions(X_raw: pd.DataFrame, models_dir: str = "./models") -> np.ndarray:
    """Get Gaussian curve model predictions from local models directory."""
    try:
        model_path = Path(models_dir) / "GAUSSIAN_optimized_model.pkl"
        if not model_path.exists():
            print("  gaussian...[FAIL] GAUSSIAN_optimized_model.pkl not found in ./models/")
            return None

        # Try loading with joblib first (best for sklearn models)
        bundle = None
        try:
            bundle = joblib.load(model_path)
        except Exception as joblib_err:
            # Fallback to pickle with proper sklearn imports available
            try:
                with open(model_path, 'rb') as f:
                    bundle = pickle.load(f)
            except Exception as pickle_err:
                print(f"  gaussian...[FAIL] could not load model: {pickle_err}")
                return None

        if bundle is None:
            print("  gaussian...[FAIL] model bundle is empty")
            return None

        # Extract temperature columns
        temp_cols = sorted([c for c in X_raw.columns], key=lambda c: float(c))
        X_axis = np.array([float(c) for c in temp_cols], dtype=float)

        # Extract best parameters from bundle
        params = bundle.get('best_params', {})
        if not params:
            print("  gaussian...[WARN] no best_params in bundle, using defaults")
            params = {
                'decimate_step': 5, 'savgol_win_temp': 1.5, 'polyorder': 3,
                'baseline_quantile': 0.3, 'scale_quantile': 0.99,
                'prom_factor': 0.15, 'distance_divisor': 150, 'rel_height': 0.5,
                'min_w_temp': 0.2, 'min_w_factor': 4, 'max_w_divisor': 3.0,
                'mu_bound_mult': 3, 'k_max': 5, 'asym_width': 0.5, 'amp_hi_mult': 5.0
            }

        # Extract features for all samples using the tuned parameters
        features_list = []
        for idx, row in X_raw.iterrows():
            # Use parametrized feature extraction matching training
            y0 = preprocess_curve(X_axis, np.asarray(row.values, float), params)
            x, y = decimate(X_axis, y0, step=params.get('decimate_step', 5))
            best = fit_best_K(x, y, params)
            if best is None:
                feat_dict = {
                    "peak1_mu":0,"peak1_amp":0,"peak1_sigma":0,
                    "peak2_mu":0,"peak2_amp":0,"peak2_sigma":0,
                    "delta_mu_12":0,"amp_ratio_12":0,"best_K":0,
                    "total_area":0.0,"asym_0p5C":0.0,"total_amp":0.0
                }
            else:
                popt = best["popt"]
                peaks = [{"amp": float(popt[i]), "mu": float(popt[i+1]), "sigma": abs(float(popt[i+2]))}
                         for i in range(0, len(popt), 3)]
                peaks.sort(key=lambda d: d["amp"], reverse=True)

                feat_dict = {}
                for i in range(2):
                    if i < len(peaks):
                        feat_dict[f"peak{i+1}_mu"] = peaks[i]["mu"]
                        feat_dict[f"peak{i+1}_amp"] = peaks[i]["amp"]
                        feat_dict[f"peak{i+1}_sigma"] = peaks[i]["sigma"]
                    else:
                        feat_dict[f"peak{i+1}_mu"] = 0.0
                        feat_dict[f"peak{i+1}_amp"] = 0.0
                        feat_dict[f"peak{i+1}_sigma"] = 0.0

                feat_dict["delta_mu_12"] = feat_dict["peak1_mu"] - feat_dict["peak2_mu"]
                feat_dict["amp_ratio_12"] = (feat_dict["peak1_amp"]+1e-9)/(feat_dict["peak2_amp"]+1e-9) if feat_dict["peak2_amp"]>0 else 1e9

                areas = [p["amp"] * p["sigma"] * np.sqrt(2*np.pi) for p in peaks]
                feat_dict["total_area"] = float(np.sum(areas)) if areas else 0.0
                feat_dict["total_amp"] = sum(p["amp"] for p in peaks)
                feat_dict["best_K"] = best["k"]

                main_mu = float(popt[1]) if len(popt) >= 2 else x[np.argmax(y)]
                left = y[(x >= main_mu - params.get('asym_width', 0.5)) & (x < main_mu)]
                right = y[(x > main_mu) & (x <= main_mu + params.get('asym_width', 0.5))]
                feat_dict["asym_0p5C"] = float((right.mean() - left.mean()) if (len(left) > 3 and len(right) > 3) else 0.0)

            features_list.append(feat_dict)

        X_eng = pd.DataFrame(features_list)

        # Handle case where bundle itself is the model
        if isinstance(bundle, dict):
            model = bundle.get("model", bundle)
        else:
            model = bundle

        proba = model.predict_proba(X_eng)

        print(f"  gaussian...[OK] ({proba.shape})")
        return proba

    except Exception as e:
        print(f"  gaussian...[FAIL] error: {e}")
        import traceback
        traceback.print_exc()
        return None
