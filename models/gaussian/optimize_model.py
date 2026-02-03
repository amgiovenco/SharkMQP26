"""
Comprehensive optimization script for Gaussian curve shark classification
Optimizes preprocessing and model hyperparameters with 5-fold CV
"""
import os
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
import optuna
from optuna.storages import RDBStorage
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from pathlib import Path

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_TRIALS = 300  # Reduced for speed with parametrized preprocessing

# Output directory for results
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Optuna persistent storage
STORAGE_PATH = RESULTS_DIR / "optuna_studies"
STORAGE_PATH.mkdir(exist_ok=True)
STORAGE_URL = f"sqlite:///{STORAGE_PATH}/optuna_studies.db"

# ============== Enhanced Feature Extraction ==============

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)

def gaussian_sum(x, *p):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(p), 3):
        amp, mu, sigma = p[i:i+3]
        y += gaussian(x, float(amp), float(mu), abs(float(sigma)))
    return y

def preprocess_curve(x, y, params):
    """Smooth + quadratic baseline removal + normalization (parametrized)."""
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
    return x[::step], y[::step]

def seed_peaks(x, y, k, params):
    """Robust peak seeds with parametrized settings."""
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
        w_c = np.array([(x[-1] - x[0]) / (3 * k)] * k)

    dx = x[1] - x[0]
    min_w = max(params['min_w_temp'], params['min_w_factor'] * dx)
    max_w = (x[-1] - x[0]) / params['max_w_divisor']
    if np.ndim(w_c) == 0: w_c = np.array([w_c])
    w_c = np.clip(w_c, min_w, max_w)

    order = np.argsort(prominences)[::-1] if len(peaks) else np.arange(len(peaks))
    peaks = peaks[order][:k]
    w_c = w_c[order][:k]

    sort_lr = np.argsort(peaks)
    return peaks[sort_lr], w_c[sort_lr]

def fit_k(x, y, k, params):
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
    return np.log(n)*n_params + n*np.log(rss/n + 1e-12)

def fit_best_K(x, y, params):
    """Fit Gaussian mixture with best K using BIC."""
    best = None
    for k in range(1, params['k_max'] + 1):
        try:
            popt = fit_k(x, y, k, params)
            yhat = gaussian_sum(x, *popt)
            rss = float(np.sum((y - yhat)**2))
            bic = float(BIC(3*k, rss, len(x)))
            if best is None or bic < best["bic"]:
                best = {"k": k, "popt": popt, "bic": bic, "rss": rss}
        except Exception:
            pass
    return best

def extract_enhanced_features(row_vals, X_axis, params):
    """Enhanced feature extraction with parametrized preprocessing."""
    y0 = preprocess_curve(X_axis, np.asarray(row_vals, float), params)
    x, y = decimate(X_axis, y0, step=params['decimate_step'])
    best = fit_best_K(x, y, params)

    if best is None:
        return {
            "peak1_mu": 0, "peak1_amp": 0, "peak1_sigma": 0,
            "peak2_mu": 0, "peak2_amp": 0, "peak2_sigma": 0,
            "delta_mu_12": 0, "amp_ratio_12": 0, "best_K": 0,
            "total_area": 0.0, "asym_0p5C": 0.0, "total_amp": 0.0
        }

    popt = best["popt"]

    # Peak features (top 2 peaks)
    peaks = [{"amp": float(popt[i]), "mu": float(popt[i+1]), "sigma": abs(float(popt[i+2]))}
             for i in range(0, len(popt), 3)]
    peaks.sort(key=lambda d: d["amp"], reverse=True)

    feats = {}
    for i in range(2):
        if i < len(peaks):
            feats[f"peak{i+1}_mu"] = peaks[i]["mu"]
            feats[f"peak{i+1}_amp"] = peaks[i]["amp"]
            feats[f"peak{i+1}_sigma"] = peaks[i]["sigma"]
        else:
            feats[f"peak{i+1}_mu"] = 0.0
            feats[f"peak{i+1}_amp"] = 0.0
            feats[f"peak{i+1}_sigma"] = 0.0

    # Peak ratios
    feats["delta_mu_12"] = feats["peak1_mu"] - feats["peak2_mu"]
    feats["amp_ratio_12"] = (feats["peak1_amp"] + 1e-9) / (feats["peak2_amp"] + 1e-9) if feats["peak2_amp"] > 0 else 1e9

    # Total features
    areas = [p["amp"] * p["sigma"] * np.sqrt(2*np.pi) for p in peaks]
    feats["total_area"] = float(np.sum(areas)) if areas else 0.0
    feats["total_amp"] = sum(p["amp"] for p in peaks)
    feats["best_K"] = best["k"]

    # Asymmetry around main peak
    main_mu = float(popt[1]) if len(popt) >= 2 else x[np.argmax(y)]
    left = y[(x >= main_mu - params['asym_width']) & (x < main_mu)]
    right = y[(x > main_mu) & (x <= main_mu + params['asym_width'])]
    feats["asym_0p5C"] = float((right.mean() - left.mean()) if (len(left) > 3 and len(right) > 3) else 0.0)

    return feats

def get_feat_df(df_subset, params):
    """Extract features for a dataframe subset with parametrized preprocessing (parallelized)."""
    def process_row(i):
        feats = extract_enhanced_features(df_subset.loc[i, temp_cols].values, X_axis, params)
        feats[SPECIES_COL] = df_subset.loc[i, SPECIES_COL]
        return feats

    rows = Parallel(n_jobs=-1, backend='loky')(delayed(process_row)(i) for i in df_subset.index)
    return pd.DataFrame(rows).fillna(0.0)

# ============== Load Data ==============
print("Loading data...")
df = pd.read_csv(CSV_PATH)
temp_cols = sorted([c for c in df.columns if c != SPECIES_COL], key=lambda c: float(c))
X_axis = np.array([float(c) for c in temp_cols], dtype=float)

print("Data shape:", df.shape)
print(f"Classes: {df[SPECIES_COL].nunique()} species")

# ============== Baseline with default preprocessing ==============
print("\n" + "="*60)
print("BASELINE: Random Forest with default preprocessing")
print("="*60)

# Default parameters from tomakeroomforthetoona.py
default_params = {
    'decimate_step': 6,
    'savgol_win_temp': 1.5,
    'polyorder': 3,
    'baseline_quantile': 0.3,
    'scale_quantile': 0.99,
    'prom_factor': 0.15,
    'distance_divisor': 150,
    'rel_height': 0.5,
    'min_w_temp': 0.2,
    'min_w_factor': 4,
    'max_w_divisor': 3.0,
    'mu_bound_mult': 3.0,
    'k_max': 5,
    'asym_width': 1.0,
    'amp_hi_mult': 5
}

print("Extracting features with default parameters...")
base_feat_df = get_feat_df(df, default_params)
base_X = base_feat_df.drop(columns=[SPECIES_COL]).to_numpy(float)
base_y = df[SPECIES_COL].values

print(f"Feature shape: {base_X.shape}")
print(f"Features: {list(base_feat_df.columns[:10])}...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
base_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    max_depth=None,
    min_samples_leaf=1
)
base_scores = cross_val_score(base_rf, base_X, base_y, cv=cv, scoring='f1_macro', n_jobs=1)
print(f"Baseline CV macro F1: {base_scores.mean():.4f} ± {base_scores.std():.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in base_scores]}")

best_overall_score = base_scores.mean()
best_overall_model = "baseline_rf"
best_overall_params = default_params.copy()

# ============== Hyperparameter + Preprocessing Optimization ==============
print("\n" + "="*60)
print("OPTIMIZING: Random Forest (features + hyperparameters)")
print("="*60)

def objective_rf(trial):
    """Optimize Random Forest with parametrized feature extraction."""
    # Preprocessing parameters
    preprocess_params = {
        'decimate_step': trial.suggest_int('decimate_step', 3, 10),
        'savgol_win_temp': trial.suggest_float('savgol_win_temp', 0.5, 3.0),
        'polyorder': trial.suggest_int('polyorder', 2, 5),
        'baseline_quantile': trial.suggest_float('baseline_quantile', 0.1, 0.5),
        'scale_quantile': trial.suggest_float('scale_quantile', 0.95, 0.999),
        'prom_factor': trial.suggest_float('prom_factor', 0.05, 0.3),
        'distance_divisor': trial.suggest_int('distance_divisor', 100, 200),
        'rel_height': trial.suggest_float('rel_height', 0.3, 0.7),
        'min_w_temp': trial.suggest_float('min_w_temp', 0.1, 0.5),
        'min_w_factor': trial.suggest_int('min_w_factor', 2, 6),
        'max_w_divisor': trial.suggest_float('max_w_divisor', 2.0, 5.0),
        'mu_bound_mult': trial.suggest_float('mu_bound_mult', 2.0, 4.0),
        'k_max': trial.suggest_int('k_max', 3, 6),
        'asym_width': trial.suggest_float('asym_width', 0.3, 1.0),
        'amp_hi_mult': trial.suggest_float('amp_hi_mult', 2, 10)
    }

    # Extract features with trial parameters
    feat_df = get_feat_df(df, preprocess_params)
    X_trial = feat_df.drop(columns=[SPECIES_COL]).to_numpy(float)
    y_trial = df[SPECIES_COL].values

    # Model parameters
    model_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 300),
        'max_depth': trial.suggest_categorical('rf_max_depth', [None, 15, 20, 30]),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('rf_class_weight', ['balanced', 'balanced_subsample']),
        'random_state': RANDOM_STATE
    }

    clf = RandomForestClassifier(**model_params)
    scores = cross_val_score(clf, X_trial, y_trial, cv=cv, scoring='f1_macro', n_jobs=1)
    return scores.mean()

storage = RDBStorage(STORAGE_URL)
study_rf = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    storage=storage,
    study_name="gaussian_rf",
    load_if_exists=True
)
print(f"Study: gaussian_rf | Completed trials: {len(study_rf.trials)}")
study_rf.optimize(objective_rf, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest RF CV macro F1: {study_rf.best_value:.4f}")
print(f"Best RF params: {study_rf.best_params}")

if study_rf.best_value > best_overall_score:
    best_overall_score = study_rf.best_value
    best_overall_model = "optimized_rf"
    best_overall_params = study_rf.best_params.copy()

# ============== ExtraTrees ==============
print("\n" + "="*60)
print("OPTIMIZING: ExtraTrees (features + hyperparameters)")
print("="*60)

def objective_et(trial):
    """Optimize ExtraTrees with parametrized feature extraction."""
    # Preprocessing parameters
    preprocess_params = {
        'decimate_step': trial.suggest_int('decimate_step', 3, 10),
        'savgol_win_temp': trial.suggest_float('savgol_win_temp', 0.5, 3.0),
        'polyorder': trial.suggest_int('polyorder', 2, 5),
        'baseline_quantile': trial.suggest_float('baseline_quantile', 0.1, 0.5),
        'scale_quantile': trial.suggest_float('scale_quantile', 0.95, 0.999),
        'prom_factor': trial.suggest_float('prom_factor', 0.05, 0.3),
        'distance_divisor': trial.suggest_int('distance_divisor', 100, 200),
        'rel_height': trial.suggest_float('rel_height', 0.3, 0.7),
        'min_w_temp': trial.suggest_float('min_w_temp', 0.1, 0.5),
        'min_w_factor': trial.suggest_int('min_w_factor', 2, 6),
        'max_w_divisor': trial.suggest_float('max_w_divisor', 2.0, 5.0),
        'mu_bound_mult': trial.suggest_float('mu_bound_mult', 2.0, 4.0),
        'k_max': trial.suggest_int('k_max', 3, 6),
        'asym_width': trial.suggest_float('asym_width', 0.3, 1.0),
        'amp_hi_mult': trial.suggest_float('amp_hi_mult', 2, 10)
    }

    # Extract features with trial parameters
    feat_df = get_feat_df(df, preprocess_params)
    X_trial = feat_df.drop(columns=[SPECIES_COL]).to_numpy(float)
    y_trial = df[SPECIES_COL].values

    # Model parameters
    model_params = {
        'n_estimators': trial.suggest_int('et_n_estimators', 100, 300),
        'max_depth': trial.suggest_categorical('et_max_depth', [None, 15, 20, 30]),
        'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('et_max_features', ['sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('et_class_weight', ['balanced', 'balanced_subsample']),
        'random_state': RANDOM_STATE
    }

    clf = ExtraTreesClassifier(**model_params)
    scores = cross_val_score(clf, X_trial, y_trial, cv=cv, scoring='f1_macro', n_jobs=1)
    return scores.mean()

print("\n" + "="*60)
print("OPTIMIZING: ExtraTrees")
print("="*60)

study_et = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    storage=storage,
    study_name="gaussian_et",
    load_if_exists=True
)
print(f"Study: gaussian_et | Completed trials: {len(study_et.trials)}")
study_et.optimize(objective_et, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest ET CV macro F1: {study_et.best_value:.4f}")
print(f"Best ET params: {study_et.best_params}")

if study_et.best_value > best_overall_score:
    best_overall_score = study_et.best_value
    best_overall_model = "optimized_et"
    best_overall_params = study_et.best_params.copy()

# ============== Results Summary ==============
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"\nBest model: {best_overall_model}")
print(f"Best CV macro F1: {best_overall_score:.4f}")
print(f"Best params: {best_overall_params}")
print(f"\nImprovement over baseline: {(best_overall_score - base_scores.mean())*100:.2f}%")

# Export results to JSON
results_dict = {
    "baseline_cv_macro_f1": float(base_scores.mean()),
    "baseline_fold_scores": [float(s) for s in base_scores],
    "best_model": best_overall_model,
    "best_cv_macro_f1": float(best_overall_score),
    "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_overall_params.items()},
    "improvement_percentage": float((best_overall_score - base_scores.mean()) * 100)
}

with open("./optimization_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nSaved optimization results to ./optimization_results.json")

# Train final model on all data
print("\n" + "="*60)
print("Training final model on all data...")
print("="*60)

# Separate preprocessing params from model params
def extract_preprocess_params(params):
    """Extract preprocessing-related params from trial params."""
    preprocess_keys = {
        'decimate_step', 'savgol_win_temp', 'polyorder', 'baseline_quantile',
        'scale_quantile', 'prom_factor', 'distance_divisor', 'rel_height',
        'min_w_temp', 'min_w_factor', 'max_w_divisor', 'mu_bound_mult',
        'k_max', 'asym_width', 'amp_hi_mult'
    }
    return {k: v for k, v in params.items() if k in preprocess_keys}

def extract_model_params(params, model_type):
    """Extract model-specific params from trial params."""
    prefix = f"{model_type}_"
    model_params = {}
    for k, v in params.items():
        if k.startswith(prefix):
            clean_key = k[len(prefix):]
            model_params[clean_key] = v
    return model_params

if best_overall_model == "baseline_rf":
    # Use baseline params (no preprocessing optimization)
    preprocess_params = default_params
    final_X = base_X
    final_y = base_y
    model_params = {'n_estimators': 200, 'random_state': RANDOM_STATE, 'class_weight': 'balanced_subsample', 'max_depth': None, 'min_samples_leaf': 1}
    final_model = RandomForestClassifier(**model_params)
elif best_overall_model == "optimized_rf":
    preprocess_params = extract_preprocess_params(best_overall_params)
    feat_df_final = get_feat_df(df, preprocess_params)
    final_X = feat_df_final.drop(columns=[SPECIES_COL]).to_numpy(float)
    final_y = df[SPECIES_COL].values
    model_params = extract_model_params(best_overall_params, 'rf')
    model_params['random_state'] = RANDOM_STATE
    final_model = RandomForestClassifier(**model_params)
elif best_overall_model == "optimized_et":
    preprocess_params = extract_preprocess_params(best_overall_params)
    feat_df_final = get_feat_df(df, preprocess_params)
    final_X = feat_df_final.drop(columns=[SPECIES_COL]).to_numpy(float)
    final_y = df[SPECIES_COL].values
    model_params = extract_model_params(best_overall_params, 'et')
    model_params['random_state'] = RANDOM_STATE
    final_model = ExtraTreesClassifier(**model_params)

# Train the final model on all data
final_model.fit(final_X, final_y)

bundle = {
    "model": final_model,
    "feature_names": list(feat_df_final.drop(columns=[SPECIES_COL]).columns) if best_overall_model != "baseline_rf" else list(base_feat_df.drop(columns=[SPECIES_COL]).columns),
    "model_type": best_overall_model,
    "cv_macro_f1": best_overall_score,
    "best_params": best_overall_params
}

joblib.dump(bundle, RESULTS_DIR / "optimized_model.pkl")
print(f"Saved optimized model to {RESULTS_DIR / 'optimized_model.pkl'}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final CV macro F1: {best_overall_score:.4f}")
print(f"Improvement: {(best_overall_score - base_scores.mean())*100:.2f}%")
print("\nDone!")
