"""
Comprehensive optimization script for Gaussian curve shark classification
Tries: enhanced features, hyperparameter tuning, different models, ensembling
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
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
K_RANGE = (1, 6)
DECIMATE_STEP = 6
RANDOM_STATE = 8
N_TRIALS = 100  # Optuna trials per model type

# Optuna persistent storage
STORAGE_PATH = Path("./optuna_studies")
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

def preprocess_curve(x, y):
    """Smooth + quadratic baseline removal + normalization."""
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(1.5 / dx)) | 1)
    if win >= len(y):
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
        w_c = np.array([(x[-1]-x[0])/(3*k)]*k)

    min_w = max(0.2, 4*(x[1]-x[0]))
    max_w = (x[-1]-x[0]) / 3.0
    if np.ndim(w_c) == 0: w_c = np.array([w_c])
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
        lo += [0.0, mu0 - 3*sigma0, (x[1]-x[0])*1e-3]
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
                best = {"k": k, "popt": popt, "bic": bic, "rss": rss}
        except Exception:
            pass
    return best

def extract_enhanced_features(row_vals, X_axis):
    """Enhanced feature extraction with more curve statistics"""
    y0 = preprocess_curve(X_axis, np.asarray(row_vals, float))
    x, y = decimate(X_axis, y0, step=DECIMATE_STEP)
    best = fit_best_K(x, y, K=K_RANGE)

    if best is None:
        return {f"f{i}": 0.0 for i in range(30)}  # Return zeros if fitting fails

    popt = best["popt"]

    # Original peak features (top 3 peaks this time)
    peaks = [{"amp": float(popt[i]), "mu": float(popt[i+1]), "sigma": abs(float(popt[i+2]))}
             for i in range(0, len(popt), 3)]
    peaks.sort(key=lambda d: d["amp"], reverse=True)

    feats = {}
    for i in range(3):  # Top 3 peaks instead of 2
        if i < len(peaks):
            feats[f"peak{i+1}_mu"] = peaks[i]["mu"]
            feats[f"peak{i+1}_amp"] = peaks[i]["amp"]
            feats[f"peak{i+1}_sigma"] = peaks[i]["sigma"]
        else:
            feats[f"peak{i+1}_mu"] = 0.0
            feats[f"peak{i+1}_amp"] = 0.0
            feats[f"peak{i+1}_sigma"] = 0.0

    # Peak ratios and differences
    feats["delta_mu_12"] = feats["peak1_mu"] - feats["peak2_mu"]
    feats["delta_mu_23"] = feats["peak2_mu"] - feats["peak3_mu"]
    feats["amp_ratio_12"] = (feats["peak1_amp"]+1e-9)/(feats["peak2_amp"]+1e-9)
    feats["amp_ratio_23"] = (feats["peak2_amp"]+1e-9)/(feats["peak3_amp"]+1e-9)
    feats["sigma_ratio_12"] = (feats["peak1_sigma"]+1e-9)/(feats["peak2_sigma"]+1e-9)

    # Total features
    feats["total_amp"] = sum(p["amp"] for p in peaks)
    feats["total_area"] = sum(p["amp"] * p["sigma"] * np.sqrt(2*np.pi) for p in peaks)

    # Fit quality
    feats["best_K"] = best["k"]
    feats["fit_rss"] = best["rss"]
    feats["fit_bic"] = best["bic"]

    # Raw curve statistics on preprocessed signal
    feats["curve_mean"] = float(np.mean(y0))
    feats["curve_std"] = float(np.std(y0))
    feats["curve_max"] = float(np.max(y0))
    feats["curve_skew"] = float(skew(y0))
    feats["curve_kurtosis"] = float(kurtosis(y0))

    # Asymmetry around main peak
    main_mu = peaks[0]["mu"] if peaks else x[np.argmax(y)]
    left = y0[(X_axis >= main_mu-1.0) & (X_axis < main_mu)]
    right = y0[(X_axis > main_mu) & (X_axis <= main_mu+1.0)]
    feats["asym_1C"] = float((right.mean() - left.mean()) if (len(left)>3 and len(right)>3) else 0.0)

    # Temperature at max
    feats["temp_at_max"] = float(X_axis[np.argmax(y0)])

    # Width metrics
    try:
        half_max = np.max(y0) / 2
        above_half = y0 >= half_max
        if above_half.any():
            indices = np.where(above_half)[0]
            feats["fwhm"] = float(X_axis[indices[-1]] - X_axis[indices[0]])
        else:
            feats["fwhm"] = 0.0
    except:
        feats["fwhm"] = 0.0

    return feats

# ============== Load Data ==============
print("Loading data...")
df = pd.read_csv(CSV_PATH)
temp_cols = sorted([c for c in df.columns if c != SPECIES_COL], key=lambda c: float(c))
X_axis = np.array([float(c) for c in temp_cols], dtype=float)

print("Extracting enhanced features...")
rows = []
for i in range(len(df)):
    if i % 100 == 0:
        print(f"  {i}/{len(df)}")
    feats = extract_enhanced_features(df.loc[i, temp_cols].values, X_axis)
    feats[SPECIES_COL] = df.loc[i, SPECIES_COL]
    rows.append(feats)

feat_df = pd.DataFrame(rows).fillna(0.0)
X = feat_df.drop(columns=[SPECIES_COL]).to_numpy(float)
y = df[SPECIES_COL].astype(str).to_numpy()

print(f"\nFeature shape: {X.shape}")
print(f"Features: {list(feat_df.columns[:10])}...")

# ============== Baseline (current approach) ==============
print("\n" + "="*60)
print("BASELINE: Current Random Forest")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
base_rf = RandomForestClassifier(
    n_estimators=800,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    max_depth=None,
    min_samples_leaf=1
)
base_scores = cross_val_score(base_rf, X, y, cv=cv, scoring='f1_macro', n_jobs=1)
print(f"Baseline CV macro F1: {base_scores.mean():.4f} ± {base_scores.std():.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in base_scores]}")

best_overall_score = base_scores.mean()
best_overall_model = "baseline_rf"
best_overall_params = {"n_estimators": 800, "class_weight": "balanced_subsample"}

# ============== Hyperparameter Optimization ==============

def objective_rf(trial):
    """Optimize Random Forest"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'max_depth': trial.suggest_categorical('max_depth', [None, 20, 30, 40, 50]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'random_state': RANDOM_STATE
    }

    clf = RandomForestClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro', n_jobs=1)
    return scores.mean()

print("\n" + "="*60)
print("OPTIMIZING: Random Forest Hyperparameters")
print("="*60)

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

print(f"\nBest RF CV accuracy: {study_rf.best_value:.4f}")
print(f"Best RF params: {study_rf.best_params}")

if study_rf.best_value > best_overall_score:
    best_overall_score = study_rf.best_value
    best_overall_model = "optimized_rf"
    best_overall_params = study_rf.best_params

# ============== ExtraTrees ==============
def objective_et(trial):
    """Optimize ExtraTrees"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'max_depth': trial.suggest_categorical('max_depth', [None, 20, 30, 40, 50]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'random_state': RANDOM_STATE
    }

    clf = ExtraTreesClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro', n_jobs=1)
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

print(f"\nBest ET CV accuracy: {study_et.best_value:.4f}")
print(f"Best ET params: {study_et.best_params}")

if study_et.best_value > best_overall_score:
    best_overall_score = study_et.best_value
    best_overall_model = "optimized_et"
    best_overall_params = study_et.best_params

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

if best_overall_model == "baseline_rf":
    final_model = RandomForestClassifier(**best_overall_params)
elif best_overall_model == "optimized_rf":
    final_model = RandomForestClassifier(**best_overall_params)
elif best_overall_model == "optimized_et":
    final_model = ExtraTreesClassifier(**best_overall_params)

# Train the final model on all data
final_model.fit(X, y)

bundle = {
    "model": final_model,
    "feature_names": list(feat_df.drop(columns=[SPECIES_COL]).columns),
    "model_type": best_overall_model,
    "cv_macro_f1": best_overall_score
}

joblib.dump(bundle, "./results/optimized_model.pkl")
print(f"Saved optimized model to ./results/optimized_model.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final CV macro F1: {best_overall_score:.4f}")
print(f"Improvement: {(best_overall_score - base_scores.mean())*100:.2f}%")
print("\nDone!")
