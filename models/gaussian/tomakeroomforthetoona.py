# %%
import numpy as np, pandas as pd, joblib, warnings
from math import isfinite
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
import optuna
from optuna.storages import RDBStorage
from joblib import Parallel, delayed

# Suppress scikit-learn warnings about class imbalance and regression-like classification
warnings.filterwarnings('ignore', category=UserWarning)

# Config
CSV_PATH  = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
CALIBRATE = True
RANDOM_STATE = 8
MODEL_OUT = "gauss_species_model_optuna_fast.joblib"
FEATURES_OUT = "gaussian_peak_features_all_optuna_fast.csv"
TRIALS=1

# %%
# Load data
df = pd.read_csv(CSV_PATH)
if SPECIES_COL not in df.columns:
    raise ValueError(f"Could not find '{SPECIES_COL}' in columns: {df.columns[:10]}...")
temp_cols = sorted([c for c in df.columns if c != SPECIES_COL], key=lambda c: float(c))
X_axis = np.array([float(c) for c in temp_cols], dtype=float)

print("Data shape:", df.shape)
print("Species counts:\n", df[SPECIES_COL].value_counts().sort_values(ascending=False).to_string())

# 60/20/20 split
df_train_val, df_test = train_test_split(df, test_size=0.2, stratify=df[SPECIES_COL], random_state=RANDOM_STATE)
df_train, df_val = train_test_split(df_train_val, test_size=0.25, stratify=df_train_val[SPECIES_COL], random_state=RANDOM_STATE)
print(f"Split sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

# %%
# Gaussians, preprocess, fitting (parametrized)
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)

def gaussian_sum(x, *p):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(p), 3):
        amp, mu, sigma = p[i:i+3]
        y += gaussian(x, float(amp), float(mu), abs(float(sigma)))
    return y

def preprocess_curve(x, y, params):
    """Smooth + quadratic baseline removal + normalization."""
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

def decimate(x, y, step):
    return x[::step], y[::step]

def seed_peaks(x, y, k, params):
    """Robust peak seeds: use prominence from spread, clamp widths."""
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
    if np.ndim(w_c) == 0:
        w_c = np.array([w_c])
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
        sigma0 = float(max(w_c[j] / (2 * np.sqrt(2 * np.log(2))), (x[1] - x[0]) * 2))
        p0 += [amp0, mu0, sigma0]
        lo += [0.0, mu0 - params['mu_bound_mult'] * sigma0, (x[1] - x[0]) * 1e-3]
        hi += [y_max * params['amp_hi_mult'] + 1e-6, mu0 + params['mu_bound_mult'] * sigma0, (x[-1] - x[0])]
    popt, _ = curve_fit(gaussian_sum, x, y, p0=p0, bounds=(lo, hi), maxfev=5000)  # Reduced maxfev for speed
    return popt

def BIC(n_params, rss, n):
    return np.log(n) * n_params + n * np.log(rss / n + 1e-12)

def fit_best_K(x, y, params):
    best = None
    for k in range(1, params['k_max'] + 1):
        try:
            popt = fit_k(x, y, k, params)
            yhat = gaussian_sum(x, *popt)
            rss = float(np.sum((y - yhat) ** 2))
            bic = float(BIC(3 * k, rss, len(x)))
            if best is None or bic < best["bic"]:
                best = {"k": k, "popt": popt, "bic": bic}
        except Exception:
            pass
    return best

def peaks_to_features(popt, k_keep=2):
    peaks = [{"amp": float(popt[i]), "mu": float(popt[i + 1]), "sigma": abs(float(popt[i + 2]))}
             for i in range(0, len(popt), 3)]
    peaks.sort(key=lambda d: d["amp"], reverse=True)
    feats = {}
    for i in range(k_keep):
        if i < len(peaks):
            feats[f"peak{i + 1}_mu"] = peaks[i]["mu"]
            feats[f"peak{i + 1}_amp"] = peaks[i]["amp"]
            feats[f"peak{i + 1}_sigma"] = peaks[i]["sigma"]
        else:
            feats[f"peak{i + 1}_mu"] = feats[f"peak{i + 1}_amp"] = feats[f"peak{i + 1}_sigma"] = 0.0
    feats["delta_mu_12"] = feats["peak1_mu"] - feats["peak2_mu"]
    feats["amp_ratio_12"] = (feats["peak1_amp"] + 1e-9) / (feats["peak2_amp"] + 1e-9) if feats["peak2_amp"] > 0 else 1e9
    return feats

def extra_features_from_fit(x, y, popt, params):
    areas = []
    for i in range(0, len(popt), 3):
        amp, mu, sigma = float(popt[i]), float(popt[i + 1]), abs(float(popt[i + 2]))
        areas.append(amp * sigma * np.sqrt(2 * np.pi))
    total_area = float(np.sum(areas)) if areas else 0.0

    main_mu = float(popt[1]) if len(popt) >= 2 else x[np.argmax(y)]
    left = y[(x >= main_mu - params['asym_width']) & (x < main_mu)]
    right = y[(x > main_mu) & (x <= main_mu + params['asym_width'])]
    asym = (right.mean() - left.mean()) if (len(left) > 3 and len(right) > 3) else 0.0
    return {"total_area": total_area, "asym_0p5C": asym}

def extract_features_for_row(row_vals, params):
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
    feats = peaks_to_features(best["popt"], k_keep=2)
    extras = extra_features_from_fit(x, y, best["popt"], params)
    feats.update(extras)
    feats["best_K"] = best["k"]
    feats["total_amp"] = feats["peak1_amp"] + feats["peak2_amp"]
    return feats

def get_feat_df(df_subset, params):
    def process_row(i):
        feats = extract_features_for_row(df_subset.loc[i, temp_cols].values, params)
        feats[SPECIES_COL] = df_subset.loc[i, SPECIES_COL]
        return feats
    
    rows = Parallel(n_jobs=-1, backend='loky')(delayed(process_row)(i) for i in df_subset.index)
    return pd.DataFrame(rows).fillna(0.0)

# %%
# Optuna objective
def objective(trial):
    params = {
        'decimate_step': trial.suggest_int('decimate_step', 3, 10),  # Increased min for fewer points, faster fits
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
        'k_max': trial.suggest_int('k_max', 3, 6),  # Reduced max for fewer fits per row
        'asym_width': trial.suggest_float('asym_width', 0.3, 1.0),
        'amp_hi_mult': trial.suggest_float('amp_hi_mult', 2, 10)
    }
    # Train on df_train, evaluate on df_val
    feat_train = get_feat_df(df_train, params)
    X_train = feat_train.drop(columns=[SPECIES_COL]).values
    y_train = feat_train[SPECIES_COL].values

    feat_val = get_feat_df(df_val, params)
    X_val = feat_val.drop(columns=[SPECIES_COL]).values
    y_val = feat_val[SPECIES_COL].values

    clf = RandomForestClassifier(
        n_estimators=200,  # Reduced for faster training
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    return f1

# %%
# Run Optuna for RandomForest
print("=" * 80)
print("RANDOMFOREST OPTUNA OPTIMIZATION")
print("=" * 80)
storage_rf = RDBStorage("sqlite:///optuna_rf_trials_fast.db")
study_rf = optuna.create_study(direction='maximize', storage=storage_rf, study_name='RandomForest', load_if_exists=True)
study_rf.optimize(objective, n_trials=TRIALS)  # Reduced trials for speed
print("Best value (macro F1):", study_rf.best_value)
print("Best params:", study_rf.best_params)
best_rf_value = study_rf.best_value
best_rf_params = study_rf.best_params
print(f"Saved {len(study_rf.trials)} RandomForest trials to optuna_rf_trials_fast.db")

# %%
# ExtraTrees objective
def objective_extratrees(trial):
    params = {
        'decimate_step': trial.suggest_int('decimate_step', 3, 10),  # Increased min for fewer points, faster fits
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
        'k_max': trial.suggest_int('k_max', 3, 6),  # Reduced max for fewer fits per row
        'asym_width': trial.suggest_float('asym_width', 0.3, 1.0),
        'amp_hi_mult': trial.suggest_float('amp_hi_mult', 2, 10)
    }
    # Train on df_train, evaluate on df_val
    feat_train = get_feat_df(df_train, params)
    X_train = feat_train.drop(columns=[SPECIES_COL]).values
    y_train = feat_train[SPECIES_COL].values

    feat_val = get_feat_df(df_val, params)
    X_val = feat_val.drop(columns=[SPECIES_COL]).values
    y_val = feat_val[SPECIES_COL].values

    clf = ExtraTreesClassifier(
        n_estimators=200,  # Reduced for faster training
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    return f1

# %%
# Run Optuna for ExtraTrees
print("\n" + "=" * 80)
print("EXTRATREES OPTUNA OPTIMIZATION")
print("=" * 80)
storage_et = RDBStorage("sqlite:///optuna_et_trials_fast.db")
study_et = optuna.create_study(direction='maximize', storage=storage_et, study_name='ExtraTrees', load_if_exists=True)
study_et.optimize(objective_extratrees, n_trials=TRIALS)  # Reduced trials for speed
print("Best value (macro F1):", study_et.best_value)
print("Best params:", study_et.best_params)
best_et_value = study_et.best_value
best_et_params = study_et.best_params
print(f"Saved {len(study_et.trials)} ExtraTrees trials to optuna_et_trials_fast.db")

# %%
# Report best model overall
print("\n" + "=" * 80)
print("COMPARISON: RANDOMFOREST vs EXTRATREES")
print("=" * 80)
print(f"RandomForest best F1: {best_rf_value:.6f}")
print(f"ExtraTrees best F1:   {best_et_value:.6f}")
print()
if best_rf_value >= best_et_value:
    print(f"WINNER: RandomForest (+{best_rf_value - best_et_value:.6f})")
    study = study_rf
    best_params = best_rf_params
else:
    print(f"WINNER: ExtraTrees (+{best_et_value - best_rf_value:.6f})")
    study = study_et
    best_params = best_et_params
print("=" * 80)

# %%
# Final model with best params
df_full_train = pd.concat([df_train, df_val])
feat_train = get_feat_df(df_full_train, best_params)
X_train = feat_train.drop(columns=[SPECIES_COL]).values
y_train = feat_train[SPECIES_COL].values

base_clf = RandomForestClassifier(
    n_estimators=800,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    max_depth=None,
    min_samples_leaf=1
)

base_clf.fit(X_train, y_train)

if CALIBRATE:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    # Manual calibration with cross_val_predict using StratifiedKFold
    try:
        # Use StratifiedKFold(n_splits=5) like optimize_model.py does with cross_val_score
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        y_pred_proba = cross_val_predict(base_clf, X_train, y_train, cv=skf, method='predict_proba')
        # Fit isotonic regression for calibration
        calibrators = {}
        for i, cls in enumerate(base_clf.classes_):
            calibrators[i] = IsotonicRegression(out_of_bounds='clip')
            calibrators[i].fit(y_pred_proba[:, i], (y_train == cls).astype(int))

        # Wrap the base classifier with calibration
        class CalibratedRF:
            def __init__(self, base, calibrators, classes):
                self.base = base
                self.calibrators = calibrators
                self.classes_ = classes

            def predict(self, X):
                return self.base.predict(X)

            def predict_proba(self, X):
                proba = self.base.predict_proba(X)
                for i in range(len(self.classes_)):
                    proba[:, i] = self.calibrators[i].transform(proba[:, i])
                # Renormalize
                proba = proba / proba.sum(axis=1, keepdims=True)
                return proba

        final_clf = CalibratedRF(base_clf, calibrators, base_clf.classes_)
    except Exception as e:
        # If calibration fails, just use uncalibrated
        print(f"Calibration failed ({e}), using uncalibrated model")
        final_clf = base_clf
else:
    final_clf = base_clf

# Test evaluation
feat_test = get_feat_df(df_test, best_params)
X_test = feat_test.drop(columns=[SPECIES_COL]).values
y_test = feat_test[SPECIES_COL].values
pred_test = final_clf.predict(X_test)
acc_test = accuracy_score(y_test, pred_test)
print("\nTest accuracy:", acc_test)
print("Test classification report:\n", classification_report(y_test, pred_test))

# Save features and model
feat_all = get_feat_df(df, best_params)
feat_all.to_csv(FEATURES_OUT, index=False)
print("Saved all features ->", FEATURES_OUT)

bundle = {
    "model": final_clf,
    "feature_names": feat_train.drop(columns=[SPECIES_COL]).columns.tolist(),
    "classes_": final_clf.classes_,
    "best_params": best_params
}
joblib.dump(bundle, MODEL_OUT)
print("Saved model ->", MODEL_OUT)