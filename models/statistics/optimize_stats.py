"""
Optimize Statistical model with hyperparameter tuning and additional features
"""
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
import optuna
from optuna.storages import RDBStorage
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_TRIALS = 100
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Optuna persistent storage
STORAGE_PATH = Path("./optuna_studies")
STORAGE_PATH.mkdir(exist_ok=True)
STORAGE_URL = f"sqlite:///{STORAGE_PATH}/optuna_studies.db"
storage = RDBStorage(STORAGE_URL)

# Load data
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
X = feat_df.drop(columns=[SPECIES_COL]).to_numpy(float)
y = df[SPECIES_COL].astype(str).to_numpy()
feature_names = feat_df.drop(columns=[SPECIES_COL]).columns.tolist()

print(f"\nFeature matrix: {feat_df.shape}")
print(f"Features: {feature_names[:10]}...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Baseline
print("\n" + "="*60)
print("BASELINE: Random Forest")
print("="*60)

base_rf = RandomForestClassifier(
    n_estimators=800,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=1
)

base_scores = cross_val_score(base_rf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
print(f"Baseline CV accuracy: {base_scores.mean():.4f} ± {base_scores.std():.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in base_scores]}")

best_overall_score = base_scores.mean()
best_overall_model = "baseline_rf"
best_overall_params = {"n_estimators": 800, "class_weight": "balanced_subsample"}

# Optimize Random Forest
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 500, 2000, step=100),
        'max_depth': trial.suggest_categorical('rf_max_depth', [None, 20, 30, 40, 50]),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 0.5, 0.7]),
        'class_weight': trial.suggest_categorical('rf_class_weight', ['balanced', 'balanced_subsample']),
        'random_state': RANDOM_STATE,
        'n_jobs': 1
    }

    clf = RandomForestClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.mean()

print("\n" + "="*60)
print("OPTIMIZING: Random Forest")
print("="*60)

study_rf = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    storage=storage,
    study_name="stats_randomforest",
    load_if_exists=True
)
print(f"Study: stats_randomforest | Completed trials: {len(study_rf.trials)}")
study_rf.optimize(objective_rf, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest RF CV accuracy: {study_rf.best_value:.4f}")
print(f"Best RF params: {study_rf.best_params}")

if study_rf.best_value > best_overall_score:
    best_overall_score = study_rf.best_value
    best_overall_model = "optimized_rf"
    best_overall_params = study_rf.best_params

# Optimize ExtraTrees
def objective_et(trial):
    params = {
        'n_estimators': trial.suggest_int('et_n_estimators', 500, 2000, step=100),
        'max_depth': trial.suggest_categorical('et_max_depth', [None, 20, 30, 40, 50]),
        'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('et_max_features', ['sqrt', 'log2', 0.5, 0.7]),
        'class_weight': trial.suggest_categorical('et_class_weight', ['balanced', 'balanced_subsample']),
        'random_state': RANDOM_STATE,
        'n_jobs': 1
    }

    clf = ExtraTreesClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.mean()

print("\n" + "="*60)
print("OPTIMIZING: ExtraTrees")
print("="*60)

study_et = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    storage=storage,
    study_name="stats_extratrees",
    load_if_exists=True
)
print(f"Study: stats_extratrees | Completed trials: {len(study_et.trials)}")
study_et.optimize(objective_et, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest ExtraTrees CV accuracy: {study_et.best_value:.4f}")
print(f"Best ExtraTrees params: {study_et.best_params}")

if study_et.best_value > best_overall_score:
    best_overall_score = study_et.best_value
    best_overall_model = "optimized_extratrees"
    best_overall_params = study_et.best_params

# Optimize LightGBM
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 300, 1500, step=100),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 150),
        'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE,
        'verbose': -1
    }

    clf = lgb.LGBMClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.mean()

print("\n" + "="*60)
print("OPTIMIZING: LightGBM")
print("="*60)

study_lgb = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    storage=storage,
    study_name="stats_lightgbm",
    load_if_exists=True
)
print(f"Study: stats_lightgbm | Completed trials: {len(study_lgb.trials)}")
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest LightGBM CV accuracy: {study_lgb.best_value:.4f}")
print(f"Best LightGBM params: {study_lgb.best_params}")

if study_lgb.best_value > best_overall_score:
    best_overall_score = study_lgb.best_value
    best_overall_model = "optimized_lightgbm"
    best_overall_params = study_lgb.best_params

# Results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"\nBest model: {best_overall_model}")
print(f"Best CV accuracy: {best_overall_score:.4f}")
print(f"Best params: {best_overall_params}")
print(f"\nImprovement over baseline: {(best_overall_score - base_scores.mean())*100:.2f}%")

# Export results to JSON
results_dict = {
    "baseline_cv_accuracy": float(base_scores.mean()),
    "baseline_fold_scores": [float(s) for s in base_scores],
    "best_model": best_overall_model,
    "best_cv_accuracy": float(best_overall_score),
    "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_overall_params.items()},
    "improvement_percentage": float((best_overall_score - base_scores.mean()) * 100)
}

with open("./optimization_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nSaved optimization results to ./optimization_results.json")

# Train final model
print("\n" + "="*60)
print("Training final model on all data...")
print("="*60)

# Helper function to strip prefix from parameter names
def clean_params(params, prefix):
    """Remove prefix from parameter names (e.g., 'rf_n_estimators' -> 'n_estimators')"""
    cleaned = {}
    for key, value in params.items():
        if key.startswith(prefix + '_'):
            new_key = key[len(prefix)+1:]
        else:
            new_key = key
        cleaned[new_key] = value
    return cleaned

if best_overall_model == "optimized_rf":
    cleaned_params = clean_params(best_overall_params, 'rf')
    final_model = RandomForestClassifier(**cleaned_params)
elif best_overall_model == "optimized_extratrees":
    cleaned_params = clean_params(best_overall_params, 'et')
    final_model = ExtraTreesClassifier(**cleaned_params)
elif best_overall_model == "optimized_lightgbm":
    cleaned_params = clean_params(best_overall_params, 'lgb')
    final_model = lgb.LGBMClassifier(**cleaned_params)
else:
    cleaned_params = clean_params(best_overall_params, 'rf')
    final_model = RandomForestClassifier(**cleaned_params)

# Apply calibration
calibrated_model = CalibratedClassifierCV(final_model, cv=3, method="isotonic")
calibrated_model.fit(X, y)

bundle = {
    "model": calibrated_model,
    "feature_names": feature_names,
    "model_type": best_overall_model,
    "cv_accuracy": best_overall_score,
    "params": cleaned_params
}

with open(results_dir / "statistics_final.pkl", 'wb') as f:
    pickle.dump(bundle, f)

print(f"Saved optimized model to {results_dir}/statistics_final.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final CV accuracy: {best_overall_score:.4f}")
print(f"Improvement: {(best_overall_score - base_scores.mean())*100:.2f}%")
print("\nDone!")
