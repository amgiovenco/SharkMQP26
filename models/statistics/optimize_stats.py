"""
Optimize Statistical model with proper 5-fold CV on train+val (80%)
Baseline: 5-fold CV on 80%
Optuna: 5-fold CV (train 4 folds → val 1 fold) → avg val acc
Final model: train+val (80%) → test (20%)
"""
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import optuna
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================= CONFIG =============================
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_TRIALS = 100
N_CV_FOLDS = 5
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

TOP_FEATURES = [
    'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
    'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
    'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
    'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
    'slope_std', 'y_left_max'
]
# =================================================================

# Load data
df = pd.read_csv(CSV_PATH)
temp_cols = sorted([c for c in df.columns if c != SPECIES_COL], key=lambda c: float(c))
x_axis = np.array([float(c) for c in temp_cols], dtype=float)
print(f"Loaded {len(df)} samples, {df[SPECIES_COL].nunique()} species")

# ============================= PREPROCESS =============================
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

# ============================= FEATURE EXTRACTION =============================
def extract_features(x, y):
    feat = {}
    feat["mean"] = float(np.mean(y))
    feat["std"] = float(np.std(y))
    feat["min"] = float(np.min(y))
    feat["max"] = float(np.max(y))
    feat["range"] = float(np.ptp(y))
    feat["skewness"] = float(pd.Series(y).skew())
    feat["kurtosis"] = float(pd.Series(y).kurtosis())
    dy = np.gradient(y, x)
    feat["max_slope"] = float(np.max(np.abs(dy)))
    feat["mean_abs_slope"] = float(np.mean(np.abs(dy)))
    feat["slope_std"] = float(np.std(dy))
    d2y = np.gradient(dy, x)
    feat["max_curvature"] = float(np.max(np.abs(d2y)))
    feat["mean_abs_curvature"] = float(np.mean(np.abs(d2y)))
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
    n = len(y)
    for region, start, end in [("left", 0, n//3), ("middle", n//3, 2*n//3), ("right", 2*n//3, n)]:
        feat[f"y_{region}_mean"] = float(np.mean(y[start:end]))
        feat[f"y_{region}_std"] = float(np.std(y[start:end]))
        feat[f"y_{region}_max"] = float(np.max(y[start:end]))
    q = np.percentile(y, [25, 50, 75])
    feat["q25"] = float(q[0])
    feat["q50"] = float(q[1])
    feat["q75"] = float(q[2])
    feat["iqr"] = float(q[2] - q[0])
    fft_vals = np.abs(fft(y - np.mean(y)))
    fft_power = fft_vals ** 2
    for i, idx in enumerate(np.argsort(fft_power)[-5:][::-1]):
        feat[f"fft_power_{i}"] = float(fft_power[idx])
    feat["fft_total_power"] = float(np.sum(fft_power))
    feat["fft_entropy"] = float(entropy(fft_power + 1e-10))
    feat["cv"] = feat["std"] / (feat["mean"] + 1e-10)
    feat["peak_to_mean_ratio"] = feat["max"] / (feat["mean"] + 1e-10)
    return feat

print("Extracting features...")
feat_list = []
for i in range(len(df)):
    if i % 100 == 0:
        print(f" {i}/{len(df)}")
    f = extract_features(x_axis, x_proc[i])
    f[SPECIES_COL] = df.iloc[i][SPECIES_COL]
    feat_list.append(f)

feat_df = pd.DataFrame(feat_list).fillna(0.0)
X = feat_df[TOP_FEATURES].to_numpy(float)
y = df[SPECIES_COL].astype(str).to_numpy()
feature_names = TOP_FEATURES
print(f"\nFiltered to {len(feature_names)} features")

# ============================= 60/20/20 SPLIT =============================
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\n60/20/20 Split:")
print(f" Train+Val: {len(X_train_val)} samples")
print(f" Test:      {len(X_test)} samples")

# Encode labels
le = LabelEncoder()
y_train_val_encoded = le.fit_transform(y_train_val)
y_test_encoded = le.transform(y_test)

# ============================= 5-FOLD CV ON TRAIN+VAL (80%) =============================
cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ============================= BASELINE: 5-FOLD CV ON TRAIN+VAL =============================
print("\n" + "="*60)
print("BASELINE: Random Forest (5-fold CV on 80%)")
print("="*60)

base_rf = RandomForestClassifier(
    n_estimators=800,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1
)

fold_scores = []
for train_idx, val_idx in cv.split(X_train_val, y_train_val):
    X_tr, X_va = X_train_val[train_idx], X_train_val[val_idx]
    y_tr, y_va = y_train_val[train_idx], y_train_val[val_idx]
    base_rf.fit(X_tr, y_tr)
    fold_scores.append(accuracy_score(y_va, base_rf.predict(X_va)))

base_cv_mean = np.mean(fold_scores)
base_cv_std = np.std(fold_scores)
print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV accuracy: {base_cv_mean:.4f} ± {base_cv_std:.4f}")

# Also fit on full train+val for final test eval
base_rf.fit(X_train_val, y_train_val)
base_test_score = accuracy_score(y_test, base_rf.predict(X_test))
print(f"Test accuracy (baseline): {base_test_score:.4f}")

best_overall_score = base_cv_mean
best_overall_model = "baseline_rf"
best_overall_params = {
    "n_estimators": 800,
    "class_weight": "balanced_subsample",
    "max_depth": None,
    "min_samples_leaf": 1
}

# ============================= OPTUNA: 5-FOLD CV ON TRAIN+VAL =============================
def create_objective(model_cls, is_xgb=False):
    def objective(trial):
        if model_cls == RandomForestClassifier:
            params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 500, 2000, step=100),
                'max_depth': trial.suggest_categorical('rf_max_depth', [None, 20, 30, 40, 50]),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 0.5, 0.7]),
                'class_weight': trial.suggest_categorical('rf_class_weight', ['balanced', 'balanced_subsample']),
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
        elif model_cls == ExtraTreesClassifier:
            params = {
                'n_estimators': trial.suggest_int('et_n_estimators', 500, 2000, step=100),
                'max_depth': trial.suggest_categorical('et_max_depth', [None, 20, 30, 40, 50]),
                'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('et_max_features', ['sqrt', 'log2', 0.5, 0.7]),
                'class_weight': trial.suggest_categorical('et_class_weight', ['balanced', 'balanced_subsample']),
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
        elif model_cls == lgb.LGBMClassifier:
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
                'verbose': -1,
                'n_jobs': -1
            }
        elif model_cls == xgb.XGBClassifier:
            params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 300, 1500, step=100),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
                'random_state': RANDOM_STATE,
                'verbosity': 0,
                'n_jobs': -1
            }

        val_scores = []
        for train_idx, val_idx in cv.split(X_train_val, y_train_val):
            X_tr, X_va = X_train_val[train_idx], X_train_val[val_idx]
            y_tr = y_train_val_encoded[train_idx] if is_xgb else y_train_val[train_idx]
            y_va = y_train_val_encoded[val_idx] if is_xgb else y_train_val[val_idx]

            clf = model_cls(**params)
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_va)
            val_scores.append(accuracy_score(y_va, pred))

        return np.mean(val_scores)
    return objective

# ============================= RUN OPTIMIZATION =============================
models = [
    ("Random Forest", RandomForestClassifier, False, "rf"),
    ("ExtraTrees", ExtraTreesClassifier, False, "et"),
    ("LightGBM", lgb.LGBMClassifier, False, "lgb"),
    ("XGBoost", xgb.XGBClassifier, True, "xgb")
]

for name, cls, is_xgb, prefix in models:
    print("\n" + "="*60)
    print(f"OPTIMIZING: {name} (5-fold CV on 80%)")
    print("="*60)

    objective = create_objective(cls, is_xgb)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"Best {name} CV accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    if study.best_value > best_overall_score:
        best_overall_score = study.best_value
        best_overall_model = f"optimized_{prefix}"
        best_overall_params = study.best_params

# ============================= FINAL MODEL ON TRAIN+VAL =============================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Best model: {best_overall_model}")
print(f"Best CV accuracy (on 80%): {best_overall_score:.4f}")

def clean_params(params, prefix):
    return {k[len(prefix)+1:] if k.startswith(prefix + '_') else k: v for k, v in params.items()}

# Refit best model on full train+val
if best_overall_model == "optimized_rf":
    params = clean_params(best_overall_params, 'rf')
    final_model = RandomForestClassifier(**params, n_jobs=-1)
    final_model.fit(X_train_val, y_train_val)
    test_score = accuracy_score(y_test, final_model.predict(X_test))
elif best_overall_model == "optimized_extratrees":
    params = clean_params(best_overall_params, 'et')
    final_model = ExtraTreesClassifier(**params, n_jobs=-1)
    final_model.fit(X_train_val, y_train_val)
    test_score = accuracy_score(y_test, final_model.predict(X_test))
elif best_overall_model == "optimized_lightgbm":
    params = clean_params(best_overall_params, 'lgb')
    final_model = lgb.LGBMClassifier(**params, n_jobs=-1)
    final_model.fit(X_train_val, y_train_val)
    test_score = accuracy_score(y_test, final_model.predict(X_test))
elif best_overall_model == "optimized_xgboost":
    params = clean_params(best_overall_params, 'xgb')
    final_model = xgb.XGBClassifier(**params, n_jobs=-1)
    final_model.fit(X_train_val, y_train_val_encoded)
    test_score = accuracy_score(y_test_encoded, final_model.predict(X_test))
else:
    final_model = base_rf
    test_score = base_test_score

print(f"Test accuracy: {test_score:.4f}")
print(f"CV → Test gap: {(best_overall_score - test_score)*100:.2f} pp")

# ============================= CALIBRATE & SAVE =============================
calibrated_model = CalibratedClassifierCV(final_model, cv=3, method="isotonic")
calibrated_model.fit(X_train_val, y_train_val if 'xgb' not in best_overall_model else y_train_val_encoded)

bundle = {
    "model": calibrated_model,
    "feature_names": feature_names,
    "model_type": best_overall_model,
    "cv_accuracy": float(best_overall_score),
    "test_accuracy": float(test_score),
    "params": clean_params(best_overall_params, best_overall_model.split('_')[1]) if 'optimized' in best_overall_model else best_overall_params
}

with open(results_dir / "statistics_final.pkl", 'wb') as f:
    pickle.dump(bundle, f)
print(f"Saved model to {results_dir}/statistics_final.pkl")

# ============================= EXPORT RESULTS =============================
results_dict = {
    "baseline_cv_mean": float(base_cv_mean),
    "baseline_cv_std": float(base_cv_std),
    "baseline_cv_folds": [float(s) for s in fold_scores],
    "baseline_test_accuracy": float(base_test_score),
    "best_model": best_overall_model,
    "best_cv_accuracy": float(best_overall_score),
    "best_test_accuracy": float(test_score),
    "best_params": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in best_overall_params.items()},
    "cv_improvement_pct": float((best_overall_score - base_cv_mean) * 100),
    "test_improvement_pct": float((test_score - base_test_score) * 100),
    "data_split": {
        "train_val_samples": int(len(X_train_val)),
        "test_samples": int(len(X_test))
    }
}

with open("./optimization_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)
print("Saved results to ./optimization_results.json")

# ============================= SUMMARY =============================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Baseline  → CV: {base_cv_mean:.4f} ± {base_cv_std:.4f} | Test: {base_test_score:.4f}")
print(f"Optimized → CV: {best_overall_score:.4f} | Test: {test_score:.4f}")
print(f"CV ↑:  +{(best_overall_score - base_cv_mean)*100:.2f}%")
print(f"Test ↑: +{(test_score - base_test_score)*100:.2f}%")
print("\nDone!")