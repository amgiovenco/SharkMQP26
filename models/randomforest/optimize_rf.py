"""
Optimize RandomForest model with enhanced feature engineering and hyperparameter tuning
"""
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.integrate import simpson
from scipy.signal import find_peaks
import optuna
from optuna.storages import RDBStorage
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_TRIALS = 100

# Optuna persistent storage
STORAGE_PATH = Path("./optuna_studies")
STORAGE_PATH.mkdir(exist_ok=True)
STORAGE_URL = f"sqlite:///{STORAGE_PATH}/optuna_studies.db"
storage = RDBStorage(STORAGE_URL)

print("Loading data...")
data = pd.read_csv(CSV_PATH)

X_raw = data.drop(columns=[SPECIES_COL])
y = data[SPECIES_COL]

# Drop species with <2 samples
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
mask = y.isin(valid_classes)
X_raw = X_raw[mask]
y = y[mask].to_numpy()

# Encode labels to numeric values for XGBoost compatibility
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(f"Data shape: {X_raw.shape}")
print(f"Classes: {len(np.unique(y))}")

# Enhanced feature engineering
def enhanced_feature_engineering(df):
    features = pd.DataFrame()
    temps = df.columns.astype(float)

    features['max'] = df.max(axis=1)
    features['min'] = df.min(axis=1)
    features['mean'] = df.mean(axis=1)
    features['std'] = df.std(axis=1)
    
    features['auc'] = df.apply(lambda row: simpson(row, temps), axis=1)
    features['centroid'] = df.apply(lambda row: np.sum(row*temps)/np.sum(row), axis=1)
    
    features['temp_peak'] = df.apply(lambda row: temps[np.argmax(row)], axis=1)
    features['fwhm'] = df.apply(lambda row: np.sum(row > 0.5*row.max()), axis=1)
    features['rise_time'] = df.apply(lambda row: np.argmax(row), axis=1)
    features['decay_time'] = df.apply(lambda row: len(row) - np.argmax(row[::-1]), axis=1)
    
    features['auc_left'] = df.apply(lambda row: simpson(row[:np.argmax(row)+1], temps[:np.argmax(row)+1]), axis=1)
    features['auc_right'] = df.apply(lambda row: simpson(row[np.argmax(row):], temps[np.argmax(row):]), axis=1)
    
    features['asymmetry'] = features['auc_left'] / (features['auc_right'] + 1e-8)

    # Gradient features
    def max_gradient(row):
        grad = np.gradient(row.values, temps)
        return np.max(np.abs(grad))

    features['max_gradient'] = df.apply(max_gradient, axis=1)

    # Peak count features
    def count_peaks(row):
        peaks, _ = find_peaks(row.values, prominence=row.max()*0.1)
        return len(peaks)

    features['n_peaks'] = df.apply(count_peaks, axis=1)

    # Range features at different temperature regions
    n = len(temps)
    third = n // 3
    features['range_low'] = df.iloc[:, :third].max(axis=1) - df.iloc[:, :third].min(axis=1)
    features['range_mid'] = df.iloc[:, third:2*third].max(axis=1) - df.iloc[:, third:2*third].min(axis=1)
    features['range_high'] = df.iloc[:, 2*third:].max(axis=1) - df.iloc[:, 2*third:].min(axis=1)

    # Variance across temperature
    features['temp_variance'] = df.var(axis=1)

    return features

print("\nExtracting enhanced features...")
X_features = enhanced_feature_engineering(X_raw)
X = X_features.to_numpy(float)

print(f"Feature shape: {X.shape}")
print(f"Features: {list(X_features.columns[:10])}...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Baseline
print("\n" + "="*60)
print("BASELINE: Random Forest with enhanced features")
print("="*60)

base_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=1
)

base_scores = cross_val_score(base_rf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
print(f"Baseline CV accuracy: {base_scores.mean():.4f} ± {base_scores.std():.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in base_scores]}")

best_overall_score = base_scores.mean()
best_overall_model = "baseline_rf"
best_overall_params = {"n_estimators": 300, "class_weight": "balanced"}

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
    study_name="rf_randomforest",
    load_if_exists=True
)
print(f"Study: rf_randomforest | Completed trials: {len(study_rf.trials)}")
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
    study_name="rf_extratrees",
    load_if_exists=True
)
print(f"Study: rf_extratrees | Completed trials: {len(study_et.trials)}")
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
    study_name="rf_lightgbm",
    load_if_exists=True
)
print(f"Study: rf_lightgbm | Completed trials: {len(study_lgb.trials)}")
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest LightGBM CV accuracy: {study_lgb.best_value:.4f}")
print(f"Best LightGBM params: {study_lgb.best_params}")

if study_lgb.best_value > best_overall_score:
    best_overall_score = study_lgb.best_value
    best_overall_model = "optimized_lightgbm"
    best_overall_params = study_lgb.best_params

# Optimize XGBoost
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 300, 1500, step=100),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 5),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE,
        'verbosity': 0,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }

    clf = xgb.XGBClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.mean()

print("\n" + "="*60)
print("OPTIMIZING: XGBoost")
print("="*60)

study_xgb = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    storage=storage,
    study_name="rf_xgboost",
    load_if_exists=True
)
print(f"Study: rf_xgboost | Completed trials: {len(study_xgb.trials)}")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest XGBoost CV accuracy: {study_xgb.best_value:.4f}")
print(f"Best XGBoost params: {study_xgb.best_params}")

if study_xgb.best_value > best_overall_score:
    best_overall_score = study_xgb.best_value
    best_overall_model = "optimized_xgboost"
    best_overall_params = study_xgb.best_params

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

# Strip prefixes from parameter names for model instantiation
def strip_prefix(params):
    cleaned = {}
    for key, value in params.items():
        # Remove rf_, et_, lgb_, or xgb_ prefix if present
        if key.startswith('rf_'):
            new_key = key[3:]
        elif key.startswith('et_'):
            new_key = key[3:]
        elif key.startswith('lgb_'):
            new_key = key[4:]
        elif key.startswith('xgb_'):
            new_key = key[4:]
        else:
            new_key = key
        cleaned[new_key] = value
    return cleaned

cleaned_params = strip_prefix(best_overall_params)

if best_overall_model == "optimized_rf":
    final_model = RandomForestClassifier(**cleaned_params)
elif best_overall_model == "optimized_extratrees":
    final_model = ExtraTreesClassifier(**cleaned_params)
elif best_overall_model == "optimized_lightgbm":
    final_model = lgb.LGBMClassifier(**cleaned_params)
elif best_overall_model == "optimized_xgboost":
    final_model = xgb.XGBClassifier(**cleaned_params)
else:
    final_model = RandomForestClassifier(**cleaned_params)

final_model.fit(X, y)

bundle = {
    "model": final_model,
    "feature_names": list(X_features.columns),
    "model_type": best_overall_model,
    "cv_accuracy": best_overall_score,
    "params": best_overall_params,
    "label_encoder": label_encoder
}

joblib.dump(bundle, "./randomforest_final.pkl")
print(f"Saved optimized model to ./randomforest_final.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final CV accuracy: {best_overall_score:.4f}")
print(f"Improvement: {(best_overall_score - base_scores.mean())*100:.2f}%")
print("\nDone!")
