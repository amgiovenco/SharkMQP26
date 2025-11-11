"""
Optimize RandomForest model with enhanced feature engineering and hyperparameter tuning
"""
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.integrate import simpson
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

# Feature engineering (matching notebook exactly)
def feature_engineering(df):
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
    return features

def enhanced_features(features):
    enhanced = features.copy()

    # Interaction features from top performers
    enhanced['fwhm_rise_ratio'] = features['fwhm'] / (features['rise_time'] + 1e-8)
    enhanced['peak_temp_std'] = features['temp_peak'] * features['std']
    enhanced['asymmetry_fwhm'] = features['asymmetry'] * features['fwhm']
    enhanced['rise_decay_ratio'] = features['rise_time'] / (features['decay_time'] + 1e-8)

    return enhanced

print("\nExtracting features...")
X_features = feature_engineering(X_raw)
X_features = enhanced_features(X_features)
X = X_features.to_numpy(float)

print(f"Feature shape: {X.shape}")
print(f"Features: {list(X_features.columns[:10])}...")

# 60/20/20 split: train/val/test
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_temp
)

print(f"\nData split:")
print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Baseline
print("\n" + "="*60)
print("BASELINE: Random Forest with enhanced features")
print("="*60)

# Compute 5-fold CV on FULL dataset for baseline comparison
from sklearn.model_selection import StratifiedKFold as SKFold
cv_splitter = SKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
baseline_cv_scores = []

for train_idx, val_idx in cv_splitter.split(X, y):
    X_cv_train, X_cv_val = X[train_idx], X[val_idx]
    y_cv_train, y_cv_val = y[train_idx], y[val_idx]

    base_rf_cv = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=1
    )
    base_rf_cv.fit(X_cv_train, y_cv_train)
    fold_score = accuracy_score(y_cv_val, base_rf_cv.predict(X_cv_val))
    baseline_cv_scores.append(fold_score)

print(f"Baseline 5-fold CV accuracies: {[f'{s:.4f}' for s in baseline_cv_scores]}")
print(f"Baseline CV accuracy (mean): {np.mean(baseline_cv_scores):.4f} ± {np.std(baseline_cv_scores):.4f}")

# Train baseline on full training set
base_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=1
)

base_rf.fit(X_train, y_train)
base_val_score = accuracy_score(y_val, base_rf.predict(X_val))
base_test_score = accuracy_score(y_test, base_rf.predict(X_test))
print(f"Baseline Test accuracy: {base_test_score:.4f}")

best_overall_score = base_val_score
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

    # Use 5-fold CV on FULL dataset for robust evaluation
    cv_splitter = SKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]

        clf = RandomForestClassifier(**params)
        clf.fit(X_cv_train, y_cv_train)
        fold_score = accuracy_score(y_cv_val, clf.predict(X_cv_val))
        cv_scores.append(fold_score)

    return np.mean(cv_scores)  # Return mean CV score across all 5 folds

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

print(f"\nBest RF Val accuracy: {study_rf.best_value:.4f}")
print(f"Best RF params: {study_rf.best_params}")

# Evaluate on test set
best_rf_params_clean = {k[3:]: v for k, v in study_rf.best_params.items()}
best_rf_test_model = RandomForestClassifier(**best_rf_params_clean)
best_rf_test_model.fit(X_train, y_train)
best_rf_test_score = accuracy_score(y_test, best_rf_test_model.predict(X_test))
print(f"Best RF Test accuracy: {best_rf_test_score:.4f}")

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

    # Use 5-fold CV on FULL dataset for robust evaluation
    cv_splitter = SKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]

        clf = ExtraTreesClassifier(**params)
        clf.fit(X_cv_train, y_cv_train)
        fold_score = accuracy_score(y_cv_val, clf.predict(X_cv_val))
        cv_scores.append(fold_score)

    return np.mean(cv_scores)  # Return mean CV score across all 5 folds

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

print(f"\nBest ExtraTrees Val accuracy: {study_et.best_value:.4f}")
print(f"Best ExtraTrees params: {study_et.best_params}")

# Evaluate on test set
best_et_params_clean = {k[3:]: v for k, v in study_et.best_params.items()}
best_et_test_model = ExtraTreesClassifier(**best_et_params_clean)
best_et_test_model.fit(X_train, y_train)
best_et_test_score = accuracy_score(y_test, best_et_test_model.predict(X_test))
print(f"Best ExtraTrees Test accuracy: {best_et_test_score:.4f}")

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

    # Use 5-fold CV on FULL dataset for robust evaluation
    cv_splitter = SKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_cv_train, y_cv_train)
        fold_score = accuracy_score(y_cv_val, clf.predict(X_cv_val))
        cv_scores.append(fold_score)

    return np.mean(cv_scores)  # Return mean CV score across all 5 folds

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

print(f"\nBest LightGBM Val accuracy: {study_lgb.best_value:.4f}")
print(f"Best LightGBM params: {study_lgb.best_params}")

# Evaluate on test set
best_lgb_params_clean = {k[4:]: v for k, v in study_lgb.best_params.items()}
best_lgb_test_model = lgb.LGBMClassifier(**best_lgb_params_clean)
best_lgb_test_model.fit(X_train, y_train)
best_lgb_test_score = accuracy_score(y_test, best_lgb_test_model.predict(X_test))
print(f"Best LightGBM Test accuracy: {best_lgb_test_score:.4f}")

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

    # Use 5-fold CV on FULL dataset for robust evaluation
    cv_splitter = SKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_cv_train, y_cv_train)
        fold_score = accuracy_score(y_cv_val, clf.predict(X_cv_val))
        cv_scores.append(fold_score)

    return np.mean(cv_scores)  # Return mean CV score across all 5 folds

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

print(f"\nBest XGBoost Val accuracy: {study_xgb.best_value:.4f}")
print(f"Best XGBoost params: {study_xgb.best_params}")

# Evaluate on test set
best_xgb_params_clean = {k[4:]: v for k, v in study_xgb.best_params.items()}
best_xgb_test_model = xgb.XGBClassifier(**best_xgb_params_clean)
best_xgb_test_model.fit(X_train, y_train)
best_xgb_test_score = accuracy_score(y_test, best_xgb_test_model.predict(X_test))
print(f"Best XGBoost Test accuracy: {best_xgb_test_score:.4f}")

if study_xgb.best_value > best_overall_score:
    best_overall_score = study_xgb.best_value
    best_overall_model = "optimized_xgboost"
    best_overall_params = study_xgb.best_params

# Results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"\nBest model: {best_overall_model}")
print(f"Best Val accuracy: {best_overall_score:.4f}")
print(f"Best params: {best_overall_params}")
print(f"\nImprovement over baseline: {(best_overall_score - base_val_score)*100:.2f}%")

# Export results to JSON
results_dict = {
    "baseline_cv_fold_scores": [float(s) for s in baseline_cv_scores],
    "baseline_val_accuracy_mean": float(np.mean(baseline_cv_scores)),
    "baseline_val_accuracy_std": float(np.std(baseline_cv_scores)),
    "baseline_test_accuracy": float(base_test_score),
    "best_model": best_overall_model,
    "best_val_accuracy": float(best_overall_score),
    "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_overall_params.items()},
    "improvement_percentage": float((best_overall_score - np.mean(baseline_cv_scores)) * 100),
    "rf_test_accuracy": float(best_rf_test_score) if best_overall_model == "optimized_rf" else None,
    "et_test_accuracy": float(best_et_test_score) if best_overall_model == "optimized_extratrees" else None,
    "lgb_test_accuracy": float(best_lgb_test_score) if best_overall_model == "optimized_lightgbm" else None,
    "xgb_test_accuracy": float(best_xgb_test_score) if best_overall_model == "optimized_xgboost" else None
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

final_model.fit(X_train, y_train)
final_test_score = accuracy_score(y_test, final_model.predict(X_test))

bundle = {
    "model": final_model,
    "feature_names": list(X_features.columns),
    "model_type": best_overall_model,
    "val_accuracy": best_overall_score,
    "test_accuracy": final_test_score,
    "params": best_overall_params,
    "label_encoder": label_encoder
}

joblib.dump(bundle, "./randomforest_final.pkl")
print(f"Saved optimized model to ./randomforest_final.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final Val accuracy: {best_overall_score:.4f}")
print(f"Final Test accuracy: {final_test_score:.4f}")
print(f"Improvement: {(best_overall_score - base_val_score)*100:.2f}%")
print("\nDone!")
