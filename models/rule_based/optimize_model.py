"""
Optimize rule-based model hyperparameters with Optuna.
Tunes margin, model type (RF vs LR), and RF hyperparameters if selected.
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import optuna
from optuna.storages import RDBStorage
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_TRIALS = 100

# Optuna persistent storage
STUDY_NAME = "rule_based_ensemble"
STORAGE_PATH = Path("./optuna_studies")
STORAGE_PATH.mkdir(exist_ok=True)
STORAGE_URL = f"sqlite:///{STORAGE_PATH}/optuna_studies.db"

# Feature engineering (copied from rule_based.py)
def _curve_features(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Engineer ~14 features from a single curve y(t)."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    k = max(1, int(0.05 * len(y)))
    baseline = np.median(y[:k])
    yb = y - baseline
    yb = np.clip(yb, 0.0, None)

    idx_max = int(np.argmax(yb))
    ymax = float(yb[idx_max])
    tmax = float(t[idx_max])

    auc = float(np.trapezoid(yb, t))
    centroid = float(np.trapezoid(yb * t, t) / (auc + 1e-12))

    half = 0.5 * ymax
    above = np.where(yb >= half)[0]
    fwhm = float(t[above[-1]] - t[above[0]]) if above.size > 0 else 0.0

    def cross(level):
        idx = np.where(yb >= level)[0]
        return (int(idx[0]) if idx.size else idx_max), (int(idx[-1]) if idx.size else idx_max)

    lo, hi = 0.1 * ymax, 0.9 * ymax
    lo_i1, hi_i1 = cross(lo)[0], cross(hi)[0]
    lo_i2, hi_i2 = cross(lo)[1], cross(hi)[1]
    rise_time = float(t[max(hi_i1, lo_i1)] - t[min(hi_i1, lo_i1)]) if ymax > 0 else 0.0
    decay_time = float(t[max(lo_i2, hi_i2)] - t[min(lo_i2, hi_i2)]) if ymax > 0 else 0.0

    auc_left = float(np.trapezoid(yb[:idx_max + 1], t[:idx_max + 1]))
    auc_right = float(np.trapezoid(yb[idx_max:], t[idx_max:]))
    asymmetry = float((auc_right - auc_left) / (auc + 1e-12))

    mean_val = float(np.mean(y))
    std_val = float(np.std(y))
    max_val = float(np.max(y))
    min_val = float(np.min(y))

    return np.array([
        ymax, tmax, auc, centroid, fwhm, rise_time, decay_time,
        auc_left, auc_right, asymmetry, mean_val, std_val, max_val, min_val
    ], dtype=float)

def engineer_features(X_raw: pd.DataFrame):
    """Convert full sequence to compact 14-dim curve features."""
    t = X_raw.columns.astype(float).to_numpy()
    M = X_raw.to_numpy(float)
    F = np.vstack([_curve_features(M[i, :], t) for i in range(M.shape[0])])
    names = [
        "ymax", "tmax", "auc", "centroid", "fwhm", "rise", "decay",
        "auc_left", "auc_right", "asym", "mean", "std", "max", "min"
    ]
    return pd.DataFrame(F, columns=names), names

# Load data
print("Loading data...")
df = pd.read_csv(CSV_PATH)
X_raw = df.drop(columns=[SPECIES_COL])
y = df[SPECIES_COL].astype(str)

print("Extracting features...")
Xf, feature_names = engineer_features(X_raw)
X = Xf.to_numpy(float)

print(f"Data shape: {X.shape}")
print(f"Features: {feature_names}")

# Encode labels (required for XGBoost during cross-validation)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Baseline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n" + "="*60)
print("BASELINE: Rule-based with RandomForest")
print("="*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

base_clf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
)

base_scores = cross_val_score(base_clf, X, y_encoded, cv=cv, scoring='accuracy', n_jobs=1)
print(f"Baseline CV accuracy: {base_scores.mean():.4f} ± {base_scores.std():.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in base_scores]}")

best_overall_score = base_scores.mean()
best_overall_model = "baseline_rb"
best_overall_params = {
    "model_type": "rf",
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "margin": 0.1
}

# Objective function for optimization
def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['rf', 'et', 'lr', 'lgb', 'xgb'])

    if model_type == 'rf':
        n_estimators = trial.suggest_int('rf_n_estimators', 200, 800)
        min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 8)
        max_depth = trial.suggest_categorical('rf_max_depth', [None, 15, 20, 30])
        max_features = trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None])

        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                max_features=max_features,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    elif model_type == 'et':
        n_estimators = trial.suggest_int('et_n_estimators', 200, 800)
        min_samples_leaf = trial.suggest_int('et_min_samples_leaf', 1, 8)
        max_depth = trial.suggest_categorical('et_max_depth', [None, 15, 20, 30])
        max_features = trial.suggest_categorical('et_max_features', ['sqrt', 'log2', None])

        clf = make_pipeline(
            StandardScaler(),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                max_features=max_features,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    elif model_type == 'lgb':
        n_estimators = trial.suggest_int('lgb_n_estimators', 100, 500)
        learning_rate = trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int('lgb_num_leaves', 20, 150)
        lgb_max_depth = trial.suggest_int('lgb_max_depth', 3, 12)

        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=lgb_max_depth,
            random_state=RANDOM_STATE,
            verbose=-1
        )
    elif model_type == 'xgb':
        n_estimators = trial.suggest_int('xgb_n_estimators', 100, 500)
        learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('xgb_max_depth', 3, 12)
        subsample = trial.suggest_float('xgb_subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0)
        min_child_weight = trial.suggest_int('xgb_min_child_weight', 1, 5)

        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=RANDOM_STATE,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    else:  # lr
        max_iter = trial.suggest_int('lr_max_iter', 1000, 5000)
        C = trial.suggest_float('lr_C', 0.001, 100.0, log=True)
        solver = trial.suggest_categorical('lr_solver', ['lbfgs', 'newton-cholesky'])

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=max_iter,
                C=C,
                solver=solver,
                multi_class="multinomial",
                random_state=RANDOM_STATE
            )
        )

    scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.mean()

print("\n" + "="*60)
print("OPTIMIZING: Rule-based model hyperparameters")
print("="*60)

storage = RDBStorage(STORAGE_URL)
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    storage=storage,
    study_name=STUDY_NAME,
    load_if_exists=True
)
print(f"Study: {STUDY_NAME} | Completed trials: {len(study.trials)}")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest CV accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

if study.best_value > best_overall_score:
    best_overall_score = study.best_value
    best_overall_model = "optimized_rb"
    best_overall_params = study.best_params

# Results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"\nBest model: {best_overall_model}")
print(f"Best CV accuracy: {best_overall_score:.4f}")
print(f"Best params: {best_overall_params}")
print(f"Improvement over baseline: {(best_overall_score - base_scores.mean())*100:.2f}%")

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

# Train final model on all data
print("\n" + "="*60)
print("Training final model on all data...")
print("="*60)

model_params = {k: v for k, v in best_overall_params.items() if k != 'margin'}
model_type = model_params.pop('model_type', 'rf')

if model_type == 'rf':
    # Extract RF-specific params
    rf_params = {k: v for k, v in model_params.items() if k in [
        'n_estimators', 'min_samples_leaf', 'max_depth', 'max_features'
    ]}
    rf_params['random_state'] = RANDOM_STATE
    rf_params['n_jobs'] = -1

    final_clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(**rf_params)
    )
elif model_type == 'et':
    # Extract ExtraTrees-specific params
    et_params = {k: v for k, v in model_params.items() if k in [
        'n_estimators', 'min_samples_leaf', 'max_depth', 'max_features'
    ]}
    et_params['random_state'] = RANDOM_STATE
    et_params['n_jobs'] = -1

    final_clf = make_pipeline(
        StandardScaler(),
        ExtraTreesClassifier(**et_params)
    )
elif model_type == 'lgb':
    # Extract LightGBM-specific params
    lgb_params = {k: v for k, v in model_params.items() if k in [
        'n_estimators', 'learning_rate', 'num_leaves', 'max_depth'
    ]}
    lgb_params['random_state'] = RANDOM_STATE
    lgb_params['verbose'] = -1

    final_clf = lgb.LGBMClassifier(**lgb_params)
elif model_type == 'xgb':
    # Extract XGBoost-specific params
    xgb_params = {k.replace('xgb_', ''): v for k, v in model_params.items() if k.startswith('xgb_')}
    xgb_params['random_state'] = RANDOM_STATE
    xgb_params['verbosity'] = 0
    xgb_params['use_label_encoder'] = False
    xgb_params['eval_metric'] = 'mlogloss'

    final_clf = xgb.XGBClassifier(**xgb_params)
else:  # lr
    lr_params = {k: v for k, v in model_params.items() if k in [
        'max_iter', 'C', 'solver'
    ]}
    lr_params['random_state'] = RANDOM_STATE
    lr_params['multi_class'] = 'multinomial'

    final_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(**lr_params)
    )

final_clf.fit(X, y_encoded)

bundle = {
    "model": final_clf,
    "label_encoder": le,
    "feature_names": feature_names,
    "model_type": best_overall_model,
    "cv_accuracy": best_overall_score,
    "params": best_overall_params
}

joblib.dump(bundle, "./rulebased_final.pkl")
print(f"Saved optimized model to ./rulebased_final.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final CV accuracy: {best_overall_score:.4f}")
print(f"Improvement: {(best_overall_score - base_scores.mean())*100:.2f}%")
print("\nDone!")
