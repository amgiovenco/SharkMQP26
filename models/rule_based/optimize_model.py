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
from sklearn.metrics import f1_score
import optuna
from optuna.storages import RDBStorage
import warnings
warnings.filterwarnings('ignore')

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_TRIALS = 1000

# Output directory for results
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Optuna persistent storage
STUDY_NAME = "rule_based_ensemble"
STORAGE_PATH = RESULTS_DIR / "optuna_studies"
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
print("BASELINE: Rule-based with ExtraTrees")
print("="*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

base_clf = make_pipeline(
    StandardScaler(),
    ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
)

base_scores = cross_val_score(base_clf, X, y_encoded, cv=cv, scoring='f1_macro', n_jobs=1)
print(f"Baseline CV macro F1: {base_scores.mean():.4f} ± {base_scores.std():.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in base_scores]}")

best_overall_score = base_scores.mean()
best_overall_model = "baseline_rb"
best_overall_params = {
    "model_type": "et",
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "margin": 0.1
}

# Objective function for optimization - ExtraTrees only
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 800, 3000, step=100)
    max_depth = trial.suggest_categorical('max_depth', [None, 20, 30, 40, 50, 60, 80])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.6, 0.7, 0.8, 0.9])
    class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    max_samples = trial.suggest_float('max_samples', 0.5, 1.0) if bootstrap else None
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.05, step=0.001)
    warm_start = trial.suggest_categorical('warm_start', [True, False])

    clf = make_pipeline(
        StandardScaler(),
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            criterion=criterion,
            bootstrap=bootstrap,
            max_samples=max_samples,
            ccp_alpha=ccp_alpha,
            warm_start=warm_start,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring='f1_macro', n_jobs=1)
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

print(f"\nBest CV macro F1: {study.best_value:.4f}")
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
print(f"Best CV macro F1: {best_overall_score:.4f}")
print(f"Best params: {best_overall_params}")
print(f"Improvement over baseline: {(best_overall_score - base_scores.mean())*100:.2f}%")

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

model_params = {k: v for k, v in best_overall_params.items() if k != 'margin'}
model_params.pop('model_type', None)  # Remove model_type if present

# Extract ExtraTrees-specific params
et_params = {k: v for k, v in model_params.items() if k in [
    'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
    'max_features', 'class_weight', 'criterion', 'bootstrap', 'max_samples',
    'ccp_alpha', 'warm_start'
]}
et_params['random_state'] = RANDOM_STATE
et_params['n_jobs'] = -1

final_clf = make_pipeline(
    StandardScaler(),
    ExtraTreesClassifier(**et_params)
)

final_clf.fit(X, y_encoded)

bundle = {
    "model": final_clf,
    "label_encoder": le,
    "feature_names": feature_names,
    "model_type": best_overall_model,
    "cv_macro_f1": best_overall_score,
    "params": best_overall_params
}

joblib.dump(bundle, RESULTS_DIR / "rulebased_final.pkl")
print(f"Saved optimized model to {RESULTS_DIR / 'rulebased_final.pkl'}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final CV macro F1: {best_overall_score:.4f}")
print(f"Improvement: {(best_overall_score - base_scores.mean())*100:.2f}%")
print("\nDone!")
