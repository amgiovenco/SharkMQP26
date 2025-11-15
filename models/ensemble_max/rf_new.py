#!/usr/bin/env python3
# rf_new_train_only_cv.py
#
# Clean pipeline (no leakage):
#  - Load RAW TRAIN / VAL / TEST CSVs
#  - Extract 36-ish features per split
#  - Tune RF hyperparameters on TRAIN only (StratifiedKFold, n_splits=3)
#  - GA feature selection on TRAIN only, fitness = 3-fold CV accuracy
#    (uses a lighter single-thread RF to avoid compute stalls)
#  - Fit & save:
#      (1) TRAIN-only calibrated model  -> ensemble_extratrees_calibrated_TRAINONLY.joblib
#      (2) TRAIN+VAL calibrated model   -> ensemble_extratrees_calibrated.joblib
#  - Evaluate once on TEST with TRAIN+VAL model
#
# Artifacts in ./results plus .joblib files in working dir.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from joblib import dump

# SciPy pieces (savgol + peaks)
try:
    from scipy.signal import savgol_filter, find_peaks
except Exception:
    savgol_filter = None
    find_peaks = None

from scipy.stats import entropy
from scipy.fft import fft

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

# --------------------
# Config / Paths
# --------------------
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

TRAIN_CSV = "shark_training_data.csv"
VAL_CSV   = "shark_validation_data.csv"
TEST_CSV  = "shark_test_data.csv"

SPECIES_COL = "Species"
RANDOM_STATE = 8
np.random.seed(RANDOM_STATE)

# ==========================================================
# Utilities: numeric axis handling, preprocess, features (36)
# ==========================================================
def _numeric_cols(d: pd.DataFrame):
    return sorted([c for c in d.columns if c != SPECIES_COL], key=lambda c: float(c))

def preprocess_curve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """smooth + baseline remove + normalize (robust, same style as before)"""
    y = np.asarray(y, float)
    if y.size < 3:
        return np.clip(y - np.min(y), 0.0, None)

    dx = x[1] - x[0] if len(x) > 1 else 1.0

    # savgol smooth (fallback to moving average if SciPy missing)
    if savgol_filter is None:
        k = max(3, min(7, (len(y)//2)*2 - 1))
        y_smooth = np.convolve(y, np.ones(k)/k, mode="same")
    else:
        win = max(7, int(round(1.5 / dx)) | 1)  # odd
        if win >= len(y):
            win = max(7, (len(y)//2)*2 - 1)
        y_smooth = savgol_filter(y, window_length=win, polyorder=3, mode="interp")

    # baseline removal: fit quadratic through low points
    q = np.quantile(y_smooth, 0.3)
    mask = y_smooth <= q
    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_smooth[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_base = y_smooth - baseline
    else:
        y_base = y_smooth - np.min(y_smooth)

    # normalize to [0, 1] via 99th percentile
    scale = np.quantile(y_base, 0.99)
    y_norm = y_base / scale if scale > 0 else y_base
    y_norm = np.maximum(y_norm, 0.0)
    return y_norm

def extract_36_features(x: np.ndarray, y: np.ndarray) -> dict:
    """36-ish features from preprocessed curve y(x)."""
    feat = {}

    # basic stats (7)
    feat["mean"] = float(np.mean(y))
    feat["std"] = float(np.std(y))
    feat["min"] = float(np.min(y))
    feat["max"] = float(np.max(y))
    feat["range"] = float(np.ptp(y))
    feat["skewness"] = float(pd.Series(y).skew())
    feat["kurtosis"] = float(pd.Series(y).kurtosis())

    # derivatives (5)
    dy = np.gradient(y, x) if len(x) == len(y) else np.gradient(y)
    feat["max_slope"] = float(np.max(np.abs(dy)))
    feat["mean_abs_slope"] = float(np.mean(np.abs(dy)))
    feat["slope_std"] = float(np.std(dy))
    d2y = np.gradient(dy, x) if len(x) == len(y) else np.gradient(dy)
    feat["max_curvature"] = float(np.max(np.abs(d2y)))
    feat["mean_abs_curvature"] = float(np.mean(np.abs(d2y)))

    # peaks (4)
    if find_peaks is None:
        # heuristic fallback
        n_peaks = int((y > (y.mean() + y.std())).sum() // 2)
        feat["n_peaks"] = float(max(0, n_peaks))
        feat["max_prominence"] = 0.0
        feat["mean_prominence"] = 0.0
        feat["peak_max_x"] = float(x[int(np.argmax(y))])
    else:
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
            feat["peak_max_x"] = float(x[int(np.argmax(y))])

    # regional stats (9): left/middle/right
    n = len(y)
    thirds = [(0, n//3), (n//3, 2*n//3), (2*n//3, n)]
    labels = ["left", "middle", "right"]
    for (s, e), nm in zip(thirds, labels):
        seg = y[s:e] if e > s else y
        feat[f"y_{nm}_mean"] = float(np.mean(seg))
        feat[f"y_{nm}_std"] = float(np.std(seg))
        feat[f"y_{nm}_max"] = float(np.max(seg))

    # quartiles (4)
    q = np.percentile(y, [25, 50, 75])
    feat["q25"] = float(q[0]); feat["q50"] = float(q[1]); feat["q75"] = float(q[2])
    feat["iqr"] = float(q[2] - q[0])

    # frequency domain (7): top-5 powers + total + entropy
    fft_vals = np.abs(fft(y - np.mean(y)))
    fft_power = fft_vals ** 2
    top_idx = np.argsort(fft_power)[-5:][::-1]
    for i, idx in enumerate(top_idx):
        feat[f"fft_power_{i}"] = float(fft_power[idx])
    feat["fft_total_power"] = float(np.sum(fft_power))
    feat["fft_entropy"] = float(entropy(fft_power + 1e-10))

    return feat

def build_features_for_split(df_raw: pd.DataFrame, temp_cols: list[str]) -> pd.DataFrame:
    x = np.array([float(c) for c in temp_cols], dtype=float)
    rows = []
    for i in range(len(df_raw)):
        y_raw = df_raw.iloc[i][temp_cols].to_numpy(float)
        y_proc = preprocess_curve(x, y_raw)
        f = extract_36_features(x, y_proc)
        f[SPECIES_COL] = df_raw.iloc[i][SPECIES_COL]
        rows.append(f)
    return pd.DataFrame(rows).fillna(0.0)

# =========================================
# RF factories (final + cheaper GA version)
# =========================================
def make_rf_from_params(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        **params,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

def make_rf_for_ga(best_params: dict) -> RandomForestClassifier:
    """
    Cheaper, single-threaded RF for GA fitness to avoid nested parallel stalls.
    """
    return RandomForestClassifier(
        n_estimators=min(300, best_params.get("n_estimators", 300)),
        max_depth=min(14, best_params.get("max_depth", 14)) if best_params.get("max_depth", None) else None,
        min_samples_split=best_params.get("min_samples_split", 5),
        min_samples_leaf=best_params.get("min_samples_leaf", 2),
        max_features=best_params.get("max_features", 0.7),
        bootstrap=best_params.get("bootstrap", True),
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,  # important: no nested parallelism during GA fitness
    )

# =========================================
# GA feature selection (TRAIN-only, 3-fold)
# =========================================
def ga_select_features_cv(
    X_df: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    make_rf_fn,  # pass lambda: make_rf_for_ga(best_params)
    *,
    pop=16,                 # lighter defaults
    gens=15,
    cx_rate=0.8,
    mut_rate=0.08,
    k_tour=3,
    random_state=RANDOM_STATE,
    n_splits=3,
    patience=6             # early stop if no improvement
):
    """
    GA on TRAIN only with 3-fold CV accuracy as fitness.
    - Single-threaded RF inside fitness to prevent nested-parallel stalls.
    - Verbose per-generation progress + early stopping.
    """
    rng = np.random.RandomState(random_state)
    n = len(feature_names)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Precompute folds once; use row indices over X_df
    folds = list(skf.split(np.arange(len(X_df)), y))
    X_np = X_df.to_numpy(float)  # speed

    def ensure_nonempty(mask):
        if mask.sum() == 0:
            mask[rng.randint(0, n)] = 1
        return mask

    def init_pop():
        P = []
        for _ in range(pop):
            m = (rng.rand(n) < 0.5).astype(int)
            P.append(ensure_nonempty(m))
        return P

    feat_idx_map = np.arange(n)

    def fitness(mask):
        mask = ensure_nonempty(mask.copy())
        cols = feat_idx_map[mask.astype(bool)]
        scores = []
        for tri, vai in folds:
            clf = make_rf_fn()
            clf.fit(X_np[tri][:, cols], y[tri])
            pred = clf.predict(X_np[vai][:, cols])
            scores.append(accuracy_score(y[vai], pred))
        return float(np.mean(scores)), int(cols.size)

    def tournament(F):
        idxs = rng.choice(len(F), size=k_tour, replace=False)
        b = idxs[0]
        for i in idxs[1:]:
            if F[i] > F[b]:
                b = i
        return b

    def crossover(m1, m2):
        if n == 1 or rng.rand() > cx_rate:
            return m1.copy(), m2.copy()
        pt = rng.randint(1, n)
        return np.concatenate([m1[:pt], m2[pt:]]), np.concatenate([m2[:pt], m1[pt:]])

    def mutate(m):
        flips = rng.rand(n) < mut_rate
        child = m.copy()
        child[flips] = 1 - child[flips]
        return ensure_nonempty(child)

    cache = {}
    def key(m): return "".join(map(str, m.tolist()))
    def eval_pop(P):
        F, K = [], []
        for m in P:
            km = key(m)
            if km in cache:
                f, k = cache[km]
            else:
                f, k = fitness(m)
                cache[km] = (f, k)
            F.append(f); K.append(k)
        return np.array(F, float), np.array(K, int)

    # --- GA loop
    P = init_pop()
    print(f"\n[GA] Starting feature selection (pop={pop}, gens={gens}, n_features={n})...")
    F, K = eval_pop(P)
    best_score = -np.inf
    gens_since_improve = 0

    for g in range(1, gens + 1):
        # Elitism
        elite_idx = np.argsort(-F)[:2]
        elites = [P[i].copy() for i in elite_idx]

        # Next population
        nextP = elites.copy()
        while len(nextP) < pop:
            p1 = P[tournament(F)]
            p2 = P[tournament(F)]
            c1, c2 = crossover(p1, p2)
            nextP.append(mutate(c1))
            if len(nextP) < pop:
                nextP.append(mutate(c2))

        P = nextP
        F, K = eval_pop(P)

        g_best_idx = int(np.argmax(F))
        g_best = float(F[g_best_idx])
        g_avg = float(np.mean(F))
        g_feats = int(K[g_best_idx])

        print(f"  Gen {g:02d}/{gens}: best={g_best:.4f}, avg={g_avg:.4f}, features={g_feats}")

        if g_best > best_score + 1e-6:
            best_score = g_best
            gens_since_improve = 0
        else:
            gens_since_improve += 1
            if gens_since_improve >= patience:
                print(f"  Early stopping: no improvement for {patience} generations.")
                break

    # Final best
    best = int(np.argmax(F))
    best_mask = P[best]
    selected_cols = [name for name, bit in zip(feature_names, best_mask) if bit]
    print(f"\n[GA] Done. Best CV acc={F[best]:.4f} with {len(selected_cols)} features.")
    return selected_cols, float(F[best])

# =========================================
# TRAIN-only hyperparameter tuning (3-fold)
# =========================================
def tune_rf_hyperparams_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = RANDOM_STATE,
    n_iter: int = 200,
    scoring: str = "accuracy"
) -> dict:
    """
    RandomizedSearchCV over RF on TRAIN only with StratifiedKFold(n_splits=3).
    Returns best_params_.
    """
    rf_base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": [150, 200, 300, 400, 500, 700, 900, 1200, 1700],
        "max_depth": [None, 10, 12, 14, 16, 18, 24, 30],
        "min_samples_split": [2, 5, 7, 9, 10, 12],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
        "bootstrap": [True, False],
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        rf_base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=2,
        refit=True  # refit best on full TRAIN at the end (still TRAIN-only)
    )
    search.fit(X_train, y_train)

    print("\n===== TRAIN-only CV (3-fold) best params =====")
    print(search.best_params_)
    print(f"Mean CV accuracy: {search.best_score_:.4f}")

    return search.best_params_

# ================
# Main flow
# ================
def main():
    # 1) Load splits
    df_train_raw = pd.read_csv(TRAIN_CSV)
    df_val_raw   = pd.read_csv(VAL_CSV)
    df_test_raw  = pd.read_csv(TEST_CSV)

    for name, d in [("train", df_train_raw), ("val", df_val_raw), ("test", df_test_raw)]:
        if SPECIES_COL not in d.columns:
            raise ValueError(f"{name} CSV missing required column '{SPECIES_COL}'")

    cols_train = _numeric_cols(df_train_raw)
    cols_val   = _numeric_cols(df_val_raw)
    cols_test  = _numeric_cols(df_test_raw)
    if cols_train != cols_val or cols_train != cols_test:
        raise ValueError("Numeric axis headers differ across splits; align the CSV headers.")

    temp_cols = cols_train
    x_axis = np.array([float(c) for c in temp_cols], dtype=float)

    print(f"Loaded: train={len(df_train_raw)}, val={len(df_val_raw)}, test={len(df_test_raw)}")
    print(f"Species (train): {df_train_raw[SPECIES_COL].nunique()} | (all): {pd.concat([df_train_raw, df_val_raw, df_test_raw])[SPECIES_COL].nunique()}")

    # 2) Build 36 features per split
    feat_train = build_features_for_split(df_train_raw, temp_cols)
    feat_val   = build_features_for_split(df_val_raw,   temp_cols)
    feat_test  = build_features_for_split(df_test_raw,  temp_cols)

    print(f"Feature shapes: TRAIN={feat_train.shape}, VAL={feat_val.shape}, TEST={feat_test.shape}")

    # 3) Encode labels (consistent across all)
    le = LabelEncoder()
    _ = le.fit(pd.concat([feat_train[SPECIES_COL], feat_val[SPECIES_COL], feat_test[SPECIES_COL]]))
    y_train = le.transform(feat_train[SPECIES_COL])
    y_val   = le.transform(feat_val[SPECIES_COL])
    y_test  = le.transform(feat_test[SPECIES_COL])

    # 4) TRAIN-only hyperparam tuning (3-fold) on ALL 36 features
    X_train_all = feat_train.drop(columns=[SPECIES_COL]).to_numpy(float)
    feature_names_all = feat_train.drop(columns=[SPECIES_COL]).columns.tolist()

    best_params = tune_rf_hyperparams_cv(
        X_train=X_train_all,
        y_train=y_train,
        random_state=RANDOM_STATE,
        n_iter=200,
        scoring="accuracy"
    )

    # 5) GA feature selection (TRAIN-only, 3-fold) using tuned RF factory (cheaper for GA)
    X_train_df = feat_train.drop(columns=[SPECIES_COL])
    def make_rf_ga(): return make_rf_for_ga(best_params)

    selected_cols, best_cv = ga_select_features_cv(
        X_df=X_train_df,
        y=y_train,
        feature_names=feature_names_all,
        make_rf_fn=make_rf_ga,
        pop=16, gens=15, cx_rate=0.80, mut_rate=0.08,
        k_tour=3, random_state=RANDOM_STATE, n_splits=3, patience=6
    )

    print(f"\n[GA] Selected {len(selected_cols)} features (TRAIN-only CV mean acc = {best_cv:.4f}):")
    print(", ".join(selected_cols))

    # 6) Train-only calibrated model (for ensemble tuning on VAL)
    Xtr_sel = feat_train[selected_cols].to_numpy(float)
    rf_train_only = make_rf_from_params(best_params)  # full params for final fit
    clf_train_only = CalibratedClassifierCV(rf_train_only, cv=3, method="isotonic")
    clf_train_only.fit(Xtr_sel, y_train)

    train_only_artifact = {
        "estimator": clf_train_only,
        "classes": le.inverse_transform(np.arange(len(le.classes_))).tolist(),
        "feature_names": selected_cols,
        "model_type": "extratrees_calibrated",
        "params": best_params,
        "random_state": RANDOM_STATE,
        "meta": {
            "cv_folds": 3,
            "selection": "GA on TRAIN (3-fold CV)",
            "note": "Use this artifact for ensemble VAL weight tuning (no VAL seen during fit)."
        }
    }
    dump(train_only_artifact, "ensemble_extratrees_calibrated_TRAINONLY.joblib")
    print("Saved TRAIN-only ensemble artifact -> ensemble_extratrees_calibrated_TRAINONLY.joblib")

    # 7) Final TRAIN+VAL calibrated model (for evaluation & deploy)
    feat_trainval = pd.concat([feat_train, feat_val], ignore_index=True)
    X_tv_sel = feat_trainval[selected_cols].to_numpy(float)
    y_tv     = le.transform(feat_trainval[SPECIES_COL])

    rf_tv = make_rf_from_params(best_params)
    clf_tv = CalibratedClassifierCV(rf_tv, cv=3, method="isotonic")
    clf_tv.fit(X_tv_sel, y_tv)

    final_artifact = {
        "estimator": clf_tv,
        "classes": le.inverse_transform(np.arange(len(le.classes_))).tolist(),
        "feature_names": selected_cols,
        "model_type": "extratrees_calibrated",
        "params": best_params,
        "random_state": RANDOM_STATE,
        "meta": {
            "cv_folds": 3,
            "selection": "GA on TRAIN (3-fold CV), then refit on TRAIN+VAL",
            "note": "This is the final model used for TEST evaluation and deployment."
        }
    }
    dump(final_artifact, "ensemble_extratrees_calibrated.joblib")
    print("Saved TRAIN+VAL ensemble artifact -> ensemble_extratrees_calibrated.joblib")

    # 8) TEST evaluation (single, clean) with TRAIN+VAL model
    X_te_sel = feat_test[selected_cols].to_numpy(float)
    y_pred = clf_tv.predict(X_te_sel)
    y_pred_proba = clf_tv.predict_proba(X_te_sel)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nFINAL TEST accuracy (held-out): {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Per-class metrics on TEST
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_test, y_pred, labels=np.arange(len(le.classes_)), zero_division=0
    )
    per_class = pd.DataFrame({
        'Species': le.inverse_transform(np.arange(len(le.classes_))),
        'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'Support': supp
    }).sort_values('Support', ascending=False)

    print(f"\nmacro avg metrics ({len(le.classes_)} species):")
    print(f"  precision: {per_class['Precision'].mean():.4f}")
    print(f"  recall:    {per_class['Recall'].mean():.4f}")
    print(f"  f1:        {per_class['F1-Score'].mean():.4f}")
    print("\ntop 10 species performance (by support):")
    print(per_class.head(10).to_string(index=False))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_norm, annot=False, cmap='Blues', square=True,
                xticklabels=le.inverse_transform(np.arange(len(le.classes_))),
                yticklabels=le.inverse_transform(np.arange(len(le.classes_))),
                cbar_kws={'label': 'accuracy'})
    plt.xlabel('predicted', fontsize=12, fontweight='bold')
    plt.ylabel('true', fontsize=12, fontweight='bold')
    plt.title(f'confusion matrix (TEST)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save supporting artifacts / CSVs (optional)
    with open(results_dir / 'trained_model.pkl', 'wb') as f:
        pickle.dump(clf_tv, f)

    with open(results_dir / 'base_extratrees.pkl', 'wb') as f:
        pickle.dump(make_rf_from_params(best_params), f)

    bundle = {
        "model": clf_tv,
        "feature_names": selected_cols,
        "model_type": "optimized_extratrees_calibrated",
        "test_accuracy": float(test_acc),
        "params": best_params
    }
    with open(results_dir / 'model_bundle.pkl', 'wb') as f:
        pickle.dump(bundle, f)

    meta = {
        'features': selected_cols,
        'classes': le.inverse_transform(np.arange(len(le.classes_))).tolist(),
        'n_features': len(selected_cols),
        'n_classes': len(le.classes_),
        'test_acc': float(test_acc),
        'best_params': best_params,
        'model_type': 'optimized_extratrees',
        'cv_note': "Hyperparameters and GA selected using TRAIN-only 3-fold CV"
    }
    with open(results_dir / 'model_metadata.pkl', 'wb') as f:
        pickle.dump(meta, f)

    # (Optional) Feature importances from TRAIN+VAL final ET (using all 36) for CSV
    et_all = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=9,
        min_samples_leaf=1,
        max_features=0.7,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    et_all.fit(feat_trainval.drop(columns=[SPECIES_COL]).to_numpy(float), y_tv)
    imp_df = pd.DataFrame({
        "Feature": feat_trainval.drop(columns=[SPECIES_COL]).columns,
        "Importance": et_all.feature_importances_
    }).sort_values("Importance", ascending=False)
    imp_df.to_csv(results_dir / "feature_importance.csv", index=False)
    per_class.to_csv(results_dir / "per_class_metrics.csv", index=False)

    print("\nSaved artifacts and reports:")
    print("  - ensemble_extratrees_calibrated_TRAINONLY.joblib  (use for ensemble weight tuning)")
    print("  - ensemble_extratrees_calibrated.joblib            (final model for TEST/deploy)")
    print("  - results/trained_model.pkl, base_extratrees.pkl, model_bundle.pkl, model_metadata.pkl")
    print("  - results/feature_importance.csv, per_class_metrics.csv")
    print("  - results/confusion_matrix_test.png")
    print("Done.")

if __name__ == "__main__":
    main()
