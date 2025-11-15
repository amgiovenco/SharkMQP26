# --- Random Forest + Genetic Feature Selection (now with in-script feature extraction + normalization) ---
# Leak-free: all CV (hyperparam search + GA fitness) is done on TRAIN only.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# ========================= USER CONFIG =========================
# Point these to your RAW (pre-feature) CSVs
TRAIN_CSV = "shark_training_data.csv"
VAL_CSV   = "shark_validation_data.csv"
TEST_CSV  = "shark_test_data.csv"

RANDOM_STATE = 8
OPT_SCORING = "accuracy"
N_ITER_RANDOM_SEARCH = 200
CV_FOLDS = 3  # TRAIN-only CV folds (set to 3 for smallest class >= 3)

# Genetic Algorithm (TRAIN-only CV fitness) settings
GA_POPULATION = 25
GA_GENERATIONS = 25
GA_TOURNAMENT_K = 3
GA_CROSSOVER_RATE = 0.80
GA_MUTATION_RATE = 0.08
GA_ELITISM = 2  # keep best few each gen

np.random.seed(RANDOM_STATE)

# ========================= FEATURE EXTRACTION HELPERS =========================
def _interp_row(vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = vals.astype(float).copy()
    mask = np.isfinite(y)
    if mask.sum() == 0:
        return np.zeros_like(y)
    y[~mask] = np.interp(x[~mask], x[mask], y[mask])
    first, last = np.argmax(mask), len(y) - np.argmax(mask[::-1]) - 1
    y[:first] = y[first]
    y[last+1:] = y[last]
    return y

def _fwhm(t, yb):
    if yb.size == 0:
        return np.nan
    peak = yb.max()
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    half = 0.5 * peak
    above = yb >= half
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    return t[idx[-1]] - t[idx[0]]

def _cross_time(t, y, level, start, end):
    if end <= start:
        return np.nan
    seg_t = t[start:end+1]
    seg_y = y[start:end+1]
    if seg_y[0] <= seg_y[-1]:
        cond = seg_y >= level
    else:
        cond = seg_y <= level
    idx = np.argmax(cond)
    if not cond[idx]:
        return np.nan
    if idx == 0:
        return seg_t[0]
    t0, t1 = seg_t[idx-1], seg_t[idx]
    y0, y1 = seg_y[idx-1], seg_y[idx]
    if y1 == y0:
        return t1
    return t0 + (level - y0) * (t1 - t0) / (y1 - y0)

def _curve_features(y: np.ndarray, t: np.ndarray) -> dict:
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    orig_mean = float(np.nanmean(y))
    orig_std  = float(np.nanstd(y, ddof=0))
    orig_max  = float(np.nanmax(y))
    orig_min  = float(np.nanmin(y))

    finite_mask = np.isfinite(y)
    if not finite_mask.all():
        y = _interp_row(y, t)

    n = len(y)
    k = max(1, int(round(0.05 * n)))
    base = float(np.median(y[:k]))
    yb = np.clip(y - base, 0.0, None)

    i_max = int(np.argmax(yb))
    ymax = float(yb[i_max])
    tmax = float(t[i_max])

    auc = float(np.trapz(yb, t))
    centroid = float(np.trapz(yb * t, t) / auc) if auc > 0 else np.nan
    fwhm = float(_fwhm(t, yb))

    if ymax > 0 and i_max > 0:
        t10 = _cross_time(t, yb, 0.1 * ymax, 0, i_max)
        t90 = _cross_time(t, yb, 0.9 * ymax, 0, i_max)
        rise = (t90 - t10) if (np.isfinite(t10) and np.isfinite(t90)) else np.nan
    else:
        rise = np.nan

    if ymax > 0 and i_max < len(yb) - 1:
        t90d = _cross_time(t, yb, 0.9 * ymax, i_max, len(yb) - 1)
        t10d = _cross_time(t, yb, 0.1 * ymax, i_max, len(yb) - 1)
        decay = (t10d - t90d) if (np.isfinite(t90d) and np.isfinite(t10d)) else np.nan
    else:
        decay = np.nan

    auc_left  = float(np.trapz(yb[:i_max+1], t[:i_max+1])) if i_max > 0 else 0.0
    auc_right = float(np.trapz(yb[i_max:], t[i_max:])) if i_max < len(yb)-1 else 0.0
    asym = float((auc_right - auc_left) / auc) if auc > 0 else np.nan

    return {
        "ymax": ymax,
        "tmax": tmax,
        "auc": auc,
        "centroid": centroid,
        "fwhm": fwhm,
        "rise": float(rise) if np.isfinite(rise) else np.nan,
        "decay": float(decay) if np.isfinite(decay) else np.nan,
        "auc_left": auc_left,
        "auc_right": auc_right,
        "asym": asym,
        "mean": orig_mean,
        "std": orig_std,
        "max": orig_max,
        "min": orig_min,
    }

def extract_features_from_df(df: pd.DataFrame, species_col: str = "Species") -> pd.DataFrame:
    if species_col not in df.columns:
        raise ValueError(f"Expected a '{species_col}' column in the input data.")

    # numeric axis from headers (excluding Species)
    feature_cols = [c for c in df.columns if c != species_col]
    t_vals, kept_cols = [], []
    for c in feature_cols:
        try:
            t_vals.append(float(c))
            kept_cols.append(c)
        except Exception:
            pass
    if not kept_cols:
        raise ValueError("No numeric sequential columns found (headers should be numeric).")

    t = np.array(t_vals, dtype=float)
    order = np.argsort(t)
    t = t[order]
    kept_cols = [kept_cols[i] for i in order]

    mat = (
        df[kept_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=float)
    )

    for i in range(mat.shape[0]):
        row = mat[i]
        mask = np.isfinite(row)
        if not mask.all():
            row = np.interp(t, t[mask], row[mask])
            first, last = np.argmax(mask), len(row) - np.argmax(mask[::-1]) - 1
            row[:first] = row[first]
            row[last+1:] = row[last]
            mat[i] = row

    out_rows = []
    for i in range(mat.shape[0]):
        y = mat[i]
        out_rows.append(_curve_features(y, t))

    out_df = pd.DataFrame(out_rows)
    out_df.insert(0, "Species", df[species_col].values)

    col_order = [
        "Species",
        "ymax","tmax",
        "auc","centroid","fwhm","rise","decay","auc_left","auc_right","asym",
        "mean","std","max","min"
    ]
    return out_df[col_order]

# ========================= NORMALIZATION (fit on TRAIN, apply to others) =========================
class Normalizer14:
    """
    - Z-score on 'min' using training fit -> 'min_z'
    - log scaling on 'rise' and 'std' -> '{col}_log'
      (if nonpositive values exist in TRAIN, we compute shift = |min| + 1 and apply to all splits)
    - Drop originals: 'min','rise','std'
    """
    def __init__(self):
        self.min_scaler = StandardScaler()
        self.rise_shift = 0.0
        self.std_shift  = 0.0
        self.fitted = False

    def fit(self, df_train: pd.DataFrame):
        if (df_train['rise'] <= 0).any():
            self.rise_shift = float(abs(df_train['rise'].min()) + 1.0)
        if (df_train['std'] <= 0).any():
            self.std_shift = float(abs(df_train['std'].min()) + 1.0)
        self.min_scaler.fit(df_train[['min']])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "Normalizer14 must be fit on training data first."
        out = df.copy()
        out['min_z'] = self.min_scaler.transform(out[['min']])
        out['rise_log'] = np.log(out['rise'] + self.rise_shift) if self.rise_shift > 0 else np.log1p(out['rise'])
        out['std_log']  = np.log(out['std']  + self.std_shift)  if self.std_shift  > 0 else np.log1p(out['std'])
        out = out.drop(columns=['min','rise','std'])
        return out

# ========================= LOAD & PREPARE DATA =========================
def load_raw_csv_and_prepare(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    feats = extract_features_from_df(raw, species_col="Species")
    return feats

def save_model_artifact(path: str,
                        model,
                        normalizer: Normalizer14,
                        selected_features,
                        label_encoder: LabelEncoder,
                        all_feature_names,
                        random_state: int):
    """
    Save everything needed to reuse this model in an ensemble:
      - trained RF
      - Normalizer14
      - GA-selected feature subset
      - LabelEncoder
      - full feature name list (for sanity)
    """
    artifact = {
        "model": model,
        "normalizer": normalizer,
        "selected_features": selected_features,
        "label_encoder": label_encoder,
        "feature_names_all": all_feature_names,
        "random_state": random_state,
        "model_type": "rf_ga_leakfree"
    }
    dump(artifact, path)
    print(f"[save_model_artifact] Saved model artifact to: {path}")

def main():
    # 1) Extract features for each split
    train_feats = load_raw_csv_and_prepare(TRAIN_CSV)
    val_feats   = load_raw_csv_and_prepare(VAL_CSV)
    test_feats  = load_raw_csv_and_prepare(TEST_CSV)

    # 2) Fit normalizer on TRAIN features; apply to all splits (no leakage)
    normalizer = Normalizer14().fit(train_feats)
    train_df = normalizer.transform(train_feats)
    val_df   = normalizer.transform(val_feats)
    test_df  = normalizer.transform(test_feats)

    # 3) Prepare matrices
    X_train = train_df.drop(columns=["Species"])
    X_val   = val_df.drop(columns=["Species"])
    X_test  = test_df.drop(columns=["Species"])

    feature_names = X_train.columns.tolist()
    n_features = len(feature_names)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Species"])
    y_val   = le.transform(val_df["Species"])
    y_test  = le.transform(test_df["Species"])

    # ========================= TRAIN-ONLY RF HYPERPARAM TUNING (StratifiedKFold) =========================
    rf_base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": [150, 200, 250, 300, 350, 400, 450, 500, 700, 900, 1200],
        "max_depth": [None, 10, 12, 14, 16, 18, 20, 25, 30],
        "min_samples_split": [2, 5, 7, 9, 10, 12],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.3, 0.4, 0.5, 0.7, 1.0],
        "bootstrap": [True, False],
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        rf_base,
        param_distributions=param_dist,
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring=OPT_SCORING,
        cv=skf,                   # <-- TRAIN-only CV
        random_state=RANDOM_STATE,
        verbose=2,
        n_jobs=-1,
        refit=True               # refit best on full TRAIN at end (still TRAIN-only)
    )
    search.fit(X_train.values, y_train)
    best_params = search.best_params_
    print("\n===== TRAIN-only CV best RF params =====")
    print(best_params)
    print(f"Mean TRAIN-CV {OPT_SCORING}: {search.best_score_:.4f}")

    def make_rf():
        return RandomForestClassifier(
            **best_params,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    # ========================= TRAIN-ONLY GA FEATURE SELECTION (fitness = TRAIN 3-fold CV) =========================
    def ensure_nonempty(mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            idx = np.random.randint(0, n_features)
            mask[idx] = 1
        return mask

    def subset_from_mask(X: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        cols = [name for name, m in zip(feature_names, mask) if m]
        return X[cols], cols

    def cv_fitness_accuracy(mask: np.ndarray) -> float:
        """Mean accuracy over TRAIN-only CV using selected features."""
        mask = ensure_nonempty(mask.copy())
        Xsel, _ = subset_from_mask(X_train, mask)
        Xsel_np = Xsel.values
        scores = []
        for tri, vai in skf.split(Xsel_np, y_train):
            clf = make_rf()
            clf.fit(Xsel_np[tri], y_train[tri])
            preds = clf.predict(Xsel_np[vai])
            scores.append(accuracy_score(y_train[vai], preds))
        return float(np.mean(scores))

    def init_population(pop_size: int) -> list:
        pop = []
        for _ in range(pop_size):
            mask = (np.random.rand(n_features) < 0.5).astype(int)
            pop.append(ensure_nonempty(mask))
        return pop

    def tournament_select(pop_masks, pop_fits, k=GA_TOURNAMENT_K):
        idxs = np.random.choice(len(pop_masks), size=k, replace=False)
        best_i = idxs[0]
        best_fit = pop_fits[best_i]
        for i in idxs[1:]:
            if pop_fits[i] > best_fit:
                best_i = i
                best_fit = pop_fits[i]
        return best_i

    def single_point_crossover(m1: np.ndarray, m2: np.ndarray) -> tuple:
        if n_features == 1 or np.random.rand() > GA_CROSSOVER_RATE:
            return m1.copy(), m2.copy()
        point = np.random.randint(1, n_features)
        c1 = np.concatenate([m1[:point], m2[point:]])
        c2 = np.concatenate([m2[:point], m1[point:]])
        return c1, c2

    def mutate(mask: np.ndarray) -> np.ndarray:
        flips = np.random.rand(n_features) < GA_MUTATION_RATE
        child = mask.copy()
        child[flips] = 1 - child[flips]
        return ensure_nonempty(child)

    print("\n===== STARTING GENETIC FEATURE SELECTION (TRAIN-only CV fitness) =====")
    population = init_population(GA_POPULATION)
    fitness_cache = {}

    def mask_key(m: np.ndarray) -> str:
        return "".join(map(str, m.tolist()))

    def evaluate_population(pop):
        fits = []
        for m in pop:
            key = mask_key(m)
            if key in fitness_cache:
                fits.append(fitness_cache[key])
            else:
                f = cv_fitness_accuracy(m)
                fitness_cache[key] = f
                fits.append(f)
        return np.array(fits, dtype=float)

    fits = evaluate_population(population)

    for gen in range(GA_GENERATIONS):
        elite_idx = np.argsort(-fits)[:GA_ELITISM]
        elites = [population[i].copy() for i in elite_idx]

        next_pop = elites.copy()
        while len(next_pop) < GA_POPULATION:
            p1 = population[tournament_select(population, fits)]
            p2 = population[tournament_select(population, fits)]
            c1, c2 = single_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            next_pop.append(c1)
            if len(next_pop) < GA_POPULATION:
                next_pop.append(c2)

        population = next_pop
        fits = evaluate_population(population)

        best_gen_idx = int(np.argmax(fits))
        best_gen_fit = float(fits[best_gen_idx])
        best_gen_mask = population[best_gen_idx]
        selected_cols = [n for n, b in zip(feature_names, best_gen_mask) if b]
        print(
            f"Gen {gen+1:02d}/{GA_GENERATIONS} | "
            f"TRAIN-CV {OPT_SCORING}: {best_gen_fit:.4f} | "
            f"subset size: {best_gen_mask.sum():2d} | {selected_cols}"
        )

    best_idx = int(np.argmax(fits))
    best_mask = population[best_idx]
    best_cv_acc = float(fits[best_idx])
    SELECTED_FEATURES = [n for n, b in zip(feature_names, best_mask) if b]
    print("\n===== GA DONE (TRAIN-only CV) =====")
    print(f"Best TRAIN-CV accuracy: {best_cv_acc:.4f}")
    print(f"Selected {len(SELECTED_FEATURES)} / {n_features} features:")
    print(SELECTED_FEATURES)

    # ========================= FINAL FIT (TRAIN+VAL) & TEST EVAL =========================
    X_train_sel = X_train[SELECTED_FEATURES]
    X_val_sel   = X_val[SELECTED_FEATURES]
    X_tv_sel    = pd.concat([X_train_sel, X_val_sel], axis=0).reset_index(drop=True)
    y_tv        = np.concatenate([y_train, y_val], axis=0)
    X_test_sel  = X_test[SELECTED_FEATURES]

    final_rf = make_rf()
    final_rf.fit(X_tv_sel, y_tv)

    # --- SAVE LEAK-FREE MODEL ARTIFACT FOR ENSEMBLE USE ---
    save_model_artifact(
        path="rf_ga_model.joblib",
        model=final_rf,
        normalizer=normalizer,
        selected_features=SELECTED_FEATURES,
        label_encoder=le,
        all_feature_names=X_train.columns.tolist(),
        random_state=RANDOM_STATE,
    )

    y_pred = final_rf.predict(X_test_sel)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n=== FINAL TEST ACCURACY (RF + GA-selected features): {test_acc:.3f} ===\n")

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    species_names = le.inverse_transform(np.arange(len(le.classes_)))

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=species_names,
        yticklabels=species_names,
        cbar_kws={'label': 'Proportion of True Species'}
    )
    plt.title(f"Normalized Confusion Matrix - RF w/ GA Features | Test Accuracy = {test_acc:.3f}")
    plt.xlabel("Predicted Species")
    plt.ylabel("True Species")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=species_names, zero_division=0))

if __name__ == "__main__":
    main()
