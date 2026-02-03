# basic_ensemble.py
#
# Usage:
#   python basic_ensemble.py shark_training_data.csv shark_validation_data.csv shark_test_data.csv
# Options:
#   --reject-thresh 0.85     # confidence threshold for rejection (default: None -> no rejection)
#   --outdir results         # where to save confusion-matrix plots (default: current folder)
#
#   (A) TRAIN-ONLY fit for each base -> get VAL probabilities
#   (B) Tune blend weight w on VAL by log loss (tie-break by accuracy)
#   (C) Refit BOTH bases on TRAIN+VAL (same settings), evaluate ensemble on TEST
import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, classification_report

# --- Import *helpers* (safe) ---
from GaussianCurveCV import (
    build_features_with_axis,
    CALIBRATE, RANDOM_STATE
)
from rf_classifier import (
    load_raw_csv_and_prepare,  # 14-feature extraction
    Normalizer14               # normalizer fitted on TRAIN only
)

# --------------------------
# Utility helpers
# --------------------------
def align_probas(p_mat, src_classes, tgt_classes):
    idx_map = {c: i for i, c in enumerate(src_classes)}
    out = np.zeros((p_mat.shape[0], len(tgt_classes)), dtype=float)
    for j, c in enumerate(tgt_classes):
        i = idx_map.get(c, None)
        if i is not None:
            out[:, j] = p_mat[:, i]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s

def soft_blend(p1, p2, w):
    return w * p1 + (1.0 - w) * p2

def make_gaussian_model():
    base_rf = RandomForestClassifier(
        n_estimators=800,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
    )
    if CALIBRATE:
        return CalibratedClassifierCV(base_rf, cv=3, method="isotonic")
    return base_rf

def tune_rf_hyperparams(X_train, y_train, X_val, y_val, random_state=RANDOM_STATE, n_iter=200):
    rf_base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    param_dist = {
        "n_estimators": [150, 200, 250, 300, 350, 400, 450, 500],
        "max_depth": [None, 10, 12, 14, 16, 18, 20, 25, 30],
        "min_samples_split": [2, 5, 7, 10, 12],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.3, 0.4, 0.5, 0.7, 1.0],
        "bootstrap": [True, False],
    }
    X_concat = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)], axis=0).reset_index(drop=True)
    y_concat = np.concatenate([y_train, y_val], axis=0)

    test_fold = np.array([-1]*len(y_train) + [0]*len(y_val))
    ps = PredefinedSplit(test_fold)

    search = RandomizedSearchCV(
        rf_base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=ps,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        refit=False
    )
    search.fit(X_concat, y_concat)
    return search.best_params_

def ga_select_features(Xtr, ytr, Xv, yv, feature_names, make_rf_fn, *,
                       pop=25, gens=25, k_tour=3, cx_rate=0.80, mut_rate=0.08, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    n = len(feature_names)

    def ensure_nonempty(m):
        if m.sum() == 0:
            m[rng.randint(0, n)] = 1
        return m

    def init_pop():
        P = []
        for _ in range(pop):
            m = (rng.rand(n) < 0.5).astype(int)
            P.append(ensure_nonempty(m))
        return P

    def subset(X, m):
        cols = [name for name, bit in zip(feature_names, m) if bit]
        return X[cols].values, cols

    def fitness(m):
        m = ensure_nonempty(m.copy())
        Xtr_s, _ = subset(pd.DataFrame(Xtr, columns=feature_names), m)
        Xv_s,  _ = subset(pd.DataFrame(Xv,  columns=feature_names), m)
        clf = make_rf_fn()
        clf.fit(Xtr_s, ytr)
        pred = clf.predict(Xv_s)
        return accuracy_score(yv, pred)

    def tour_select(P, F):
        idxs = rng.choice(len(P), size=k_tour, replace=False)
        best = idxs[0]
        for i in idxs[1:]:
            if F[i] > F[best]:
                best = i
        return best

    def crossover(m1, m2):
        if n == 1 or rng.rand() > cx_rate:
            return m1.copy(), m2.copy()
        pt = rng.randint(1, n)
        c1 = np.concatenate([m1[:pt], m2[pt:]])
        c2 = np.concatenate([m2[:pt], m1[pt:]])
        return c1, c2

    def mutate(m):
        flips = rng.rand(n) < mut_rate
        child = m.copy()
        child[flips] = 1 - child[flips]
        return ensure_nonempty(child)

    P = init_pop()
    cache = {}
    def key(m): return "".join(map(str, m.tolist()))
    def eval_pop(P):
        F = []
        for m in P:
            km = key(m)
            if km in cache:
                F.append(cache[km])
            else:
                f = fitness(m)
                cache[km] = f
                F.append(f)
        return np.array(F, float)

    F = eval_pop(P)
    for _ in range(gens):
        elite_idx = np.argsort(-F)[:2]
        elites = [P[i].copy() for i in elite_idx]
        nextP = elites.copy()
        while len(nextP) < pop:
            p1 = P[tour_select(P, F)]
            p2 = P[tour_select(P, F)]
            c1, c2 = crossover(p1, p2)
            nextP.append(mutate(c1))
            if len(nextP) < pop:
                nextP.append(mutate(c2))
        P = nextP
        F = eval_pop(P)

    best = int(np.argmax(F))
    best_mask = P[best]
    selected_cols = [name for name, bit in zip(feature_names, best_mask) if bit]
    return selected_cols, float(F[best])

# --------------------------
# Plotting helpers
# --------------------------
def _row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums

def plot_cm(y_true, y_pred, labels, title, out_path, normalize=True, figsize=(10, 6)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = _row_normalize(cm) if normalize else cm

    plt.figure(figsize=figsize, dpi=140)
    ax = sns.heatmap(
        cmn,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv")
    parser.add_argument("val_csv")
    parser.add_argument("test_csv")
    parser.add_argument("--reject-thresh", type=float, default=None,
                        help="If set, predictions with max prob < thresh are mapped to <REJECT>.")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Directory to save confusion-matrix images.")
    args = parser.parse_args()

    train_csv, val_csv, test_csv = args.train_csv, args.val_csv, args.test_csv

    # ---------------- TRAIN/VAL/TEST raw
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)
    df_test  = pd.read_csv(test_csv)

    # =========================
    # A) TRAIN-ONLY fits -> VAL probabilities
    # =========================

    # ---- A1: Gaussian path (TRAIN -> VAL)
    temp_cols = sorted([c for c in df_train.columns if c != "Species"], key=lambda c: float(c))
    Xg_tr, yg_tr = build_features_with_axis(df_train, temp_cols)
    Xg_va, yg_va = build_features_with_axis(df_val,   temp_cols)

    g_tr = make_gaussian_model()
    g_tr.fit(Xg_tr, yg_tr)
    p_g_val = g_tr.predict_proba(Xg_va)
    classes_g = list(g_tr.classes_)

    # ---- A2: RF+GA path (TRAIN -> VAL)
    tr14 = load_raw_csv_and_prepare(train_csv)
    va14 = load_raw_csv_and_prepare(val_csv)
    te14 = load_raw_csv_and_prepare(test_csv)

    normalizer = Normalizer14().fit(tr14)
    trN = normalizer.transform(tr14)
    vaN = normalizer.transform(va14)
    teN = normalizer.transform(te14)

    Xr_tr_df = trN.drop(columns=["Species"])
    Xr_va_df = vaN.drop(columns=["Species"])
    feature_names = Xr_tr_df.columns.tolist()

    species = sorted(pd.concat([tr14["Species"], va14["Species"], te14["Species"]]).unique())
    lab2idx = {s:i for i, s in enumerate(species)}
    y_r_tr = tr14["Species"].map(lab2idx).values
    y_r_va = va14["Species"].map(lab2idx).values
    classes_r = species[:]

    best_params = tune_rf_hyperparams(Xr_tr_df.values, y_r_tr, Xr_va_df.values, y_r_va)

    def make_rf():
        return RandomForestClassifier(
            **best_params,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    selected_cols, _ = ga_select_features(
        Xr_tr_df.values, y_r_tr, Xr_va_df.values, y_r_va,
        feature_names, make_rf
    )
    rf_tr = make_rf()
    rf_tr.fit(Xr_tr_df[selected_cols].values, y_r_tr)
    p_r_val = rf_tr.predict_proba(Xr_va_df[selected_cols].values)

    # =========================
    # B) Tune blend weight on VAL (log loss, tie-break accuracy)
    # =========================
    classes_all = sorted(set(classes_g).union(set(classes_r)))
    y_val_str = df_val["Species"].astype(str).values

    p_g_val_a = align_probas(p_g_val, classes_g, classes_all)
    p_r_val_a = align_probas(p_r_val, classes_r, classes_all)

    ws = np.linspace(0.0, 1.0, 51)
    best = {"w": None, "logloss": np.inf, "acc": -1.0}
    for w in ws:
        p_blend = soft_blend(p_g_val_a, p_r_val_a, w)
        ll = log_loss(y_val_str, p_blend, labels=classes_all)
        pred = np.array(classes_all)[np.argmax(p_blend, axis=1)]
        acc = accuracy_score(y_val_str, pred)
        if (ll < best["logloss"]) or (np.isclose(ll, best["logloss"]) and acc > best["acc"]):
            best = {"w": float(w), "logloss": float(ll), "acc": float(acc)}

    print(f"Best VAL blend: w={best['w']:.2f} | logloss={best['logloss']:.4f} | acc={best['acc']:.4f}")

    # =========================
    # C) Refit bases on TRAIN+VAL, evaluate on TEST
    # =========================

    # ---- C1: Gaussian refit on TRAIN+VAL
    Xg_tv = np.vstack([Xg_tr, Xg_va])
    yg_tv = np.concatenate([yg_tr, yg_va])
    g_tv = make_gaussian_model()
    g_tv.fit(Xg_tv, yg_tv)

    Xg_te, yg_te = build_features_with_axis(df_test, temp_cols)
    p_g_test = g_tv.predict_proba(Xg_te)
    classes_g_tv = list(g_tv.classes_)

    # ---- C2: RF+GA refit on TRAIN+VAL (same params and subset)
    Xr_te_df = teN.drop(columns=["Species"])
    Xr_tv_df = pd.concat([Xr_tr_df, Xr_va_df], axis=0).reset_index(drop=True)
    y_r_tv   = np.concatenate([y_r_tr, y_r_va])

    rf_tv = make_rf()
    rf_tv.fit(Xr_tv_df[selected_cols].values, y_r_tv)
    p_r_test = rf_tv.predict_proba(Xr_te_df[selected_cols].values)

    # ---- Blend & report on TEST
    p_g_test_a = align_probas(p_g_test, classes_g_tv, classes_all)
    p_r_test_a = align_probas(p_r_test, classes_r,     classes_all)

    p_test = soft_blend(p_g_test_a, p_r_test_a, best["w"])
    y_pred = np.array(classes_all)[np.argmax(p_test, axis=1)]
    y_test = df_test["Species"].astype(str).values

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (leak-free ensemble): {test_acc:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=classes_all)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=classes_all, columns=classes_all))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=classes_all, zero_division=0))

    # =======================================
    # D) (NEW) Confidence-based rejection
    # =======================================
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Base model confusion matrix (no rejects) — plot
    base_title = f"Normalized Confusion Matrix - Ensemble | Test Acc = {test_acc:.3f}"
    base_path = os.path.join(args.outdir, "cm_base.png")
    plot_cm(y_test, y_pred, classes_all, base_title, base_path, normalize=True)
    print(f"\nSaved base confusion matrix to: {base_path}")

    # 2) With rejection (if threshold provided)
    if args.reject_thresh is not None:
        reject_label = "<REJECT>"

        max_conf = p_test.max(axis=1)
        y_pred_thr = y_pred.copy()
        y_pred_thr[max_conf < float(args.reject_thresh)] = reject_label

        labels_with_reject = classes_all + [reject_label]

        kept_mask = (y_pred_thr != reject_label)
        coverage = kept_mask.mean()
        selective_acc = accuracy_score(y_test[kept_mask], y_pred_thr[kept_mask]) if coverage > 0 else 0.0

        print(f"\nRejection threshold = {args.reject_thresh:.3f}")
        print(f"Coverage (not rejected): {coverage:.3f}  |  Selective accuracy: {selective_acc:.4f}")

        rej_title = (f"Normalized Confusion Matrix - Ensemble + Reject @ {args.reject_thresh:.2f} | "
                     f"Coverage={coverage:.2f}, SelAcc={selective_acc:.3f}")
        rej_path = os.path.join(args.outdir, "cm_with_rejects.png")
        plot_cm(y_test, y_pred_thr, labels_with_reject, rej_title, rej_path, normalize=True)
        print(f"Saved confusion matrix (with rejects) to: {rej_path}")
    else:
        print("\nNo rejection threshold provided. To enable, pass --reject-thresh, e.g. --reject-thresh 0.85")


if __name__ == "__main__":
    main()
