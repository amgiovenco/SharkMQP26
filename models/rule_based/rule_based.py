# rule_based.py
# End-to-end multiclass thresholding pipeline for melting-curve data.

import os, json, sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

# ---------------------------
# 1) Load & row-wise cleaning
# ---------------------------

def load_dataset(csv_path: str, target_name: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV, assume the first column is the target if target_name is None.
    Cleans features: numeric coercion, inf->NaN, row-wise interpolate, fill edges.
    """
    # More robust read in case delimiter isn't a comma
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding_errors="replace")

    if target_name is None:
        target_name = df.columns[0]

    X = df.drop(columns=[target_name]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Interpolate along each row (sequence-wise); then fill edges
    X = X.interpolate(axis=1, limit_direction="both")
    X = X.fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)

    y = df[target_name].astype(str)
    return X, y


# -------------------------------------------------
# 2) Compact curve features (fast & interpretable)
# -------------------------------------------------

def _curve_features(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Engineer ~14 features from a single curve y(t).
    """
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    # Robust baseline: median of first 5% of points
    k = max(1, int(0.05 * len(y)))
    baseline = np.median(y[:k])
    yb = y - baseline
    yb = np.clip(yb, 0.0, None)  # no negative area

    # Peak
    idx_max = int(np.argmax(yb))
    ymax = float(yb[idx_max])
    tmax = float(t[idx_max])

    # Area & centroid
    auc = float(np.trapz(yb, t))
    centroid = float(np.trapz(yb * t, t) / (auc + 1e-12))

    # FWHM
    half = 0.5 * ymax
    above = np.where(yb >= half)[0]
    fwhm = float(t[above[-1]] - t[above[0]]) if above.size > 0 else 0.0

    # Rise/decay (10%->90% and 90%->10%)
    def cross(level):
        idx = np.where(yb >= level)[0]
        return (int(idx[0]) if idx.size else idx_max), (int(idx[-1]) if idx.size else idx_max)

    lo, hi = 0.1 * ymax, 0.9 * ymax
    lo_i1, hi_i1 = cross(lo)[0], cross(hi)[0]
    lo_i2, hi_i2 = cross(lo)[1], cross(hi)[1]
    rise_time = float(t[max(hi_i1, lo_i1)] - t[min(hi_i1, lo_i1)]) if ymax > 0 else 0.0
    decay_time = float(t[max(lo_i2, hi_i2)] - t[min(lo_i2, hi_i2)]) if ymax > 0 else 0.0

    # Left/right area & asymmetry
    auc_left = float(np.trapz(yb[:idx_max + 1], t[:idx_max + 1]))
    auc_right = float(np.trapz(yb[idx_max:], t[idx_max:]))
    asymmetry = float((auc_right - auc_left) / (auc + 1e-12))

    # Global stats on raw y
    mean_val = float(np.mean(y))
    std_val = float(np.std(y))
    max_val = float(np.max(y))
    min_val = float(np.min(y))

    return np.array([
        ymax, tmax, auc, centroid, fwhm, rise_time, decay_time,
        auc_left, auc_right, asymmetry, mean_val, std_val, max_val, min_val
    ], dtype=float)

def engineer_features(X_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert full sequence to compact 14-dim curve features.
    """
    # Ensure columns are castable to float temperatures/time
    t = X_raw.columns.astype(float).to_numpy()
    M = X_raw.to_numpy(float)
    F = np.vstack([_curve_features(M[i, :], t) for i in range(M.shape[0])])
    names = [
        "ymax", "tmax", "auc", "centroid", "fwhm", "rise", "decay",
        "auc_left", "auc_right", "asym", "mean", "std", "max", "min"
    ]
    return pd.DataFrame(F, columns=names), names


# ----------------------------
# 3) Train a quick baseline
# ----------------------------

def train_baseline(Xf: pd.DataFrame, y: pd.Series, model: str = "rf", seed: int = 8):
    """
    Train RF or LR on engineered features.
    Splits: 60% train, 20% val, 20% test (stratified).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    Xtr_te, Xte, ytr_te, yte = train_test_split(
        Xf, y_enc, test_size=0.20, random_state=seed, stratify=y_enc
    )
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr_te, ytr_te, test_size=0.25, random_state=seed, stratify=ytr_te
    )  # 0.25 of 0.8 = 0.2

    if model == "rf":
        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1
            )
        )
    else:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                solver="lbfgs"
            )
        )

    clf.fit(Xtr, ytr)
    return clf, le, (Xtr, ytr, Xva, yva, Xte, yte)


# --------------------------------------
# 4) Per-class thresholds & margin rule
# --------------------------------------

def classwise_thresholds_by_f1(proba_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    For each class c, sweep thresholds on validation to maximize F1 in a one-vs-rest manner.
    Returns vector of class-specific thresholds.
    """
    C = proba_val.shape[1]
    thr = np.zeros(C)
    for c in range(C):
        y_one = (y_val == c).astype(int)
        p, r, t = precision_recall_curve(y_one, proba_val[:, c])
        if t.size == 0:
            thr[c] = 0.5
        else:
            f1 = (2 * p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-12)
            thr[c] = t[int(np.argmax(f1))]
    return thr

def predict_with_thresholds(proba: np.ndarray, thr: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    Accept top-1 class only if its probability >= per-class threshold and (optionally)
    top1 - top2 >= margin. Otherwise return -1 (abstain).
    """
    top1 = np.argmax(proba, axis=1)
    top1v = proba[np.arange(len(proba)), top1]

    if margin > 0:
        srt = np.argsort(-proba, axis=1)
        top2 = srt[:, 1]
        top2v = proba[np.arange(len(proba)), top2]
        ok_margin = (top1v - top2v) >= margin
    else:
        ok_margin = np.ones_like(top1v, dtype=bool)

    ok_thr = top1v >= thr[top1]
    return np.where(ok_thr & ok_margin, top1, -1)

def accepted_accuracy(decisions: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """
    Accuracy over accepted samples and abstain rate.
    """
    mask = decisions != -1
    if mask.sum() == 0:
        return 0.0, 1.0
    return accuracy_score(y_true[mask], decisions[mask]), 1.0 - mask.mean()


# ---------------------------------------------
# 5) Evaluation helpers (plain vs rule-based)
# ---------------------------------------------

def eval_plain(clf, X, y_true, split_name: str):
    """Evaluate the raw classifier (always argmax)."""
    y_pred = clf.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "split": split_name,
        "mode": "plain",
        "accuracy": acc,
        "macro_f1": f1m,
        "n_samples": int(len(y_true)),
    }

def eval_rule_based(proba, y_true, thresholds, margin: float, split_name: str):
    """Evaluate the rule-based classifier (per-class thresholds + margin)."""
    decisions = predict_with_thresholds(proba, thresholds, margin=margin)
    mask = decisions != -1
    abstain_rate = 1.0 - mask.mean()
    acc_accept = accuracy_score(y_true[mask], decisions[mask]) if mask.any() else 0.0
    f1m_accept = f1_score(y_true[mask], decisions[mask], average="macro", zero_division=0) if mask.any() else 0.0
    return {
        "split": split_name,
        "mode": "rule_based",
        "accepted_accuracy": acc_accept,
        "accepted_macro_f1": f1m_accept,
        "abstain_rate": abstain_rate,
        "n_samples": int(len(y_true)),
        "n_accepted": int(mask.sum()),
    }

def pretty_print_eval_table(rows, title="=== Evaluation Summary ==="):
    df = pd.DataFrame(rows)
    print("\n" + title)
    cols = [c for c in [
        "split","mode","accuracy","macro_f1","accepted_accuracy","accepted_macro_f1",
        "abstain_rate","n_samples","n_accepted"
    ] if c in df.columns]
    print(df[cols].to_string(index=False, float_format="%.3f"))


# -----------------------------------------
# 6) Human-readable rules via small tree
# -----------------------------------------

def extract_rules(Xf: pd.DataFrame, y_enc: np.ndarray, feature_names: List[str], max_depth: int = 3) -> str:
    """
    Fit a small decision tree and return human-readable if-then rules.
    """
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5, random_state=8)
    tree.fit(Xf, y_enc)
    rules = export_text(tree, feature_names=list(feature_names))
    return rules


# -------------------------
# 7) End-to-end pipeline
# -------------------------

def run_pipeline(csv_path: str, target_name: str = None,
                 model: str = "rf", margin: float = 0.1,
                 export_dir: str = None):
    """
    Train baseline, compute thresholds on validation, and evaluate
    on train/val/test (plain & rule-based). Optionally export artifacts.
    """
    X_raw, y = load_dataset(csv_path, target_name)
    Xf, names = engineer_features(X_raw)
    clf, le, (Xtr, ytr, Xva, yva, Xte, yte) = train_baseline(Xf, y, model=model)

    # Predict probabilities for all splits
    proba_tr = clf.predict_proba(Xtr)
    proba_va = clf.predict_proba(Xva)
    proba_te = clf.predict_proba(Xte)

    # Thresholds chosen on validation by F1 (one-vs-rest)
    thr = classwise_thresholds_by_f1(proba_va, yva)

    # Evaluate plain vs rule-based across splits
    rows = []
    rows.append(eval_plain(clf, Xtr, ytr, "train"))
    rows.append(eval_plain(clf, Xva, yva, "val"))
    rows.append(eval_plain(clf, Xte, yte, "test"))

    rows.append(eval_rule_based(proba_tr, ytr, thr, margin, "train"))
    rows.append(eval_rule_based(proba_va, yva, thr, margin, "val"))
    rows.append(eval_rule_based(proba_te, yte, thr, margin, "test"))

    # Optional: small tree for readable feature rules (fit on all data)
    rules = extract_rules(Xf, le.transform(y), names, max_depth=3)

    results = {
        "classes": list(le.classes_),
        "thresholds_f1": thr.tolist(),
        "eval_rows": rows,
        "rules_depth3": rules,
    }

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        pd.Series(thr, index=le.classes_).to_csv(os.path.join(export_dir, "thresholds_f1.csv"))
        pd.DataFrame(rows).to_csv(os.path.join(export_dir, "eval_summary.csv"), index=False)
        with open(os.path.join(export_dir, "rules_depth3.txt"), "w") as f:
            f.write(rules)
        with open(os.path.join(export_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results


# -------------------------
# 8) CLI entry point
# -------------------------

if __name__ == "__main__":
    import argparse

    # 👇 set your default CSV path here
    DEFAULT_CSV = Path(r"C:/Users/maxho/OneDrive/Documents/cs1004/Rule_based/cleaned_data_but_in_rows.csv")

    parser = argparse.ArgumentParser(description="Run multiclass thresholding pipeline on a melting-curve CSV.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                        help="Path to the CSV file (defaults to cleaned_data_but_in_rows.csv)")
    parser.add_argument("--model", default="rf", choices=["rf", "lr"],
                        help="Classifier: rf (RandomForest) or lr (LogisticRegression)")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Top1-Top2 probability margin for abstention")
    parser.add_argument("--export_dir", default=None,
                        help="Optional directory to export thresholds/rules/results")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    print(f"[info] Using CSV: {csv_path}")
    if not csv_path.exists():
        print(f"[error] File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out = run_pipeline(
        str(csv_path),
        target_name=None,
        model=args.model,
        margin=args.margin,
        export_dir=args.export_dir
    )

    # Pretty console output
    pretty_print_eval_table(out["eval_rows"], title="=== Train vs Val vs Test (Plain & Rule-based) ===")

    print("\n=== Per-Class Probability Thresholds (F1-optimized) ===")
    thr_df = pd.DataFrame({"Class": out["classes"], "Threshold": out["thresholds_f1"]})
    print(thr_df.to_string(index=False, float_format="%.3f"))

    print("\n=== Interpretable Rules (depth=3 tree) ===")
    print(out["rules_depth3"])
