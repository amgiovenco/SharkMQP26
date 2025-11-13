"""
svm_simple.py
-------------
Minimal SVM classifier for melting-curve (or similar) data WITH PCA feature extraction
and automatic threshold selection based on training data.

Key behavior:
- Visuals open interactively (no image files saved).
- Threshold is chosen on TRAIN (by default: max selective accuracy) and applied to TEST.
"""

# -------------------------------
# Standard library imports
# -------------------------------
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# -------------------------------
# Third-party imports
# -------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# -------------------------------
# Project-specific helpers
# -------------------------------
# Expects rule_based.py in the same folder (or on PYTHONPATH) with:
#   - load_dataset(csv_path, target_name)
#   - engineer_features(X_raw)
from rule_based import load_dataset, engineer_features


# ===============================
# Utilities
# ===============================

def split_data(Xf: pd.DataFrame, y: pd.Series, seed: int = 42):
    """Stratified 60/20/20 split with label encoding."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    Xtr_te, Xte, ytr_te, yte = train_test_split(
        Xf, y_enc, test_size=0.20, random_state=seed, stratify=y_enc
    )
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr_te, ytr_te, test_size=0.25, random_state=seed, stratify=ytr_te
    )
    return (Xtr, ytr, Xva, yva, Xte, yte, le)


def build_svm_pipeline_with_pca(
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    pca_components: float | int = 0.95,
    seed: int = 42,
):
    """Pipeline: StandardScaler -> PCA -> SVC(probability=True)."""
    clf = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components, random_state=seed),
        SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=seed),
    )
    return clf


def evaluate_split_basic(clf, X, y, split_name: str) -> Dict[str, Any]:
    """Accuracy, macro-F1, full per-class report, confusion matrix."""
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro", zero_division=0)
    report = classification_report(y, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()
    return {
        "split": split_name,
        "accuracy": acc,
        "macro_f1": f1m,
        "n_samples": int(len(y)),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def selective_metrics(y_true: np.ndarray, proba: np.ndarray, thresholds: List[float]) -> pd.DataFrame:
    """
    For each threshold t:
      - accept if max_proba >= t else reject
      - report rejection_rate and selective_accuracy (on accepted only)
    """
    max_proba = proba.max(axis=1)
    y_pred = proba.argmax(axis=1)
    rows = []
    n = len(y_true)
    for t in thresholds:
        accepted = max_proba >= t
        n_acc = int(accepted.sum())
        rej_rate = 1.0 - (n_acc / n)
        sel_acc = np.nan if n_acc == 0 else float((y_pred[accepted] == y_true[accepted]).mean())
        rows.append({
            "threshold": float(t),
            "rejection_rate": rej_rate,
            "n_rejected": n - n_acc,
            "n": n,
            "selective_accuracy": sel_acc
        })
    return pd.DataFrame(rows)


def choose_thresholds(df: pd.DataFrame, base_acc: float) -> Dict[str, Dict[str, float]]:
    """Report-style picks used for the printed summary on the report split."""
    under_10 = df[df["rejection_rate"] < 0.10].copy()
    best_perf = None
    if not under_10.empty:
        idx = under_10["selective_accuracy"].idxmax()
        best_perf = under_10.loc[idx]
    best_acc_idx = df["selective_accuracy"].idxmax()
    best_acc = df.loc[best_acc_idx]
    ok = df[df["selective_accuracy"] >= base_acc].copy()
    best_rej = None
    if not ok.empty:
        ok = ok.sort_values(by=["rejection_rate", "threshold"], ascending=[True, True])
        best_rej = ok.iloc[0]
    return {
        "best_perf_under_10": best_perf.to_dict() if best_perf is not None else None,
        "best_accuracy": best_acc.to_dict(),
        "best_rejection": best_rej.to_dict() if best_rej is not None else None,
    }


# ===============================
# Threshold selection (train) and application (test)
# ===============================

def pick_best_threshold(
    df: pd.DataFrame,
    strategy: str = "max_selective_accuracy",
    base_acc: float | None = None
) -> dict:
    """
    Choose a single threshold from a TRAIN selective table.
    Strategies:
      - "max_selective_accuracy" (default)
      - "under_10_rejection"
      - "min_rejection_over_base" (requires base_acc)
    """
    if df.empty:
        raise ValueError("Selective table is empty; cannot pick a threshold.")

    if strategy == "max_selective_accuracy":
        idx = df["selective_accuracy"].idxmax()
        return df.loc[idx].to_dict()

    if strategy == "under_10_rejection":
        under_10 = df[df["rejection_rate"] < 0.10]
        if under_10.empty:
            raise ValueError("No threshold under 10% rejection found on TRAIN.")
        idx = under_10["selective_accuracy"].idxmax()
        return under_10.loc[idx].to_dict()

    if strategy == "min_rejection_over_base":
        if base_acc is None:
            raise ValueError("base_acc required for 'min_rejection_over_base'.")
        ok = df[df["selective_accuracy"] >= base_acc].copy()
        if ok.empty:
            raise ValueError("No threshold on TRAIN achieves at least base accuracy.")
        ok = ok.sort_values(by=["rejection_rate", "threshold"], ascending=[True, True])
        return ok.iloc[0].to_dict()

    raise ValueError(f"Unknown strategy: {strategy}")


def apply_threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute deployment-style metrics when applying a single threshold."""
    max_p = proba.max(axis=1)
    y_pred = proba.argmax(axis=1)
    accepted = max_p >= threshold
    n_total = int(len(y_true))
    n_acc = int(accepted.sum())
    n_rej = n_total - n_acc
    rej_rate = 1.0 - (n_acc / n_total)
    sel_acc = float("nan") if n_acc == 0 else float((y_pred[accepted] == y_true[accepted]).mean())
    cm = None if n_acc == 0 else confusion_matrix(y_true[accepted], y_pred[accepted]).tolist()
    return {
        "threshold": float(threshold),
        "rejection_rate": rej_rate,
        "n_rejected": n_rej,
        "n_accepted": n_acc,
        "n_total": n_total,
        "selective_accuracy": sel_acc,
        "confusion_matrix_on_accepted": cm,
    }


# ===============================
# Visualization (interactive only)
# ===============================

def _show_visuals(export_dir, clf, X_s, y_s, le, pca_step, proba):
    """
    Show interactive visuals and keep them open until you close the windows.
    """
    try:
        # Confusion Matrix
        plt.figure()
        ConfusionMatrixDisplay.from_estimator(
            clf, X_s, y_s, display_labels=le.classes_, xticks_rotation=90
        )
        plt.title("Confusion Matrix — SVM + PCA")
        plt.tight_layout()

        # Probability histogram
        plt.figure()
        max_proba = proba.max(axis=1)
        plt.hist(max_proba, bins=20)
        plt.title("Max Predicted Probability (Confidence)")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.tight_layout()

        # PCA explained variance
        if hasattr(pca_step, "explained_variance_ratio_") and pca_step.explained_variance_ratio_ is not None:
            evr = pca_step.explained_variance_ratio_
            plt.figure()
            plt.bar(range(len(evr)), evr)
            plt.title("PCA Explained Variance by Component")
            plt.xlabel("Component Index")
            plt.ylabel("Variance Ratio")
            plt.tight_layout()

        # 🚦 Block here until you close the plot windows
        print("Close the plot windows to continue...")
        plt.show()

    except Exception as e:
        print(f"[warning] Could not display visuals: {e}")


# ===============================
# Main training / evaluation
# ===============================

def run(
    csv_path: str,
    target_name: str | None,
    kernel: str,
    C: float,
    gamma: str | float,
    export_dir: str | None = None,
    seed: int = 42,
    report_split: str = "test",
    threshold_grid: Tuple[float, float, float] = (0.0, 1.0, 0.01),
    pca_components: float | int = 0.95,
) -> Dict[str, Any]:

    # 1) Load + feature engineering
    X_raw, y = load_dataset(csv_path, target_name)
    Xf, feat_names = engineer_features(X_raw)

    # 2) Splits
    Xtr, ytr, Xva, yva, Xte, yte, le = split_data(Xf, y, seed=seed)

    # 3) Pipeline
    clf = build_svm_pipeline_with_pca(kernel, C, gamma, pca_components, seed)
    clf.fit(Xtr, ytr)

    # PCA summary
    pca_step: PCA = clf.named_steps["pca"]
    n_components_retained = int(pca_step.n_components_) if hasattr(pca_step, "n_components_") else None
    explained_variance_ratio = getattr(pca_step, "explained_variance_ratio_", None)

    # 4) Basic metrics
    basic_rows = [
        evaluate_split_basic(clf, Xtr, ytr, "train"),
        evaluate_split_basic(clf, Xva, yva, "val"),
        evaluate_split_basic(clf, Xte, yte, "test"),
    ]

    # 5) Threshold sweep on TRAIN (choose deployment threshold)
    thresholds = np.round(np.arange(*threshold_grid), 6)
    train_proba = clf.predict_proba(Xtr)
    sel_train_df = selective_metrics(ytr, train_proba, thresholds)

    train_base_acc = basic_rows[0]["accuracy"]
    chosen_strategy = "max_selective_accuracy"  # change if desired
    chosen_row_train = pick_best_threshold(sel_train_df, strategy=chosen_strategy, base_acc=train_base_acc)
    chosen_threshold = float(chosen_row_train["threshold"])

    # 6) Build report table on chosen split (for rich reporting)
    split_map = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
    Xs, ys = split_map[report_split]
    proba_split = clf.predict_proba(Xs)
    sel_df = selective_metrics(ys, proba_split, thresholds)

    base_acc = basic_rows[["train", "val", "test"].index(report_split)]["accuracy"]
    picks = choose_thresholds(sel_df, base_acc=base_acc)

    # 7) Apply the TRAIN-picked threshold to TEST (deployment-style)
    test_proba = clf.predict_proba(Xte)
    applied_on_test = apply_threshold_metrics(yte, test_proba, chosen_threshold)

    # 8) Results payload
    results = {
        "classes": list(le.classes_),
        "svm_params": {"kernel": kernel, "C": C, "gamma": gamma},
        "pca": {
            "requested_components": pca_components,
            "n_components_retained": n_components_retained,
            "explained_variance_ratio": (explained_variance_ratio.tolist() if explained_variance_ratio is not None else None),
        },
        "basic_eval": basic_rows,
        "selective_table": sel_df.to_dict(orient="records"),
        "selections": picks,
        "report_split": report_split,
        "deployment_threshold": {
            "strategy": chosen_strategy,
            "train_selection": chosen_row_train,
            "applied_to_test": applied_on_test,
        },
    }

    # 9) Interactive visuals
    _show_visuals(export_dir, clf, Xs, ys, le, pca_step, proba_split)

    # 10) Optional JSON export
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        with open(Path(export_dir) / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def print_report(results: Dict[str, Any]):
    """Human-readable report mirroring teammate-style summary."""
    split = results["report_split"]
    sel = results["selections"]
    sel_table = pd.DataFrame(results["selective_table"])
    n = int(sel_table["n"].iloc[0]) if not sel_table.empty else 0

    def fmt_block(title: str, row: dict | None):
        print(title)
        if row is None:
            print("  (no threshold met the criteria)")
            return
        thr = float(row["threshold"])
        rej_rate = float(row["rejection_rate"])
        n_rej = int(row["n_rejected"])
        sel_acc = float(row["selective_accuracy"]) if not np.isnan(row["selective_accuracy"]) else float("nan")
        print(f"  Confidence threshold: {thr:.2f}")
        print(f"  Rejection rate: {rej_rate*100:.1f}% ({n_rej} of {n})")
        print(f"  Selective accuracy (accepted only): {sel_acc:.3f}\n")

    print(f"Optimized SVM with PCA (reporting on {split} set)\n")
    pca_info = results.get("pca", {})
    print("PCA summary:")
    print(f"  requested components: {pca_info.get('requested_components')}")
    print(f"  retained components:  {pca_info.get('n_components_retained')}")
    if pca_info.get("explained_variance_ratio") is not None:
        evr = np.array(pca_info["explained_variance_ratio"])
        cum = evr.cumsum()
        print(f"  first 5 explained variance ratios: {np.round(evr[:5], 4).tolist()}")
        print(f"  cumulative explained variance (first 5): {np.round(cum[:5], 4).tolist()}")
    print()

    fmt_block("Best performance threshold (<10% rejection):", sel.get("best_perf_under_10"))
    fmt_block("Optimizing on accuracy:", sel.get("best_accuracy"))
    fmt_block("Optimizing on rejection rate:", sel.get("best_rejection"))

    dep = results.get("deployment_threshold")
    if dep:
        print("\nChosen deployment threshold (selected on TRAIN):")
        tr = dep.get("train_selection", {})
        thr = tr.get("threshold", float("nan"))
        print(f"  Strategy: {dep.get('strategy')}")
        print(f"  TRAIN threshold: {thr:.2f}")
        print(f"  TRAIN selective accuracy: {tr.get('selective_accuracy'):.3f}")
        print(f"  TRAIN rejection rate: {tr.get('rejection_rate')*100:.1f}% "
              f"({tr.get('n_rejected')} of {tr.get('n')})")

        te = dep.get("applied_to_test", {})
        print("\nApplied to TEST:")
        print(f"  TEST threshold: {te.get('threshold'):.2f}")
        print(f"  TEST selective accuracy: {te.get('selective_accuracy'):.3f}")
        print(f"  TEST rejection rate: {te.get('rejection_rate')*100:.1f}% "
              f"({te.get('n_rejected')} of {te.get('n_total')})")


# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SVM + PCA classifier with train-chosen threshold and interactive visuals."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--target_name", default=None, help="Target column name if not first column")
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "poly", "sigmoid"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--export_dir", default=None, help="Optional directory to export results.json")
    parser.add_argument("--report_split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_components", default=0.95, type=float)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out = run(
        csv_path=str(csv_path),
        target_name=args.target_name,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        export_dir=args.export_dir,
        seed=args.seed,
        report_split=args.report_split,
        pca_components=args.pca_components,
    )
    print_report(out)