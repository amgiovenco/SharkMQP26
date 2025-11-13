# to run: 
# python train_xgb_species.py --csv /mnt/data/species_14_features_normalized2.csv --random_state 42 --folds 5

# train_xgb_species.py
# Usage:
#   python train_xgb_species.py --csv /path/to/species_14_features_normalized2.csv --random_state 42 --folds 5
# Produces:
#   - Prints stratified 60/20/20 split summary
#   - 5-fold CV validation accuracies on the 60% training split
#   - Test accuracy on the 20% held-out test split
#   - confusion_matrix.png saved to the current directory

# train_xgb_species.py
# Run with:
#   python train_xgb_species.py
# train_xgb_species.py
# Run:
#   python train_xgb_species.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

CSV_FILENAME = "species_14_features_normalized2.csv"

def make_confusion_matrix_png(y_true, y_pred, class_names, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix (Test Set)'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out_path

def print_distribution(label_array, name, class_names):
    classes, counts = np.unique(label_array, return_counts=True)
    total = counts.sum()
    print(f"{name} distribution (count, percent):")
    for c, n in zip(classes, counts):
        pct = 100.0 * n / total
        print(f"  {c:>2} ({class_names[c]:>20}): {n:>4} ({pct:5.1f}%)")
    print(f"  Majority baseline for {name.lower()}: {counts.max()/total:.4f}\n")

def warn_if_single_class(pred_array, where, class_names):
    uniq = np.unique(pred_array)
    if len(uniq) == 1:
        only = uniq[0]
        print(f"⚠️  Warning: {where} predicts a SINGLE class only -> {only} ({class_names[only]}).")
        print("   This usually means heavy class imbalance or underfitting.\n")

def per_class_recall(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(cm) / cm.sum(axis=1)
    print("Per-class recall on test:")
    for idx, r in enumerate(recall):
        r_str = "nan" if np.isnan(r) else f"{r:.4f}"
        print(f"  {idx:>2} ({class_names[idx]:>20}): {r_str}")
    print()

def main():
    if not os.path.exists(CSV_FILENAME):
        raise FileNotFoundError(f"CSV not found: {CSV_FILENAME}")

    df = pd.read_csv(CSV_FILENAME)
    X = df.drop(columns=["Species"])
    y_text = df["Species"].astype(str)

    # Encode labels to integers for XGBoost
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    class_names = list(le.classes_)

    # Stratified 60/20/20 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=8
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=8
    )

    print("== Split Sizes ==")
    print(f"Train: {X_train.shape[0]} rows")
    print(f"Val:   {X_val.shape[0]} rows")
    print(f"Test:  {X_test.shape[0]} rows\n")

    # Print true distributions (useful sanity check)
    print_distribution(y_train, "TRAIN (true)", class_names)
    print_distribution(y_val,   "VAL   (true)", class_names)
    print_distribution(y_test,  "TEST  (true)", class_names)

    # Train on train split, validate on val split
    clf = XGBClassifier(
        n_estimators = 3000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="merror",   # multi-class error rate
        tree_method="hist",
        random_state=8
    )
    clf.fit(X_train, y_train)

    # ----- Validation diagnostics -----
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_acc:.4f}\n")
    print_distribution(val_pred, "VAL (pred)", class_names)
    warn_if_single_class(val_pred, "Validation", class_names)
    print("Validation classification report:")
    print(classification_report(y_val, val_pred, target_names=class_names, digits=4))

    # Final model on TRAIN+VAL
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    clf.fit(X_trainval, y_trainval)
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n== Final Model Performance ==")
    print(f"Test accuracy: {test_acc:.4f}\n")

    # ----- Test diagnostics -----
    print_distribution(test_pred, "TEST (pred)", class_names)
    warn_if_single_class(test_pred, "Test", class_names)
    print("Test classification report:")
    print(classification_report(y_test, test_pred, target_names=class_names, digits=4))
    per_class_recall(y_test, test_pred, class_names)

    # Confusion matrix saved as image
    out_img = make_confusion_matrix_png(y_test, test_pred, class_names)
    print(f"Saved confusion matrix: {out_img}")

    # Label reference
    print("\n== Label Mapping ==")
    for idx, name in enumerate(class_names):
        print(f"{idx} -> {name}")

if __name__ == "__main__":
    main()
