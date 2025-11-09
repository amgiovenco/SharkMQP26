"""
pre-compute base model predictions
===================================

run once to generate predictions from all 5 base models on the entire dataset.
saves to disk so ensemble pipeline doesn't need to recalculate.

usage:
    python precompute_predictions.py

output:
    base_predictions.npz - contains all predictions + metadata
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# import model prediction functions
from predict_rf import get_rf_predictions
from predict_rb import get_rb_predictions
from predict_stats import get_stats_predictions
from predict_resnet import get_resnet_predictions
from predict_cnn import get_cnn_predictions
from predict_gaussian import get_gaussian_predictions

# ============================================================================
# CONFIG
# ============================================================================

DATA_PATH = "../../data/shark_dataset.csv"
MODELS_DIR = "./models"
OUTPUT_FILE = "base_predictions.npz"
SEED = 8

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """load raw fluorescence data."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Species"])
    y = df["Species"]

    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    mask = y.isin(valid_classes)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    return X, y

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("pre-computing base model predictions")
    print("="*80)

    # load data
    print(f"\n[load] loading data from {DATA_PATH}...", end=" ", flush=True)
    X_raw, y = load_data()
    print(f"[OK] ({len(X_raw)} samples, {len(y.unique())} species)")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # get all predictions
    print(f"\n[*] getting base model predictions:")

    predictions = {}

    pred_rf = get_rf_predictions(X_raw, MODELS_DIR)
    if pred_rf is not None:
        predictions['rf'] = pred_rf

    pred_rb = get_rb_predictions(X_raw, MODELS_DIR)
    if pred_rb is not None:
        predictions['rb'] = pred_rb

    pred_stats = get_stats_predictions(X_raw, MODELS_DIR)
    if pred_stats is not None:
        predictions['stats'] = pred_stats

    pred_resnet = get_resnet_predictions(X_raw, len(le.classes_), MODELS_DIR)
    if pred_resnet is not None:
        predictions['resnet'] = pred_resnet

    pred_cnn = get_cnn_predictions(X_raw, len(le.classes_), MODELS_DIR)
    if pred_cnn is not None:
        predictions['cnn'] = pred_cnn

    pred_gaussian = get_gaussian_predictions(X_raw, MODELS_DIR)
    if pred_gaussian is not None:
        predictions['gaussian'] = pred_gaussian

    if not predictions:
        print("\n[FAIL] no predictions generated!")
        return

    print(f"\n[OK] generated predictions for {len(predictions)} models")

    # stack all predictions
    print(f"\n[*] stacking predictions...", end=" ", flush=True)
    pred_list = list(predictions.values())
    X_stacked = np.hstack(pred_list)
    print(f"[OK] ({X_stacked.shape})")

    # save to disk (NPZ)
    print(f"\n[save] saving to {OUTPUT_FILE}...", end=" ", flush=True)
    np.savez(
        OUTPUT_FILE,
        X_stacked=X_stacked,
        y=y_encoded,
        y_original=y.values,
        species_classes=le.classes_,
        model_names=np.array(list(predictions.keys())),
        **predictions
    )
    print(f"[OK]")

    # save to CSV
    csv_file = OUTPUT_FILE.replace('.npz', '.csv')
    print(f"[save] saving to {csv_file}...", end=" ", flush=True)

    # create column names for each model's predictions
    column_names = ['Species']
    for model_name in predictions.keys():
        for class_idx, class_name in enumerate(le.classes_):
            column_names.append(f"{model_name}_{class_name}")

    # create DataFrame with stacked predictions
    df_csv = pd.DataFrame(X_stacked, columns=[col for col in column_names[1:]])
    df_csv.insert(0, 'Species', y.values)

    # save to CSV
    df_csv.to_csv(csv_file, index=False)
    print(f"[OK]")

    # summary
    print(f"\n[data] summary:")
    print(f"   total samples: {len(X_stacked)}")
    print(f"   stacked features: {X_stacked.shape[1]} ({len(le.classes_)} classes × {len(predictions)} models)")
    print(f"   species: {len(le.classes_)}")
    print(f"   model predictions: {', '.join(predictions.keys())}")

    print(f"\n[dir] files saved:")
    print(f"   - {OUTPUT_FILE}")
    print(f"   - {csv_file}")

    print(f"\n[OK] done! open ensemble_experiments.ipynb to train and experiment.")

if __name__ == "__main__":
    main()
