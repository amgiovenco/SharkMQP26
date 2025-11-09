"""
production inference script
===========================

use trained ensemble meta-learner to make predictions on fluorescence data.
reads from data file, outputs predictions.

configuration is set at the top of this file.
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import torch
import torch.nn as nn

# import model prediction functions
from predict_rf import get_rf_predictions
from predict_rb import get_rb_predictions
from predict_stats import get_stats_predictions
from predict_resnet import get_resnet_predictions
from predict_cnn import get_cnn_predictions

# input data
DATA_PATH = "../../data/shark_dataset.csv"

# model locations
PRODUCTION_DIR = "./production"
BASE_MODELS_DIR = "./models"

# output
OUTPUT_FILE = "predictions.csv"

# which meta-learner to use: "xgb" or "nn"
META_LEARNER_TYPE = "xgb"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# META-LEARNER LOADING
# ============================================================================

def load_meta_learner(model_type: str = "xgb"):
    """load trained meta-learner model."""

    if model_type == "xgb":
        meta_path = f"{PRODUCTION_DIR}/final_xgb_meta.pkl"
        if not Path(meta_path).exists():
            raise FileNotFoundError(f"meta-learner not found: {meta_path}\nrun the notebook to train models first!")

        with open(meta_path, 'rb') as f:
            data = pickle.load(f)

        return {
            'type': 'xgb',
            'model': data['model'],
            'label_encoder': data['label_encoder'],
            'config': data['config']
        }

    elif model_type == "nn":
        meta_path = f"{PRODUCTION_DIR}/final_nn_meta.pth"
        if not Path(meta_path).exists():
            raise FileNotFoundError(f"meta-learner not found: {meta_path}\nrun the notebook to train models first!")

        checkpoint = torch.load(meta_path, map_location=DEVICE)

        # rebuild model
        model = nn.Sequential(
            nn.Linear(checkpoint['input_dim'], 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, checkpoint['num_classes'])
        ).to(DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return {
            'type': 'nn',
            'model': model,
            'label_encoder_classes': checkpoint['label_encoder_classes'],
            'config': checkpoint['config']
        }

    else:
        raise ValueError(f"unknown model type: {model_type}")

# ============================================================================
# PREDICTION
# ============================================================================

def predict(data_path: str, model_type: str = "xgb") -> Tuple[np.ndarray, list]:
    """
    make predictions on fluorescence data.

    args:
        data_path: path to fluorescence csv file
        model_type: "xgb" or "nn"

    returns:
        predictions: predicted species labels
        probabilities: confidence scores for each species
    """

    print(f"\n{'='*80}")
    print("ensemble prediction")
    print(f"{'='*80}\n")

    # load data
    print(f"[load] loading fluorescence data from {data_path}...")
    X_raw = pd.read_csv(data_path, index_col=0)
    print(f"[OK] loaded {len(X_raw)} samples, {X_raw.shape[1]} time points")

    # get predictions from base models
    print(f"\n[predict] getting base model predictions:")
    predictions_list = []
    model_names = []

    # get rf predictions
    pred_rf = get_rf_predictions(X_raw, BASE_MODELS_DIR)
    if pred_rf is not None:
        predictions_list.append(pred_rf)
        model_names.append('rf')

    # get rb predictions
    pred_rb = get_rb_predictions(X_raw, BASE_MODELS_DIR)
    if pred_rb is not None:
        predictions_list.append(pred_rb)
        model_names.append('rb')

    # get stats predictions
    pred_stats = get_stats_predictions(X_raw, BASE_MODELS_DIR)
    if pred_stats is not None:
        predictions_list.append(pred_stats)
        model_names.append('stats')

    # get resnet predictions
    pred_resnet = get_resnet_predictions(X_raw, num_classes=57, models_dir=BASE_MODELS_DIR)
    if pred_resnet is not None:
        predictions_list.append(pred_resnet)
        model_names.append('resnet')

    # get cnn predictions
    pred_cnn = get_cnn_predictions(X_raw, num_classes=57, models_dir=BASE_MODELS_DIR)
    if pred_cnn is not None:
        predictions_list.append(pred_cnn)
        model_names.append('cnn')

    if not predictions_list:
        raise RuntimeError("no base models could generate predictions!")

    # stack predictions
    print(f"\n[stack] stacking predictions...")
    X_stacked = np.hstack(predictions_list)
    print(f"[OK] stacked features: {X_stacked.shape}")

    # load meta-learner
    print(f"\n[load] loading meta-learner ({model_type})...")
    meta_learner = load_meta_learner(model_type)
    print(f"[OK] meta-learner loaded")

    # get final predictions
    print(f"\n[predict] getting meta-learner predictions...")

    if model_type == "xgb":
        dtest = xgb.DMatrix(X_stacked)
        proba = meta_learner['model'].predict(dtest)
        pred_indices = np.argmax(proba, axis=1)
        predictions = meta_learner['label_encoder'].inverse_transform(pred_indices)
        probabilities = np.max(proba, axis=1)

    elif model_type == "nn":
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_stacked).to(DEVICE)
            outputs = meta_learner['model'](X_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_indices = torch.argmax(probs, dim=1)
            probabilities = torch.max(probs, dim=1)[0]
            predictions = meta_learner['label_encoder_classes'][pred_indices.cpu().numpy()]
            probabilities = probabilities.cpu().numpy()

    print(f"[OK] predictions complete!")

    return predictions, probabilities

# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        predictions, confidences = predict(
            DATA_PATH,
            model_type=META_LEARNER_TYPE
        )

        # create results dataframe
        results_df = pd.DataFrame({
            'predicted_species': predictions,
            'confidence': confidences
        })

        print(f"\n{'='*80}")
        print("results")
        print(f"{'='*80}\n")
        print(results_df.head(10))
        print(f"\ntotal predictions: {len(results_df)}")

        # save results
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n[save] results saved to {OUTPUT_FILE}")

        print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"\n[error] {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
