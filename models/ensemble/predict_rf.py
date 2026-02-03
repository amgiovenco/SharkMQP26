"""Random Forest model prediction from local models directory."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import simpson
from sklearn.ensemble import ExtraTreesClassifier


def engineer_features_randomforest(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for random forest."""
    features = pd.DataFrame(index=X_raw.index)
    temps = X_raw.columns.astype(float).values

    features['max'] = X_raw.max(axis=1)
    features['min'] = X_raw.min(axis=1)
    features['mean'] = X_raw.mean(axis=1)
    features['std'] = X_raw.std(axis=1)
    features['auc'] = X_raw.apply(lambda row: simpson(row.values, temps), axis=1)
    features['centroid'] = X_raw.apply(lambda row: np.sum(row.values * temps) / (np.sum(row.values) + 1e-8), axis=1)
    features['temp_peak'] = X_raw.apply(lambda row: temps[np.argmax(row.values)], axis=1)
    features['fwhm'] = X_raw.apply(lambda row: np.sum(row.values > 0.5 * row.max()), axis=1)
    features['rise_time'] = X_raw.apply(lambda row: np.argmax(row.values), axis=1)
    features['decay_time'] = X_raw.apply(lambda row: len(row) - np.argmax(row.values[::-1]), axis=1)
    features['auc_left'] = X_raw.apply(
        lambda row: simpson(row.values[:np.argmax(row.values)+1], temps[:np.argmax(row.values)+1]),
        axis=1
    )
    features['auc_right'] = X_raw.apply(
        lambda row: simpson(row.values[np.argmax(row.values):], temps[np.argmax(row.values):]),
        axis=1
    )
    features['asymmetry'] = features['auc_left'] / (features['auc_right'] + 1e-8)
    features['fwhm_rise_ratio'] = features['fwhm'] / (features['rise_time'] + 1e-8)
    features['peak_temp_std'] = features['temp_peak'] * features['std']
    features['asymmetry_fwhm'] = features['asymmetry'] * features['fwhm']
    features['rise_decay_ratio'] = features['rise_time'] / (features['decay_time'] + 1e-8)

    return features


def get_rf_predictions(X_raw: pd.DataFrame, models_dir: str = "./models") -> np.ndarray:
    """Get random forest predictions from ensemble/models directory."""
    try:
        model_path = Path(models_dir) / "RANDOMFOREST_randomforest_final.pkl"
        if not model_path.exists():
            print("  random forest...[FAIL] RANDOMFOREST_randomforest_final.pkl not found in ./models/")
            return None

        data = joblib.load(model_path)

        # Extract model from bundle dict if needed
        if isinstance(data, dict) and "model" in data:
            model = data["model"]
        else:
            model = data

        X_eng = engineer_features_randomforest(X_raw)
        proba = model.predict_proba(X_eng)
        print(f"  random forest...[OK] ({proba.shape})")
        return proba
    except Exception as e:
        print(f"  random forest...[FAIL] error: {e}")
        return None
