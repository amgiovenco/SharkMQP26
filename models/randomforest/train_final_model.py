"""
Train final Random Forest model with known best parameters
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from scipy.integrate import simpson

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8

print("Loading data...")
data = pd.read_csv(CSV_PATH)

X_raw = data.drop(columns=[SPECIES_COL])
y = data[SPECIES_COL]

# Drop species with <2 samples
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
mask = y.isin(valid_classes)
X_raw = X_raw[mask]
y = y[mask].to_numpy()

print(f"Data shape: {X_raw.shape}")
print(f"Classes: {len(np.unique(y))}")

# Feature engineering (matches randomforest.ipynb exactly)
def feature_engineering(df):
    features = pd.DataFrame()
    temps = df.columns.astype(float)

    features['max'] = df.max(axis=1)
    features['min'] = df.min(axis=1)
    features['mean'] = df.mean(axis=1)
    features['std'] = df.std(axis=1)

    features['auc'] = df.apply(lambda row: simpson(row, temps), axis=1)
    features['centroid'] = df.apply(lambda row: np.sum(row*temps)/np.sum(row), axis=1)

    features['temp_peak'] = df.apply(lambda row: temps[np.argmax(row)], axis=1)
    features['fwhm'] = df.apply(lambda row: np.sum(row > 0.5*row.max()), axis=1)
    features['rise_time'] = df.apply(lambda row: np.argmax(row), axis=1)
    features['decay_time'] = df.apply(lambda row: len(row) - np.argmax(row[::-1]), axis=1)

    features['auc_left'] = df.apply(lambda row: simpson(row[:np.argmax(row)+1], temps[:np.argmax(row)+1]), axis=1)
    features['auc_right'] = df.apply(lambda row: simpson(row[np.argmax(row):], temps[np.argmax(row):]), axis=1)

    features['asymmetry'] = features['auc_left'] / (features['auc_right'] + 1e-8)

    return features


def enhanced_features(features):
    """Add interaction features (matches randomforest.ipynb exactly)"""
    enhanced = features.copy()

    # Interaction features from top performers
    enhanced['fwhm_rise_ratio'] = features['fwhm'] / (features['rise_time'] + 1e-8)
    enhanced['peak_temp_std'] = features['temp_peak'] * features['std']
    enhanced['asymmetry_fwhm'] = features['asymmetry'] * features['fwhm']
    enhanced['rise_decay_ratio'] = features['rise_time'] / (features['decay_time'] + 1e-8)

    return enhanced

print("\nExtracting features...")
X_features = feature_engineering(X_raw)
X_features = enhanced_features(X_features)
X = X_features.to_numpy(float)

print(f"Feature shape: {X.shape}")

# Train final model with best parameters
print("\n" + "="*60)
print("Training final model on all data...")
print("="*60)

best_params = {
    'n_estimators': 900,
    'max_depth': 40,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'max_features': 0.7,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': 1
}

final_model = RandomForestClassifier(**best_params)
final_model.fit(X, y)

bundle = {
    "model": final_model,
    "feature_names": list(X_features.columns),
    "model_type": "optimized_rf",
    "cv_accuracy": 0.9232,
    "params": best_params
}

joblib.dump(bundle, "./randomforest_final.pkl")
print(f"Saved optimized model to ./randomforest_final.pkl")

print("\n" + "="*60)
print("Done!")
print("="*60)
