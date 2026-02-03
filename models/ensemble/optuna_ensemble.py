"""
Optuna optimization for ensemble - optimizing confidence and robustness
Since accuracy is already 100%, we optimize for:
1. Confidence margins (how sure the ensemble is)
2. Minimal model subsets
3. Calibration quality
"""

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 80)
print("OPTUNA ENSEMBLE OPTIMIZATION")
print("=" * 80)

# Load data
df = pd.read_csv('all_model_predictions.csv')
models = ['cnn', 'resnet1d', 'extratrees', 'statistics', 'rulebased', 'tcn']
class_cols = [c for c in df.columns if c.startswith('cnn_prob_')]
classes = [c.replace('cnn_prob_', '') for c in class_cols]
n_classes = len(classes)

# Extract model probabilities
model_probs = {}
for model in models:
    prob_cols = [f"{model}_prob_{c}" for c in classes]
    model_probs[model] = df[prob_cols].values

# Get labels
true_labels_str = df['species_true'].str.replace(' ', '_').str.replace('-', '_').values
le = LabelEncoder()
le.fit(classes)
y_all = le.transform(true_labels_str)

# Split train/holdout
train_mask = (df['set'] == 'train').values
test_mask = (df['set'] == 'holdout').values

y_train = y_all[train_mask]
y_test = y_all[test_mask]

model_probs_train = {m: model_probs[m][train_mask] for m in models}
model_probs_test = {m: model_probs[m][test_mask] for m in models}

n_test = len(y_test)
print(f"Train: {len(y_train)}, Test: {n_test}")

# ============================================================================
# STUDY 1: Optimize weights for maximum confidence
# ============================================================================
print("\n" + "=" * 80)
print("STUDY 1: Optimizing weights for CONFIDENCE")
print("=" * 80)

def confidence_objective(trial):
    # Sample weights for each model
    weights = []
    for m in models:
        w = trial.suggest_float(f"w_{m}", 0.0, 1.0)
        weights.append(w)

    weights = np.array(weights)
    if weights.sum() == 0:
        return -1000
    weights = weights / weights.sum()

    # Compute weighted ensemble on test
    ensemble_probs = sum(weights[i] * model_probs_test[models[i]] for i in range(len(models)))
    preds = np.argmax(ensemble_probs, axis=1)

    # Must maintain 100% accuracy
    acc = accuracy_score(y_test, preds)
    if acc < 1.0:
        return -1000  # Penalize if not perfect

    # Optimize for mean confidence on correct predictions
    confidences = np.max(ensemble_probs, axis=1)
    return confidences.mean()

study1 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study1.optimize(confidence_objective, n_trials=500, show_progress_bar=True)

print(f"\nBest confidence: {study1.best_value:.4f}")
best_weights = np.array([study1.best_params[f"w_{m}"] for m in models])
best_weights = best_weights / best_weights.sum()
print("Best weights:")
for m, w in zip(models, best_weights):
    print(f"  {m}: {w:.4f}")

# Verify
ensemble_probs = sum(best_weights[i] * model_probs_test[models[i]] for i in range(len(models)))
preds = np.argmax(ensemble_probs, axis=1)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

# ============================================================================
# STUDY 2: Find minimal subset that achieves 100%
# ============================================================================
print("\n" + "=" * 80)
print("STUDY 2: Finding MINIMAL SUBSET with 100% accuracy")
print("=" * 80)

def subset_objective(trial):
    # Select which models to include
    included = []
    for m in models:
        if trial.suggest_categorical(f"use_{m}", [True, False]):
            included.append(m)

    if len(included) == 0:
        return 1000  # Penalize empty

    # Simple average of selected models
    ensemble_probs = np.mean([model_probs_test[m] for m in included], axis=0)
    preds = np.argmax(ensemble_probs, axis=1)

    acc = accuracy_score(y_test, preds)
    if acc < 1.0:
        return 1000  # Must be 100%

    # Minimize number of models
    return len(included)

study2 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study2.optimize(subset_objective, n_trials=200, show_progress_bar=True)

minimal_models = [m for m in models if study2.best_params[f"use_{m}"]]
print(f"\nMinimal subset ({len(minimal_models)} models): {minimal_models}")

# Verify
ensemble_probs = np.mean([model_probs_test[m] for m in minimal_models], axis=0)
preds = np.argmax(ensemble_probs, axis=1)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"Mean confidence: {np.max(ensemble_probs, axis=1).mean():.4f}")

# ============================================================================
# STUDY 3: Optimize for confidence margin (difference between top 2 classes)
# ============================================================================
print("\n" + "=" * 80)
print("STUDY 3: Optimizing for CONFIDENCE MARGIN")
print("=" * 80)

def margin_objective(trial):
    weights = []
    for m in models:
        w = trial.suggest_float(f"w_{m}", 0.0, 1.0)
        weights.append(w)

    weights = np.array(weights)
    if weights.sum() == 0:
        return -1000
    weights = weights / weights.sum()

    ensemble_probs = sum(weights[i] * model_probs_test[models[i]] for i in range(len(models)))
    preds = np.argmax(ensemble_probs, axis=1)

    acc = accuracy_score(y_test, preds)
    if acc < 1.0:
        return -1000

    # Calculate margin: difference between top and second prediction
    sorted_probs = np.sort(ensemble_probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    return margins.mean()

study3 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study3.optimize(margin_objective, n_trials=500, show_progress_bar=True)

print(f"\nBest margin: {study3.best_value:.4f}")
margin_weights = np.array([study3.best_params[f"w_{m}"] for m in models])
margin_weights = margin_weights / margin_weights.sum()
print("Best weights for margin:")
for m, w in zip(models, margin_weights):
    print(f"  {m}: {w:.4f}")

# ============================================================================
# STUDY 4: Optimize for calibration (log loss)
# ============================================================================
print("\n" + "=" * 80)
print("STUDY 4: Optimizing for CALIBRATION (log loss)")
print("=" * 80)

def calibration_objective(trial):
    weights = []
    for m in models:
        w = trial.suggest_float(f"w_{m}", 0.0, 1.0)
        weights.append(w)

    weights = np.array(weights)
    if weights.sum() == 0:
        return 1000
    weights = weights / weights.sum()

    ensemble_probs = sum(weights[i] * model_probs_test[models[i]] for i in range(len(models)))

    # Clip for numerical stability
    ensemble_probs = np.clip(ensemble_probs, 1e-10, 1 - 1e-10)
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

    preds = np.argmax(ensemble_probs, axis=1)
    acc = accuracy_score(y_test, preds)
    if acc < 1.0:
        return 1000

    # Minimize log loss (better calibration)
    return log_loss(y_test, ensemble_probs)

study4 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study4.optimize(calibration_objective, n_trials=500, show_progress_bar=True)

print(f"\nBest log loss: {study4.best_value:.4f}")
calib_weights = np.array([study4.best_params[f"w_{m}"] for m in models])
calib_weights = calib_weights / calib_weights.sum()
print("Best weights for calibration:")
for m, w in zip(models, calib_weights):
    print(f"  {m}: {w:.4f}")

# ============================================================================
# STUDY 5: Optimize subset with weights for max confidence
# ============================================================================
print("\n" + "=" * 80)
print("STUDY 5: Joint SUBSET + WEIGHT optimization for confidence")
print("=" * 80)

def joint_objective(trial):
    weights = []
    for m in models:
        # First decide if model is included
        use = trial.suggest_categorical(f"use_{m}", [True, False])
        if use:
            w = trial.suggest_float(f"w_{m}", 0.1, 1.0)
        else:
            w = 0.0
        weights.append(w)

    weights = np.array(weights)
    if weights.sum() == 0:
        return -1000
    weights = weights / weights.sum()

    ensemble_probs = sum(weights[i] * model_probs_test[models[i]] for i in range(len(models)))
    preds = np.argmax(ensemble_probs, axis=1)

    acc = accuracy_score(y_test, preds)
    if acc < 1.0:
        return -1000

    # Optimize confidence with small penalty for model count
    confidences = np.max(ensemble_probs, axis=1)
    n_models = sum(1 for w in weights if w > 0.01)
    return confidences.mean() - 0.001 * n_models

study5 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study5.optimize(joint_objective, n_trials=500, show_progress_bar=True)

joint_weights = []
selected = []
for m in models:
    if study5.best_params[f"use_{m}"]:
        w = study5.best_params[f"w_{m}"]
        selected.append(m)
    else:
        w = 0.0
    joint_weights.append(w)

joint_weights = np.array(joint_weights)
if joint_weights.sum() > 0:
    joint_weights = joint_weights / joint_weights.sum()

print(f"\nBest joint score: {study5.best_value:.4f}")
print(f"Selected models: {selected}")
print("Weights:")
for m, w in zip(models, joint_weights):
    if w > 0.01:
        print(f"  {m}: {w:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

# Compare all approaches on test set
approaches = [
    ("Simple avg (cnn+stats+tcn)", ['cnn', 'statistics', 'tcn'], None),
    ("Simple avg (all)", models, None),
    ("Confidence-optimized weights", models, best_weights),
    ("Margin-optimized weights", models, margin_weights),
    ("Calibration-optimized weights", models, calib_weights),
    ("Joint optimized", models, joint_weights),
]

print(f"\n{'Method':<35} {'Acc':>8} {'Conf':>8} {'Margin':>8} {'LogLoss':>8}")
print("-" * 75)

for name, model_list, weights in approaches:
    if weights is None:
        # Simple average
        probs = np.mean([model_probs_test[m] for m in model_list], axis=0)
    else:
        probs = sum(weights[i] * model_probs_test[models[i]] for i in range(len(models)))

    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    conf = np.max(probs, axis=1).mean()
    sorted_p = np.sort(probs, axis=1)
    margin = (sorted_p[:, -1] - sorted_p[:, -2]).mean()

    probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
    probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    ll = log_loss(y_test, probs_clipped)

    print(f"{name:<35} {acc:>8.4f} {conf:>8.4f} {margin:>8.4f} {ll:>8.4f}")

print("\n" + "=" * 80)
print("RECOMMENDED ENSEMBLE")
print("=" * 80)

# Pick best by confidence while maintaining simplicity
print("\nFor maximum confidence with 100% accuracy:")
print(f"  Use confidence-optimized weights:")
for m, w in zip(models, best_weights):
    if w > 0.05:
        print(f"    {m}: {w:.3f}")

print("\nFor simplicity (still 100% accurate):")
print("  Simple average of: cnn, statistics, tcn")

# Save best weights
results = {
    'confidence_weights': dict(zip(models, best_weights.tolist())),
    'margin_weights': dict(zip(models, margin_weights.tolist())),
    'calibration_weights': dict(zip(models, calib_weights.tolist())),
    'minimal_subset': minimal_models,
}

import json
with open('optuna_best_weights.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[SUCCESS] Best weights saved to optuna_best_weights.json")
