"""
Final Production Ensemble - Logistic Regression Meta-Learner
Learns to combine model predictions optimally.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import json

print("=" * 80)
print("FINAL ENSEMBLE - LOGISTIC REGRESSION META-LEARNER")
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

# Split
train_mask = (df['set'] == 'train').values
test_mask = (df['set'] == 'holdout').values

y_train = y_all[train_mask]
y_test = y_all[test_mask]

# Build stacking features
X_train = np.hstack([model_probs[m][train_mask] for m in models])
X_test = np.hstack([model_probs[m][test_mask] for m in models])

print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")
print(f"Features: {X_train.shape[1]} (6 models x {n_classes} classes)")

# Train Logistic Regression meta-learner
print("\n" + "=" * 80)
print("TRAINING META-LEARNER")
print("=" * 80)

# C=1.0 worked well in experiments
meta_learner = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

print("\nTraining Logistic Regression (C=1.0)...")
meta_learner.fit(X_train, y_train)

# Evaluate on train
train_preds = meta_learner.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)
train_f1 = f1_score(y_train, train_preds, average='macro')
print(f"Train accuracy: {train_acc:.4f}")
print(f"Train F1-macro: {train_f1:.4f}")

# Evaluate on test
test_preds = meta_learner.predict(X_test)
test_probs = meta_learner.predict_proba(X_test)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average='macro')

print(f"\nTest accuracy: {test_acc:.4f} ({int(test_acc * len(y_test))}/{len(y_test)})")
print(f"Test F1-macro: {test_f1:.4f}")

# Confidence stats
confidences = np.max(test_probs, axis=1)
print(f"\nConfidence stats:")
print(f"  Mean: {confidences.mean():.4f}")
print(f"  Min:  {confidences.min():.4f}")
print(f"  Max:  {confidences.max():.4f}")

# Compare with individual models and simple average
print("\n" + "=" * 80)
print("COMPARISON WITH BASELINES")
print("=" * 80)

print(f"\n{'Method':<35} {'Test Acc':>10} {'Test F1':>10}")
print("-" * 60)

# Individual models
for m in models:
    preds = np.argmax(model_probs[m][test_mask], axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    print(f"{m:<35} {acc:>10.4f} {f1:>10.4f}")

# Simple average
avg_probs = np.mean([model_probs[m][test_mask] for m in models], axis=0)
avg_preds = np.argmax(avg_probs, axis=1)
avg_acc = accuracy_score(y_test, avg_preds)
avg_f1 = f1_score(y_test, avg_preds, average='macro')
print(f"{'Simple average (all 6)':<35} {avg_acc:>10.4f} {avg_f1:>10.4f}")

# Top 3 average
top3 = ['cnn', 'statistics', 'tcn']
top3_probs = np.mean([model_probs[m][test_mask] for m in top3], axis=0)
top3_preds = np.argmax(top3_probs, axis=1)
top3_acc = accuracy_score(y_test, top3_preds)
top3_f1 = f1_score(y_test, top3_preds, average='macro')
print(f"{'Simple average (top 3)':<35} {top3_acc:>10.4f} {top3_f1:>10.4f}")

# Meta-learner
print(f"{'Logistic Regression meta-learner':<35} {test_acc:>10.4f} {test_f1:>10.4f}")

# Analyze what the meta-learner learned
print("\n" + "=" * 80)
print("WHAT THE META-LEARNER LEARNED")
print("=" * 80)

# Get feature importances (coefficient magnitudes per model)
coef = meta_learner.coef_  # shape: (n_classes, n_features)
n_features_per_model = n_classes

model_importance = {}
for i, m in enumerate(models):
    start_idx = i * n_features_per_model
    end_idx = start_idx + n_features_per_model
    # Sum of absolute coefficients for this model's features
    importance = np.abs(coef[:, start_idx:end_idx]).sum()
    model_importance[m] = importance

# Normalize
total = sum(model_importance.values())
model_importance = {k: v/total for k, v in model_importance.items()}

print("\nModel importance (by coefficient magnitude):")
for m, imp in sorted(model_importance.items(), key=lambda x: -x[1]):
    bar = "#" * int(imp * 50)
    print(f"  {m:<12}: {imp:.3f} {bar}")

# Save model and metadata
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

# Save the trained model
with open('ensemble_meta_learner.pkl', 'wb') as f:
    pickle.dump(meta_learner, f)

# Save metadata
metadata = {
    'models': models,
    'classes': classes,
    'n_classes': n_classes,
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'train_f1': float(train_f1),
    'test_f1': float(test_f1),
    'model_importance': model_importance,
    'meta_learner_type': 'LogisticRegression',
    'meta_learner_params': {'C': 1.0, 'max_iter': 1000}
}

with open('ensemble_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nSaved:")
print("  - ensemble_meta_learner.pkl (trained model)")
print("  - ensemble_metadata.json (metadata)")

# Example usage
print("\n" + "=" * 80)
print("USAGE EXAMPLE")
print("=" * 80)

print("""
# Load and use the ensemble:

import pickle
import numpy as np

# Load meta-learner
with open('ensemble_meta_learner.pkl', 'rb') as f:
    meta_learner = pickle.load(f)

# Get predictions from your 6 models (each shape: n_samples x 57)
# models = ['cnn', 'resnet1d', 'extratrees', 'statistics', 'rulebased', 'tcn']

# Stack features
X = np.hstack([cnn_probs, resnet1d_probs, extratrees_probs,
               statistics_probs, rulebased_probs, tcn_probs])

# Predict
predictions = meta_learner.predict(X)
probabilities = meta_learner.predict_proba(X)
""")

print("\n[SUCCESS] Final ensemble ready for production!")
