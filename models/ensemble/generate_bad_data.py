"""
Generate fake/bad data to stress test the ensemble.
See if confidence drops for out-of-distribution samples.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

print("=" * 80)
print("STRESS TEST: FAKE/BAD DATA")
print("=" * 80)

# Load real data
df = pd.read_csv('all_model_predictions.csv')
models = ['cnn', 'statistics', 'tcn']
class_cols = [c for c in df.columns if c.startswith('cnn_prob_')]
classes = [c.replace('cnn_prob_', '') for c in class_cols]
n_classes = len(classes)

model_probs = {}
for m in models:
    model_probs[m] = df[[f'{m}_prob_{c}' for c in classes]].values

true_labels = df['species_true'].str.replace(' ', '_').str.replace('-', '_').values
le = LabelEncoder()
le.fit(classes)
y_all = le.transform(true_labels)

train_mask = (df['set'] == 'train').values
test_mask = (df['set'] == 'holdout').values
y_train, y_test = y_all[train_mask], y_all[test_mask]

X_train = np.hstack([model_probs[m][train_mask] for m in models])
X_test = np.hstack([model_probs[m][test_mask] for m in models])

# Train the ensemble
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

T = 0.3  # Temperature scaling

def get_confidence(X):
    """Get predictions and confidence for samples."""
    logits = lr.decision_function(X)
    probs = softmax(logits / T, axis=1)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    return preds, confs

# Real data baseline
print("\n" + "=" * 80)
print("BASELINE: REAL DATA")
print("=" * 80)
preds, confs = get_confidence(X_test)
print(f"Real test data (n={len(X_test)}):")
print(f"  Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"  Confidence: {confs.mean():.4f} (min={confs.min():.4f}, max={confs.max():.4f})")

# Generate various types of bad data
n_fake = 100
np.random.seed(42)

print("\n" + "=" * 80)
print("FAKE DATA TYPES")
print("=" * 80)

results = []

# 1. Pure random noise (uniform probabilities)
print("\n1. RANDOM NOISE (uniform random probs)")
X_random = np.random.rand(n_fake, X_train.shape[1])
# Normalize each model's probs to sum to 1
for i in range(3):
    start = i * n_classes
    end = start + n_classes
    X_random[:, start:end] = X_random[:, start:end] / X_random[:, start:end].sum(axis=1, keepdims=True)

preds, confs = get_confidence(X_random)
print(f"  Confidence: {confs.mean():.4f} (min={confs.min():.4f}, max={confs.max():.4f})")
results.append(("Random noise", confs.mean(), confs.min()))

# 2. All models disagree (each confident in different class)
print("\n2. MODEL DISAGREEMENT (each model picks different class)")
X_disagree = np.zeros((n_fake, X_train.shape[1]))
for i in range(n_fake):
    for j in range(3):
        start = j * n_classes
        cls = np.random.randint(0, n_classes)
        X_disagree[i, start + cls] = 0.95
        # Small prob for others
        for k in range(n_classes):
            if k != cls:
                X_disagree[i, start + k] = 0.05 / (n_classes - 1)

preds, confs = get_confidence(X_disagree)
print(f"  Confidence: {confs.mean():.4f} (min={confs.min():.4f}, max={confs.max():.4f})")
results.append(("Model disagreement", confs.mean(), confs.min()))

# 3. Uniform predictions (all classes equal prob)
print("\n3. UNIFORM (all classes equal probability)")
X_uniform = np.ones((n_fake, X_train.shape[1])) / n_classes

preds, confs = get_confidence(X_uniform)
print(f"  Confidence: {confs.mean():.4f} (min={confs.min():.4f}, max={confs.max():.4f})")
results.append(("Uniform probs", confs.mean(), confs.min()))

# 4. Perturbed real data (add noise to real samples)
print("\n4. PERTURBED REAL (real data + gaussian noise)")
noise_level = 0.3
X_perturbed = X_test[:n_fake].copy() + np.random.randn(n_fake, X_train.shape[1]) * noise_level
X_perturbed = np.clip(X_perturbed, 0, 1)
# Renormalize
for i in range(3):
    start = i * n_classes
    end = start + n_classes
    X_perturbed[:, start:end] = X_perturbed[:, start:end] / (X_perturbed[:, start:end].sum(axis=1, keepdims=True) + 1e-10)

preds, confs = get_confidence(X_perturbed)
acc = accuracy_score(y_test[:n_fake], preds)
print(f"  Accuracy: {acc:.4f}")
print(f"  Confidence: {confs.mean():.4f} (min={confs.min():.4f}, max={confs.max():.4f})")
results.append(("Perturbed real", confs.mean(), confs.min()))

# 5. Swapped model outputs (wrong model order)
print("\n5. SWAPPED MODELS (model outputs in wrong order)")
X_swapped = np.hstack([
    model_probs['tcn'][test_mask][:n_fake],      # tcn pretending to be cnn
    model_probs['cnn'][test_mask][:n_fake],      # cnn pretending to be statistics
    model_probs['statistics'][test_mask][:n_fake] # statistics pretending to be tcn
])

preds, confs = get_confidence(X_swapped)
acc = accuracy_score(y_test[:n_fake], preds)
print(f"  Accuracy: {acc:.4f}")
print(f"  Confidence: {confs.mean():.4f} (min={confs.min():.4f}, max={confs.max():.4f})")
results.append(("Swapped models", confs.mean(), confs.min()))

# 6. Interpolation between random classes
print("\n6. CLASS INTERPOLATION (blend between 2 random classes)")
X_interp = np.zeros((n_fake, X_train.shape[1]))
for i in range(n_fake):
    c1, c2 = np.random.choice(n_classes, 2, replace=False)
    alpha = np.random.rand()
    for j in range(3):
        start = j * n_classes
        X_interp[i, start + c1] = alpha * 0.9
        X_interp[i, start + c2] = (1 - alpha) * 0.9
        # Small prob for others
        remaining = 0.1
        for k in range(n_classes):
            if k != c1 and k != c2:
                X_interp[i, start + k] = remaining / (n_classes - 2)

preds, confs = get_confidence(X_interp)
print(f"  Confidence: {confs.mean():.4f} (min={confs.min():.4f}, max={confs.max():.4f})")
results.append(("Class interpolation", confs.mean(), confs.min()))

# 7. Adversarial-like: all models very confident but in different classes
print("\n7. ADVERSARIAL (all models 99% confident, different classes)")
X_adv = np.zeros((n_fake, X_train.shape[1]))
for i in range(n_fake):
    classes_picked = np.random.choice(n_classes, 3, replace=False)
    for j in range(3):
        start = j * n_classes
        X_adv[i, start + classes_picked[j]] = 0.99
        for k in range(n_classes):
            if k != classes_picked[j]:
                X_adv[i, start + k] = 0.01 / (n_classes - 1)

preds, confs = get_confidence(X_adv)
print(f"  Confidence: {confs.mean():.4f} (min={confs.min():.4f}, max={confs.max():.4f})")
results.append(("Adversarial", confs.mean(), confs.min()))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Data Type':<25} {'Mean Conf':>12} {'Min Conf':>12}")
print("-" * 50)
print(f"{'Real test data':<25} {0.9993:>12.4f} {0.9531:>12.4f}")
for name, mean_conf, min_conf in results:
    print(f"{name:<25} {mean_conf:>12.4f} {min_conf:>12.4f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Find threshold
real_min = 0.9531
bad_maxes = [r[1] for r in results]  # mean confidence of bad data

print(f"\nReal data min confidence: {real_min:.4f}")
print(f"Bad data max mean confidence: {max(bad_maxes):.4f}")

if real_min > max(bad_maxes):
    print(f"\nGOOD NEWS: You can set a threshold!")
    threshold = (real_min + max(bad_maxes)) / 2
    print(f"Suggested threshold: {threshold:.4f}")
    print(f"  - Real data: all above {real_min:.4f}")
    print(f"  - Most bad data: below {max(bad_maxes):.4f}")
else:
    print(f"\nWARNING: Some bad data has higher confidence than real data.")
    print("Consider using additional validation.")
