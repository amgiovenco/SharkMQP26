"""
Comprehensive Ensemble Experiments for Shark Classification
Tries many different ensemble strategies to maximize accuracy.
Uses 80/20 train/test split for proper evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

# Output directory for results
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE ENSEMBLE EXPERIMENTS (with train/test split)")
print("=" * 80)

# Load data
print("\nLoading data...")
df = pd.read_csv('all_model_predictions.csv')

# Extract models and classes
models = ['cnn', 'resnet1d', 'extratrees', 'statistics', 'rulebased', 'tcn']
class_cols = [c for c in df.columns if c.startswith('cnn_prob_')]
classes = [c.replace('cnn_prob_', '') for c in class_cols]
n_classes = len(classes)

print(f"Total samples: {len(df)}, Classes: {n_classes}, Models: {len(models)}")

# Extract predictions for each model
model_probs = {}
model_preds = {}
for model in models:
    prob_cols = [f"{model}_prob_{c}" for c in classes]
    model_probs[model] = df[prob_cols].values
    model_preds[model] = np.argmax(model_probs[model], axis=1)

# Get true labels
true_labels_str = df['species_true'].str.replace(' ', '_').str.replace('-', '_').values
le = LabelEncoder()
le.fit(classes)
y_all = le.transform(true_labels_str)

# Split into train/test based on 'set' column
train_mask = (df['set'] == 'train').values
test_mask = (df['set'] == 'holdout').values

n_train = train_mask.sum()
n_test = test_mask.sum()
print(f"Train samples: {n_train}, Test samples: {n_test}")

y_train = y_all[train_mask]
y_test = y_all[test_mask]

# Split model probs/preds
model_probs_train = {m: model_probs[m][train_mask] for m in models}
model_probs_test = {m: model_probs[m][test_mask] for m in models}
model_preds_train = {m: model_preds[m][train_mask] for m in models}
model_preds_test = {m: model_preds[m][test_mask] for m in models}

# Store results
results = []

def evaluate(y_pred, name):
    """Evaluate on TEST set"""
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append({
        'method': name,
        'accuracy': acc,
        'f1_macro': f1,
        'correct': int(np.sum(y_pred == y_test)),
        'total': len(y_test)
    })
    print(f"  {name}: {acc:.4f} acc, {f1:.4f} F1")
    return acc, f1

print("\n" + "=" * 80)
print("1. BASELINE: INDIVIDUAL MODELS")
print("=" * 80)
for model in models:
    evaluate(model_preds_test[model], f"individual_{model}")

print("\n" + "=" * 80)
print("2. SIMPLE AVERAGING / VOTING")
print("=" * 80)

# Simple average
avg_probs = np.mean([model_probs_test[m] for m in models], axis=0)
evaluate(np.argmax(avg_probs, axis=1), "simple_average_all")

# Majority voting
all_preds = np.stack([model_preds_test[m] for m in models], axis=1)
majority_preds = mode(all_preds, axis=1, keepdims=False)[0]
evaluate(majority_preds, "majority_vote_all")

# Subsets
for subset_name, subset in [('cnn_stats_tcn', ['cnn', 'statistics', 'tcn']),
                             ('cnn_resnet1d', ['cnn', 'resnet1d']),
                             ('top4', ['cnn', 'statistics', 'tcn', 'resnet1d'])]:
    avg = np.mean([model_probs_test[m] for m in subset], axis=0)
    evaluate(np.argmax(avg, axis=1), f"avg_{subset_name}")

print("\n" + "=" * 80)
print("3. WEIGHTED AVERAGING (weights from train accuracy)")
print("=" * 80)

# Get train accuracies
train_accs = {m: accuracy_score(y_train, model_preds_train[m]) for m in models}
print(f"  Train accs: {', '.join([f'{m}={train_accs[m]:.3f}' for m in models])}")

# Weight by accuracy
weights = np.array([train_accs[m] for m in models])
weights = weights / weights.sum()

weighted = np.zeros((n_test, n_classes))
for i, m in enumerate(models):
    weighted += weights[i] * model_probs_test[m]
evaluate(np.argmax(weighted, axis=1), "weighted_by_train_acc")

# Squared weights
weights2 = np.array([train_accs[m]**2 for m in models])
weights2 = weights2 / weights2.sum()
weighted2 = sum(weights2[i] * model_probs_test[models[i]] for i in range(len(models)))
evaluate(np.argmax(weighted2, axis=1), "weighted_by_train_acc_sq")

print("\n" + "=" * 80)
print("4. CONFIDENCE-WEIGHTED VOTING")
print("=" * 80)

# Weight by per-sample confidence
def conf_weighted(probs_dict, models_list):
    weighted = np.zeros((len(list(probs_dict.values())[0]), n_classes))
    for m in models_list:
        probs = probs_dict[m]
        conf = np.max(probs, axis=1, keepdims=True)
        weighted += probs * conf
    return np.argmax(weighted, axis=1)

evaluate(conf_weighted(model_probs_test, models), "confidence_weighted_all")
evaluate(conf_weighted(model_probs_test, ['cnn', 'statistics', 'tcn']), "confidence_weighted_top3")

print("\n" + "=" * 80)
print("5. OPTIMIZED WEIGHTS (grid search on train, eval on test)")
print("=" * 80)

# Grid search on train set
best_train_acc = 0
best_weights = None
for w1 in np.arange(0.2, 0.6, 0.05):
    for w2 in np.arange(0.1, 0.4, 0.05):
        for w3 in np.arange(0.1, 0.4, 0.05):
            remaining = 1 - w1 - w2 - w3
            if remaining < 0.05:
                continue
            w_other = remaining / 3
            weights = [w1, w_other, w_other, w2, w_other, w3]  # cnn, resnet, extra, stats, rule, tcn

            weighted = sum(weights[i] * model_probs_train[models[i]] for i in range(len(models)))
            preds = np.argmax(weighted, axis=1)
            acc = accuracy_score(y_train, preds)
            if acc > best_train_acc:
                best_train_acc = acc
                best_weights = weights

print(f"  Best weights (train acc={best_train_acc:.4f}):")
print(f"    {dict(zip(models, [f'{w:.3f}' for w in best_weights]))}")

# Apply to test
weighted_test = sum(best_weights[i] * model_probs_test[models[i]] for i in range(len(models)))
evaluate(np.argmax(weighted_test, axis=1), "grid_optimized_weights")

print("\n" + "=" * 80)
print("6. SCIPY OPTIMIZATION (on train)")
print("=" * 80)

def neg_accuracy(w, probs_list, y):
    w = np.abs(w)
    w = w / w.sum()
    weighted = sum(w[i] * probs_list[i] for i in range(len(probs_list)))
    return -accuracy_score(y, np.argmax(weighted, axis=1))

train_probs_list = [model_probs_train[m] for m in models]
x0 = np.ones(len(models)) / len(models)
result = minimize(neg_accuracy, x0, args=(train_probs_list, y_train), method='Nelder-Mead')
opt_w = np.abs(result.x)
opt_w = opt_w / opt_w.sum()

print(f"  Optimized: {dict(zip(models, [f'{w:.3f}' for w in opt_w]))}")
test_probs_list = [model_probs_test[m] for m in models]
weighted_opt = sum(opt_w[i] * test_probs_list[i] for i in range(len(models)))
evaluate(np.argmax(weighted_opt, axis=1), "scipy_optimized")

print("\n" + "=" * 80)
print("7. STACKING (train on train, predict on test)")
print("=" * 80)

X_train = np.hstack([model_probs_train[m] for m in models])
X_test = np.hstack([model_probs_test[m] for m in models])
print(f"  Features: {X_train.shape[1]}")

# Various meta-learners
meta_learners = [
    ('logreg_C0.1', LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
    ('logreg_C1', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ('logreg_C10', LogisticRegression(max_iter=1000, C=10.0, random_state=42)),
    ('ridge', RidgeClassifier(alpha=1.0, random_state=42)),
    ('gradboost', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ('mlp_100', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
    ('mlp_50', MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)),
]

for name, clf in meta_learners:
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    evaluate(preds, f"stack_{name}")

print("\n" + "=" * 80)
print("8. STACKING WITH REDUCED FEATURES")
print("=" * 80)

# Top 3 only
top3 = ['cnn', 'statistics', 'tcn']
X_train_top3 = np.hstack([model_probs_train[m] for m in top3])
X_test_top3 = np.hstack([model_probs_test[m] for m in top3])

lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train_top3, y_train)
evaluate(lr.predict(X_test_top3), "stack_logreg_top3")

# With L2 regularization tuned
for C in [0.01, 0.1, 1.0, 10.0]:
    lr = LogisticRegression(max_iter=1000, C=C, random_state=42)
    lr.fit(X_train, y_train)
    # Don't print all, just track
    acc = accuracy_score(y_test, lr.predict(X_test))
    if C == 0.01:
        evaluate(lr.predict(X_test), "stack_logreg_C0.01")

print("\n" + "=" * 80)
print("9. CLASS-SPECIFIC ROUTING")
print("=" * 80)

# Learn best model per class on train
best_model_per_class = {}
for c_idx in range(n_classes):
    mask = y_train == c_idx
    if mask.sum() == 0:
        best_model_per_class[c_idx] = 'cnn'
        continue
    best_acc = 0
    best_m = 'cnn'
    for m in models:
        acc = accuracy_score(y_train[mask], model_preds_train[m][mask])
        if acc > best_acc:
            best_acc = acc
            best_m = m
    best_model_per_class[c_idx] = best_m

# Apply: use ensemble prediction to pick specialist
avg_test = np.mean([model_probs_test[m] for m in models], axis=0)
ensemble_preds = np.argmax(avg_test, axis=1)
routed = np.array([model_preds_test[best_model_per_class[ensemble_preds[i]]][i] for i in range(n_test)])
evaluate(routed, "class_routing")

print("\n" + "=" * 80)
print("10. CALIBRATED ENSEMBLE")
print("=" * 80)

# Temperature scale extratrees (worst calibrated)
def temp_scale(probs, T):
    logits = np.log(probs + 1e-10)
    return softmax(logits / T, axis=1)

# Find best T on train
best_T = 1.0
best_acc = 0
for T in np.arange(0.3, 3.0, 0.1):
    scaled = temp_scale(model_probs_train['extratrees'], T)
    # Replace in ensemble
    train_probs = [model_probs_train[m] if m != 'extratrees' else scaled for m in models]
    avg = np.mean(train_probs, axis=0)
    acc = accuracy_score(y_train, np.argmax(avg, axis=1))
    if acc > best_acc:
        best_acc = acc
        best_T = T

print(f"  Best temperature for extratrees: {best_T:.1f}")
scaled_test = temp_scale(model_probs_test['extratrees'], best_T)
test_probs = [model_probs_test[m] if m != 'extratrees' else scaled_test for m in models]
evaluate(np.argmax(np.mean(test_probs, axis=0), axis=1), "calibrated_avg")

print("\n" + "=" * 80)
print("11. RANK-BASED ENSEMBLE")
print("=" * 80)

def prob_to_rank(probs):
    ranks = np.zeros_like(probs)
    for i in range(len(probs)):
        ranks[i] = np.argsort(np.argsort(-probs[i]))
    return ranks

model_ranks = {m: prob_to_rank(model_probs_test[m]) for m in models}
avg_ranks = np.mean([model_ranks[m] for m in models], axis=0)
evaluate(np.argmin(avg_ranks, axis=1), "rank_average")

# Borda count
borda = sum(n_classes - 1 - model_ranks[m] for m in models)
evaluate(np.argmax(borda, axis=1), "borda_count")

print("\n" + "=" * 80)
print("12. PRODUCT OF EXPERTS")
print("=" * 80)

log_probs = sum(np.log(model_probs_test[m] + 1e-10) for m in models)
geo_mean = np.exp(log_probs / len(models))
evaluate(np.argmax(geo_mean, axis=1), "geometric_mean")

print("\n" + "=" * 80)
print("13. MAX CONFIDENCE SELECTION")
print("=" * 80)

max_conf_preds = np.zeros(n_test, dtype=int)
for i in range(n_test):
    best_conf = 0
    for m in models:
        conf = np.max(model_probs_test[m][i])
        if conf > best_conf:
            best_conf = conf
            max_conf_preds[i] = np.argmax(model_probs_test[m][i])
evaluate(max_conf_preds, "max_confidence")

print("\n" + "=" * 80)
print("14. DISAGREEMENT-AWARE")
print("=" * 80)

preds_out = np.zeros(n_test, dtype=int)
for i in range(n_test):
    votes = [model_preds_test[m][i] for m in models]
    unique, counts = np.unique(votes, return_counts=True)
    if counts.max() >= 4:
        preds_out[i] = unique[counts.argmax()]
    else:
        weighted = sum(train_accs[m] * model_probs_test[m][i] for m in models)
        preds_out[i] = np.argmax(weighted)
evaluate(preds_out, "disagreement_aware")

print("\n" + "=" * 80)
print("15. BLENDING META-LEARNERS")
print("=" * 80)

lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)

blended = (lr_probs + rf_probs) / 2
evaluate(np.argmax(blended, axis=1), "blended_lr_rf")

print("\n" + "=" * 80)
print("=" * 80)
print("FINAL RESULTS - SORTED BY ACCURACY")
print("=" * 80)

results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)

print(f"\n{'Method':<40} {'Accuracy':>10} {'F1-Macro':>10} {'Correct':>10}")
print("-" * 75)
for _, row in results_df.iterrows():
    print(f"{row['method']:<40} {row['accuracy']:>10.4f} {row['f1_macro']:>10.4f} {row['correct']:>7}/{row['total']}")

print("\n" + "=" * 80)
print("TOP 10 METHODS")
print("=" * 80)
for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
    print(f"{i:2}. {row['method']:<35} {row['accuracy']:.4f} ({row['correct']}/{row['total']})")

print("\n" + "=" * 80)
print("BEST METHOD")
print("=" * 80)
best = results_df.iloc[0]
print(f"\n  {best['method']}")
print(f"  Accuracy: {best['accuracy']:.4f} ({best['correct']}/{best['total']})")
print(f"  F1-Macro: {best['f1_macro']:.4f}")

best_ind = results_df[results_df['method'].str.startswith('individual_')]['accuracy'].max()
print(f"\n  Improvement over best individual: +{best['accuracy'] - best_ind:.4f}")

results_df.to_csv(RESULTS_DIR / 'ensemble_experiments_results.csv', index=False)
print(f"\n[SUCCESS] Results saved to {RESULTS_DIR / 'ensemble_experiments_results.csv'}")
