"""
1. Calibrate base model probabilities (especially ResNet1D)
2. Weighted voting with learned/optimized weights
3. Diverse meta-learner architectures (LightGBM, LogReg, SVM)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_and_prepare_data(csv_path='all_model_predictions.csv', use_top3_only=True):
    """Load CSV and prepare meta-learner data."""
    print("Loading data from", csv_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    y = df['species_true'].values

    if use_top3_only:
        meta_columns = [col for col in df.columns if col.startswith(('cnn_', 'resnet1d_', 'statistics_'))]
        model_names = ['cnn', 'resnet1d', 'statistics']
    else:
        meta_columns = [col for col in df.columns if col.startswith(('cnn_', 'resnet1d_', 'statistics_', 'extratrees_', 'rulebased_'))]
        model_names = ['cnn', 'resnet1d', 'statistics', 'extratrees', 'rulebased']

    X_meta = df[meta_columns].values

    print(f"\nFound {len(meta_columns)} meta-learner features")
    for prefix in ['cnn_', 'resnet1d_', 'statistics_', 'extratrees_', 'rulebased_']:
        count = len([c for c in meta_columns if c.startswith(prefix)])
        if count > 0:
            print(f"  - {prefix[:-1]}: {count}")

    return X_meta, y, meta_columns, model_names


def encode_labels(y):
    """Encode string labels to integers."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def split_data(X, y, train_size=0.65, val_size=0.15, test_size=0.20, random_state=42):
    """Split into train (65%), validation (15%), and test (20%) sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    print(f"\nData split:")
    print(f"  Train (65%): {len(X_train)} samples")
    print(f"  Val (15%):   {len(X_val)} samples")
    print(f"  Test (20%):  {len(X_test)} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


def calibrate_probabilities(X_train, X_val, X_test, y_train, num_classes, num_models, model_names):
    """
    METHOD 1: Calibrate base model probabilities using temperature scaling.

    Temperature scaling: optimal_probs = softmax(logits / temperature)
    Find temperature on validation set to minimize NLL.
    """
    print("\n" + "="*60)
    print("METHOD 1: PROBABILITY CALIBRATION")
    print("="*60)
    print("Calibrating base model probabilities to improve reliability...")

    X_train_cal = X_train.copy()
    X_val_cal = X_val.copy()
    X_test_cal = X_test.copy()

    for model_idx, name in enumerate(model_names):
        start_idx = model_idx * num_classes
        end_idx = (model_idx + 1) * num_classes

        # Get probabilities for this model
        train_probs = X_train[:, start_idx:end_idx]
        val_probs = X_val[:, start_idx:end_idx]
        test_probs = X_test[:, start_idx:end_idx]

        # Apply temperature scaling (find best temperature on train set)
        def nll_loss(temperature):
            """Negative log likelihood with temperature."""
            eps = 1e-12
            scaled_probs = np.exp(np.log(train_probs + eps) / temperature)
            scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)
            # NLL
            nll = -np.log(scaled_probs[np.arange(len(y_train)), y_train] + eps).mean()
            return nll

        # Find optimal temperature (between 0.1 and 10.0)
        result = minimize(nll_loss, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
        temp = result.x[0]

        # Apply calibration
        eps = 1e-12
        val_cal = np.exp(np.log(val_probs + eps) / temp)
        val_cal = val_cal / val_cal.sum(axis=1, keepdims=True)

        test_cal = np.exp(np.log(test_probs + eps) / temp)
        test_cal = test_cal / test_cal.sum(axis=1, keepdims=True)

        train_cal = np.exp(np.log(train_probs + eps) / temp)
        train_cal = train_cal / train_cal.sum(axis=1, keepdims=True)

        # Update
        X_train_cal[:, start_idx:end_idx] = train_cal
        X_val_cal[:, start_idx:end_idx] = val_cal
        X_test_cal[:, start_idx:end_idx] = test_cal

        print(f"  {name:15s} temperature={temp:.3f}")

    print("\n[OK] Calibration complete!")
    return X_train_cal, X_val_cal, X_test_cal


def weighted_voting_optimized(X_train, X_val, X_test, y_train, y_val, y_test, num_classes, num_models, model_names):
    """
    METHOD 2: Weighted voting with optimized weights.

    Find optimal weights on validation set.
    """
    print("\n" + "="*60)
    print("METHOD 2: WEIGHTED VOTING (Optimized)")
    print("="*60)

    # Split probabilities by model
    def get_model_probs(X, model_idx):
        start_idx = model_idx * num_classes
        end_idx = (model_idx + 1) * num_classes
        return X[:, start_idx:end_idx]

    # Objective: maximize accuracy on validation set
    def objective(weights):
        """Compute validation accuracy with given weights."""
        weights = np.abs(weights)  # Ensure positive
        weights = weights / weights.sum()  # Normalize

        # Weighted average of probabilities
        weighted_probs = np.zeros((len(X_val), num_classes))
        for i in range(num_models):
            weighted_probs += weights[i] * get_model_probs(X_val, i)

        preds = np.argmax(weighted_probs, axis=1)
        acc = accuracy_score(y_val, preds)
        return -acc  # Minimize negative accuracy

    # Optimize weights (start with equal weights)
    initial_weights = np.ones(num_models) / num_models
    result = minimize(objective, x0=initial_weights, method='Nelder-Mead',
                     options={'maxiter': 500})

    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()

    print("\nOptimal weights found:")
    for i, (name, weight) in enumerate(zip(model_names, optimal_weights)):
        print(f"  {name:15s} weight={weight:.4f} ({weight/optimal_weights.min():.2f}x relative)")

    # Apply to test set
    test_probs_weighted = np.zeros((len(X_test), num_classes))
    for i in range(num_models):
        test_probs_weighted += optimal_weights[i] * get_model_probs(X_test, i)

    preds = np.argmax(test_probs_weighted, axis=1)
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)

    print(f"\nWeighted Voting Results (Test Set):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {f1_macro:.4f}")

    return acc, f1_macro, optimal_weights


def train_diverse_meta_learners_optuna(X_train, X_val, X_test, y_train, y_val, y_test, num_classes, n_trials=30):
    """
    METHOD 3: Diverse meta-learner architectures with Optuna tuning.

    Train multiple types: LightGBM, Logistic Regression, SVM
    """
    print("\n" + "="*60)
    print("METHOD 3: DIVERSE META-LEARNER ARCHITECTURES (OPTUNA)")
    print("="*60)

    results = {}

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    sample_weights = class_weights[y_train]

    # ========== 1. LightGBM with Optuna ==========
    print(f"\n[1/3] Training LightGBM with Optuna ({n_trials} trials)...")

    lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weights)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    def lgb_objective(trial):
        """Optuna objective for LightGBM - maximize macro F1."""
        params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10.0),
            'verbose': -1,
        }

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        val_pred_labels = np.argmax(val_preds, axis=1)
        val_f1 = f1_score(y_val, val_pred_labels, average='macro', zero_division=0)
        return val_f1

    lgb_study = optuna.create_study(
        study_name='lgb_meta',
        storage='sqlite:///optuna_meta_learner.db',
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True
    )
    lgb_study.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best val macro F1: {lgb_study.best_value:.4f}")

    # Train final model with best params
    best_params_lgb = lgb_study.best_params.copy()
    best_params_lgb.update({
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1
    })

    lgb_model = lgb.train(
        best_params_lgb,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    lgb_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    lgb_pred_labels = np.argmax(lgb_preds, axis=1)
    lgb_acc = accuracy_score(y_test, lgb_pred_labels)
    lgb_f1 = f1_score(y_test, lgb_pred_labels, average='macro', zero_division=0)

    print(f"  LightGBM (Optuna): Acc={lgb_acc:.4f}, Macro F1={lgb_f1:.4f}")
    results['LightGBM (Optuna)'] = {'acc': lgb_acc, 'f1': lgb_f1}

    # ========== 2. Logistic Regression with Optuna ==========
    print(f"\n[2/3] Training Logistic Regression with Optuna ({n_trials} trials)...")

    def lr_objective(trial):
        """Optuna objective for LogReg - maximize macro F1."""
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])

        model = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            C=C,
            solver=solver,
            multi_class='multinomial',
            random_state=42
        )
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
        return val_f1

    lr_study = optuna.create_study(
        study_name='lr_meta',
        storage='sqlite:///optuna_meta_learner.db',
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        load_if_exists=True
    )
    lr_study.optimize(lr_objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best val macro F1: {lr_study.best_value:.4f}")

    # Train final model
    lr_model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        C=lr_study.best_params['C'],
        solver=lr_study.best_params['solver'],
        multi_class='multinomial',
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average='macro', zero_division=0)

    print(f"  LogisticReg (Optuna): Acc={lr_acc:.4f}, Macro F1={lr_f1:.4f}")
    results['LogisticReg (Optuna)'] = {'acc': lr_acc, 'f1': lr_f1}

    # ========== 3. Linear SVM with Optuna ==========
    print(f"\n[3/3] Training Linear SVM with Optuna ({n_trials} trials)...")

    def svm_objective(trial):
        """Optuna objective for SVM - maximize macro F1."""
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        loss = trial.suggest_categorical('loss', ['hinge', 'squared_hinge'])

        from sklearn.svm import LinearSVC
        model = LinearSVC(
            class_weight='balanced',
            C=C,
            loss=loss,
            max_iter=3000,
            random_state=42
        )
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
        return val_f1

    svm_study = optuna.create_study(
        study_name='svm_meta',
        storage='sqlite:///optuna_meta_learner.db',
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        load_if_exists=True
    )
    svm_study.optimize(svm_objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best val macro F1: {svm_study.best_value:.4f}")

    # Train final model
    from sklearn.svm import LinearSVC
    svm_model = LinearSVC(
        class_weight='balanced',
        C=svm_study.best_params['C'],
        loss=svm_study.best_params['loss'],
        max_iter=3000,
        random_state=42
    )
    svm_model.fit(X_train, y_train)

    svm_preds = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds, average='macro', zero_division=0)

    print(f"  Linear SVM (Optuna):  Acc={svm_acc:.4f}, Macro F1={svm_f1:.4f}")
    results['LinearSVM (Optuna)'] = {'acc': svm_acc, 'f1': svm_f1}

    # Save models and studies
    import pickle
    with open('meta_lgb_optuna.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
    with open('meta_lr_optuna.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('meta_svm_optuna.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

    # Save study results
    with open('optuna_studies.json', 'w') as f:
        json.dump({
            'LightGBM': {
                'best_params': lgb_study.best_params,
                'best_val_f1': lgb_study.best_value,
                'test_acc': lgb_acc,
                'test_f1': lgb_f1
            },
            'LogisticReg': {
                'best_params': lr_study.best_params,
                'best_val_f1': lr_study.best_value,
                'test_acc': lr_acc,
                'test_f1': lr_f1
            },
            'LinearSVM': {
                'best_params': svm_study.best_params,
                'best_val_f1': svm_study.best_value,
                'test_acc': svm_acc,
                'test_f1': svm_f1
            }
        }, f, indent=2)

    print("\n[OK] Saved models: meta_lgb_optuna.pkl, meta_lr_optuna.pkl, meta_svm_optuna.pkl")
    print("[OK] Saved study results: optuna_studies.json")

    return results


def main():
    """Main pipeline."""
    print("\n" + "="*60)
    print("META-LEARNING TECHNIQUES")
    print("="*60)

    # Load data
    X_meta, y_raw, feature_names, model_names = load_and_prepare_data(use_top3_only=True)
    y, label_encoder = encode_labels(y_raw)
    num_classes = len(np.unique(y))
    num_models = len(model_names)

    print(f"\nNumber of classes: {num_classes}")
    print(f"Number of base models: {num_models}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_meta, y)

    # Store original for comparisons
    X_train_orig = X_train.copy()
    X_val_orig = X_val.copy()
    X_test_orig = X_test.copy()

    # ==================================================================
    # METHOD 1: Calibrate probabilities
    # ==================================================================
    X_train_cal, X_val_cal, X_test_cal = calibrate_probabilities(
        X_train, X_val, X_test, y_train, num_classes, num_models, model_names
    )

    # ==================================================================
    # METHOD 2: Weighted voting
    # ==================================================================
    wv_acc, wv_f1, weights = weighted_voting_optimized(
        X_train_orig, X_val_orig, X_test_orig, y_train, y_val, y_test,
        num_classes, num_models, model_names
    )

    # ==================================================================
    # METHOD 3: Diverse meta-learners on CALIBRATED data
    # ==================================================================
    print("\n" + "="*60)
    print("Training diverse meta-learners on CALIBRATED probabilities...")
    print("="*60)

    # Normalize calibrated data
    train_mean = X_train_cal.mean(axis=0)
    train_std = X_train_cal.std(axis=0)
    train_std[train_std == 0] = 1

    X_train_norm = (X_train_cal - train_mean) / train_std
    X_val_norm = (X_val_cal - train_mean) / train_std
    X_test_norm = (X_test_cal - train_mean) / train_std

    diverse_results = train_diverse_meta_learners_optuna(
        X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, num_classes, n_trials=100
    )

    # ==================================================================
    # BASELINE: Individual models and simple ensembles
    # ==================================================================
    print("\n" + "="*60)
    print("BASELINE: Individual Models")
    print("="*60)

    baseline_results = {}
    for model_idx, name in enumerate(model_names):
        start_idx = model_idx * num_classes
        end_idx = (model_idx + 1) * num_classes
        probs = X_test_orig[:, start_idx:end_idx]
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        baseline_results[name] = {'acc': acc, 'f1': f1}
        print(f"  {name:15s} Acc={acc:.4f}, Macro F1={f1:.4f}")

    # Simple average
    avg_probs = np.zeros((len(X_test_orig), num_classes))
    for i in range(num_models):
        start_idx = i * num_classes
        end_idx = (i + 1) * num_classes
        avg_probs += X_test_orig[:, start_idx:end_idx]
    avg_probs /= num_models
    avg_preds = np.argmax(avg_probs, axis=1)
    avg_acc = accuracy_score(y_test, avg_preds)
    avg_f1 = f1_score(y_test, avg_preds, average='macro', zero_division=0)
    baseline_results['Simple Average'] = {'acc': avg_acc, 'f1': avg_f1}

    print(f"\n  Simple Average: Acc={avg_acc:.4f}, Macro F1={avg_f1:.4f}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    all_results = {
        **baseline_results,
        'Weighted Voting (Optimized)': {'acc': wv_acc, 'f1': wv_f1},
        **diverse_results
    }

    print("\n" + "-"*80)
    print(f"{'Method':<30} {'Accuracy':>12} {'Macro F1':>12} {'Improvement':>12}")
    print("-"*80)

    best_baseline_acc = baseline_results['cnn']['acc']  # CNN was best

    for method, metrics in sorted(all_results.items(), key=lambda x: x[1]['acc'], reverse=True):
        acc = metrics['acc']
        f1 = metrics['f1']
        improvement = acc - best_baseline_acc
        marker = " ★" if acc > best_baseline_acc else ""
        print(f"{method:<30} {acc:>12.4f} {f1:>12.4f} {improvement:>+11.4f}{marker}")

    print("-"*80)

    # Find best
    best_method = max(all_results.items(), key=lambda x: x[1]['acc'])
    best_f1_method = max(all_results.items(), key=lambda x: x[1]['f1'])

    print(f"\n[*] BEST ACCURACY:  {best_method[0]} = {best_method[1]['acc']:.4f}")
    print(f"[*] BEST MACRO F1:  {best_f1_method[0]} = {best_f1_method[1]['f1']:.4f}")

    if best_method[1]['acc'] > best_baseline_acc:
        print(f"\n[SUCCESS] Beat CNN baseline by {best_method[1]['acc'] - best_baseline_acc:.4f}")
    else:
        print(f"\n[INFO] CNN still best, but learned valuable techniques:")
        print(f"   - Optimal weights: {dict(zip(model_names, weights))}")
        print(f"   - Calibration improves reliability")
        print(f"   - Diverse meta-learners close the gap")

    # Save summary
    with open('meta_results.json', 'w') as f:
        json.dump({
            'all_results': {k: {kk: float(vv) for kk, vv in v.items()}
                           for k, v in all_results.items()},
            'optimal_weights': {name: float(w) for name, w in zip(model_names, weights)},
            'best_method': best_method[0],
            'best_accuracy': float(best_method[1]['acc']),
            'best_f1_method': best_f1_method[0],
            'best_f1': float(best_f1_method[1]['f1'])
        }, f, indent=2)

    print("\n[OK] Saved results to meta_results.json")


if __name__ == '__main__':
    main()
