"""
Meta-learner optimization for shark ensemble using Optuna.
Uses 80/20 train/holdout split from all_model_predictions.csv.
- Train on 80% with 5-fold CV for hyperparameter tuning
- Test on 20% holdout set for final evaluation
"""

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pickle
import json
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from datetime import datetime

# ======================================================================
# 0. DEVICE SETUP (CUDA/CPU)
# ======================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    torch.cuda.reset_peak_memory_stats()
else:
    print("CUDA not available - using CPU")

# ======================================================================
# 1. OPTUNA DATABASE SETUP
# ======================================================================

# SQLite database for Optuna studies
OPTUNA_DB_PATH = Path("optuna_studies.db")
OPTUNA_DB_URL = f"sqlite:///{OPTUNA_DB_PATH.absolute()}"

print(f"Optuna database: {OPTUNA_DB_PATH.absolute()}")

# ======================================================================
# 1. LOAD PREDICTIONS FROM CSV (80/20 SPLIT)
# ======================================================================

print("Loading predictions from all_model_predictions.csv...")
df = pd.read_csv('all_model_predictions.csv')

# Extract metadata
true_labels_str = df['species_true'].values
set_labels = df['set'].values
indices = df['index'].values

# Get train (80%) and holdout (20%) splits
train_mask = set_labels == 'train'
holdout_mask = set_labels == 'holdout'

X_train_full = df[train_mask].copy()
X_holdout_full = df[holdout_mask].copy()

y_train = X_train_full['species_true'].values
y_holdout = X_holdout_full['species_true'].values

print(f"Loaded {len(df)} total samples")
print(f"  Train (80%): {len(X_train_full)} samples")
print(f"  Holdout (20%): {len(X_holdout_full)} samples")

# Create species mapping
species_list = sorted(df['species_true'].unique())
species_to_idx = {sp: idx for idx, sp in enumerate(species_list)}
print(f"  Number of species: {len(species_list)}")

# Convert labels to indices
y_train_idx = np.array([species_to_idx[sp] for sp in y_train])
y_holdout_idx = np.array([species_to_idx[sp] for sp in y_holdout])

# ======================================================================
# 2. EXTRACT MODEL PREDICTIONS
# ======================================================================

print("\nExtracting model predictions...")

models = ['cnn', 'tcn', 'statistics']
species_cols_clean = [sp.replace(' ', '_').replace('-', '_') for sp in species_list]

# Extract probability columns for each model
def extract_model_probs(df, model_name, species_cols):
    """Extract probability columns for a specific model."""
    prob_cols = [f'{model_name}_prob_{sp}' for sp in species_cols]
    available_cols = [col for col in prob_cols if col in df.columns]
    
    if len(available_cols) == 0:
        print(f"  WARNING: No columns found for {model_name}")
        return None
    
    return df[available_cols].values

# Extract for train set
X_train_meta = []
for model in models:
    probs = extract_model_probs(X_train_full, model, species_cols_clean)
    if probs is not None:
        X_train_meta.append(probs)
        print(f"  {model} train: {probs.shape}")

X_train_meta = np.hstack(X_train_meta)

# Extract for holdout set
X_holdout_meta = []
for model in models:
    probs = extract_model_probs(X_holdout_full, model, species_cols_clean)
    if probs is not None:
        X_holdout_meta.append(probs)
        print(f"  {model} holdout: {probs.shape}")

X_holdout_meta = np.hstack(X_holdout_meta)

print(f"\nMeta-learner input shapes:")
print(f"  Train: {X_train_meta.shape} ({X_train_meta.shape[0]} samples, {X_train_meta.shape[1]} features)")
print(f"  Holdout: {X_holdout_meta.shape}")

# ======================================================================
# 3. LOGISTIC REGRESSION WITH OPTUNA
# ======================================================================

def objective_logistic(trial):
    """Optuna objective for logistic regression hyperparameters."""
    
    # Hyperparameters to tune
    C = trial.suggest_float('C', 0.001, 100, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 2000)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
    
    # Avoid incompatible combinations
    if solver == 'liblinear' and max_iter > 1000:
        raise optuna.TrialPruned()
    
    try:
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=42,
            n_jobs=-1,
            multi_class='multinomial'
        )
        
        # Cross-validation on train set only
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X_train_meta, y_train_idx,
            cv=cv, scoring='f1_macro', n_jobs=-1
        )
        
        return scores.mean()
    
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.TrialPruned()

print("\n" + "="*70)
print("OPTIMIZING LOGISTIC REGRESSION META-LEARNER")
print("="*70)
print("Tuning on 80% train set with 5-fold CV")
print("Testing on 20% holdout set")

sampler_lr = TPESampler(seed=42)
pruner_lr = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

study_lr = optuna.create_study(
    study_name=f"logistic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    direction='maximize',
    sampler=sampler_lr,
    pruner=pruner_lr,
    storage=OPTUNA_DB_URL,
    load_if_exists=False
)

print(f"Study name: {study_lr.study_name}")
study_lr.optimize(objective_logistic, n_trials=30, show_progress_bar=True)

best_trial_lr = study_lr.best_trial
print(f"\nBest trial: {best_trial_lr.number}")
print(f"Best CV F1-Macro (train): {best_trial_lr.value:.6f}")
print(f"Best hyperparameters:")
for key, value in best_trial_lr.params.items():
    print(f"  {key}: {value}")

# Train final logistic regression model on full train set
best_lr_model = LogisticRegression(
    C=best_trial_lr.params['C'],
    max_iter=best_trial_lr.params['max_iter'],
    solver=best_trial_lr.params['solver'],
    random_state=42,
    n_jobs=-1,
    multi_class='multinomial'
)
best_lr_model.fit(X_train_meta, y_train_idx)

# Evaluate on holdout set
lr_holdout_pred = best_lr_model.predict(X_holdout_meta)
lr_holdout_proba = best_lr_model.predict_proba(X_holdout_meta)

lr_holdout_accuracy = accuracy_score(y_holdout_idx, lr_holdout_pred)
lr_holdout_f1 = f1_score(y_holdout_idx, lr_holdout_pred, average='macro')
lr_holdout_precision = precision_score(y_holdout_idx, lr_holdout_pred, average='macro')
lr_holdout_recall = recall_score(y_holdout_idx, lr_holdout_pred, average='macro')

# Also get train performance
lr_train_pred = best_lr_model.predict(X_train_meta)
lr_train_accuracy = accuracy_score(y_train_idx, lr_train_pred)
lr_train_f1 = f1_score(y_train_idx, lr_train_pred, average='macro')

print(f"\nLogistic Regression Performance:")
print(f"  Train (full):")
print(f"    Accuracy: {lr_train_accuracy:.6f}")
print(f"    F1-Macro: {lr_train_f1:.6f}")
print(f"  Holdout (20%):")
print(f"    Accuracy: {lr_holdout_accuracy:.6f}")
print(f"    F1-Macro: {lr_holdout_f1:.6f}")
print(f"    Precision: {lr_holdout_precision:.6f}")
print(f"    Recall: {lr_holdout_recall:.6f}")

# ======================================================================
# 4. NEURAL NETWORK META-LEARNER WITH OPTUNA
# ======================================================================

class MetaLearnerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def objective_nn(trial):
    """Optuna objective for neural network meta-learner."""
    
    # Hyperparameters to tune
    hidden_dim = trial.suggest_int('hidden_dim', 32, 512, step=32)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    epochs = 50
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train_meta)
    y_tensor = torch.LongTensor(y_train_idx)
    
    # Use stratified k-fold manually
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_idx, val_idx in cv.split(X_train_meta, y_train_idx):
        X_train_fold, X_val_fold = X_tensor[train_idx], X_tensor[val_idx]
        y_train_fold, y_val_fold = y_tensor[train_idx], y_tensor[val_idx]
        
        # Normalize
        scaler = StandardScaler()
        X_train_fold_scaled = torch.FloatTensor(
            scaler.fit_transform(X_train_fold.numpy())
        ).to(DEVICE)
        X_val_fold_scaled = torch.FloatTensor(
            scaler.transform(X_val_fold.numpy())
        ).to(DEVICE)
        
        # Create model on device
        model = MetaLearnerNN(
            input_dim=X_train_meta.shape[1],
            hidden_dim=hidden_dim,
            output_dim=len(species_list),
            dropout_rate=dropout_rate
        ).to(DEVICE)
        
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        train_dataset = TensorDataset(X_train_fold_scaled, y_train_fold.to(DEVICE))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_fold_scaled)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            fold_score = f1_score(y_val_fold.cpu().numpy(), val_preds, average='macro')
        
        fold_scores.append(fold_score)
    
    return np.mean(fold_scores)

print("\n" + "="*70)
print("OPTIMIZING NEURAL NETWORK META-LEARNER")
print("="*70)
print("Tuning on 80% train set with 5-fold CV")
print("Testing on 20% holdout set")

sampler_nn = TPESampler(seed=42)
pruner_nn = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

study_nn = optuna.create_study(
    study_name=f"neural_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    direction='maximize',
    sampler=sampler_nn,
    pruner=pruner_nn,
    storage=OPTUNA_DB_URL,
    load_if_exists=False
)

print(f"Study name: {study_nn.study_name}")
study_nn.optimize(objective_nn, n_trials=30, show_progress_bar=True)

best_trial_nn = study_nn.best_trial
print(f"\nBest trial: {best_trial_nn.number}")
print(f"Best CV F1-Macro (train): {best_trial_nn.value:.6f}")
print(f"Best hyperparameters:")
for key, value in best_trial_nn.params.items():
    print(f"  {key}: {value}")

# Train final NN model on full train set
X_tensor = torch.FloatTensor(X_train_meta)
y_tensor = torch.LongTensor(y_train_idx)

scaler_final = StandardScaler()
X_train_scaled = torch.FloatTensor(scaler_final.fit_transform(X_train_meta))

# Also scale holdout set with train scaler
X_holdout_scaled = torch.FloatTensor(scaler_final.transform(X_holdout_meta))

nn_model = MetaLearnerNN(
    input_dim=X_train_meta.shape[1],
    hidden_dim=best_trial_nn.params['hidden_dim'],
    output_dim=len(species_list),
    dropout_rate=best_trial_nn.params['dropout_rate']
)

optimizer = Adam(nn_model.parameters(), lr=best_trial_nn.params['learning_rate'])
criterion = nn.CrossEntropyLoss()

train_dataset = TensorDataset(X_train_scaled, y_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=best_trial_nn.params['batch_size'],
    shuffle=True
)

for epoch in range(100):
    nn_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = nn_model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

# Evaluate on both sets
nn_model.eval()

with torch.no_grad():
    # Train performance
    nn_train_logits = nn_model(X_train_scaled)
    nn_train_pred = nn_train_logits.argmax(dim=1).numpy()
    nn_train_accuracy = accuracy_score(y_train_idx, nn_train_pred)
    nn_train_f1 = f1_score(y_train_idx, nn_train_pred, average='macro')
    
    # Holdout performance
    nn_holdout_logits = nn_model(X_holdout_scaled)
    nn_holdout_proba = torch.softmax(nn_holdout_logits, dim=1).numpy()
    nn_holdout_pred = nn_holdout_logits.argmax(dim=1).numpy()
    nn_holdout_accuracy = accuracy_score(y_holdout_idx, nn_holdout_pred)
    nn_holdout_f1 = f1_score(y_holdout_idx, nn_holdout_pred, average='macro')
    nn_holdout_precision = precision_score(y_holdout_idx, nn_holdout_pred, average='macro')
    nn_holdout_recall = recall_score(y_holdout_idx, nn_holdout_pred, average='macro')

print(f"\nNeural Network Performance:")
print(f"  Train (full):")
print(f"    Accuracy: {nn_train_accuracy:.6f}")
print(f"    F1-Macro: {nn_train_f1:.6f}")
print(f"  Holdout (20%):")
print(f"    Accuracy: {nn_holdout_accuracy:.6f}")
print(f"    F1-Macro: {nn_holdout_f1:.6f}")
print(f"    Precision: {nn_holdout_precision:.6f}")
print(f"    Recall: {nn_holdout_recall:.6f}")

# ======================================================================
# 5. COMPARISON AND RESULTS
# ======================================================================

print("\n" + "="*70)
print("META-LEARNER COMPARISON")
print("="*70)

results = {
    'logistic_regression': {
        'train_accuracy': float(lr_train_accuracy),
        'train_f1_macro': float(lr_train_f1),
        'holdout_accuracy': float(lr_holdout_accuracy),
        'holdout_f1_macro': float(lr_holdout_f1),
        'holdout_precision': float(lr_holdout_precision),
        'holdout_recall': float(lr_holdout_recall),
        'hyperparameters': best_trial_lr.params,
        'cv_score': float(best_trial_lr.value)
    },
    'neural_network': {
        'train_accuracy': float(nn_train_accuracy),
        'train_f1_macro': float(nn_train_f1),
        'holdout_accuracy': float(nn_holdout_accuracy),
        'holdout_f1_macro': float(nn_holdout_f1),
        'holdout_precision': float(nn_holdout_precision),
        'holdout_recall': float(nn_holdout_recall),
        'hyperparameters': best_trial_nn.params,
        'cv_score': float(best_trial_nn.value)
    }
}

print(f"\nLogistic Regression:")
print(f"  CV F1-Macro (train fold): {results['logistic_regression']['cv_score']:.6f}")
print(f"  Train Accuracy: {results['logistic_regression']['train_accuracy']:.6f}")
print(f"  Train F1-Macro: {results['logistic_regression']['train_f1_macro']:.6f}")
print(f"  Holdout Accuracy: {results['logistic_regression']['holdout_accuracy']:.6f}")
print(f"  Holdout F1-Macro: {results['logistic_regression']['holdout_f1_macro']:.6f}")

print(f"\nNeural Network:")
print(f"  CV F1-Macro (train fold): {results['neural_network']['cv_score']:.6f}")
print(f"  Train Accuracy: {results['neural_network']['train_accuracy']:.6f}")
print(f"  Train F1-Macro: {results['neural_network']['train_f1_macro']:.6f}")
print(f"  Holdout Accuracy: {results['neural_network']['holdout_accuracy']:.6f}")
print(f"  Holdout F1-Macro: {results['neural_network']['holdout_f1_macro']:.6f}")

# Determine winner by holdout performance
if results['logistic_regression']['holdout_f1_macro'] > results['neural_network']['holdout_f1_macro']:
    print(f"\n[WINNER] Logistic Regression (holdout F1-Macro: {results['logistic_regression']['holdout_f1_macro']:.6f})")
    best_model_name = 'logistic_regression'
else:
    print(f"\n[WINNER] Neural Network (holdout F1-Macro: {results['neural_network']['holdout_f1_macro']:.6f})")
    best_model_name = 'neural_network'

# ======================================================================
# 6. SAVE RESULTS
# ======================================================================

# Save models
with open('best_lr_model.pkl', 'wb') as f:
    pickle.dump(best_lr_model, f)

torch.save(nn_model.state_dict(), 'best_nn_model.pt')
torch.save(scaler_final, 'nn_scaler.pt')

# Save results JSON
with open('meta_learner_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save detailed per-species performance
print("\n" + "="*70)
print("PER-SPECIES HOLDOUT PERFORMANCE")
print("="*70)

print(f"\n{best_model_name.upper()} (WINNER)")
if best_model_name == 'logistic_regression':
    pred = lr_holdout_pred
    proba = lr_holdout_proba
else:
    pred = nn_holdout_pred
    proba = nn_holdout_proba

# Map predictions back to species names
pred_species = np.array([species_list[idx] for idx in pred])
true_species = np.array([species_list[idx] for idx in y_holdout_idx])

# Per-species accuracy
print("\nPer-species accuracy on holdout set:")
for species in species_list:
    mask = true_species == species
    if np.sum(mask) > 0:
        acc = np.mean(pred_species[mask] == true_species[mask])
        count = np.sum(mask)
        print(f"  {species}: {np.sum(pred_species[mask] == true_species[mask])}/{count} ({acc*100:.1f}%)")

print(f"\n[SUCCESS] Models and results saved")
print(f"  - best_lr_model.pkl")
print(f"  - best_nn_model.pt")
print(f"  - nn_scaler.pt")
print(f"  - meta_learner_results.json")

# ======================================================================
# 7. OPTUNA DATABASE SUMMARY
# ======================================================================

print("\n" + "="*70)
print("OPTUNA STUDIES DATABASE")
print("="*70)

# Load and display all studies from the database
from optuna.study import load_study

try:
    all_studies = optuna.study.get_all_study_summaries(storage=OPTUNA_DB_URL)
    
    print(f"\nTotal studies in database: {len(all_studies)}")
    for summary in all_studies:
        print(f"\n  Study: {summary.study_name}")
        print(f"    Direction: {summary.direction}")
        print(f"    Trials: {summary.n_trials}")
        print(f"    Best value: {summary.best_value:.6f}" if summary.best_value is not None else "    Best value: N/A")
        
        # Load study and show best trial
        study = load_study(study_name=summary.study_name, storage=OPTUNA_DB_URL)
        if study.best_trial:
            print(f"    Best trial: {study.best_trial.number}")
    
    print(f"\nDatabase file: {OPTUNA_DB_PATH}")
    print(f"To analyze studies further, use:")
    print(f"  optuna studies {OPTUNA_DB_PATH}")
    
except Exception as e:
    print(f"Could not retrieve studies: {e}")