"""
Train a stacked ensemble (meta-learner) using model predictions.
Trains both XGBoost and Neural Network meta-learners.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path='all_model_predictions.csv'):
    """Load CSV and prepare meta-learner data."""
    print("Loading data from", csv_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Extract true labels
    y = df['species_true'].values

    # Auto-detect columns by prefix (all probability columns from different models)
    meta_columns = [col for col in df.columns if col.startswith(('cnn_', 'resnet1d_', 'statistics_', 'extratrees_', 'rulebased_'))]
    X_meta = df[meta_columns].values

    print(f"Found {len(meta_columns)} meta-learner features")
    print(f"Feature groups:")
    print(f"  - CNN: {len([c for c in meta_columns if c.startswith('cnn_')])}")
    print(f"  - ResNet1D: {len([c for c in meta_columns if c.startswith('resnet1d_')])}")
    print(f"  - Statistics: {len([c for c in meta_columns if c.startswith('statistics_')])}")
    print(f"  - ExtraTrees: {len([c for c in meta_columns if c.startswith('extratrees_')])}")
    print(f"  - RuleBased: {len([c for c in meta_columns if c.startswith('rulebased_')])}")

    return X_meta, y, meta_columns


def encode_labels(y):
    """Encode string labels to integers."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def split_data(X, y, train_size=0.65, val_size=0.15, test_size=0.20, random_state=42):
    """Split into train (65%), validation (15%), and test (20%) sets."""
    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: separate val from train on remaining 80%
    val_ratio = val_size / (1 - test_size)  # Adjust ratio for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    print(f"\nData split:")
    print(f"  Train (65%): {len(X_train)} samples")
    print(f"  Val (15%):   {len(X_val)} samples")
    print(f"  Test (20%):  {len(X_test)} samples")
    print(f"  Total:       {len(X)} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_oof_predictions(X_train, X_val, X_test, y_train, y_val, y_test, num_classes, n_splits=5, random_state=42):
    """Generate out-of-fold (OOF) predictions using k-fold cross-validation."""
    print("\n" + "="*60)
    print("Generating OOF Predictions with K-Fold Cross-Validation")
    print("="*60)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # OOF predictions for train set
    oof_train = np.zeros((len(X_train), num_classes))
    oof_val = np.zeros((len(X_val), num_classes))
    oof_test = np.zeros((len(X_test), num_classes))

    fold = 0
    for train_idx, val_idx in skf.split(X_train, y_train):
        fold += 1
        print(f"Fold {fold}/{n_splits}...")

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_vl = X_train[val_idx]

        # Train XGBoost on this fold
        dtrain_fold = xgb.DMatrix(X_tr, label=y_tr)
        dvl_fold = xgb.DMatrix(X_vl)

        params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
        }

        model_fold = xgb.train(
            params,
            dtrain_fold,
            num_boost_round=200,
            verbose_eval=False
        )

        # Get OOF predictions
        oof_train[val_idx] = model_fold.predict(dvl_fold)

    # Train on full train set to get val and test predictions
    dtrain_full = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val)
    dtest = xgb.DMatrix(X_test)

    model_full = xgb.train(
        params,
        dtrain_full,
        num_boost_round=200,
        verbose_eval=False
    )

    oof_val = model_full.predict(dval)
    oof_test = model_full.predict(dtest)

    print(f"OOF predictions generated (shape: {oof_train.shape})")
    return oof_train, oof_val, oof_test


def train_xgboost(oof_train, oof_val, oof_test, y_train, y_val, y_test, num_classes):
    """Train XGBoost meta-learner on OOF predictions with early stopping."""
    print("\n" + "="*60)
    print("Training XGBoost Meta-Learner on OOF Predictions")
    print("="*60)

    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(oof_train, label=y_train)
    dval = xgb.DMatrix(oof_val, label=y_val)
    dtest = xgb.DMatrix(oof_test, label=y_test)

    # XGBoost parameters
    params = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }

    # Train with early stopping on validation set
    evals = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}

    model_xgb = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Evaluate on test set
    y_pred = model_xgb.predict(dtest)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nXGBoost Results (Test Set):")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Per-class accuracy
    per_class_acc = []
    for class_id in range(num_classes):
        mask = y_test == class_id
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            per_class_acc.append(class_acc)

    print("Per-Class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Class {i}: {acc:.4f}")

    # Feature importance (top predictions from OOF)
    importance = model_xgb.get_score(importance_type='weight')
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 OOF Features Used (by frequency):")
    for i, (feat, score) in enumerate(importance_sorted[:10]):
        print(f"  {i+1}. {feat}: {score}")

    # Save model
    model_xgb.save_model('meta_xgb.json')
    print("\nSaved XGBoost model to meta_xgb.json")

    return model_xgb, importance_sorted


class MetaLearnerNN(nn.Module):
    """Small feed-forward neural network for meta-learning."""
    def __init__(self, input_size, num_classes, hidden_size=128):
        super(MetaLearnerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


def train_neural_network(oof_train, oof_val, oof_test, y_train, y_val, y_test, num_classes):
    """Train neural network meta-learner on OOF predictions with early stopping."""
    print("\n" + "="*60)
    print("Training Neural Network Meta-Learner on OOF Predictions")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to tensors
    oof_train_t = torch.FloatTensor(oof_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    oof_val_t = torch.FloatTensor(oof_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    oof_test_t = torch.FloatTensor(oof_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(oof_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model
    input_size = oof_train.shape[1]
    model = MetaLearnerNN(input_size, num_classes, hidden_size=128).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training with early stopping
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    num_epochs = 200

    print(f"\nTraining for up to {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(oof_val_t)
            val_loss = criterion(val_outputs, y_val_t)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(oof_test_t)
        y_pred = y_pred_logits.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nNeural Network Results (Test Set):")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Per-class accuracy
    per_class_acc = []
    for class_id in range(num_classes):
        mask = y_test == class_id
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            per_class_acc.append(class_acc)

    print("Per-Class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Class {i}: {acc:.4f}")

    # Save model
    torch.save(model.state_dict(), 'meta_nn.pt')
    print("\nSaved Neural Network model to meta_nn.pt")

    return model, device


def main():
    """Main training pipeline with k-fold OOF predictions."""
    print("\n" + "="*60)
    print("META-LEARNER TRAINING PIPELINE")
    print("="*60)

    # Load and prepare data
    X_meta, y_raw, feature_names = load_and_prepare_data()

    # Encode labels
    y, label_encoder = encode_labels(y_raw)
    num_classes = len(np.unique(y))
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split data into 65% train, 15% val, 20% test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_meta, y)

    # Normalize features (based on training set statistics)
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1  # Avoid division by zero

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    print("Features normalized (z-score)")

    # Generate OOF predictions using k-fold cross-validation
    oof_train, oof_val, oof_test = generate_oof_predictions(
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes, n_splits=5
    )

    # Train XGBoost meta-learner on OOF predictions
    model_xgb, importance = train_xgboost(
        oof_train, oof_val, oof_test, y_train, y_val, y_test, num_classes
    )

    # Train Neural Network meta-learner on OOF predictions
    model_nn, device = train_neural_network(
        oof_train, oof_val, oof_test, y_train, y_val, y_test, num_classes
    )

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nModels saved:")
    print("  - meta_xgb.json (XGBoost)")
    print("  - meta_nn.pt (Neural Network)")
    print("\nData split summary:")
    print(f"  Train (65%): {len(X_train)} samples")
    print(f"  Val (15%):   {len(X_val)} samples")
    print(f"  Test (20%):  {len(X_test)} samples")
    print("\nLabel mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")


if __name__ == '__main__':
    main()
