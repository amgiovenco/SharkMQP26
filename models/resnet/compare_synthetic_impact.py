"""
Comparative Cross-Validation Analysis: Normal Data vs. Normal + Synthetic Data
5-fold stratified CV on seed 8 with ResNet1D model

This script compares:
1. Baseline: 5-fold CV with normal data only
2. Enhanced: 5-fold CV with real + synthetic data (synthetic only in training)

Synthetic data is NEVER in validation or test sets, only training sets.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        # Set seeds for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("PyTorch not available!")
    exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - visualization will be skipped")


def normalize_confusion_matrix(cm):
    """Normalize confusion matrix by row (True label distribution)."""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm_normalized

# ============================================================================
# CONFIG
# ============================================================================
NORMAL_DATA_PATH = "../../data/shark_dataset.csv"
SYNTHETIC_DATA_PATH = "../../data/synthetic_only.csv"
SPECIES_COL = "Species"
NUM_EPOCHS = 100
RANDOM_STATE = 8
N_SPLITS = 5

# Best hyperparameters from Trial 67
BEST_PARAMS = {
    "initial_filters": 80,
    "dropout": 0.20796879885018393,
    "learning_rate": 0.0004313869594239175,
    "batch_size": 16,
    "weight_decay": 0.0001560845747200455
}

# ============================================================================
# SET RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

# ============================================================================
# RESNET1D MODEL (identical to train_cv_model.py and train_final_model.py)
# ============================================================================
class ResidualBlock1D(nn.Module):
    """Residual block for 1d convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                              stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """1d residual network for sequence processing."""
    def __init__(self, num_classes, input_channels=1, initial_filters=64, dropout=0.5):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(initial_filters, initial_filters, blocks=2)
        self.layer2 = self._make_layer(initial_filters, initial_filters*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(initial_filters*2, initial_filters*4, blocks=2, stride=2)
        self.layer4 = self._make_layer(initial_filters*4, initial_filters*8, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(initial_filters*8, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        layers = [ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_and_normalize_data(csv_path, mean_vals=None, std_vals=None):
    """Load and normalize data from CSV.

    Args:
        csv_path: Path to CSV file
        mean_vals: Pre-computed mean values (for synthetic data normalization)
        std_vals: Pre-computed std values (for synthetic data normalization)
    """
    df = pd.read_csv(csv_path)
    X_raw = df.drop(columns=[SPECIES_COL], errors='ignore')
    y = df[SPECIES_COL].astype(str)

    # Normalize X
    if mean_vals is None:
        # First time: compute mean/std from this dataset
        mean_vals = X_raw.mean().values
        std_vals = X_raw.std().values

    # Normalize using provided or computed mean/std
    X = (X_raw - mean_vals) / (std_vals + 1e-8)
    X = X.fillna(0).astype(np.float32).values
    X = np.expand_dims(X, axis=1)  # Add channel dimension for Conv1D

    return X, y.values, mean_vals, std_vals


print("="*80)
print("LOADING DATA")
print("="*80)
print(f"Loading normal data from: {NORMAL_DATA_PATH}")
X_normal, y_normal, normal_mean, normal_std = load_and_normalize_data(NORMAL_DATA_PATH)
print(f"  Shape: {X_normal.shape}")

print(f"Loading synthetic data from: {SYNTHETIC_DATA_PATH}")
# IMPORTANT: Normalize synthetic data using NORMAL data's mean/std
X_synthetic, y_synthetic, _, _ = load_and_normalize_data(SYNTHETIC_DATA_PATH, normal_mean, normal_std)
print(f"  Shape: {X_synthetic.shape}")
print(f"\nNormalization: Both datasets normalized using normal data's statistics")

# Encode labels (using normal data labels as reference)
le = LabelEncoder()
y_normal_encoded = le.fit_transform(y_normal)
y_synthetic_encoded = le.transform(y_synthetic)

print(f"Classes: {le.classes_}")
print(f"Number of classes: {len(le.classes_)}")

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_model(X_train, y_train, num_classes, epochs=NUM_EPOCHS):
    """Train a ResNet1D model."""
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.LongTensor(y_train).to(DEVICE)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=True)

    # Create model
    model = ResNet1D(
        num_classes=num_classes,
        input_channels=1,
        initial_filters=BEST_PARAMS['initial_filters'],
        dropout=BEST_PARAMS['dropout']
    ).to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(model.parameters(),
                          lr=BEST_PARAMS['learning_rate'],
                          weight_decay=BEST_PARAMS['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    y_test_tensor = torch.LongTensor(y_test).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred_tensor = torch.argmax(outputs, dim=1)
        y_pred = y_pred_tensor.cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_labels': y_test
    }


# ============================================================================
# SCENARIO 1: NORMAL DATA ONLY
# ============================================================================
print("\n" + "="*80)
print("SCENARIO 1: BASELINE (Normal Data Only)")
print("="*80)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

baseline_results = {
    "scenario": "baseline_normal_data_only",
    "seed": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in BEST_PARAMS.items()},
    "folds": []
}

baseline_accuracies = []
baseline_f1_scores = []
baseline_precisions = []
baseline_recalls = []
baseline_cms = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_normal, y_normal_encoded), 1):
    X_train, X_test = X_normal[train_idx], X_normal[test_idx]
    y_train, y_test = y_normal_encoded[train_idx], y_normal_encoded[test_idx]

    print(f"\n  Fold {fold_idx}/{N_SPLITS}")
    print(f"    Training size: {len(X_train)} samples (normal only)")
    print(f"    Test size: {len(X_test)} samples")

    # Train model
    model = train_model(X_train, y_train, len(le.classes_))

    # Evaluate
    fold_metrics = evaluate_model(model, X_test, y_test)

    baseline_accuracies.append(fold_metrics['accuracy'])
    baseline_f1_scores.append(fold_metrics['f1'])
    baseline_precisions.append(fold_metrics['precision'])
    baseline_recalls.append(fold_metrics['recall'])
    baseline_cms.append(fold_metrics['confusion_matrix'])

    baseline_results["folds"].append({
        "fold": fold_idx,
        "accuracy": float(fold_metrics['accuracy']),
        "f1": float(fold_metrics['f1']),
        "precision": float(fold_metrics['precision']),
        "recall": float(fold_metrics['recall']),
        "test_size": len(y_test),
        "train_size": len(X_train)
    })

    print(f"    Accuracy: {fold_metrics['accuracy']:.4f} | F1: {fold_metrics['f1']:.4f} | Precision: {fold_metrics['precision']:.4f} | Recall: {fold_metrics['recall']:.4f}")

    del model

# Summary for baseline
baseline_results["mean_accuracy"] = float(np.mean(baseline_accuracies))
baseline_results["std_accuracy"] = float(np.std(baseline_accuracies))
baseline_results["mean_f1"] = float(np.mean(baseline_f1_scores))
baseline_results["std_f1"] = float(np.std(baseline_f1_scores))
baseline_results["mean_precision"] = float(np.mean(baseline_precisions))
baseline_results["std_precision"] = float(np.std(baseline_precisions))
baseline_results["mean_recall"] = float(np.mean(baseline_recalls))
baseline_results["std_recall"] = float(np.std(baseline_recalls))

print("\n" + "-"*80)
print("BASELINE SUMMARY")
print("-"*80)
print(f"Mean Accuracy:  {np.mean(baseline_accuracies):.4f} ± {np.std(baseline_accuracies):.4f}")
print(f"Mean F1:        {np.mean(baseline_f1_scores):.4f} ± {np.std(baseline_f1_scores):.4f}")
print(f"Mean Precision: {np.mean(baseline_precisions):.4f} ± {np.std(baseline_precisions):.4f}")
print(f"Mean Recall:    {np.mean(baseline_recalls):.4f} ± {np.std(baseline_recalls):.4f}")


# ============================================================================
# SCENARIO 2: NORMAL + SYNTHETIC DATA (Synthetic in Training Only)
# ============================================================================
print("\n" + "="*80)
print("SCENARIO 2: ENHANCED (Normal + Synthetic in Training Only)")
print("="*80)

enhanced_results = {
    "scenario": "enhanced_normal_plus_synthetic_training",
    "seed": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in BEST_PARAMS.items()},
    "folds": []
}

enhanced_accuracies = []
enhanced_f1_scores = []
enhanced_precisions = []
enhanced_recalls = []
enhanced_cms = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_normal, y_normal_encoded), 1):
    X_train, X_test = X_normal[train_idx], X_normal[test_idx]
    y_train, y_test = y_normal_encoded[train_idx], y_normal_encoded[test_idx]

    # IMPORTANT: Add synthetic data to training set ONLY
    X_train_combined = np.concatenate([X_train, X_synthetic], axis=0)
    y_train_combined = np.concatenate([y_train, y_synthetic_encoded], axis=0)

    print(f"\n  Fold {fold_idx}/{N_SPLITS}")
    print(f"    Training size: {len(X_train)} real + {len(X_synthetic)} synthetic = {len(X_train_combined)} total")
    print(f"    Test size: {len(X_test)} samples (normal only)")

    # Train model
    model = train_model(X_train_combined, y_train_combined, len(le.classes_))

    # Evaluate on original test set (normal data only)
    fold_metrics = evaluate_model(model, X_test, y_test)

    enhanced_accuracies.append(fold_metrics['accuracy'])
    enhanced_f1_scores.append(fold_metrics['f1'])
    enhanced_precisions.append(fold_metrics['precision'])
    enhanced_recalls.append(fold_metrics['recall'])
    enhanced_cms.append(fold_metrics['confusion_matrix'])

    enhanced_results["folds"].append({
        "fold": fold_idx,
        "accuracy": float(fold_metrics['accuracy']),
        "f1": float(fold_metrics['f1']),
        "precision": float(fold_metrics['precision']),
        "recall": float(fold_metrics['recall']),
        "test_size": len(y_test),
        "train_size_real": len(X_train),
        "train_size_synthetic": len(X_synthetic),
        "train_size_total": len(X_train_combined)
    })

    print(f"    Accuracy: {fold_metrics['accuracy']:.4f} | F1: {fold_metrics['f1']:.4f} | Precision: {fold_metrics['precision']:.4f} | Recall: {fold_metrics['recall']:.4f}")

    del model

# Summary for enhanced
enhanced_results["mean_accuracy"] = float(np.mean(enhanced_accuracies))
enhanced_results["std_accuracy"] = float(np.std(enhanced_accuracies))
enhanced_results["mean_f1"] = float(np.mean(enhanced_f1_scores))
enhanced_results["std_f1"] = float(np.std(enhanced_f1_scores))
enhanced_results["mean_precision"] = float(np.mean(enhanced_precisions))
enhanced_results["std_precision"] = float(np.std(enhanced_precisions))
enhanced_results["mean_recall"] = float(np.mean(enhanced_recalls))
enhanced_results["std_recall"] = float(np.std(enhanced_recalls))

print("\n" + "-"*80)
print("ENHANCED SUMMARY")
print("-"*80)
print(f"Mean Accuracy:  {np.mean(enhanced_accuracies):.4f} ± {np.std(enhanced_accuracies):.4f}")
print(f"Mean F1:        {np.mean(enhanced_f1_scores):.4f} ± {np.std(enhanced_f1_scores):.4f}")
print(f"Mean Precision: {np.mean(enhanced_precisions):.4f} ± {np.std(enhanced_precisions):.4f}")
print(f"Mean Recall:    {np.mean(enhanced_recalls):.4f} ± {np.std(enhanced_recalls):.4f}")


# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON ANALYSIS")
print("="*80)

accuracy_improvement = (np.mean(enhanced_accuracies) - np.mean(baseline_accuracies)) * 100
f1_improvement = (np.mean(enhanced_f1_scores) - np.mean(baseline_f1_scores)) * 100
precision_improvement = (np.mean(enhanced_precisions) - np.mean(baseline_precisions)) * 100
recall_improvement = (np.mean(enhanced_recalls) - np.mean(baseline_recalls)) * 100

print(f"\nMetric Improvements (Enhanced vs Baseline):")
print(f"  Accuracy:  {accuracy_improvement:+.2f}% ({np.mean(baseline_accuracies):.4f} → {np.mean(enhanced_accuracies):.4f})")
print(f"  F1 Score:  {f1_improvement:+.2f}% ({np.mean(baseline_f1_scores):.4f} → {np.mean(enhanced_f1_scores):.4f})")
print(f"  Precision: {precision_improvement:+.2f}% ({np.mean(baseline_precisions):.4f} → {np.mean(enhanced_precisions):.4f})")
print(f"  Recall:    {recall_improvement:+.2f}% ({np.mean(baseline_recalls):.4f} → {np.mean(enhanced_recalls):.4f})")

# ============================================================================
# SAVE RESULTS TO JSON
# ============================================================================
results_dir = Path("./results")
results_dir.mkdir(exist_ok=True)

comparison_summary = {
    "experiment": "synthetic_data_impact_analysis",
    "seed": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "classes": list(le.classes_),
    "data_paths": {
        "normal": NORMAL_DATA_PATH,
        "synthetic": SYNTHETIC_DATA_PATH
    },
    "baseline": baseline_results,
    "enhanced": enhanced_results,
    "confusion_matrices": {
        "baseline": [cm.tolist() for cm in baseline_cms],
        "enhanced": [cm.tolist() for cm in enhanced_cms]
    },
    "comparison": {
        "accuracy_improvement_percent": float(accuracy_improvement),
        "f1_improvement_percent": float(f1_improvement),
        "precision_improvement_percent": float(precision_improvement),
        "recall_improvement_percent": float(recall_improvement),
        "baseline_mean_accuracy": float(np.mean(baseline_accuracies)),
        "enhanced_mean_accuracy": float(np.mean(enhanced_accuracies)),
        "baseline_mean_f1": float(np.mean(baseline_f1_scores)),
        "enhanced_mean_f1": float(np.mean(enhanced_f1_scores)),
        "baseline_mean_precision": float(np.mean(baseline_precisions)),
        "enhanced_mean_precision": float(np.mean(enhanced_precisions)),
        "baseline_mean_recall": float(np.mean(baseline_recalls)),
        "enhanced_mean_recall": float(np.mean(enhanced_recalls))
    }
}

summary_file = results_dir / "synthetic_comparison_summary.json"
with open(summary_file, 'w') as f:
    json.dump(comparison_summary, f, indent=2)

print(f"\n[OK] Summary saved to {summary_file}")


# ============================================================================
# VISUALIZATION
# ============================================================================
if MATPLOTLIB_AVAILABLE:
    print("\nGenerating visualizations...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 12)

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))

    # 1. Accuracy comparison over folds
    ax1 = plt.subplot(3, 3, 1)
    folds = list(range(1, N_SPLITS + 1))
    ax1.plot(folds, baseline_accuracies, 'o-', label='Baseline (Normal Only)', linewidth=2, markersize=8)
    ax1.plot(folds, enhanced_accuracies, 's-', label='Enhanced (Normal + Synthetic)', linewidth=2, markersize=8)
    ax1.set_xlabel('Fold', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy Across Folds', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # 2. F1 Score comparison over folds
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(folds, baseline_f1_scores, 'o-', label='Baseline (Normal Only)', linewidth=2, markersize=8)
    ax2.plot(folds, enhanced_f1_scores, 's-', label='Enhanced (Normal + Synthetic)', linewidth=2, markersize=8)
    ax2.set_xlabel('Fold', fontsize=11)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title('F1 Score Across Folds', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # 3. Precision comparison over folds
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(folds, baseline_precisions, 'o-', label='Baseline (Normal Only)', linewidth=2, markersize=8)
    ax3.plot(folds, enhanced_precisions, 's-', label='Enhanced (Normal + Synthetic)', linewidth=2, markersize=8)
    ax3.set_xlabel('Fold', fontsize=11)
    ax3.set_ylabel('Precision', fontsize=11)
    ax3.set_title('Precision Across Folds', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

    # 4. Mean metrics comparison (Bar chart)
    ax4 = plt.subplot(3, 3, 4)
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    baseline_means = [np.mean(baseline_accuracies), np.mean(baseline_f1_scores),
                     np.mean(baseline_precisions), np.mean(baseline_recalls)]
    enhanced_means = [np.mean(enhanced_accuracies), np.mean(enhanced_f1_scores),
                     np.mean(enhanced_precisions), np.mean(enhanced_recalls)]

    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, baseline_means, width, label='Baseline', alpha=0.8)
    ax4.bar(x + width/2, enhanced_means, width, label='Enhanced', alpha=0.8)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Mean Metrics Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.set_ylim([0, 1.05])
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Recall comparison over folds
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(folds, baseline_recalls, 'o-', label='Baseline (Normal Only)', linewidth=2, markersize=8)
    ax5.plot(folds, enhanced_recalls, 's-', label='Enhanced (Normal + Synthetic)', linewidth=2, markersize=8)
    ax5.set_xlabel('Fold', fontsize=11)
    ax5.set_ylabel('Recall', fontsize=11)
    ax5.set_title('Recall Across Folds', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.05])

    # 6. Performance improvement chart
    ax6 = plt.subplot(3, 3, 6)
    improvements = [accuracy_improvement, f1_improvement, precision_improvement, recall_improvement]
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax6.barh(metrics, improvements, color=colors, alpha=0.7)
    ax6.set_xlabel('Improvement (%)', fontsize=11)
    ax6.set_title('Performance Improvement\n(Enhanced vs Baseline)', fontsize=12, fontweight='bold')
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax6.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(improvements):
        ax6.text(v + 0.1 if v > 0 else v - 0.1, i, f'{v:+.2f}%', va='center', fontsize=10)

    # 7. Baseline confusion matrix for Fold 1 (Unweighted)
    ax7 = plt.subplot(3, 3, 7)
    sns.heatmap(baseline_cms[0], annot=False, cmap='Blues', ax=ax7, cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax7.set_title('Baseline - Fold 1 (Unweighted)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('True Label', fontsize=10)
    ax7.set_xlabel('Predicted Label', fontsize=10)
    ax7.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=6)
    ax7.set_yticklabels(le.classes_, rotation=0, fontsize=6)

    # 8. Enhanced confusion matrix for Fold 1 (Unweighted)
    ax8 = plt.subplot(3, 3, 8)
    sns.heatmap(enhanced_cms[0], annot=False, cmap='Greens', ax=ax8, cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax8.set_title('Enhanced - Fold 1 (Unweighted)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('True Label', fontsize=10)
    ax8.set_xlabel('Predicted Label', fontsize=10)
    ax8.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=6)
    ax8.set_yticklabels(le.classes_, rotation=0, fontsize=6)

    # 9. Accuracy variance across folds
    ax9 = plt.subplot(3, 3, 9)
    baseline_var = [np.mean(baseline_accuracies) - 1.96*np.std(baseline_accuracies)/np.sqrt(N_SPLITS),
                   np.mean(baseline_accuracies),
                   np.mean(baseline_accuracies) + 1.96*np.std(baseline_accuracies)/np.sqrt(N_SPLITS)]
    enhanced_var = [np.mean(enhanced_accuracies) - 1.96*np.std(enhanced_accuracies)/np.sqrt(N_SPLITS),
                   np.mean(enhanced_accuracies),
                   np.mean(enhanced_accuracies) + 1.96*np.std(enhanced_accuracies)/np.sqrt(N_SPLITS)]

    x_pos = [0, 1]
    ax9.bar([0], [baseline_var[1]], color='steelblue', alpha=0.7, label='Baseline')
    ax9.bar([1], [enhanced_var[1]], color='seagreen', alpha=0.7, label='Enhanced')
    ax9.errorbar([0], [baseline_var[1]],
                 yerr=[[baseline_var[1]-baseline_var[0]], [baseline_var[2]-baseline_var[1]]],
                 fmt='none', ecolor='steelblue', capsize=10, capthick=2)
    ax9.errorbar([1], [enhanced_var[1]],
                 yerr=[[enhanced_var[1]-enhanced_var[0]], [enhanced_var[2]-enhanced_var[1]]],
                 fmt='none', ecolor='seagreen', capsize=10, capthick=2)
    ax9.set_ylabel('Accuracy', fontsize=11)
    ax9.set_title('Accuracy with 95% CI', fontsize=12, fontweight='bold')
    ax9.set_xticks([0, 1])
    ax9.set_xticklabels(['Baseline', 'Enhanced'])
    ax9.set_ylim([0, 1.05])
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    comparison_fig = results_dir / "comparison_metrics.png"
    plt.savefig(comparison_fig, dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison metrics chart saved to {comparison_fig}")
    plt.close()

    # Create weighted confusion matrices for Fold 1 comparison
    print("\nGenerating weighted confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    baseline_cm_weighted = normalize_confusion_matrix(baseline_cms[0])
    enhanced_cm_weighted = normalize_confusion_matrix(enhanced_cms[0])

    sns.heatmap(baseline_cm_weighted, annot=False, cmap='Blues', ax=axes[0],
                cbar=True, cbar_kws={'label': 'Proportion'},
                xticklabels=le.classes_, yticklabels=le.classes_, vmin=0, vmax=1)
    axes[0].set_title('Baseline - Fold 1 (Weighted - Normalized)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(le.classes_, rotation=0, fontsize=8)

    sns.heatmap(enhanced_cm_weighted, annot=False, cmap='Greens', ax=axes[1],
                cbar=True, cbar_kws={'label': 'Proportion'},
                xticklabels=le.classes_, yticklabels=le.classes_, vmin=0, vmax=1)
    axes[1].set_title('Enhanced - Fold 1 (Weighted - Normalized)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(le.classes_, rotation=0, fontsize=8)

    plt.tight_layout()
    weighted_cm_fig = results_dir / "confusion_matrices_fold_01_weighted.png"
    plt.savefig(weighted_cm_fig, dpi=300, bbox_inches='tight')
    print(f"[OK] Weighted confusion matrices saved to {weighted_cm_fig}")
    plt.close()

    # Create individual confusion matrices for all folds (both unweighted and weighted)
    for fold_idx in range(N_SPLITS):
        baseline_cm_unweighted = baseline_cms[fold_idx]
        enhanced_cm_unweighted = enhanced_cms[fold_idx]
        baseline_cm_weighted = normalize_confusion_matrix(baseline_cm_unweighted)
        enhanced_cm_weighted = normalize_confusion_matrix(enhanced_cm_unweighted)

        # 2x2 grid: Baseline/Enhanced x Unweighted/Weighted
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Row 1: Unweighted
        sns.heatmap(baseline_cm_unweighted, annot=False, cmap='Blues', ax=axes[0, 0],
                    cbar=True, cbar_kws={'label': 'Count'},
                    xticklabels=le.classes_, yticklabels=le.classes_)
        axes[0, 0].set_title(f'Baseline - Fold {fold_idx+1} (Unweighted)', fontsize=13, fontweight='bold')
        axes[0, 0].set_ylabel('True Label', fontsize=11)
        axes[0, 0].set_xlabel('Predicted Label', fontsize=11)
        axes[0, 0].set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=8)
        axes[0, 0].set_yticklabels(le.classes_, rotation=0, fontsize=8)

        sns.heatmap(enhanced_cm_unweighted, annot=False, cmap='Greens', ax=axes[0, 1],
                    cbar=True, cbar_kws={'label': 'Count'},
                    xticklabels=le.classes_, yticklabels=le.classes_)
        axes[0, 1].set_title(f'Enhanced - Fold {fold_idx+1} (Unweighted)', fontsize=13, fontweight='bold')
        axes[0, 1].set_ylabel('True Label', fontsize=11)
        axes[0, 1].set_xlabel('Predicted Label', fontsize=11)
        axes[0, 1].set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=8)
        axes[0, 1].set_yticklabels(le.classes_, rotation=0, fontsize=8)

        # Row 2: Weighted (normalized)
        sns.heatmap(baseline_cm_weighted, annot=False, cmap='Blues', ax=axes[1, 0],
                    cbar=True, cbar_kws={'label': 'Proportion'},
                    xticklabels=le.classes_, yticklabels=le.classes_, vmin=0, vmax=1)
        axes[1, 0].set_title(f'Baseline - Fold {fold_idx+1} (Weighted - Normalized)', fontsize=13, fontweight='bold')
        axes[1, 0].set_ylabel('True Label', fontsize=11)
        axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
        axes[1, 0].set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=8)
        axes[1, 0].set_yticklabels(le.classes_, rotation=0, fontsize=8)

        sns.heatmap(enhanced_cm_weighted, annot=False, cmap='Greens', ax=axes[1, 1],
                    cbar=True, cbar_kws={'label': 'Proportion'},
                    xticklabels=le.classes_, yticklabels=le.classes_, vmin=0, vmax=1)
        axes[1, 1].set_title(f'Enhanced - Fold {fold_idx+1} (Weighted - Normalized)', fontsize=13, fontweight='bold')
        axes[1, 1].set_ylabel('True Label', fontsize=11)
        axes[1, 1].set_xlabel('Predicted Label', fontsize=11)
        axes[1, 1].set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=8)
        axes[1, 1].set_yticklabels(le.classes_, rotation=0, fontsize=8)

        plt.tight_layout()
        cm_fig = results_dir / f"confusion_matrices_fold_{fold_idx+1:02d}.png"
        plt.savefig(cm_fig, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"[OK] Individual fold confusion matrices saved to {results_dir}/")

else:
    print("Matplotlib not available - skipping visualization")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
print(f"\nResults saved to: {results_dir}/")
print(f"  - synthetic_comparison_summary.json (all metrics)")
print(f"  - comparison_metrics.png (comparison charts)")
print(f"  - confusion_matrices_fold_*.png (detailed fold matrices)")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print(f"\nBaseline (Normal Data Only):")
print(f"  Mean Accuracy: {np.mean(baseline_accuracies):.4f} ± {np.std(baseline_accuracies):.4f}")
print(f"  Mean F1:       {np.mean(baseline_f1_scores):.4f} ± {np.std(baseline_f1_scores):.4f}")

print(f"\nEnhanced (Normal + Synthetic Training):")
print(f"  Mean Accuracy: {np.mean(enhanced_accuracies):.4f} ± {np.std(enhanced_accuracies):.4f}")
print(f"  Mean F1:       {np.mean(enhanced_f1_scores):.4f} ± {np.std(enhanced_f1_scores):.4f}")

print(f"\nSynthetic Data Impact:")
if accuracy_improvement > 0:
    print(f"  ✓ POSITIVE: Accuracy improved by {accuracy_improvement:.2f}%")
elif accuracy_improvement < 0:
    print(f"  ✗ NEGATIVE: Accuracy decreased by {abs(accuracy_improvement):.2f}%")
else:
    print(f"  ≈ NO CHANGE: Accuracy remained the same")

print(f"  ✓ PRECISION: {precision_improvement:+.2f}%")
print(f"  ✓ RECALL:    {recall_improvement:+.2f}%")
print("="*80)
