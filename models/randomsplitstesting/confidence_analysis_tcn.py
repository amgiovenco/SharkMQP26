"""
Confidence Calibration Analysis for TCN Model

Tests TCN model stability across different random seeds and analyzes:
1. Test accuracy consistency
2. Confidence calibration (are high confidences actually correct?)
3. Top-k prediction quality
4. Confidence when correct vs wrong

Run this to verify your TCN model is production-ready for user-facing predictions.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Test these seeds for stability
SEEDS_TO_TEST = [8, 42, 123]

# Model configuration (using 60/20/20 split as recommended)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
REAL_DATA_PATH = DATA_DIR / "shark_dataset.csv"
SYNTHETIC_DIR = DATA_DIR / "syntheticDataIndividual"

# Output
OUTPUT_DIR = SCRIPT_DIR / "confidence_analysis_tcn_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data split (0/50/50 - all synthetic for training)
REAL_VAL_SPLIT = 0.5  # 50% validation
REAL_TEST_SPLIT = 0.5  # 50% test

# Use all synthetic data (50 per species)
MAX_SYN_PER_SPECIES = 50

# TCN parameters
TCN_BATCH_SIZE = 8
TCN_EPOCHS = 200
TCN_PATIENCE = 30
TCN_LEARNING_RATE = 9.29558217542935e-05
TCN_WEIGHT_DECAY = 0.0044478395955166086
TCN_DROPOUT = 0.02208528755253426
TCN_KERNEL_SIZE = 13
TCN_NUM_CHANNELS = [32, 64, 96, 128, 160, 192, 224]
TCN_REVERSE_DILATION = True

print("=" * 80)
print("CONFIDENCE CALIBRATION ANALYSIS - TCN MODEL")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Testing seeds: {SEEDS_TO_TEST}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 80)
print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_real_data() -> pd.DataFrame:
    """Load real shark dataset."""
    if not REAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Real data not found at {REAL_DATA_PATH}")
    return pd.read_csv(REAL_DATA_PATH)

def load_synthetic_data(species_list: List[str]) -> Dict[str, pd.DataFrame]:
    """Load synthetic data from individual species CSV files."""
    synthetic_data = {}

    if not SYNTHETIC_DIR.exists():
        print(f"Warning: Synthetic data directory not found at {SYNTHETIC_DIR}")
        return synthetic_data

    for species in species_list:
        filename = f"synthetic_{species.replace(' ', '_')}.csv"
        filepath = SYNTHETIC_DIR / filename

        if filepath.exists():
            df = pd.read_csv(filepath)
            synthetic_data[species] = df.reset_index(drop=True)

    return synthetic_data

def create_synthetic_training_set(synthetic_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all synthetic data (up to 50 per species)."""
    all_synthetic = []

    for species, df in synthetic_data.items():
        # Use up to MAX_SYN_PER_SPECIES samples per species
        n_samples = min(len(df), MAX_SYN_PER_SPECIES)
        sampled = df.iloc[:n_samples].copy()
        all_synthetic.append(sampled)

    synthetic_train = pd.concat(all_synthetic, ignore_index=True)
    return synthetic_train.reset_index(drop=True)

# ============================================================================
# TCN MODEL DEFINITIONS
# ============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution with weight normalization."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                     padding=self.padding, dilation=dilation)
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class TemporalBlock(nn.Module):
    """Temporal block with residual connections and batch normalization."""
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.bn2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for time-series classification."""
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=3, dropout=0.2, reverse_dilation=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            if reverse_dilation:
                dilation_size = 2 ** (num_levels - i - 1)
            else:
                dilation_size = 2 ** i

            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size,
                            dilation_size, dropout)
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        y = self.network(x)
        y = torch.mean(y, dim=2)
        return self.fc(y)

class TCNDataset(Dataset):
    """Dataset for TCN time-series data."""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_tcn_epoch(model, loader, criterion, optimizer, device):
    """Train TCN for one epoch."""
    model.train()
    total_loss = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ============================================================================
# CONFIDENCE ANALYSIS FUNCTIONS
# ============================================================================

def evaluate_tcn_with_confidence(model, loader, device):
    """
    Evaluate TCN and return detailed confidence metrics.

    Returns:
        dict with:
        - accuracy: Overall accuracy
        - f1: Macro F1 score
        - all_probs: numpy array of all predicted probabilities
        - all_preds: numpy array of predicted classes
        - all_labels: numpy array of true labels
        - avg_confidence: Average max probability
        - confidence_when_correct: Avg confidence on correct predictions
        - confidence_when_wrong: Avg confidence on wrong predictions
        - top3_accuracy: Fraction where true label is in top 3
        - top5_accuracy: Fraction where true label is in top 5
        - confidence_gap: Avg gap between 1st and 2nd choice
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Basic metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Confidence metrics
    max_probs = np.max(all_probs, axis=1)
    avg_confidence = np.mean(max_probs)

    # Confidence when correct vs wrong
    correct_mask = (all_preds == all_labels)
    conf_when_correct = np.mean(max_probs[correct_mask]) if correct_mask.any() else 0.0
    conf_when_wrong = np.mean(max_probs[~correct_mask]) if (~correct_mask).any() else 0.0

    # Top-k accuracy
    top3_correct = []
    top5_correct = []
    confidence_gaps = []

    for probs, true_label in zip(all_probs, all_labels):
        # Get top-k indices (sorted descending)
        sorted_indices = np.argsort(probs)[::-1]
        top3 = sorted_indices[:3]
        top5 = sorted_indices[:5]

        top3_correct.append(true_label in top3)
        top5_correct.append(true_label in top5)

        # Confidence gap between 1st and 2nd choice
        if len(sorted_indices) >= 2:
            gap = probs[sorted_indices[0]] - probs[sorted_indices[1]]
            confidence_gaps.append(gap)

    top3_accuracy = np.mean(top3_correct)
    top5_accuracy = np.mean(top5_correct)
    avg_confidence_gap = np.mean(confidence_gaps) if confidence_gaps else 0.0

    return {
        'accuracy': accuracy,
        'f1': f1,
        'all_probs': all_probs,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'avg_confidence': avg_confidence,
        'confidence_when_correct': conf_when_correct,
        'confidence_when_wrong': conf_when_wrong,
        'num_correct': np.sum(correct_mask),
        'num_wrong': np.sum(~correct_mask),
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'confidence_gap': avg_confidence_gap
    }

def compute_calibration_metrics(probs, preds, labels, n_bins=10):
    """
    Compute calibration curve and Expected Calibration Error (ECE).

    Returns:
        dict with:
        - ece: Expected Calibration Error
        - bin_accuracies: Accuracy in each confidence bin
        - bin_confidences: Average confidence in each bin
        - bin_counts: Number of samples in each bin
    """
    max_probs = np.max(probs, axis=1)
    correct = (preds == labels)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(max_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    total_samples = len(max_probs)

    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = max_probs[mask].mean()
            bin_count = mask.sum()

            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)

            # ECE contribution
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    return {
        'ece': ece,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_and_evaluate_tcn(train_data, test_data, num_classes, seed):
    """Train TCN with given seed and evaluate with confidence metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prepare data
    temp_cols = sorted([c for c in train_data.columns if c != 'Species'], key=lambda c: float(c))

    X_train = train_data[temp_cols].values
    y_train_labels = train_data['Species'].values

    X_test = test_data[temp_cols].values
    y_test_labels = test_data['Species'].values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_test = label_encoder.transform(y_test_labels)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for TCN (batch, channels, sequence)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    train_dataset = TCNDataset(X_train, y_train)
    test_dataset = TCNDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=TCN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TCN_BATCH_SIZE, shuffle=False)

    model = TemporalConvNet(
        num_inputs=1,
        num_channels=TCN_NUM_CHANNELS,
        num_classes=num_classes,
        kernel_size=TCN_KERNEL_SIZE,
        dropout=TCN_DROPOUT,
        reverse_dilation=TCN_REVERSE_DILATION
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=TCN_LEARNING_RATE, weight_decay=TCN_WEIGHT_DECAY)

    best_train_loss = float('inf')
    patience_counter = 0

    print(f"  Training with seed {seed}...", flush=True)
    for epoch in range(TCN_EPOCHS):
        train_loss = train_tcn_epoch(model, train_loader, criterion, optimizer, DEVICE)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{TCN_EPOCHS}, Loss: {train_loss:.4f}, Patience: {patience_counter}/{TCN_PATIENCE}", flush=True)

        if patience_counter >= TCN_PATIENCE:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    # Evaluate with confidence metrics
    print(f"  Evaluating on test set...")
    results = evaluate_tcn_with_confidence(model, test_loader, DEVICE)

    # Compute calibration
    calibration = compute_calibration_metrics(
        results['all_probs'],
        results['all_preds'],
        results['all_labels']
    )
    results['calibration'] = calibration

    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confidence_analysis(all_results, output_dir):
    """Generate comprehensive confidence analysis plots."""

    # Extract metrics across seeds
    seeds = list(all_results.keys())
    accuracies = [all_results[s]['accuracy'] for s in seeds]
    f1s = [all_results[s]['f1'] for s in seeds]
    avg_confs = [all_results[s]['avg_confidence'] for s in seeds]
    conf_correct = [all_results[s]['confidence_when_correct'] for s in seeds]
    conf_wrong = [all_results[s]['confidence_when_wrong'] for s in seeds]
    top3_accs = [all_results[s]['top3_accuracy'] for s in seeds]
    conf_gaps = [all_results[s]['confidence_gap'] for s in seeds]
    eces = [all_results[s]['calibration']['ece'] for s in seeds]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TCN Confidence Calibration Analysis Across Seeds', fontsize=16, fontweight='bold')

    # Plot 1: Accuracy and F1
    ax = axes[0, 0]
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, f1s, width, label='Macro F1', alpha=0.8)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Score')
    ax.set_title('Accuracy and F1 Across Seeds')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.set_ylim([0.95, 1.0])
    ax.grid(True, alpha=0.3)

    # Plot 2: Confidence metrics
    ax = axes[0, 1]
    ax.bar(x, avg_confs, alpha=0.8, label='Avg Confidence')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Confidence')
    ax.set_title('Average Confidence')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim([0.9, 1.0])
    ax.grid(True, alpha=0.3)

    # Plot 3: Confidence when correct vs wrong
    ax = axes[0, 2]
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, conf_correct, width, label='When Correct', alpha=0.8, color='green')
    ax.bar(x + width/2, conf_wrong, width, label='When Wrong', alpha=0.8, color='red')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence: Correct vs Wrong Predictions')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3)

    # Plot 4: Top-k accuracy
    ax = axes[1, 0]
    ax.bar(x, top3_accs, alpha=0.8)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Top-3 Accuracy')
    ax.set_title('Top-3 Accuracy (True label in top 3 predictions)')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim([0.95, 1.0])
    ax.grid(True, alpha=0.3)

    # Plot 5: Confidence gap
    ax = axes[1, 1]
    ax.bar(x, conf_gaps, alpha=0.8)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Confidence Gap')
    ax.set_title('Avg Gap Between 1st and 2nd Choice')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.grid(True, alpha=0.3)

    # Plot 6: Expected Calibration Error
    ax = axes[1, 2]
    ax.bar(x, eces, alpha=0.8, color='orange')
    ax.set_xlabel('Seed')
    ax.set_ylabel('ECE')
    ax.set_title('Expected Calibration Error (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis_tcn.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved confidence analysis plot to {output_dir / 'confidence_analysis_tcn.png'}")

    # Calibration curve for first seed
    seed = seeds[0]
    cal = all_results[seed]['calibration']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(cal['bin_confidences'], cal['bin_accuracies'], 'o-',
            label=f'TCN Model (ECE={cal["ece"]:.4f})', linewidth=2, markersize=8)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'TCN Calibration Curve (Seed {seed})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve_tcn.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved calibration curve to {output_dir / 'calibration_curve_tcn.png'}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nConfidence Calibration Analysis - TCN Model")
    print("Testing TCN model with 0/50/50 split (all synthetic training) across multiple seeds")
    print(f"\nConfiguration:")
    print(f"  Model: TCN (Temporal Convolutional Network)")
    print(f"  Seeds to test: {SEEDS_TO_TEST}")
    print(f"  Max synthetic per species: {MAX_SYN_PER_SPECIES}")
    print(f"  Device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    real_data = load_real_data()
    print(f"  Loaded {len(real_data)} real samples")

    # Split real data 50/50 for validation and test (use seed 8 for splitting)
    print("\nSplitting real data 50/50...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=REAL_TEST_SPLIT, random_state=8)
    for val_idx, test_idx in sss.split(real_data, real_data['Species']):
        real_val = real_data.iloc[val_idx].reset_index(drop=True)
        real_test = real_data.iloc[test_idx].reset_index(drop=True)

    num_classes = len(real_data['Species'].unique())
    print(f"  Validation: {len(real_val)} samples")
    print(f"  Test: {len(real_test)} samples")
    print(f"  Number of classes: {num_classes}")

    # Load synthetic data
    print("\nLoading synthetic data...")
    synthetic_data = load_synthetic_data(real_data['Species'].unique().tolist())
    print(f"  Loaded synthetic data for {len(synthetic_data)} species")

    # Create synthetic training set
    print("\nCreating synthetic training set...")
    synthetic_train = create_synthetic_training_set(synthetic_data)
    print(f"  Synthetic training: {len(synthetic_train)} samples")

    # Test across multiple seeds
    all_results = {}

    print("\n" + "="*80)
    print("TRAINING AND EVALUATING ACROSS SEEDS")
    print("="*80)

    for seed in SEEDS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}")

        print(f"  Training data: {len(synthetic_train)} samples (all synthetic)")
        print(f"  Test data: {len(real_test)} samples (real)")

        # Train and evaluate
        results = train_and_evaluate_tcn(synthetic_train, real_test, num_classes, seed)
        all_results[seed] = results

        # Print results
        print(f"\n  Results for seed {seed}:")
        print(f"    Accuracy: {results['accuracy']:.4f}")
        print(f"    Macro F1: {results['f1']:.4f}")
        print(f"    Avg Confidence: {results['avg_confidence']:.4f}")
        print(f"    Confidence when correct: {results['confidence_when_correct']:.4f}")
        print(f"    Confidence when wrong: {results['confidence_when_wrong']:.4f}")
        print(f"    Num wrong: {results['num_wrong']}")
        print(f"    Top-3 Accuracy: {results['top3_accuracy']:.4f}")
        print(f"    Top-5 Accuracy: {results['top5_accuracy']:.4f}")
        print(f"    Confidence Gap (1st vs 2nd): {results['confidence_gap']:.4f}")
        print(f"    Expected Calibration Error: {results['calibration']['ece']:.4f}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY ACROSS SEEDS")
    print("="*80)

    accuracies = [all_results[s]['accuracy'] for s in SEEDS_TO_TEST]
    f1s = [all_results[s]['f1'] for s in SEEDS_TO_TEST]
    eces = [all_results[s]['calibration']['ece'] for s in SEEDS_TO_TEST]

    print(f"\nAccuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Macro F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"ECE:       {np.mean(eces):.4f} ± {np.std(eces):.4f}")

    # Check stability
    acc_range = max(accuracies) - min(accuracies)
    f1_range = max(f1s) - min(f1s)

    print(f"\nStability Check:")
    print(f"  Accuracy range: {acc_range:.4f} (max - min)")
    print(f"  F1 range: {f1_range:.4f} (max - min)")

    if acc_range < 0.02 and f1_range < 0.02:
        print(f"  ✓ STABLE: Results are consistent across seeds (<2% variation)")
    else:
        print(f"  ✗ UNSTABLE: Results vary significantly across seeds (>2% variation)")

    # Check calibration
    avg_ece = np.mean(eces)
    if avg_ece < 0.05:
        print(f"\nCalibration Check:")
        print(f"  ✓ WELL CALIBRATED: ECE = {avg_ece:.4f} < 0.05")
    else:
        print(f"\nCalibration Check:")
        print(f"  ⚠ POORLY CALIBRATED: ECE = {avg_ece:.4f} > 0.05")
        print(f"  Consider adding temperature scaling for better confidence estimates")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_confidence_analysis(all_results, OUTPUT_DIR)

    # Save detailed results
    summary = {
        'model': 'TCN',
        'seeds_tested': SEEDS_TO_TEST,
        'max_synthetic_per_species': MAX_SYN_PER_SPECIES,
        'summary_stats': {
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'ece_mean': float(np.mean(eces)),
            'ece_std': float(np.std(eces)),
            'accuracy_range': float(acc_range),
            'f1_range': float(f1_range)
        },
        'per_seed_results': {}
    }

    for seed in SEEDS_TO_TEST:
        r = all_results[seed]
        summary['per_seed_results'][seed] = {
            'accuracy': float(r['accuracy']),
            'f1': float(r['f1']),
            'avg_confidence': float(r['avg_confidence']),
            'confidence_when_correct': float(r['confidence_when_correct']),
            'confidence_when_wrong': float(r['confidence_when_wrong']),
            'num_wrong': int(r['num_wrong']),
            'top3_accuracy': float(r['top3_accuracy']),
            'top5_accuracy': float(r['top5_accuracy']),
            'confidence_gap': float(r['confidence_gap']),
            'ece': float(r['calibration']['ece'])
        }

    results_path = OUTPUT_DIR / 'confidence_analysis_tcn_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nCheck {OUTPUT_DIR} for:")
    print(f"  - confidence_analysis_tcn.png (comparison across seeds)")
    print(f"  - calibration_curve_tcn.png (calibration curve)")
    print(f"  - confidence_analysis_tcn_results.json (detailed results)")
