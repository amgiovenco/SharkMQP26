"""
Confidence Calibration Analysis for Synthetic Training Models

Tests model stability across different random seeds and analyzes:
1. Test accuracy consistency
2. Confidence calibration (are high confidences actually correct?)
3. Top-k prediction quality
4. Confidence when correct vs wrong

Run this to verify your models are production-ready for user-facing predictions.
"""

import os
import json
import pickle
import hashlib
import copy
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import calibration_curve

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
OUTPUT_DIR = SCRIPT_DIR / "confidence_analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "image_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Data split (0/50/50 - all synthetic for training)
REAL_VAL_SPLIT = 0.5  # 50% validation
REAL_TEST_SPLIT = 0.5  # 50% test

# Use all synthetic data (50 per species)
MAX_SYN_PER_SPECIES = 50

# Model parameters
CNN_IMAGE_SIZE = 224
CNN_BATCH_SIZE = 16
CNN_EPOCHS = 200
CNN_PATIENCE = 15
CNN_LEARNING_RATE = 0.0004303702377686196
CNN_WEIGHT_DECAY = 4.572988042665251e-06
CNN_DROPOUT_1 = 0.6217843386251581
CNN_DROPOUT_2 = 0.19498440140497733
CNN_FOCAL_ALPHA = 1.0
CNN_FOCAL_GAMMA = 1.2483412017424098
CNN_HIDDEN_DIM = 256

TCN_BATCH_SIZE = 8
TCN_EPOCHS = 200
TCN_PATIENCE = 30
TCN_LEARNING_RATE = 9.29558217542935e-05
TCN_WEIGHT_DECAY = 0.0044478395955166086
TCN_DROPOUT = 0.02208528755253426
TCN_KERNEL_SIZE = 13
TCN_NUM_CHANNELS = [32, 64, 96, 128, 160, 192, 224]
TCN_REVERSE_DILATION = True

STATS_N_ESTIMATORS = 1700
STATS_MAX_DEPTH = None
STATS_MIN_SAMPLES_SPLIT = 9
STATS_MIN_SAMPLES_LEAF = 1
STATS_MAX_FEATURES = 0.7
STATS_CLASS_WEIGHT = 'balanced'

STATS_TOP_FEATURES = [
    'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
    'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
    'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
    'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
    'slope_std', 'y_left_max'
]

print("=" * 80)
print("CONFIDENCE CALIBRATION ANALYSIS")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Testing seeds: {SEEDS_TO_TEST}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 80)
print()

# ============================================================================
# DATA LOADING (same as before)
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
# MODEL DEFINITIONS (CNN only for faster testing)
# ============================================================================

_CACHED_EFFICIENTNET_WEIGHTS = None

def get_cached_efficientnet_weights():
    """Load EfficientNet-B0 weights once and cache them."""
    global _CACHED_EFFICIENTNET_WEIGHTS
    if _CACHED_EFFICIENTNET_WEIGHTS is None:
        base_model = efficientnet_b0(weights='IMAGENET1K_V1')
        _CACHED_EFFICIENTNET_WEIGHTS = copy.deepcopy(base_model.state_dict())
    return _CACHED_EFFICIENTNET_WEIGHTS

def plot_fluorescence_curve_to_image(temps: List[float], fluor: List[float],
                                     dpi: int = 96, width: float = 3.0,
                                     height: float = 2.25) -> Image.Image:
    """Generate a 2D image from fluorescence curve data."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.plot(temps, fluor, 'b-', linewidth=2)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Fluorescence')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)

    return img

class FluorescenceDataset(Dataset):
    """PyTorch dataset for fluorescence curves as images."""

    def __init__(self, data: pd.DataFrame, label_encoder: LabelEncoder,
                 transform: Optional[transforms.Compose] = None,
                 cache_dir: Optional[Path] = None):
        self.data = data.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.transform = transform
        self.cache_dir = cache_dir

        self.temp_cols = [col for col in self.data.columns if col != 'Species']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        species = row['Species']
        fluor = row[self.temp_cols].values.astype(float)
        temps = np.linspace(20, 95, len(fluor))

        cache_path = None
        if self.cache_dir:
            cache_key = hashlib.md5(
                f"{species}_{'_'.join(f'{x:.6f}' for x in fluor)}".encode()
            ).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    img = pickle.load(f)
            else:
                img = plot_fluorescence_curve_to_image(temps.tolist(), fluor.tolist())
                with open(cache_path, 'wb') as f:
                    pickle.dump(img, f)
        else:
            img = plot_fluorescence_curve_to_image(temps.tolist(), fluor.tolist())

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = self.label_encoder.transform([species])[0]

        return img, label

class GaussianNoise(nn.Module):
    """Add Gaussian noise to tensor."""
    def __init__(self, std: float = 0.005):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

class SharkCNN(nn.Module):
    """EfficientNet-B0 with custom classifier head."""

    def __init__(self, num_classes: int, hidden_dim: int = CNN_HIDDEN_DIM):
        super().__init__()

        self.backbone = efficientnet_b0(weights=None)
        cached_weights = get_cached_efficientnet_weights()
        self.backbone.load_state_dict(cached_weights)

        in_features = self.backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(CNN_DROPOUT_1),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(CNN_DROPOUT_2),
            nn.Linear(hidden_dim, num_classes)
        )

        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha: float = 1.0, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def get_cnn_transforms():
    """Get training and validation transforms for CNN."""
    train_transform = transforms.Compose([
        transforms.Resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)),
        transforms.RandomAffine(degrees=0, translate=(0.0, 0.03)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        GaussianNoise(std=0.005),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def train_cnn_epoch(model, loader, criterion, optimizer, device):
    """Train CNN for one epoch."""
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

# ============================================================================
# CONFIDENCE ANALYSIS FUNCTIONS
# ============================================================================

def evaluate_cnn_with_confidence(model, dataloader, device, num_classes):
    """
    Evaluate CNN and return detailed confidence metrics.

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
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
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

    for i, (probs, true_label) in enumerate(zip(all_probs, all_labels)):
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

def train_and_evaluate_cnn(train_data, test_data, num_classes, seed):
    """Train CNN with given seed and evaluate with confidence metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_transform, val_transform = get_cnn_transforms()
    label_encoder = LabelEncoder()
    label_encoder.fit(train_data['Species'])

    train_dataset = FluorescenceDataset(train_data, label_encoder, train_transform, CACHE_DIR)
    test_dataset = FluorescenceDataset(test_data, label_encoder, val_transform, CACHE_DIR)

    train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=2)

    model = SharkCNN(num_classes=num_classes).to(DEVICE)
    criterion = FocalLoss(alpha=CNN_FOCAL_ALPHA, gamma=CNN_FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=CNN_LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    best_train_loss = float('inf')
    patience_counter = 0

    print(f"  Training with seed {seed}...")
    for epoch in range(CNN_EPOCHS):
        train_loss = train_cnn_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CNN_PATIENCE:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    # Evaluate with confidence metrics
    print(f"  Evaluating on test set...")
    results = evaluate_cnn_with_confidence(model, test_loader, DEVICE, num_classes)

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
    fig.suptitle('Confidence Calibration Analysis Across Seeds', fontsize=16, fontweight='bold')

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
    plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved confidence analysis plot to {output_dir / 'confidence_analysis.png'}")

    # Calibration curve for first seed
    seed = seeds[0]
    cal = all_results[seed]['calibration']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(cal['bin_confidences'], cal['bin_accuracies'], 'o-',
            label=f'Model (ECE={cal["ece"]:.4f})', linewidth=2, markersize=8)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Calibration Curve (Seed {seed})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved calibration curve to {output_dir / 'calibration_curve.png'}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nConfidence Calibration Analysis")
    print("Testing CNN model with 0/50/50 split (all synthetic training) across multiple seeds")
    print(f"\nConfiguration:")
    print(f"  Model: CNN (EfficientNet-B0)")
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
        results = train_and_evaluate_cnn(synthetic_train, real_test, num_classes, seed)
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

    results_path = OUTPUT_DIR / 'confidence_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nCheck {OUTPUT_DIR} for:")
    print(f"  - confidence_analysis.png (comparison across seeds)")
    print(f"  - calibration_curve.png (calibration curve)")
    print(f"  - confidence_analysis_results.json (detailed results)")
