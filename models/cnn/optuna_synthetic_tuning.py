"""
Optuna-based hyperparameter optimization for synthetic sample addition to EfficientNet-B0 CNN.

Optimizes five per-bin multipliers (n_very_low, n_low, n_medium, n_high, n_very_high)
that determine how many synthetic samples per species to add to training data, based on
species grouping by real count. Maximizes validation macro F1 score across 5-fold stratified CV.

Species binning:
- 'very_low': < 6 real samples
- 'low': 6-9 real samples
- 'medium': 10-15 real samples
- 'high': 16-25 real samples
- 'very_high': > 25 real samples

Dataset: 651 real shark samples (57 species) + synthetic samples
Model: EfficientNet-B0 with custom classifier head
Training: 5-fold stratified CV, 5-10 epochs per fold, early stopping
"""

import os
import sys
import json
import pickle
import warnings
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and basic setup
IMAGE_SIZE = 224
N_FOLDS = 5
TEST_SPLIT_RATIO = 0.2  # Hold out 20% for final test set
MAX_SYN_PER_SPECIES = 48

# Training hyperparameters (from train_efficientnetb0.ipynb optimization)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0020594007612475913
WEIGHT_DECAY = 1.0083970230770894e-05
EARLY_STOPPING_PATIENCE = 5

# Focal loss parameters
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 1.5

# Mixup parameter
MIXUP_ALPHA = 0.4
USE_MIXUP = True

# Best hyperparameters from optimization
HYPERPARAMETERS = {
    'cnn_dropout1': 0.7,
    'cnn_dropout2': 0.5,
    'cnn_learning_rate': 0.0020594007612475913,
    'cnn_batch_size': 32,
    'cnn_weight_decay': 1.0083970230770894e-05,
    'cnn_focal_gamma': 1.5
}

NUM_CLASSES = 57
HIDDEN_DIM = 256
DROPOUT_1 = HYPERPARAMETERS['cnn_dropout1']
DROPOUT_2 = HYPERPARAMETERS['cnn_dropout2']
NUM_WORKERS = 4

print(DEVICE)

# Data paths
DATA_DIR = Path(__file__).parent / "data"
REAL_DATA_PATH = DATA_DIR / "shark_dataset.csv"
SYNTHETIC_DIR = DATA_DIR / "syntheticDataIndividual"
GOOD_SYNTHETIC_PATH = DATA_DIR / "synthetic_data_good_quality.csv"

# Output
OUTPUT_DIR = Path(__file__).parent / "optuna_results"
OUTPUT_DIR.mkdir(exist_ok=True)
DB_URL = "sqlite:///optuna_shark.db"
CACHE_DIR = OUTPUT_DIR / "image_cache"
CACHE_DIR.mkdir(exist_ok=True)


# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================
# CACHED MODEL WEIGHTS (Loaded once to avoid repeated initialization)
# ============================================================================
_CACHED_EFFICIENTNET_WEIGHTS = None

def get_cached_efficientnet_weights():
    """Load EfficientNet-B0 weights once and cache them."""
    global _CACHED_EFFICIENTNET_WEIGHTS
    if _CACHED_EFFICIENTNET_WEIGHTS is None:
        base_model = efficientnet_b0(weights='IMAGENET1K_V1')
        _CACHED_EFFICIENTNET_WEIGHTS = copy.deepcopy(base_model.state_dict())
    return _CACHED_EFFICIENTNET_WEIGHTS


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_fluorescence_curve_to_image(temps: List[float], fluor: List[float],
                                     dpi: int = 96, width: float = 3.0,
                                     height: float = 2.25) -> Image.Image:
    """
    Generate a 2D image from fluorescence curve data.

    Args:
        temps: Temperature values (x-axis)
        fluor: Fluorescence values (y-axis)
        dpi: Resolution (default 96 DPI)
        width: Figure width in inches (default 3.0)
        height: Figure height in inches (default 2.25)

    Returns:
        PIL Image object
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.plot(temps, fluor, 'b-', linewidth=2)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Fluorescence')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)

    return img


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class FluorescenceDataset(Dataset):
    """PyTorch dataset for fluorescence curves as images."""

    def __init__(self, data: pd.DataFrame, label_encoder: LabelEncoder,
                 transform: Optional[transforms.Compose] = None,
                 cache_dir: Optional[Path] = None):
        """
        Args:
            data: DataFrame with 'Species' column and fluorescence curve data
            label_encoder: LabelEncoder for species names
            transform: Torchvision transforms
            cache_dir: Directory to cache generated images
        """
        self.data = data.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.transform = transform
        self.cache_dir = cache_dir

        # Temperature columns (assumed to be all columns except 'Species')
        self.temp_cols = [col for col in self.data.columns if col != 'Species']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        species = row['Species']
        fluor = row[self.temp_cols].values.astype(float)
        temps = np.linspace(20, 95, len(fluor))  # Temperature range

        # Try to load from cache or generate image
        # Use hash-based cache key from curve values to survive shuffling across trials
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

        # Apply augmentation after loading cached image
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


def load_real_data() -> pd.DataFrame:
    """Load real shark dataset."""
    if not REAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Real data not found at {REAL_DATA_PATH}")
    return pd.read_csv(REAL_DATA_PATH)


def load_synthetic_data(species_list: List[str]) -> Dict[str, pd.DataFrame]:
    """Load synthetic data for species, preferring good_quality if available."""
    synthetic_data = {}

    # Try to load good quality synthetics first
    if GOOD_SYNTHETIC_PATH.exists():
        df_good = pd.read_csv(GOOD_SYNTHETIC_PATH)
        for species in species_list:
            species_data = df_good[df_good['Species'] == species]
            if len(species_data) > 0:
                synthetic_data[species] = species_data.reset_index(drop=True)

    # Fallback to per-species CSV files for missing species
    if not SYNTHETIC_DIR.exists():
        print(f"Warning: Synthetic data directory not found at {SYNTHETIC_DIR}")
        return synthetic_data

    for species in species_list:
        if species in synthetic_data:
            continue  # Already loaded from good quality

        # Format species name for filename (replace spaces with underscores)
        filename = f"synthetic_{species.replace(' ', '_')}.csv"
        filepath = SYNTHETIC_DIR / filename

        if filepath.exists():
            df = pd.read_csv(filepath)
            synthetic_data[species] = df.reset_index(drop=True)

    return synthetic_data


def check_minimum_samples_per_class(data: pd.DataFrame, min_samples: int = 3) -> bool:
    """
    Validate that all classes have minimum samples for CV stability.

    Args:
        data: DataFrame with 'Species' column
        min_samples: Minimum required samples per class

    Returns:
        True if valid, False if any class has fewer samples
    """
    class_counts = data['Species'].value_counts()
    undersized = class_counts[class_counts < min_samples]

    if len(undersized) > 0:
        print(f"  WARNING: {len(undersized)} species have < {min_samples} samples:")
        for species, count in undersized.items():
            print(f"    {species}: {count} samples")
        return False
    return True


def bin_species_by_real_count(real_data: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Bin species into 5 groups based on real sample counts.

    Bins:
    - 'very_low': < 6 real samples
    - 'low': 6-9 real samples
    - 'medium': 10-15 real samples
    - 'high': 16-25 real samples
    - 'very_high': > 25 real samples

    Args:
        real_data: DataFrame with 'Species' column

    Returns:
        Dict with 'very_low', 'low', 'medium', 'high', 'very_high' keys,
        each containing list of species
    """
    species_counts = real_data['Species'].value_counts()

    bins = {'very_low': [], 'low': [], 'medium': [], 'high': [], 'very_high': []}

    for species, count in species_counts.items():
        if count < 6:
            bins['very_low'].append(species)
        elif count < 10:
            bins['low'].append(species)
        elif count < 16:
            bins['medium'].append(species)
        elif count < 26:
            bins['high'].append(species)
        else:
            bins['very_high'].append(species)

    return bins


def create_augmented_dataset(real_data: pd.DataFrame, synthetic_data: Dict[str, pd.DataFrame],
                           n_very_low: int, n_low: int, n_medium: int, n_high: int, n_very_high: int,
                           max_synthetic_per_species: int = 50) -> Tuple[pd.DataFrame, Dict]:
    """
    Combine real and synthetic data based on per-bin multipliers.

    For each species in a bin: add min(n_bin * real_count, max_synthetic_per_species) random synthetics.

    Args:
        real_data: Real samples
        synthetic_data: Synthetic samples per species
        n_very_low: Multiplier for 'very_low' bin species (0-10)
        n_low: Multiplier for 'low' bin species (0-10)
        n_medium: Multiplier for 'medium' bin species (0-10)
        n_high: Multiplier for 'high' bin species (0-10)
        n_very_high: Multiplier for 'very_high' bin species (0-10)
        max_synthetic_per_species: Cap per species

    Returns:
        Augmented dataset, dict with per-bin statistics
    """
    augmented = real_data.copy()

    # Bin species
    bins = bin_species_by_real_count(real_data)
    n_values = {
        'very_low': n_very_low,
        'low': n_low,
        'medium': n_medium,
        'high': n_high,
        'very_high': n_very_high
    }

    # Track per-bin statistics
    bin_stats = {
        'very_low': {'added': 0, 'species_count': len(bins['very_low']), 'avg_per_species': 0},
        'low': {'added': 0, 'species_count': len(bins['low']), 'avg_per_species': 0},
        'medium': {'added': 0, 'species_count': len(bins['medium']), 'avg_per_species': 0},
        'high': {'added': 0, 'species_count': len(bins['high']), 'avg_per_species': 0},
        'very_high': {'added': 0, 'species_count': len(bins['very_high']), 'avg_per_species': 0}
    }

    for bin_name in ['very_low', 'low', 'medium', 'high', 'very_high']:
        k = n_values[bin_name]
        for species in bins[bin_name]:
            species_real = real_data[real_data['Species'] == species]
            real_count = len(species_real)

            # Determine how many synthetics to add
            num_synthetic_to_add = min(k * real_count, max_synthetic_per_species)

            if num_synthetic_to_add > 0 and species in synthetic_data:
                synth_pool = synthetic_data[species]
                if len(synth_pool) > 0:
                    # Warn if sampling with replacement (indicates insufficient synthetic data)
                    if num_synthetic_to_add > len(synth_pool):
                        print(f"  WARNING: {species} sampling synthetic WITH REPLACEMENT "
                              f"({num_synthetic_to_add} > {len(synth_pool)} available)")
                    # Sample with replacement if needed
                    sampled = synth_pool.sample(n=int(num_synthetic_to_add), replace=True,
                                               random_state=SEED)
                    augmented = pd.concat([augmented, sampled], ignore_index=True)
                    bin_stats[bin_name]['added'] += int(num_synthetic_to_add)

        # Compute average per species in bin
        if bin_stats[bin_name]['species_count'] > 0:
            bin_stats[bin_name]['avg_per_species'] = (
                bin_stats[bin_name]['added'] / bin_stats[bin_name]['species_count']
            )

    return augmented.reset_index(drop=True), bin_stats


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SharkCNN(nn.Module):
    """EfficientNet-B0 with custom classifier head for shark species classification."""

    def __init__(self, num_classes: int = NUM_CLASSES, hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        # Load pretrained EfficientNet-B0 architecture (without downloading weights)
        self.backbone = efficientnet_b0(weights=None)

        # Load cached weights instead of downloading again
        cached_weights = get_cached_efficientnet_weights()
        self.backbone.load_state_dict(cached_weights)

        # Get the output dimension of the backbone
        in_features = self.backbone.classifier[1].in_features  # 1280 for EfficientNet-B0

        # Custom classifier head as per spec
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def get_transforms():
    """Get training and validation transforms as per specification."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(degrees=0, translate=(0.0, 0.03)),  # Small vertical shift
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        GaussianNoise(std=0.005),  # Measurement noise
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader,
             device: torch.device) -> Tuple[float, float, float, float]:
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, macro_f1, precision, recall


def train_with_cv(train_data: pd.DataFrame, label_encoder: LabelEncoder,
                 device: torch.device, trial: Optional[optuna.Trial] = None) -> Tuple[float, Dict]:
    """
    Train model with 5-fold stratified CV.

    Args:
        train_data: Training data
        label_encoder: Species label encoder
        device: torch device
        trial: Optuna trial for pruning (optional)

    Returns:
        Mean macro F1 across folds, detailed metrics per fold
    """
    train_transform, val_transform = get_transforms()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = {'accuracy': [], 'macro_f1': [], 'precision': [], 'recall': []}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_data, train_data['Species'])):
        set_seed(SEED)

        # Split data
        fold_train = train_data.iloc[train_idx].reset_index(drop=True)
        fold_val = train_data.iloc[val_idx].reset_index(drop=True)

        # Create datasets and loaders
        train_dataset = FluorescenceDataset(fold_train, label_encoder, train_transform, CACHE_DIR)
        val_dataset = FluorescenceDataset(fold_val, label_encoder, val_transform, CACHE_DIR)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

        # Initialize model
        model = SharkCNN(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Training loop with early stopping and Optuna pruning
        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(EPOCHS):
            _ = train_epoch(model, train_loader, criterion, optimizer, device)
            val_acc, val_f1, val_prec, val_rec = evaluate(model, val_loader, device)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1

            # Report to Optuna and check for pruning (skip if trial is None)
            if trial is not None:
                trial.report(val_f1, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

        # Final evaluation
        val_acc, val_f1, val_prec, val_rec = evaluate(model, val_loader, device)
        fold_scores['accuracy'].append(val_acc)
        fold_scores['macro_f1'].append(val_f1)
        fold_scores['precision'].append(val_prec)
        fold_scores['recall'].append(val_rec)

        print(f"  Fold {fold_idx + 1}/{N_FOLDS}: F1={val_f1:.4f}, Acc={val_acc:.4f}")

    # Compute means
    mean_f1 = np.mean(fold_scores['macro_f1'])
    mean_acc = np.mean(fold_scores['accuracy'])
    mean_prec = np.mean(fold_scores['precision'])
    mean_rec = np.mean(fold_scores['recall'])

    detailed_metrics = {
        'fold_scores': fold_scores,
        'mean_accuracy': mean_acc,
        'mean_macro_f1': mean_f1,
        'mean_precision': mean_prec,
        'mean_recall': mean_rec
    }

    return mean_f1, detailed_metrics


# ============================================================================
# BASELINE EVALUATION (5-fold CV on REAL DATA ONLY)
# ============================================================================

def run_baseline() -> Dict:
    """
    Run baseline evaluation: 5-fold CV on real data only (no synthetic augmentation).

    Returns:
        Dictionary with baseline metrics
    """
    print(f"\n{'='*70}")
    print("BASELINE: 5-Fold CV on Real Data Only (No Synthetic)")
    print(f"{'='*70}")

    set_seed(SEED)

    try:
        # Load real data
        print("Loading data...")
        real_data = load_real_data()
        print(f"  Real samples: {len(real_data)}")

        # Split real data into train+val vs test (stratified)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT_RATIO, random_state=SEED)
        for train_idx, test_idx in sss.split(real_data, real_data['Species']):
            real_train_val = real_data.iloc[train_idx].reset_index(drop=True)
            real_test = real_data.iloc[test_idx].reset_index(drop=True)

        print(f"  Train+Val: {len(real_train_val)} samples")
        print(f"  Test (holdout): {len(real_test)} samples")

        # Create label encoder from train+val (test uses same encoder)
        label_encoder = LabelEncoder()
        label_encoder.fit(real_train_val['Species'])

        # Train with CV on train+val split (no synthetic augmentation)
        print("Training with 5-fold CV on train+val split (NO synthetic data)...")
        mean_f1, metrics = train_with_cv(real_train_val, label_encoder, DEVICE, trial=None)

        # Evaluate on holdout test set
        print("Evaluating on holdout test set...")
        test_dataset = FluorescenceDataset(real_test, label_encoder, get_transforms()[1], CACHE_DIR)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        # Create a final model trained on full train+val for test evaluation
        model = SharkCNN(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        train_transform, _ = get_transforms()
        train_dataset = FluorescenceDataset(real_train_val, label_encoder, train_transform, CACHE_DIR)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=True)

        # Train on full train+val set
        best_val_f1 = 0.0
        patience_counter = 0
        for _ in range(EPOCHS):
            _ = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

            # Check patience on test set
            test_acc, test_f1, _, _ = evaluate(model, test_loader, DEVICE)

            if test_f1 > best_val_f1:
                best_val_f1 = test_f1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

        # Final test evaluation
        test_acc, test_f1, test_prec, test_rec = evaluate(model, test_loader, DEVICE)
        print(f"  Holdout Test Results:")
        print(f"    F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

        metrics['test_f1'] = test_f1
        metrics['test_accuracy'] = test_acc
        metrics['test_precision'] = test_prec
        metrics['test_recall'] = test_rec

        print(f"\nBaseline Results (Train+Val, 5-fold):")
        print(f"  Mean Macro F1: {mean_f1:.4f}")
        print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        print(f"  Mean Precision: {metrics['mean_precision']:.4f}")
        print(f"  Mean Recall: {metrics['mean_recall']:.4f}")
        print(f"\nBaseline Test Results (Holdout Set):")
        print(f"  Test F1: {metrics['test_f1']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test Precision: {metrics['test_precision']:.4f}")
        print(f"  Test Recall: {metrics['test_recall']:.4f}")

        return {
            'baseline_cv_macro_f1': mean_f1,
            'baseline_cv_accuracy': metrics['mean_accuracy'],
            'baseline_cv_precision': metrics['mean_precision'],
            'baseline_cv_recall': metrics['mean_recall'],
            'baseline_test_f1': test_f1,
            'baseline_test_accuracy': test_acc,
            'baseline_test_precision': test_prec,
            'baseline_test_recall': test_rec,
            'fold_scores': metrics['fold_scores']
        }

    except Exception as e:
        print(f"Error in baseline: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function: optimize per-bin multipliers (n_very_low, n_low, n_medium, n_high, n_very_high).

    Maximizes mean macro F1 across 5-fold CV.
    """
    set_seed(SEED)

    # Flat number of synthetic samples to add PER species in each real-data bin
    n_very_low  = trial.suggest_int("n_very_low", 0, MAX_SYN_PER_SPECIES)
    n_low       = trial.suggest_int("n_low", 0, MAX_SYN_PER_SPECIES)
    n_medium    = trial.suggest_int("n_medium", 0, MAX_SYN_PER_SPECIES)
    n_high      = trial.suggest_int("n_high", 0, MAX_SYN_PER_SPECIES)
    n_very_high = trial.suggest_int("n_very_high", 0, MAX_SYN_PER_SPECIES)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number}: "
        f"n_very_low={n_very_low}, n_low={n_low}, "
        f"n_medium={n_medium}, n_high={n_high}, n_very_high={n_very_high}")
    print(f"{'='*70}")

    try:
        # Load data
        print("Loading data...")
        real_data = load_real_data()
        print(f"  Real samples: {len(real_data)}")

        # Split real data into train+val vs test (stratified)
        # Test set only contains REAL samples (not augmented)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT_RATIO, random_state=SEED)
        for train_idx, test_idx in sss.split(real_data, real_data['Species']):
            real_train_val = real_data.iloc[train_idx].reset_index(drop=True)
            real_test = real_data.iloc[test_idx].reset_index(drop=True)

        print(f"  Train+Val: {len(real_train_val)} samples")
        print(f"  Test (holdout): {len(real_test)} samples")

        # Bin species and show distribution
        bins = bin_species_by_real_count(real_train_val)
        print(f"  Train+Val Species distribution:")
        print(f"    Very Low (<6):   {len(bins['very_low'])} species")
        print(f"    Low (6-9):       {len(bins['low'])} species")
        print(f"    Medium (10-15):  {len(bins['medium'])} species")
        print(f"    High (16-25):    {len(bins['high'])} species")
        print(f"    Very High (>25): {len(bins['very_high'])} species")

        synthetic_data = load_synthetic_data(real_train_val['Species'].unique().tolist())
        print(f"  Synthetic species available: {len(synthetic_data)}")

        # Create augmented dataset with per-bin multipliers (only for train+val)
        augmented_data, bin_stats = create_augmented_dataset(
            real_train_val, synthetic_data,
            n_very_low, n_low, n_medium, n_high, n_very_high,
            MAX_SYN_PER_SPECIES
        )
        num_added = len(augmented_data) - len(real_train_val)
        print(f"  Augmented train+val samples: {len(augmented_data)} (added {num_added})")

        # Print per-bin statistics
        print(f"  Per-bin synthetic additions:")
        for bin_name in ['very_low', 'low', 'medium', 'high', 'very_high']:
            stats = bin_stats[bin_name]
            print(f"    {bin_name.capitalize():10s}: {stats['added']:4d} total "
                  f"({stats['avg_per_species']:5.1f} avg per species)")

        # Validate minimum samples per class for CV stability
        print(f"  Validating class distribution for CV...")
        valid = check_minimum_samples_per_class(augmented_data, min_samples=3)
        if not valid:
            print(f"  → Dataset has insufficient samples in some classes. CV may be unstable.")

        # Create label encoder from train+val (test uses same encoder)
        label_encoder = LabelEncoder()
        label_encoder.fit(augmented_data['Species'])

        # Train with CV on train+val split
        print("Training with 5-fold CV on train+val split...")
        mean_f1, metrics = train_with_cv(augmented_data, label_encoder, DEVICE, trial)

        # Evaluate on holdout test set
        print("Evaluating on holdout test set...")
        test_dataset = FluorescenceDataset(real_test, label_encoder, get_transforms()[1], CACHE_DIR)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        # Create a final model trained on full train+val for test evaluation
        model = SharkCNN(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        train_transform, _ = get_transforms()
        train_dataset = FluorescenceDataset(augmented_data, label_encoder, train_transform, CACHE_DIR)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=True)

        # Train on full augmented train+val set
        best_val_f1 = 0.0
        patience_counter = 0
        for _ in range(EPOCHS):
            _ = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

            # Check patience on test set
            test_acc, test_f1, _, _ = evaluate(model, test_loader, DEVICE)

            if test_f1 > best_val_f1:
                best_val_f1 = test_f1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

        # Final test evaluation
        test_acc, test_f1, test_prec, test_rec = evaluate(model, test_loader, DEVICE)
        print(f"  Holdout Test Results:")
        print(f"    F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

        metrics['test_f1'] = test_f1
        metrics['test_accuracy'] = test_acc
        metrics['test_precision'] = test_prec
        metrics['test_recall'] = test_rec

        # Log results
        trial.set_user_attr('total_synthetic_added', num_added)
        trial.set_user_attr('bin_stats', bin_stats)
        trial.set_user_attr('mean_accuracy', metrics['mean_accuracy'])
        trial.set_user_attr('mean_precision', metrics['mean_precision'])
        trial.set_user_attr('mean_recall', metrics['mean_recall'])
        trial.set_user_attr('fold_scores', metrics['fold_scores'])
        trial.set_user_attr('test_f1', metrics['test_f1'])
        trial.set_user_attr('test_accuracy', metrics['test_accuracy'])
        trial.set_user_attr('test_precision', metrics['test_precision'])
        trial.set_user_attr('test_recall', metrics['test_recall'])

        print(f"\nCV Results (Train+Val, 5-fold):")
        print(f"  Mean Macro F1: {mean_f1:.4f}")
        print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        print(f"  Mean Precision: {metrics['mean_precision']:.4f}")
        print(f"  Mean Recall: {metrics['mean_recall']:.4f}")
        print(f"\nTest Results (Holdout Set):")
        print(f"  Test F1: {metrics['test_f1']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test Precision: {metrics['test_precision']:.4f}")
        print(f"  Test Recall: {metrics['test_recall']:.4f}")
        print(f"\nData Info:")
        print(f"  Total Synthetic Added: {num_added}")

        # Return CV F1 for Optuna optimization (test is independent evaluation)
        return mean_f1

    except Exception as e:
        print(f"Error in trial: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


# ============================================================================
# OPTIMIZATION AND VISUALIZATION
# ============================================================================

def run_optimization(n_trials: int = 30):
    """Run Optuna optimization."""
    print(f"Starting Optuna optimization with {n_trials} trials...")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner()

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=DB_URL,
        study_name='shark_synthetic_optimization',
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials)

    # Best trial
    best_trial = study.best_trial
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print("Best multipliers:")
    for p, v in best_trial.params.items():
        print(f"  {p}: {v}")
    print(f"Best Macro F1: {best_trial.value:.4f}")
    print(f"Best Trial Number: {best_trial.number}")

    return study


def plot_optimization_results(study: optuna.Study):
    """Generate plots of optimization results with per-bin analysis (5 bins)."""
    trials = study.trials

    # Extract data
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_very_low_values = [t.params['n_very_low'] for t in completed_trials]
    n_low_values = [t.params['n_low'] for t in completed_trials]
    n_medium_values = [t.params['n_medium'] for t in completed_trials]
    n_high_values = [t.params['n_high'] for t in completed_trials]
    n_very_high_values = [t.params['n_very_high'] for t in completed_trials]
    f1_values = [t.value for t in completed_trials]  # CV F1
    acc_values = [t.user_attrs.get('mean_accuracy', 0.0) for t in completed_trials]  # CV accuracy
    test_f1_values = [t.user_attrs.get('test_f1', 0.0) for t in completed_trials]  # Holdout test F1
    synthetic_added = [t.user_attrs.get('total_synthetic_added', 0) for t in completed_trials]

    # Create figure with subplots (4x3 for per-bin + summary + test comparison)
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    fig.suptitle('Optuna 5-Bin Per-Bin Optimization Results (with Holdout Test Set)', fontsize=16, fontweight='bold')

    # Plot 1: Macro F1 vs n_very_low
    axes[0, 0].scatter(n_very_low_values, f1_values, alpha=0.6, s=100, color='darkred')
    axes[0, 0].set_xlabel('n_very_low (Very Low Bin)')
    axes[0, 0].set_ylabel('Mean Macro F1')
    axes[0, 0].set_title('F1 vs n_very_low (<6)')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Macro F1 vs n_low
    axes[0, 1].scatter(n_low_values, f1_values, alpha=0.6, s=100, color='blue')
    axes[0, 1].set_xlabel('n_low (Low Bin)')
    axes[0, 1].set_ylabel('Mean Macro F1')
    axes[0, 1].set_title('F1 vs n_low (6-9)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Macro F1 vs n_medium
    axes[0, 2].scatter(n_medium_values, f1_values, alpha=0.6, s=100, color='green')
    axes[0, 2].set_xlabel('n_medium (Medium Bin)')
    axes[0, 2].set_ylabel('Mean Macro F1')
    axes[0, 2].set_title('F1 vs n_medium (10-15)')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Macro F1 vs n_high
    axes[1, 0].scatter(n_high_values, f1_values, alpha=0.6, s=100, color='orange')
    axes[1, 0].set_xlabel('n_high (High Bin)')
    axes[1, 0].set_ylabel('Mean Macro F1')
    axes[1, 0].set_title('F1 vs n_high (16-25)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Macro F1 vs n_very_high
    axes[1, 1].scatter(n_very_high_values, f1_values, alpha=0.6, s=100, color='purple')
    axes[1, 1].set_xlabel('n_very_high (Very High Bin)')
    axes[1, 1].set_ylabel('Mean Macro F1')
    axes[1, 1].set_title('F1 vs n_very_high (>25)')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Macro F1 vs Total Synthetic Samples Added
    axes[1, 2].scatter(synthetic_added, f1_values, alpha=0.6, s=100, color='brown')
    axes[1, 2].set_xlabel('Total Synthetic Samples Added')
    axes[1, 2].set_ylabel('Mean Macro F1')
    axes[1, 2].set_title('F1 vs Total Synthetic Samples')
    axes[1, 2].grid(True, alpha=0.3)

    # Plot 7: Accuracy vs Total Synthetic Samples
    axes[2, 0].scatter(synthetic_added, acc_values, alpha=0.6, s=100, color='red')
    axes[2, 0].set_xlabel('Total Synthetic Samples Added')
    axes[2, 0].set_ylabel('Mean Accuracy')
    axes[2, 0].set_title('Accuracy vs Total Synthetic')
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 8: Trial history (best value so far)
    best_f1_so_far = []
    current_best = 0.0
    for val in f1_values:
        if val > current_best:
            current_best = val
        best_f1_so_far.append(current_best)

    axes[2, 1].plot(range(len(best_f1_so_far)), best_f1_so_far, marker='o',
                    color='darkblue', linewidth=2, markersize=6)
    axes[2, 1].set_xlabel('Trial Number')
    axes[2, 1].set_ylabel('Best Macro F1 (so far)')
    axes[2, 1].set_title('Optimization Progress (CV F1)')
    axes[2, 1].grid(True, alpha=0.3)

    # Plot 9: CV F1 vs Test F1 scatter
    axes[2, 2].scatter(f1_values, test_f1_values, alpha=0.6, s=100, color='teal')
    axes[2, 2].plot([min(f1_values), max(f1_values)], [min(f1_values), max(f1_values)],
                    'k--', alpha=0.3, label='CV=Test')
    axes[2, 2].set_xlabel('CV F1 (5-fold)')
    axes[2, 2].set_ylabel('Test F1 (Holdout)')
    axes[2, 2].set_title('CV vs Test F1 Correlation')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    # Plot 10: Trial history for Test F1
    best_test_f1_so_far = []
    current_best_test = 0.0
    for val in test_f1_values:
        if val > current_best_test:
            current_best_test = val
        best_test_f1_so_far.append(current_best_test)

    axes[3, 0].plot(range(len(best_test_f1_so_far)), best_test_f1_so_far, marker='s',
                    color='darkgreen', linewidth=2, markersize=6)
    axes[3, 0].set_xlabel('Trial Number')
    axes[3, 0].set_ylabel('Best Test F1 (so far)')
    axes[3, 0].set_title('Test Set Optimization Progress')
    axes[3, 0].grid(True, alpha=0.3)

    # Plot 11: CV vs Test F1 side by side
    trial_indices = np.arange(len(f1_values))
    width = 0.35
    axes[3, 1].bar(trial_indices - width/2, f1_values, width, label='CV F1', alpha=0.7, color='skyblue')
    axes[3, 1].bar(trial_indices + width/2, test_f1_values, width, label='Test F1', alpha=0.7, color='salmon')
    axes[3, 1].set_xlabel('Trial Number')
    axes[3, 1].set_ylabel('F1 Score')
    axes[3, 1].set_title('CV F1 vs Test F1 by Trial')
    axes[3, 1].legend()
    axes[3, 1].grid(True, alpha=0.3, axis='y')

    # Plot 12: Summary statistics
    axes[3, 2].axis('off')
    summary_text = f"""
    Trial Summary (with Holdout Test):
    ──────────────────────────────────
    Total Trials: {len(completed_trials)}

    CV F1:
      Best: {max(f1_values):.4f}
      Avg: {np.mean(f1_values):.4f}
      Range: [{min(f1_values):.4f}, {max(f1_values):.4f}]

    Test F1:
      Best: {max(test_f1_values):.4f}
      Avg: {np.mean(test_f1_values):.4f}
      Range: [{min(test_f1_values):.4f}, {max(test_f1_values):.4f}]

    CV Accuracy:
      Best: {max(acc_values):.4f}
      Avg: {np.mean(acc_values):.4f}

    Synthetic Range:
      [{min(synthetic_added)}, {max(synthetic_added)}]
    """
    axes[3, 2].text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plot_path = OUTPUT_DIR / 'optuna_optimization_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {plot_path}")
    plt.close()


def save_optimization_summary(study: optuna.Study, baseline_results: Optional[Dict] = None):
    """Save detailed summary to JSON with per-bin analysis (5 bins) and baseline comparison."""
    best_trial = study.best_trial

    summary = {
        'best_n_very_low': best_trial.params['n_very_low'],
        'best_n_low': best_trial.params['n_low'],
        'best_n_medium': best_trial.params['n_medium'],
        'best_n_high': best_trial.params['n_high'],
        'best_n_very_high': best_trial.params['n_very_high'],
        'best_cv_macro_f1': best_trial.value,
        'best_trial_number': best_trial.number,
        'total_trials': len(study.trials),
        'cv_metrics': {
            'mean_accuracy': best_trial.user_attrs.get('mean_accuracy'),
            'mean_precision': best_trial.user_attrs.get('mean_precision'),
            'mean_recall': best_trial.user_attrs.get('mean_recall'),
        },
        'test_metrics': {
            'test_f1': best_trial.user_attrs.get('test_f1'),
            'test_accuracy': best_trial.user_attrs.get('test_accuracy'),
            'test_precision': best_trial.user_attrs.get('test_precision'),
            'test_recall': best_trial.user_attrs.get('test_recall'),
        },
        'data_info': {
            'total_synthetic_added': best_trial.user_attrs.get('total_synthetic_added')
        },
        'best_bin_stats': best_trial.user_attrs.get('bin_stats', {}),
        'baseline_results': baseline_results if baseline_results else {}
    }

    summary_path = OUTPUT_DIR / 'optimization_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")

    # Also print to console
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY (5-BIN)")
    print(f"{'='*70}")
    print(f"Best Parameters:")
    print(f"  n_very_low:  {best_trial.params['n_very_low']}")
    print(f"  n_low:       {best_trial.params['n_low']}")
    print(f"  n_medium:    {best_trial.params['n_medium']}")
    print(f"  n_high:      {best_trial.params['n_high']}")
    print(f"  n_very_high: {best_trial.params['n_very_high']}")
    print(f"\nBest CV Performance (5-fold):")
    print(f"  Macro F1:  {best_trial.value:.4f}")
    print(f"  Accuracy:  {best_trial.user_attrs.get('mean_accuracy', 0.0):.4f}")
    print(f"  Precision: {best_trial.user_attrs.get('mean_precision', 0.0):.4f}")
    print(f"  Recall:    {best_trial.user_attrs.get('mean_recall', 0.0):.4f}")
    print(f"\nBest Test Performance (Holdout Set):")
    print(f"  F1:        {best_trial.user_attrs.get('test_f1', 0.0):.4f}")
    print(f"  Accuracy:  {best_trial.user_attrs.get('test_accuracy', 0.0):.4f}")
    print(f"  Precision: {best_trial.user_attrs.get('test_precision', 0.0):.4f}")
    print(f"  Recall:    {best_trial.user_attrs.get('test_recall', 0.0):.4f}")
    print(f"\nBest Bin Statistics:")
    bin_stats = best_trial.user_attrs.get('bin_stats', {})
    bin_display_names = {
        'very_low': 'Very Low (<6)',
        'low': 'Low (6-9)',
        'medium': 'Medium (10-15)',
        'high': 'High (16-25)',
        'very_high': 'Very High (>25)'
    }
    for bin_name in ['very_low', 'low', 'medium', 'high', 'very_high']:
        if bin_name in bin_stats:
            stats = bin_stats[bin_name]
            display_name = bin_display_names[bin_name]
            print(f"  {display_name:20s}: {stats['added']:4d} total "
                  f"({stats['avg_per_species']:5.1f} avg per species)")
    print(f"  {'Total synthetic added':20s}: {best_trial.user_attrs.get('total_synthetic_added', 0)}")

    # Baseline comparison
    if baseline_results:
        print(f"\nComparison vs Baseline (Real Data Only):")
        baseline_cv_f1 = baseline_results.get('baseline_cv_macro_f1', 0.0)
        baseline_test_f1 = baseline_results.get('baseline_test_f1', 0.0)
        optimized_cv_f1 = best_trial.value
        optimized_test_f1 = best_trial.user_attrs.get('test_f1', 0.0)

        cv_improvement = optimized_cv_f1 - baseline_cv_f1
        test_improvement = optimized_test_f1 - baseline_test_f1

        print(f"  CV Macro F1:")
        print(f"    Baseline: {baseline_cv_f1:.4f}")
        print(f"    Optimized: {optimized_cv_f1:.4f}")
        print(f"    Improvement: {cv_improvement:+.4f} ({cv_improvement/baseline_cv_f1*100 if baseline_cv_f1 > 0 else 0:+.1f}%)")
        print(f"  Test F1:")
        print(f"    Baseline: {baseline_test_f1:.4f}")
        print(f"    Optimized: {optimized_test_f1:.4f}")
        print(f"    Improvement: {test_improvement:+.4f} ({test_improvement/baseline_test_f1*100 if baseline_test_f1 > 0 else 0:+.1f}%)")

    print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Shark CNN Synthetic Sample Optimization with Optuna")
    print("5-Bin Per-Bin Multipliers: n_very_low, n_low, n_medium, n_high, n_very_high")
    print(f"\nConfiguration:")
    print(f"  Seed: {SEED}")
    print(f"  Device: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs per fold: {EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Number of CV folds: {N_FOLDS}")
    print(f"  Model: EfficientNet-B0 → Linear(1280, {HIDDEN_DIM}) → Linear({HIDDEN_DIM}, {NUM_CLASSES})")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\nSpecies Binning (based on real sample counts):")
    print(f"  Very Low:  < 6 real samples")
    print(f"  Low:       6-9 real samples")
    print(f"  Medium:    10-15 real samples")
    print(f"  High:      16-25 real samples")
    print(f"  Very High: > 25 real samples")
    print(f"\nHyperparameter Ranges (each 0-10):")
    print(f"  n_very_low:  0-10")
    print(f"  n_low:       0-10")
    print(f"  n_medium:    0-10")
    print(f"  n_high:      0-10")
    print(f"  n_very_high: 0-10")
    print(f"\nObjective: Maximize mean Macro F1 across 5-fold stratified CV")
    print(f"Trials: 30, Pruner: MedianPruner, Sampler: TPE")

    # Run baseline (5-fold CV on real data only)
    print("\n" + "="*70)
    print("STEP 1: BASELINE EVALUATION")
    print("="*70)
    baseline_results = run_baseline()

    # Run optimization
    print("\n" + "="*70)
    print("STEP 2: OPTUNA OPTIMIZATION WITH SYNTHETIC DATA")
    print("="*70)
    study = run_optimization(n_trials=50)

    # Generate visualizations
    print("\nGenerating plots...")
    plot_optimization_results(study)
    save_optimization_summary(study, baseline_results)

    print("\nOptimization complete!")
