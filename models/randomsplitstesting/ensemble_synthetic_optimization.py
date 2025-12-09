"""
Ensemble Synthetic Data Optimization - Standalone Script

Optimizes synthetic data allocation across CNN, TCN, and Statistics models.
Uses Optuna to find the optimal number of synthetic samples per bucket (very_low, low, medium, high, very_high).
Objective: Maximize average macro F1 score across all three models.

This script is standalone - copy it to any machine with the data directory structure:
- ../../data/shark_dataset.csv (real data)
- ../../data/syntheticDataIndividual/*.csv (synthetic data, one file per species)

Models trained:
1. CNN (EfficientNet-B0) - Image-based classification
2. TCN (Temporal Convolutional Network) - Time series
3. Statistics (ExtraTrees) - Feature engineering

Bucket structure (based on real sample counts):
- very_low: < 6 real samples
- low: 6-9 real samples
- medium: 10-15 real samples
- high: 16-25 real samples
- very_high: > 25 real samples
"""

import os
import sys
import json
import pickle
import hashlib
import math
import copy
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
REAL_DATA_PATH = DATA_DIR / "shark_dataset.csv"
SYNTHETIC_DIR = DATA_DIR / "syntheticDataIndividual"

# Output
OUTPUT_DIR = SCRIPT_DIR / "ensemble_synthetic_results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "image_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Training parameters
N_FOLDS = 5
TEST_SPLIT_RATIO = 0.2
MAX_SYN_PER_SPECIES = 50

# CNN parameters
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

# TCN parameters (optimized architecture from trial 160)
TCN_BATCH_SIZE = 8
TCN_EPOCHS = 200
TCN_PATIENCE = 30
TCN_LEARNING_RATE = 9.29558217542935e-05
TCN_WEIGHT_DECAY = 0.0044478395955166086
TCN_DROPOUT = 0.02208528755253426
TCN_KERNEL_SIZE = 13
TCN_NUM_CHANNELS = [32, 64, 96, 128, 160, 192, 224]  # 7 layers with linear growth
TCN_REVERSE_DILATION = True

# Statistics parameters
STATS_N_ESTIMATORS = 1700
STATS_MAX_DEPTH = None
STATS_MIN_SAMPLES_SPLIT = 9
STATS_MIN_SAMPLES_LEAF = 1
STATS_MAX_FEATURES = 0.7
STATS_CLASS_WEIGHT = 'balanced'

# Top features for statistics model
STATS_TOP_FEATURES = [
    'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
    'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
    'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
    'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
    'slope_std', 'y_left_max'
]

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("ENSEMBLE SYNTHETIC DATA OPTIMIZATION")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Random Seed: {SEED}")
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
        # Format species name for filename (replace spaces with underscores)
        filename = f"synthetic_{species.replace(' ', '_')}.csv"
        filepath = SYNTHETIC_DIR / filename

        if filepath.exists():
            df = pd.read_csv(filepath)
            synthetic_data[species] = df.reset_index(drop=True)
        else:
            print(f"  Warning: No synthetic data found for {species} at {filepath}")

    return synthetic_data


def bin_species_by_real_count(real_data: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Bin species into 5 groups based on real sample counts.

    Bins:
    - 'very_low': < 6 real samples
    - 'low': 6-9 real samples
    - 'medium': 10-15 real samples
    - 'high': 16-25 real samples
    - 'very_high': > 25 real samples
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

    For each species in a bin: add min(n_bin, max_synthetic_per_species) synthetic samples.
    """
    augmented = real_data.copy()

    bins = bin_species_by_real_count(real_data)
    n_values = {
        'very_low': n_very_low,
        'low': n_low,
        'medium': n_medium,
        'high': n_high,
        'very_high': n_very_high
    }

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
            num_synthetic_to_add = min(k, max_synthetic_per_species) if k > 0 else 0

            if num_synthetic_to_add > 0 and species in synthetic_data:
                synth_pool = synthetic_data[species]
                if len(synth_pool) > 0:
                    # Sample with replacement if needed
                    sampled = synth_pool.sample(n=int(num_synthetic_to_add), replace=True, random_state=SEED)
                    augmented = pd.concat([augmented, sampled], ignore_index=True)
                    bin_stats[bin_name]['added'] += int(num_synthetic_to_add)

        # Compute average per species in bin
        if bin_stats[bin_name]['species_count'] > 0:
            bin_stats[bin_name]['avg_per_species'] = (
                bin_stats[bin_name]['added'] / bin_stats[bin_name]['species_count']
            )

    return augmented.reset_index(drop=True), bin_stats


# ============================================================================
# CNN MODEL (EfficientNet-B0)
# ============================================================================

# Cached EfficientNet weights
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

        # Try to load from cache or generate image
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
    """EfficientNet-B0 with custom classifier head for shark species classification."""

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


def evaluate_cnn(model, dataloader, device):
    """Evaluate CNN model."""
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

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return macro_f1


# ============================================================================
# TCN MODEL (Temporal Convolutional Network)
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


def evaluate_tcn(model, loader, device):
    """Evaluate TCN model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return macro_f1


# ============================================================================
# STATISTICS MODEL (ExtraTrees with Feature Engineering)
# ============================================================================

def preprocess_curve(x, y):
    """Preprocess curve with smoothing and baseline removal."""
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(1.5 / dx)) | 1)
    if win >= len(y):
        win = max(7, (len(y)//2)*2 - 1)
    y_smooth = savgol_filter(y, window_length=win, polyorder=3, mode="interp")
    q = np.quantile(y_smooth, 0.3)
    mask = y_smooth <= q
    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_smooth[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_baseline = y_smooth - baseline
    else:
        y_baseline = y_smooth - np.min(y_smooth)
    scale = np.quantile(y_baseline, 0.99)
    y_norm = y_baseline / scale if scale > 0 else y_baseline
    y_norm = np.maximum(y_norm, 0.0)
    return y_norm


def extract_features(x, y):
    """Extract 36 features from fluorescence curve."""
    feat = {}
    feat["mean"] = float(np.mean(y))
    feat["std"] = float(np.std(y))
    feat["min"] = float(np.min(y))
    feat["max"] = float(np.max(y))
    feat["range"] = float(np.ptp(y))
    feat["skewness"] = float(pd.Series(y).skew())
    feat["kurtosis"] = float(pd.Series(y).kurtosis())
    dy = np.gradient(y, x)
    feat["max_slope"] = float(np.max(np.abs(dy)))
    feat["mean_abs_slope"] = float(np.mean(np.abs(dy)))
    feat["slope_std"] = float(np.std(dy))
    d2y = np.gradient(dy, x)
    feat["max_curvature"] = float(np.max(np.abs(d2y)))
    feat["mean_abs_curvature"] = float(np.mean(np.abs(d2y)))
    peaks, props = find_peaks(y, prominence=0.1)
    if len(peaks) > 0:
        proms = props.get("prominences", [0])
        feat["n_peaks"] = float(len(peaks))
        feat["max_prominence"] = float(np.max(proms))
        feat["mean_prominence"] = float(np.mean(proms))
        feat["peak_max_x"] = float(x[peaks[np.argmax(proms)]])
    else:
        feat["n_peaks"] = 0.0
        feat["max_prominence"] = 0.0
        feat["mean_prominence"] = 0.0
        feat["peak_max_x"] = float(x[np.argmax(y)])
    n = len(y)
    for region, start, end in [("left", 0, n//3), ("middle", n//3, 2*n//3), ("right", 2*n//3, n)]:
        feat[f"y_{region}_mean"] = float(np.mean(y[start:end]))
        feat[f"y_{region}_std"] = float(np.std(y[start:end]))
        feat[f"y_{region}_max"] = float(np.max(y[start:end]))
    q = np.percentile(y, [25, 50, 75])
    feat["q25"] = float(q[0])
    feat["q50"] = float(q[1])
    feat["q75"] = float(q[2])
    feat["iqr"] = float(q[2] - q[0])
    fft_vals = np.abs(fft(y - np.mean(y)))
    fft_power = fft_vals ** 2
    for i, idx in enumerate(np.argsort(fft_power)[-5:][::-1]):
        feat[f"fft_power_{i}"] = float(fft_power[idx])
    feat["fft_total_power"] = float(np.sum(fft_power))
    feat["fft_entropy"] = float(entropy(fft_power + 1e-10))
    return feat


def prepare_stats_data(df: pd.DataFrame):
    """Prepare feature matrix for statistics model."""
    temp_cols = sorted([c for c in df.columns if c != 'Species'], key=lambda c: float(c))
    x_axis = np.array([float(c) for c in temp_cols], dtype=float)

    # Preprocess curves
    x_proc = np.array([preprocess_curve(x_axis, df.iloc[i, 1:].values.astype(float)) for i in range(len(df))])

    # Extract features
    feat_list = []
    for i in range(len(df)):
        f = extract_features(x_axis, x_proc[i])
        f['Species'] = df.iloc[i]['Species']
        feat_list.append(f)

    feat_df = pd.DataFrame(feat_list).fillna(0.0)
    X = feat_df[STATS_TOP_FEATURES].to_numpy(float)
    y = df['Species'].astype(str).to_numpy()

    return X, y


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_cnn_cv(train_data: pd.DataFrame, num_classes: int) -> float:
    """Train CNN with 5-fold CV and return mean macro F1."""
    train_transform, val_transform = get_cnn_transforms()
    label_encoder = LabelEncoder()
    label_encoder.fit(train_data['Species'])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_data, train_data['Species'])):
        print(f"  Fold {fold_idx + 1}/{N_FOLDS}...", end='', flush=True)
        torch.manual_seed(SEED)

        fold_train = train_data.iloc[train_idx].reset_index(drop=True)
        fold_val = train_data.iloc[val_idx].reset_index(drop=True)

        train_dataset = FluorescenceDataset(fold_train, label_encoder, train_transform, CACHE_DIR)
        val_dataset = FluorescenceDataset(fold_val, label_encoder, val_transform, CACHE_DIR)

        train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=2)

        model = SharkCNN(num_classes=num_classes).to(DEVICE)
        criterion = FocalLoss(alpha=CNN_FOCAL_ALPHA, gamma=CNN_FOCAL_GAMMA)
        optimizer = optim.AdamW(model.parameters(), lr=CNN_LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(CNN_EPOCHS):
            _ = train_cnn_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_f1 = evaluate_cnn(model, val_loader, DEVICE)
            scheduler.step()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= CNN_PATIENCE:
                break

        fold_scores.append(best_val_f1)
        print(f" F1={best_val_f1:.4f} (stopped at epoch {epoch+1})")

    return np.mean(fold_scores)


def train_tcn_cv(train_data: pd.DataFrame, num_classes: int) -> float:
    """Train TCN with 5-fold CV and return mean macro F1."""
    # Prepare TCN data
    temp_cols = sorted([c for c in train_data.columns if c != 'Species'], key=lambda c: float(c))
    X = train_data[temp_cols].values
    y_labels = train_data['Species'].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_idx + 1}/{N_FOLDS}...", end='', flush=True)
        torch.manual_seed(SEED)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Normalize - fit scaler on training data only to avoid data leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Reshape for TCN
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        train_dataset = TCNDataset(X_train, y_train)
        val_dataset = TCNDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=TCN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=TCN_BATCH_SIZE, shuffle=False)

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

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(TCN_EPOCHS):
            _ = train_tcn_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_f1 = evaluate_tcn(model, val_loader, DEVICE)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= TCN_PATIENCE:
                break

        fold_scores.append(best_val_f1)
        print(f" F1={best_val_f1:.4f} (stopped at epoch {epoch+1})")

    return np.mean(fold_scores)


def train_stats_cv(train_data: pd.DataFrame) -> float:
    """Train Statistics model with 5-fold CV and return mean macro F1."""
    X, y = prepare_stats_data(train_data)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_idx + 1}/{N_FOLDS}...", end='', flush=True)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = ExtraTreesClassifier(
            n_estimators=STATS_N_ESTIMATORS,
            max_depth=STATS_MAX_DEPTH,
            min_samples_split=STATS_MIN_SAMPLES_SPLIT,
            min_samples_leaf=STATS_MIN_SAMPLES_LEAF,
            max_features=STATS_MAX_FEATURES,
            class_weight=STATS_CLASS_WEIGHT,
            random_state=SEED,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        fold_scores.append(val_f1)
        print(f" F1={val_f1:.4f}")

    return np.mean(fold_scores)


# ============================================================================
# BASELINE EVALUATION (NO SYNTHETIC DATA)
# ============================================================================

def run_baseline(real_train_val: pd.DataFrame, num_classes: int) -> Dict:
    """
    Run baseline evaluation: train all three models on real data only (no synthetic).

    Returns:
        Dictionary with baseline F1 scores for each model
    """
    print("\n" + "="*80)
    print("BASELINE EVALUATION (Real Data Only - No Synthetic)")
    print("="*80)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Train CNN on real data only
    print("\nTraining Baseline CNN...")
    baseline_cnn_f1 = train_cnn_cv(real_train_val, num_classes)
    print(f"Baseline CNN Macro F1: {baseline_cnn_f1:.4f}")

    # Train TCN on real data only
    print("\nTraining Baseline TCN...")
    baseline_tcn_f1 = train_tcn_cv(real_train_val, num_classes)
    print(f"Baseline TCN Macro F1: {baseline_tcn_f1:.4f}")

    # Train Statistics on real data only
    print("\nTraining Baseline Statistics...")
    baseline_stats_f1 = train_stats_cv(real_train_val)
    print(f"Baseline Statistics Macro F1: {baseline_stats_f1:.4f}")

    # Average
    baseline_avg_f1 = (baseline_cnn_f1 + baseline_tcn_f1 + baseline_stats_f1) / 3.0

    print("\n" + "="*80)
    print("BASELINE RESULTS (No Synthetic Data)")
    print("="*80)
    print(f"CNN F1:        {baseline_cnn_f1:.4f}")
    print(f"TCN F1:        {baseline_tcn_f1:.4f}")
    print(f"Statistics F1: {baseline_stats_f1:.4f}")
    print(f"Average F1:    {baseline_avg_f1:.4f}")
    print("="*80)

    return {
        'baseline_cnn_f1': baseline_cnn_f1,
        'baseline_tcn_f1': baseline_tcn_f1,
        'baseline_stats_f1': baseline_stats_f1,
        'baseline_avg_f1': baseline_avg_f1
    }


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function: optimize per-bin synthetic sample counts.

    Trains all three models (CNN, TCN, Statistics) and returns average macro F1.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Suggest hyperparameters (number of synthetic samples per bucket)
    n_very_low = trial.suggest_int("n_very_low", 0, MAX_SYN_PER_SPECIES)
    n_low = trial.suggest_int("n_low", 0, MAX_SYN_PER_SPECIES)
    n_medium = trial.suggest_int("n_medium", 0, MAX_SYN_PER_SPECIES)
    n_high = trial.suggest_int("n_high", 0, MAX_SYN_PER_SPECIES)
    n_very_high = trial.suggest_int("n_very_high", 0, MAX_SYN_PER_SPECIES)

    print(f"\n{'='*80}")
    print(f"Trial {trial.number}: n_very_low={n_very_low}, n_low={n_low}, "
          f"n_medium={n_medium}, n_high={n_high}, n_very_high={n_very_high}")
    print(f"{'='*80}")

    try:
        # Load data
        print("Loading data...")
        real_data = load_real_data()
        print(f"  Real samples: {len(real_data)}")

        # Split into train+val vs test
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT_RATIO, random_state=SEED)
        for train_idx, test_idx in sss.split(real_data, real_data['Species']):
            real_train_val = real_data.iloc[train_idx].reset_index(drop=True)
            real_test = real_data.iloc[test_idx].reset_index(drop=True)

        print(f"  Train+Val: {len(real_train_val)} samples")
        print(f"  Test: {len(real_test)} samples (not used during optimization)")

        # Get species bins
        bins = bin_species_by_real_count(real_train_val)
        num_classes = len(real_train_val['Species'].unique())
        print(f"  Number of classes: {num_classes}")

        # Load synthetic data
        synthetic_data = load_synthetic_data(real_train_val['Species'].unique().tolist())
        print(f"  Synthetic species available: {len(synthetic_data)}")

        # Create augmented dataset
        augmented_data, bin_stats = create_augmented_dataset(
            real_train_val, synthetic_data,
            n_very_low, n_low, n_medium, n_high, n_very_high,
            MAX_SYN_PER_SPECIES
        )
        num_added = len(augmented_data) - len(real_train_val)
        print(f"  Augmented samples: {len(augmented_data)} (added {num_added})")

        # Train CNN
        print("\n" + "-"*80)
        print("Training CNN (EfficientNet-B0)...")
        print("-"*80)
        cnn_f1 = train_cnn_cv(augmented_data, num_classes)
        print(f"CNN Mean Macro F1: {cnn_f1:.4f}")

        # Train TCN
        print("\n" + "-"*80)
        print("Training TCN (Temporal Convolutional Network)...")
        print("-"*80)
        tcn_f1 = train_tcn_cv(augmented_data, num_classes)
        print(f"TCN Mean Macro F1: {tcn_f1:.4f}")

        # Train Statistics
        print("\n" + "-"*80)
        print("Training Statistics (ExtraTrees)...")
        print("-"*80)
        stats_f1 = train_stats_cv(augmented_data)
        print(f"Statistics Mean Macro F1: {stats_f1:.4f}")

        # Average F1 across all three models
        avg_f1 = (cnn_f1 + tcn_f1 + stats_f1) / 3.0

        print("\n" + "="*80)
        print(f"TRIAL {trial.number} RESULTS")
        print("="*80)
        print(f"CNN F1:        {cnn_f1:.4f}")
        print(f"TCN F1:        {tcn_f1:.4f}")
        print(f"Statistics F1: {stats_f1:.4f}")
        print(f"Average F1:    {avg_f1:.4f}")
        print("="*80)

        # Store individual model scores as user attributes
        trial.set_user_attr('cnn_f1', cnn_f1)
        trial.set_user_attr('tcn_f1', tcn_f1)
        trial.set_user_attr('stats_f1', stats_f1)
        trial.set_user_attr('total_synthetic_added', num_added)
        trial.set_user_attr('bin_stats', bin_stats)

        return avg_f1

    except Exception as e:
        print(f"Error in trial: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


# ============================================================================
# MAIN OPTIMIZATION
# ============================================================================

def trial_callback(study, trial):
    """Callback to print information about pruned trials."""
    if trial.state == optuna.trial.TrialState.PRUNED:
        print(f"\n{'='*80}")
        print(f"Trial {trial.number} PRUNED")
        print(f"{'='*80}")
        print(f"Reason: Trial stopped early by MedianPruner")
        print(f"Parameters:")
        print(f"  n_very_low:  {trial.params.get('n_very_low', 'N/A')}")
        print(f"  n_low:       {trial.params.get('n_low', 'N/A')}")
        print(f"  n_medium:    {trial.params.get('n_medium', 'N/A')}")
        print(f"  n_high:      {trial.params.get('n_high', 'N/A')}")
        print(f"  n_very_high: {trial.params.get('n_very_high', 'N/A')}")
        print(f"{'='*80}\n")


def run_optimization(n_trials: int = 30, baseline_results: Dict = None):
    """Run Optuna optimization."""
    print(f"\nStarting Optuna optimization with {n_trials} trials...")
    print(f"Objective: Maximize average macro F1 across CNN, TCN, and Statistics")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner()

    db_path = OUTPUT_DIR / "ensemble_optuna.db"
    db_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=db_url,
        study_name='ensemble_synthetic_optimization',
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

    # Best trial
    best_trial = study.best_trial
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print("Best synthetic sample allocation:")
    print(f"  n_very_low:  {best_trial.params['n_very_low']}")
    print(f"  n_low:       {best_trial.params['n_low']}")
    print(f"  n_medium:    {best_trial.params['n_medium']}")
    print(f"  n_high:      {best_trial.params['n_high']}")
    print(f"  n_very_high: {best_trial.params['n_very_high']}")
    print(f"\nBest Average Macro F1: {best_trial.value:.4f}")
    print(f"  CNN F1:        {best_trial.user_attrs['cnn_f1']:.4f}")
    print(f"  TCN F1:        {best_trial.user_attrs['tcn_f1']:.4f}")
    print(f"  Statistics F1: {best_trial.user_attrs['stats_f1']:.4f}")
    print(f"\nBest Trial Number: {best_trial.number}")
    print(f"Total Synthetic Added: {best_trial.user_attrs['total_synthetic_added']}")

    # Show improvement over baseline if available
    if baseline_results:
        print(f"\n{'='*80}")
        print("IMPROVEMENT OVER BASELINE (Real Data Only)")
        print(f"{'='*80}")
        baseline_avg = baseline_results['baseline_avg_f1']
        improvement = best_trial.value - baseline_avg
        improvement_pct = (improvement / baseline_avg) * 100

        print(f"Baseline Avg F1:   {baseline_avg:.4f}")
        print(f"Best Avg F1:       {best_trial.value:.4f}")
        print(f"Improvement:       {improvement:+.4f} ({improvement_pct:+.2f}%)")
        print(f"\nPer-Model Improvement:")
        print(f"  CNN:        {baseline_results['baseline_cnn_f1']:.4f} → {best_trial.user_attrs['cnn_f1']:.4f} ({(best_trial.user_attrs['cnn_f1'] - baseline_results['baseline_cnn_f1']):.4f})")
        print(f"  TCN:        {baseline_results['baseline_tcn_f1']:.4f} → {best_trial.user_attrs['tcn_f1']:.4f} ({(best_trial.user_attrs['tcn_f1'] - baseline_results['baseline_tcn_f1']):.4f})")
        print(f"  Statistics: {baseline_results['baseline_stats_f1']:.4f} → {best_trial.user_attrs['stats_f1']:.4f} ({(best_trial.user_attrs['stats_f1'] - baseline_results['baseline_stats_f1']):.4f})")
        print(f"{'='*80}")

    print(f"{'='*80}")

    # Save results
    results = {
        'baseline': baseline_results if baseline_results else {},
        'best_params': best_trial.params,
        'best_avg_f1': best_trial.value,
        'best_cnn_f1': best_trial.user_attrs['cnn_f1'],
        'best_tcn_f1': best_trial.user_attrs['tcn_f1'],
        'best_stats_f1': best_trial.user_attrs['stats_f1'],
        'total_synthetic_added': best_trial.user_attrs['total_synthetic_added'],
        'bin_stats': best_trial.user_attrs['bin_stats'],
        'total_trials': len(study.trials)
    }

    if baseline_results:
        results['improvement'] = {
            'avg_f1_improvement': best_trial.value - baseline_results['baseline_avg_f1'],
            'avg_f1_improvement_pct': ((best_trial.value - baseline_results['baseline_avg_f1']) / baseline_results['baseline_avg_f1']) * 100,
            'cnn_improvement': best_trial.user_attrs['cnn_f1'] - baseline_results['baseline_cnn_f1'],
            'tcn_improvement': best_trial.user_attrs['tcn_f1'] - baseline_results['baseline_tcn_f1'],
            'stats_improvement': best_trial.user_attrs['stats_f1'] - baseline_results['baseline_stats_f1']
        }

    results_path = OUTPUT_DIR / 'optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return study


# ============================================================================
# FINAL MODEL TRAINING & TEST EVALUATION
# ============================================================================

def train_final_cnn(train_data: pd.DataFrame, test_data: pd.DataFrame, num_classes: int) -> Tuple[float, float]:
    """Train final CNN on full training data and evaluate on test set."""
    train_transform, val_transform = get_cnn_transforms()
    label_encoder = LabelEncoder()
    label_encoder.fit(train_data['Species'])

    torch.manual_seed(SEED)

    train_dataset = FluorescenceDataset(train_data, label_encoder, train_transform, CACHE_DIR)
    test_dataset = FluorescenceDataset(test_data, label_encoder, val_transform, CACHE_DIR)

    train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=2)

    model = SharkCNN(num_classes=num_classes).to(DEVICE)
    criterion = FocalLoss(alpha=CNN_FOCAL_ALPHA, gamma=CNN_FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=CNN_LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    best_train_f1 = 0.0
    patience_counter = 0

    for epoch in range(CNN_EPOCHS):
        _ = train_cnn_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_f1 = evaluate_cnn(model, train_loader, DEVICE)
        scheduler.step()

        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CNN_PATIENCE:
            break

    # Evaluate on test set
    test_f1 = evaluate_cnn(model, test_loader, DEVICE)
    return best_train_f1, test_f1


def train_final_tcn(train_data: pd.DataFrame, test_data: pd.DataFrame, num_classes: int) -> Tuple[float, float]:
    """Train final TCN on full training data and evaluate on test set."""
    temp_cols = sorted([c for c in train_data.columns if c != 'Species'], key=lambda c: float(c))

    X_train = train_data[temp_cols].values
    y_train_labels = train_data['Species'].values

    X_test = test_data[temp_cols].values
    y_test_labels = test_data['Species'].values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_test = label_encoder.transform(y_test_labels)

    torch.manual_seed(SEED)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for TCN
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

    best_train_f1 = 0.0
    patience_counter = 0

    for epoch in range(TCN_EPOCHS):
        _ = train_tcn_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_f1 = evaluate_tcn(model, train_loader, DEVICE)

        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= TCN_PATIENCE:
            break

    # Evaluate on test set
    test_f1 = evaluate_tcn(model, test_loader, DEVICE)
    return best_train_f1, test_f1


def train_final_stats(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[float, float]:
    """Train final Statistics model on full training data and evaluate on test set."""
    X_train, y_train = prepare_stats_data(train_data)
    X_test, y_test = prepare_stats_data(test_data)

    model = ExtraTreesClassifier(
        n_estimators=STATS_N_ESTIMATORS,
        max_depth=STATS_MAX_DEPTH,
        min_samples_split=STATS_MIN_SAMPLES_SPLIT,
        min_samples_leaf=STATS_MIN_SAMPLES_LEAF,
        max_features=STATS_MAX_FEATURES,
        class_weight=STATS_CLASS_WEIGHT,
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_f1 = f1_score(y_train, y_pred_train, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

    return train_f1, test_f1


def evaluate_on_test_set(best_params: Dict, real_train_val: pd.DataFrame, real_test: pd.DataFrame,
                         synthetic_data: Dict, num_classes: int) -> Dict:
    """Train final models with best params and evaluate on test set."""
    print("\n" + "="*80)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*80)
    print(f"Training final models with best hyperparameters:")
    print(f"  n_very_low:  {best_params['n_very_low']}")
    print(f"  n_low:       {best_params['n_low']}")
    print(f"  n_medium:    {best_params['n_medium']}")
    print(f"  n_high:      {best_params['n_high']}")
    print(f"  n_very_high: {best_params['n_very_high']}")

    # Create augmented training set with best params
    augmented_data, bin_stats = create_augmented_dataset(
        real_train_val, synthetic_data,
        best_params['n_very_low'], best_params['n_low'], best_params['n_medium'],
        best_params['n_high'], best_params['n_very_high'],
        MAX_SYN_PER_SPECIES
    )

    print(f"\nTraining data: {len(augmented_data)} samples ({len(augmented_data) - len(real_train_val)} synthetic added)")
    print(f"Test data: {len(real_test)} samples")

    # Train and evaluate CNN
    print("\n" + "-"*80)
    print("Training final CNN on full training data...")
    print("-"*80)
    cnn_train_f1, cnn_test_f1 = train_final_cnn(augmented_data, real_test, num_classes)
    print(f"CNN Train F1: {cnn_train_f1:.4f}, Test F1: {cnn_test_f1:.4f}")

    # Train and evaluate TCN
    print("\n" + "-"*80)
    print("Training final TCN on full training data...")
    print("-"*80)
    tcn_train_f1, tcn_test_f1 = train_final_tcn(augmented_data, real_test, num_classes)
    print(f"TCN Train F1: {tcn_train_f1:.4f}, Test F1: {tcn_test_f1:.4f}")

    # Train and evaluate Statistics
    print("\n" + "-"*80)
    print("Training final Statistics model on full training data...")
    print("-"*80)
    stats_train_f1, stats_test_f1 = train_final_stats(augmented_data, real_test)
    print(f"Statistics Train F1: {stats_train_f1:.4f}, Test F1: {stats_test_f1:.4f}")

    # Average test F1
    avg_test_f1 = (cnn_test_f1 + tcn_test_f1 + stats_test_f1) / 3.0

    print("\n" + "="*80)
    print("FINAL TEST SET RESULTS")
    print("="*80)
    print(f"CNN Test F1:        {cnn_test_f1:.4f}")
    print(f"TCN Test F1:        {tcn_test_f1:.4f}")
    print(f"Statistics Test F1: {stats_test_f1:.4f}")
    print(f"Average Test F1:    {avg_test_f1:.4f}")
    print("="*80)

    return {
        'cnn_train_f1': cnn_train_f1,
        'cnn_test_f1': cnn_test_f1,
        'tcn_train_f1': tcn_train_f1,
        'tcn_test_f1': tcn_test_f1,
        'stats_train_f1': stats_train_f1,
        'stats_test_f1': stats_test_f1,
        'avg_test_f1': avg_test_f1,
        'test_samples': len(real_test),
        'train_samples': len(augmented_data)
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nEnsemble Synthetic Data Optimization")
    print("Optimizes synthetic data allocation across CNN, TCN, and Statistics models")
    print(f"\nConfiguration:")
    print(f"  Models: CNN (EfficientNet-B0), TCN, Statistics (ExtraTrees)")
    print(f"  Buckets: very_low (<6), low (6-9), medium (10-15), high (16-25), very_high (>25)")
    print(f"  Max synthetic per species: {MAX_SYN_PER_SPECIES}")
    print(f"  CV Folds: {N_FOLDS}")
    print(f"  Random Seed: {SEED}")
    print(f"  Device: {DEVICE}")

    # Load real data and split
    print("\nLoading data for baseline evaluation...")
    real_data = load_real_data()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT_RATIO, random_state=SEED)
    for train_idx, test_idx in sss.split(real_data, real_data['Species']):
        real_train_val = real_data.iloc[train_idx].reset_index(drop=True)
        real_test = real_data.iloc[test_idx].reset_index(drop=True)

    num_classes = len(real_train_val['Species'].unique())

    # Run baseline evaluation (real data only)
    baseline_results = run_baseline(real_train_val, num_classes)

    # Run optimization
    study = run_optimization(n_trials=20, baseline_results=baseline_results)

    # Evaluate on test set with best hyperparameters
    synthetic_data = load_synthetic_data(real_train_val['Species'].unique().tolist())
    test_results = evaluate_on_test_set(
        study.best_params,
        real_train_val,
        real_test,
        synthetic_data,
        num_classes
    )

    # Update results file with test scores
    results_path = OUTPUT_DIR / 'optimization_results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    results['test_evaluation'] = test_results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nOptimization complete!")
    print(f"Results saved to {OUTPUT_DIR}")
