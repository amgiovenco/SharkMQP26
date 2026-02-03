"""
Ensemble Synthetic Data Training with 5-Fold Cross-Validation

Simple, clean script that:
1. Uses all 50 synthetic samples per species for training
2. Splits real data 50/50 for validation (CV) and test (hold-out)
3. Trains CNN, TCN, and Statistics models independently with 5-fold CV
4. Pregenerates all CNN images upfront
5. Tracks: accuracy, recall, precision, macro F1, weighted F1, etc.
6. Seed: 8
"""

import sys
import json
import pickle
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO

print("Loading numpy...", flush=True)
import numpy as np
print("Loading pandas...", flush=True)
import pandas as pd

print("Loading matplotlib...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Loading PIL...", flush=True)
from PIL import Image

print("Loading scipy...", flush=True)
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy
from scipy.fft import fft

print("Loading torch...", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("Loading torchvision...", flush=True)
from torchvision import transforms
from torchvision.models import efficientnet_b0

print("Loading sklearn...", flush=True)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

print("All imports loaded successfully!", flush=True)
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
REAL_DATA_PATH = DATA_DIR / "shark_dataset.csv"
SYNTHETIC_DIR = DATA_DIR / "syntheticDataIndividual"

# Output
OUTPUT_DIR = SCRIPT_DIR / "ensemble_synthetic_cv_results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "image_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Training
N_FOLDS = 5
REAL_VALIDATION_SPLIT = 0.5  # 50% validation, 50% test
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

# Statistics parameters
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

print("Setting random seeds...", flush=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80, flush=True)
print("ENSEMBLE SYNTHETIC DATA TRAINING WITH 5-FOLD CV", flush=True)
print("=" * 80, flush=True)
print(f"Device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Random Seed: {SEED}", flush=True)
print(f"Output Directory: {OUTPUT_DIR}", flush=True)
print(f"Synthetic per species: {MAX_SYN_PER_SPECIES}", flush=True)
print(f"CV Folds: {N_FOLDS}", flush=True)
print("=" * 80, flush=True)
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
        print(f"Error: Synthetic data directory not found at {SYNTHETIC_DIR}")
        return synthetic_data

    for species in species_list:
        filename = f"synthetic_{species.replace(' ', '_')}.csv"
        filepath = SYNTHETIC_DIR / filename

        if filepath.exists():
            df = pd.read_csv(filepath)
            synthetic_data[species] = df.reset_index(drop=True)
            print(f"  Loaded {species}: {len(df)} samples")

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
# CNN MODEL
# ============================================================================

_CACHED_EFFICIENTNET_WEIGHTS = None

def get_cached_efficientnet_weights():
    """Load EfficientNet-B0 weights once and cache them."""
    global _CACHED_EFFICIENTNET_WEIGHTS
    if _CACHED_EFFICIENTNET_WEIGHTS is None:
        base_model = efficientnet_b0(weights='IMAGENET1K_V1')
        _CACHED_EFFICIENTNET_WEIGHTS = base_model.state_dict().copy()
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


def evaluate_cnn_full(model, dataloader, device, label_encoder):
    """Evaluate CNN model and return all metrics."""
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

    return compute_metrics(all_labels, all_preds, label_encoder)


# ============================================================================
# TCN MODEL
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
    """Temporal block with residual connections."""
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
    """Temporal Convolutional Network."""
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


def evaluate_tcn_full(model, loader, device, label_encoder):
    """Evaluate TCN model and return all metrics."""
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

    return compute_metrics(all_labels, all_preds, label_encoder)


# ============================================================================
# STATISTICS MODEL
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

    x_proc = np.array([preprocess_curve(x_axis, df.iloc[i, 1:].values.astype(float)) for i in range(len(df))])

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
# METRICS
# ============================================================================

def compute_metrics(y_true, y_pred, label_encoder):
    """Compute all metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


# ============================================================================
# PREGENERATE CNN IMAGES
# ============================================================================

def pregenerate_cnn_images(data: pd.DataFrame):
    """Pregenerate all images for CNN to cache."""
    print(f"Pregenerating {len(data)} CNN images...")
    label_encoder = LabelEncoder()
    label_encoder.fit(data['Species'])

    dataset = FluorescenceDataset(data, label_encoder, cache_dir=CACHE_DIR)

    for i in range(len(dataset)):
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(dataset)} images")
        _ = dataset[i]

    print(f"All {len(data)} images cached!")


# ============================================================================
# TRAINING WITH 5-FOLD CV
# ============================================================================

def train_and_evaluate_cnn(synthetic_train: pd.DataFrame, real_val: pd.DataFrame, real_test: pd.DataFrame, num_classes: int) -> Dict:
    """Train CNN on synthetic, validate on real_val, test on real_test."""
    print("\n" + "="*80)
    print("CNN (EfficientNet-B0)")
    print("="*80)

    train_transform, val_transform = get_cnn_transforms()
    label_encoder = LabelEncoder()
    label_encoder.fit(synthetic_train['Species'])

    # Pregenerate all images
    print("Pregenerating synthetic images...", flush=True)
    pregenerate_cnn_images(synthetic_train)
    print("Pregenerating validation images...", flush=True)
    pregenerate_cnn_images(real_val)
    print("Pregenerating test images...", flush=True)
    pregenerate_cnn_images(real_test)

    torch.manual_seed(SEED)

    train_dataset = FluorescenceDataset(synthetic_train, label_encoder, train_transform, CACHE_DIR)
    val_dataset = FluorescenceDataset(real_val, label_encoder, val_transform, CACHE_DIR)
    test_dataset = FluorescenceDataset(real_test, label_encoder, val_transform, CACHE_DIR)

    train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=2)

    model = SharkCNN(num_classes=num_classes).to(DEVICE)
    criterion = FocalLoss(alpha=CNN_FOCAL_ALPHA, gamma=CNN_FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=CNN_LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    print("\nTraining...", flush=True)
    for epoch in range(CNN_EPOCHS):
        _ = train_cnn_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate_cnn_full(model, val_loader, DEVICE, label_encoder)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: Val F1={val_metrics['f1_macro']:.4f}", flush=True)

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CNN_PATIENCE:
            print(f"  Stopped at epoch {epoch + 1} (patience)", flush=True)
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    # Evaluate on validation and test
    val_metrics = evaluate_cnn_full(model, val_loader, DEVICE, label_encoder)
    test_metrics = evaluate_cnn_full(model, test_loader, DEVICE, label_encoder)

    print("\n" + "="*80)
    print("CNN RESULTS")
    print("="*80)
    print(f"Validation Set:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision (Macro): {val_metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {val_metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro): {val_metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"\nTest Set:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {test_metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    print("="*80)

    return {
        'validation': val_metrics,
        'test': test_metrics
    }


def train_and_evaluate_tcn(synthetic_train: pd.DataFrame, real_val: pd.DataFrame, real_test: pd.DataFrame, num_classes: int) -> Dict:
    """Train TCN on synthetic, validate on real_val, test on real_test."""
    print("\n" + "="*80)
    print("TCN (Temporal Convolutional Network)")
    print("="*80)

    # Prepare data
    temp_cols = sorted([c for c in synthetic_train.columns if c != 'Species'], key=lambda c: float(c))
    X_syn = synthetic_train[temp_cols].values
    y_syn_labels = synthetic_train['Species'].values

    X_val = real_val[temp_cols].values
    y_val_labels = real_val['Species'].values

    X_test = real_test[temp_cols].values
    y_test_labels = real_test['Species'].values

    label_encoder = LabelEncoder()
    y_syn = label_encoder.fit_transform(y_syn_labels)
    y_val = label_encoder.transform(y_val_labels)
    y_test = label_encoder.transform(y_test_labels)

    torch.manual_seed(SEED)

    # Normalize
    scaler = StandardScaler()
    X_syn_scaled = scaler.fit_transform(X_syn)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Reshape
    X_syn_reshaped = X_syn_scaled.reshape(X_syn_scaled.shape[0], 1, X_syn_scaled.shape[1])
    X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    train_dataset = TCNDataset(X_syn_reshaped, y_syn)
    val_dataset = TCNDataset(X_val_reshaped, y_val)
    test_dataset = TCNDataset(X_test_reshaped, y_test)

    train_loader = DataLoader(train_dataset, batch_size=TCN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TCN_BATCH_SIZE, shuffle=False)
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

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    print("Training...", flush=True)
    for epoch in range(TCN_EPOCHS):
        _ = train_tcn_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate_tcn_full(model, val_loader, DEVICE, label_encoder)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: Val F1={val_metrics['f1_macro']:.4f}", flush=True)

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= TCN_PATIENCE:
            print(f"  Stopped at epoch {epoch + 1} (patience)", flush=True)
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    # Evaluate on validation and test
    val_metrics = evaluate_tcn_full(model, val_loader, DEVICE, label_encoder)
    test_metrics = evaluate_tcn_full(model, test_loader, DEVICE, label_encoder)

    print("\n" + "="*80)
    print("TCN RESULTS")
    print("="*80)
    print(f"Validation Set:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision (Macro): {val_metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {val_metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro): {val_metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"\nTest Set:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {test_metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    print("="*80)

    return {
        'validation': val_metrics,
        'test': test_metrics
    }


def train_and_evaluate_stats(synthetic_train: pd.DataFrame, real_val: pd.DataFrame, real_test: pd.DataFrame) -> Dict:
    """Train Statistics model on synthetic, validate on real_val, test on real_test."""
    print("\n" + "="*80)
    print("STATISTICS (ExtraTrees)")
    print("="*80)

    # Prepare data
    X_syn, y_syn = prepare_stats_data(synthetic_train)
    X_val, y_val = prepare_stats_data(real_val)
    X_test, y_test = prepare_stats_data(real_test)

    print("Training...", flush=True)
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

    model.fit(X_syn, y_syn)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Compute validation metrics
    val_acc = accuracy_score(y_val, y_pred_val)
    val_precision = precision_score(y_val, y_pred_val, average='macro', zero_division=0)
    val_recall = recall_score(y_val, y_pred_val, average='macro', zero_division=0)
    val_f1_macro = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
    val_f1_weighted = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)

    val_metrics = {
        'accuracy': val_acc,
        'precision_macro': val_precision,
        'recall_macro': val_recall,
        'f1_macro': val_f1_macro,
        'f1_weighted': val_f1_weighted
    }

    # Compute test metrics
    test_acc = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_f1_macro = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

    test_metrics = {
        'accuracy': test_acc,
        'precision_macro': test_precision,
        'recall_macro': test_recall,
        'f1_macro': test_f1_macro,
        'f1_weighted': test_f1_weighted
    }

    print("\n" + "="*80)
    print("STATISTICS RESULTS")
    print("="*80)
    print(f"Validation Set:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Precision (Macro): {val_precision:.4f}")
    print(f"  Recall (Macro): {val_recall:.4f}")
    print(f"  F1-Score (Macro): {val_f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {val_f1_weighted:.4f}")
    print(f"\nTest Set:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision (Macro): {test_precision:.4f}")
    print(f"  Recall (Macro): {test_recall:.4f}")
    print(f"  F1-Score (Macro): {test_f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {test_f1_weighted:.4f}")
    print("="*80)

    return {
        'validation': val_metrics,
        'test': test_metrics
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nLoading data...", flush=True)
    real_data = load_real_data()
    print(f"Real data: {len(real_data)} samples, {len(real_data['Species'].unique())} species", flush=True)

    # Split real data: 50% validation, 50% test
    print("Splitting real data 50/50...", flush=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=REAL_VALIDATION_SPLIT, random_state=SEED)
    for val_idx, test_idx in sss.split(real_data, real_data['Species']):
        real_val = real_data.iloc[val_idx].reset_index(drop=True)
        real_test = real_data.iloc[test_idx].reset_index(drop=True)

    print(f"  Validation set (for CV): {len(real_val)} samples", flush=True)
    print(f"  Test set (hold-out): {len(real_test)} samples", flush=True)

    # Load synthetic data
    print(f"\nLoading synthetic data...", flush=True)
    synthetic_data = load_synthetic_data(real_data['Species'].unique().tolist())

    if not synthetic_data:
        print("\nERROR: No synthetic data found!", flush=True)
        print("Please ensure synthetic data files exist in the syntheticDataGeneration/syntheticDataIndividual directory", flush=True)
        sys.exit(1)

    # Create synthetic training set
    print("Creating synthetic training set...", flush=True)
    synthetic_train = create_synthetic_training_set(synthetic_data)
    if len(synthetic_train) == 0:
        print("\nERROR: No synthetic training data available!", flush=True)
        sys.exit(1)
    print(f"Synthetic training set: {len(synthetic_train)} samples", flush=True)

    num_classes = len(real_data['Species'].unique())
    print(f"Number of classes: {num_classes}", flush=True)

    # Train and evaluate models (all on same synthetic data, validation set, and test set)
    print("\n" + "="*80)
    print("TRAINING ALL 3 MODELS")
    print("="*80)

    cnn_results = train_and_evaluate_cnn(synthetic_train, real_val, real_test, num_classes)
    tcn_results = train_and_evaluate_tcn(synthetic_train, real_val, real_test, num_classes)
    stats_results = train_and_evaluate_stats(synthetic_train, real_val, real_test)

    # Save results
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    summary = {
        'configuration': {
            'seed': SEED,
            'real_validation_split': REAL_VALIDATION_SPLIT,
            'max_synthetic_per_species': MAX_SYN_PER_SPECIES,
            'num_classes': num_classes,
            'device': str(DEVICE),
        },
        'data': {
            'synthetic_samples': len(synthetic_train),
            'validation_samples': len(real_val),
            'test_samples': len(real_test),
        },
        'results': {
            'cnn': cnn_results,
            'tcn': tcn_results,
            'statistics': stats_results,
        }
    }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("="*80)
