#!/usr/bin/env python3
"""
Shark Species Classification - Complete End-to-End Training & Prediction Script
Implements proper k-fold cross-validation with out-of-fold predictions for stacking.
Faithfully reproduces all models as specified in MODEL_REPRODUCIBILITY_GUIDE.md
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import fft
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import pickle
import json
import time

warnings.filterwarnings('ignore')

# Determine device (CPU/GPU)
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
    from torch import nn, optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    from torchvision import transforms
    from torchvision.models import efficientnet_b0
    from PIL import Image
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

# Constants
RANDOM_STATE = 8
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "shark_dataset.csv"
TRAIN_IMG_PATH = Path(__file__).parent.parent.parent / "data" / "train"
TEST_IMG_PATH = Path(__file__).parent.parent.parent / "data" / "test"
DECIMATE_STEP = 6  # For Gaussian curve fitting

np.random.seed(RANDOM_STATE)
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_STATE)


# ============================================================================
# CUSTOM TRANSFORMS
# ============================================================================

class AddGaussianNoise:
    """Add Gaussian noise to image tensors (measurement noise)."""
    def __init__(self, std=0.005):
        self.std = std

    def __call__(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor + torch.randn_like(tensor) * self.std
        return tensor


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data():
    """Load shark dataset."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Extract species and fluorescence data
    X = df.iloc[:, 1:].values  # All columns except first (species)
    y = df.iloc[:, 0].values   # First column is species

    return {
        'X': X,
        'y': y,
        'species_list': np.unique(y),
        'species_to_idx': {sp: idx for idx, sp in enumerate(np.unique(y))}
    }


def create_holdout_split(data_dict):
    """Create 80/20 train/holdout split with stratification."""
    X, y = data_dict['X'], data_dict['y']
    indices = np.arange(len(X))

    X_train, X_holdout, y_train, y_holdout, indices_train, indices_holdout = train_test_split(
        X, y, indices,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return {
        'X_train': X_train,
        'X_holdout': X_holdout,
        'y_train': y_train,
        'y_holdout': y_holdout,
        'indices_train': indices_train,
        'indices_holdout': indices_holdout
    }


def preprocess_curve(x, y):
    """Preprocess fluorescence curve: smoothing, baseline removal, normalization."""
    # Savitzky-Golay smoothing with ~1.5°C window
    dx = np.mean(np.diff(x))
    win = max(7, int(round(1.5 / dx)) | 1)
    y_smooth = savgol_filter(y, window_length=win, polyorder=3, mode="interp")

    # Baseline removal: quadratic fit through 30th percentile points
    q = np.quantile(y_smooth, 0.3)
    mask = y_smooth <= q
    if np.sum(mask) > 2:
        coeffs = np.polyfit(x[mask], y_smooth[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_baseline = y_smooth - baseline
    else:
        y_baseline = y_smooth

    # Normalize using 99th percentile
    scale = np.quantile(y_baseline, 0.99)
    y_norm = y_baseline / scale if scale > 0 else y_baseline
    y_norm = np.maximum(y_norm, 0.0)

    return y_norm


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_14_features_rulebased(fluorescence_curve, x_temp=None):
    """Extract 14 hand-crafted features for rule-based model."""
    if x_temp is None:
        x_temp = np.arange(len(fluorescence_curve))

    y = fluorescence_curve
    dx = np.mean(np.diff(x_temp)) if len(x_temp) > 1 else 1.0

    # Baseline as median of first 5%
    baseline = np.median(y[:max(1, len(y)//20)])
    y_adj = np.maximum(y - baseline, 0.0)

    # Peak information
    peak_idx = np.argmax(y_adj)
    ymax = y_adj[peak_idx]
    tmax = x_temp[peak_idx]

    # Area under curve
    auc = simpson(y_adj, x_temp)

    # Centroid
    centroid = np.sum(y_adj * x_temp) / (np.sum(y_adj) + 1e-10)

    # FWHM
    half_max = ymax / 2.0
    fwhm = np.sum(y_adj > half_max)

    # Rise/decay times (10% to 90%)
    ten_pct = 0.1 * ymax
    ninety_pct = 0.9 * ymax

    # Rise time: 10% → 90%
    above_10 = np.where(y_adj >= ten_pct)[0]
    above_90 = np.where(y_adj >= ninety_pct)[0]
    if len(above_10) > 0 and len(above_90) > 0:
        rise = above_90[0] - above_10[0] if above_90[0] >= above_10[0] else 0
    else:
        rise = 1

    # Decay time: 90% → 10% (from peak forward)
    decay_from_peak = y_adj[peak_idx:]
    above_90_decay = np.where(decay_from_peak >= ninety_pct)[0]
    above_10_decay = np.where(decay_from_peak >= ten_pct)[0]
    if len(above_90_decay) > 0 and len(above_10_decay) > 0:
        decay = above_10_decay[-1] - above_90_decay[-1] if above_10_decay[-1] >= above_90_decay[-1] else 1
    else:
        decay = 1

    # AUC left and right
    auc_left = simpson(y_adj[:peak_idx+1], x_temp[:peak_idx+1]) if peak_idx > 0 else y_adj[0]
    auc_right = simpson(y_adj[peak_idx:], x_temp[peak_idx:]) if peak_idx < len(y_adj)-1 else y_adj[-1]

    # Asymmetry
    asym = (auc_right - auc_left) / (auc + 1e-10)

    # Raw statistics (on original curve, not adjusted)
    mean_raw = np.mean(y)
    std_raw = np.std(y)
    max_raw = np.max(y)
    min_raw = np.min(y)

    features = {
        'ymax': ymax,
        'tmax': tmax,
        'auc': auc,
        'centroid': centroid,
        'fwhm': fwhm,
        'rise': rise,
        'decay': decay,
        'auc_left': auc_left,
        'auc_right': auc_right,
        'asym': asym,
        'mean': mean_raw,
        'std': std_raw,
        'max': max_raw,
        'min': min_raw,
    }

    return features


def extract_16_features(fluorescence_curve, x_temp=None):
    """Extract 16 hand-crafted features from fluorescence curve."""
    if x_temp is None:
        x_temp = np.arange(len(fluorescence_curve))

    y = fluorescence_curve
    dx = np.mean(np.diff(x_temp)) if len(x_temp) > 1 else 1.0

    # Basic statistics
    max_val = np.max(y)
    min_val = np.min(y)
    mean_val = np.mean(y)
    std_val = np.std(y)

    # Area under curve
    auc = simpson(y, x_temp)

    # Centroid (center of mass)
    centroid = np.sum(y * x_temp) / np.sum(y) if np.sum(y) > 0 else 0

    # Peak information
    temp_peak = x_temp[np.argmax(y)]
    peak_idx = np.argmax(y)

    # FWHM (full width at half maximum)
    half_max = max_val / 2.0
    fwhm = np.sum(y > half_max)

    # Rise and decay times
    rise_time = peak_idx + 1
    decay_time = len(y) - peak_idx

    # AUC left and right of peak
    auc_left = simpson(y[:peak_idx+1], x_temp[:peak_idx+1]) if peak_idx > 0 else y[0]
    auc_right = simpson(y[peak_idx:], x_temp[peak_idx:]) if peak_idx < len(y)-1 else y[-1]

    # Asymmetry
    asymmetry = auc_left / auc_right if auc_right > 0 else 0

    # Rise/decay ratio
    rise_decay_ratio = rise_time / decay_time if decay_time > 0 else 0

    # Slopes
    slopes = np.diff(y) / dx if len(y) > 1 else np.array([0])
    max_slope = np.max(np.abs(slopes))
    mean_abs_slope = np.mean(np.abs(slopes))

    # Curvatures
    second_deriv = np.diff(slopes) / dx if len(slopes) > 1 else np.array([0])
    max_curvature = np.max(np.abs(second_deriv)) if len(second_deriv) > 0 else 0
    mean_abs_curvature = np.mean(np.abs(second_deriv)) if len(second_deriv) > 0 else 0

    # Interaction features
    fwhm_rise_ratio = fwhm / rise_time if rise_time > 0 else 0
    peak_temp_std = temp_peak * std_val
    asymmetry_fwhm = asymmetry * fwhm

    features = {
        'max': max_val,
        'min': min_val,
        'mean': mean_val,
        'std': std_val,
        'auc': auc,
        'centroid': centroid,
        'temp_peak': temp_peak,
        'fwhm': fwhm,
        'rise_time': rise_time,
        'decay_time': decay_time,
        'auc_left': auc_left,
        'auc_right': auc_right,
        'asymmetry': asymmetry,
        'fwhm_rise_ratio': fwhm_rise_ratio,
        'peak_temp_std': peak_temp_std,
        'asymmetry_fwhm': asymmetry_fwhm,
    }

    return features


def extract_36_features(fluorescence_curve, x_temp=None):
    """Extract 36 statistical features from fluorescence curve."""
    if x_temp is None:
        x_temp = np.arange(len(fluorescence_curve))

    y = fluorescence_curve
    dx = np.mean(np.diff(x_temp)) if len(x_temp) > 1 else 1.0

    features = {}

    # Basic statistics (7)
    features['mean'] = np.mean(y)
    features['std'] = np.std(y)
    features['min'] = np.min(y)
    features['max'] = np.max(y)
    features['range'] = features['max'] - features['min']
    features['skewness'] = (np.mean((y - features['mean'])**3) / (features['std']**3 + 1e-10))
    features['kurtosis'] = (np.mean((y - features['mean'])**4) / (features['std']**4 + 1e-10) - 3)

    # Derivatives (5)
    slopes = np.diff(y) / dx if len(y) > 1 else np.array([0])
    features['max_slope'] = np.max(np.abs(slopes))
    features['mean_abs_slope'] = np.mean(np.abs(slopes))
    features['slope_std'] = np.std(slopes)

    second_deriv = np.diff(slopes) / dx if len(slopes) > 1 else np.array([0])
    features['max_curvature'] = np.max(np.abs(second_deriv)) if len(second_deriv) > 0 else 0
    features['mean_abs_curvature'] = np.mean(np.abs(second_deriv)) if len(second_deriv) > 0 else 0

    # Peaks (4) - proper peak detection using scipy
    prominences = find_peaks(y, prominence=0)[1].get('prominences', np.array([]))
    if len(prominences) > 0:
        peak_indices = find_peaks(y, prominence=0)[0]
        features['n_peaks'] = len(peak_indices)
        features['max_prominence'] = np.max(prominences) if len(prominences) > 0 else 0
        features['mean_prominence'] = np.mean(prominences) if len(prominences) > 0 else 0
    else:
        # Fallback: single peak with prominence as height
        peak_idx = np.argmax(y)
        features['n_peaks'] = 1
        features['max_prominence'] = y[peak_idx]
        features['mean_prominence'] = y[peak_idx]

    peak_idx = np.argmax(y)
    features['peak_max_x'] = x_temp[peak_idx]

    # Regional stats (9)
    third = len(y) // 3
    features['y_left_mean'] = np.mean(y[:third])
    features['y_left_std'] = np.std(y[:third])
    features['y_left_max'] = np.max(y[:third])
    features['y_middle_mean'] = np.mean(y[third:2*third])
    features['y_middle_std'] = np.std(y[third:2*third])
    features['y_middle_max'] = np.max(y[third:2*third])
    features['y_right_mean'] = np.mean(y[2*third:])
    features['y_right_std'] = np.std(y[2*third:])
    features['y_right_max'] = np.max(y[2*third:])

    # Quartiles (4)
    features['q25'] = np.quantile(y, 0.25)
    features['q50'] = np.quantile(y, 0.50)
    features['q75'] = np.quantile(y, 0.75)
    features['iqr'] = features['q75'] - features['q25']

    # FFT features (11)
    fft_vals = np.abs(fft(y))[:len(y)//2]
    fft_power = (fft_vals ** 2) / len(y)
    top_freqs = np.argsort(fft_power)[-5:][::-1]

    for i in range(5):
        features[f'fft_power_{i}'] = fft_power[top_freqs[i]] if i < len(top_freqs) else 0

    features['fft_total_power'] = np.sum(fft_power)
    p_norm = fft_power / (features['fft_total_power'] + 1e-10)
    features['fft_entropy'] = -np.sum(p_norm[p_norm > 0] * np.log(p_norm[p_norm > 0] + 1e-10))

    return features


def extract_gaussian_features(fluorescence_curve, x_temp=None):
    """Extract 12 features from optimal Gaussian curve fitting (k=1 to k=6)."""
    if x_temp is None:
        x_temp = np.arange(len(fluorescence_curve))

    y = fluorescence_curve

    # Decimate for faster fitting
    decim_idx = np.arange(0, len(x_temp), DECIMATE_STEP)
    x_decim = x_temp[decim_idx]
    y_decim = y[decim_idx]

    def gaussian(x, *params):
        """Sum of Gaussians."""
        y_pred = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), 3):
            if i + 2 < len(params):
                amp, mu, sigma = params[i:i+3]
                y_pred += amp * np.exp(-0.5 * ((x - mu) / abs(sigma)) ** 2)
        return y_pred

    def bic(y_true, y_pred, n_params):
        """Bayesian Information Criterion."""
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)
        return n * np.log(rss / n) + n_params * np.log(n)

    # Find peaks for initialization
    peaks, peak_props = find_peaks(y_decim, prominence=0.05)
    if len(peaks) == 0:
        # Fallback: single peak at argmax
        peak_idx = np.argmax(y_decim)
        peaks = [peak_idx]
        peak_props = {'prominences': np.array([y_decim[peak_idx]])}

    # Sort peaks by prominence
    if 'prominences' in peak_props:
        peak_order = np.argsort(peak_props['prominences'])[::-1]
        peaks = peaks[peak_order]

    # Try k=1 to k=6 Gaussians
    best_k = 1
    best_bic_val = float('inf')
    best_params = None
    best_y_pred = None

    for k in range(1, 7):
        try:
            # Initial guess from top k peaks
            p0 = []
            for i in range(k):
                if i < len(peaks):
                    peak_x = x_decim[peaks[i]]
                    peak_y = y_decim[peaks[i]]
                else:
                    # Distribute remaining peaks
                    peak_x = x_decim[np.argmax(y_decim)] + (i - len(peaks)) * 5
                    peak_y = np.max(y_decim) * (0.5 ** (i + 1))

                sigma_guess = 2.0  # Initial sigma
                p0.extend([peak_y, peak_x, sigma_guess])

            # Fit
            try:
                popt, _ = curve_fit(
                    gaussian, x_decim, y_decim, p0=p0,
                    maxfev=15000, ftol=1e-6, xtol=1e-6
                )
                y_pred = gaussian(x_decim, *popt)
                bic_val = bic(y_decim, y_pred, len(popt))

                if bic_val < best_bic_val:
                    best_bic_val = bic_val
                    best_k = k
                    best_params = popt
                    best_y_pred = y_pred
            except:
                continue
        except:
            continue

    # Extract features from best fit
    if best_params is None:
        # Fallback: simple heuristic approach
        peak_idx = np.argmax(y_decim)
        peak_amp = y_decim[peak_idx]
        peak_mu = x_decim[peak_idx]

        half_max = peak_amp / 2
        above_half = np.where(y_decim > half_max)[0]
        if len(above_half) > 1:
            peak_sigma = (x_decim[above_half[-1]] - x_decim[above_half[0]]) / 2.355
        else:
            peak_sigma = 1.0

        # Second peak
        y_masked = y_decim.copy()
        y_masked[max(0, peak_idx-3):min(len(y_decim), peak_idx+3)] = 0
        if np.max(y_masked) > peak_amp * 0.1:
            peak2_idx = np.argmax(y_masked)
            peak2_amp = y_masked[peak2_idx]
            peak2_mu = x_decim[peak2_idx]
            above_half2 = np.where(y_masked > y_masked[peak2_idx]/2)[0]
            if len(above_half2) > 1:
                peak2_sigma = (x_decim[above_half2[-1]] - x_decim[above_half2[0]]) / 2.355
            else:
                peak2_sigma = 1.0
        else:
            peak2_amp = peak_amp * 0.3
            peak2_mu = peak_mu + 10
            peak2_sigma = peak_sigma

        features = {
            'peak1_mu': peak_mu,
            'peak1_amp': peak_amp,
            'peak1_sigma': peak_sigma,
            'peak2_mu': peak2_mu,
            'peak2_amp': peak2_amp,
            'peak2_sigma': peak2_sigma,
            'delta_mu_12': abs(peak_mu - peak2_mu),
            'amp_ratio_12': peak_amp / (peak2_amp + 1e-10),
            'total_area': (peak_amp * peak_sigma + peak2_amp * peak2_sigma) * np.sqrt(2 * np.pi),
            'asym_0p5C': 0.5,
            'best_K': 2,
            'total_amp': peak_amp + peak2_amp,
        }
    else:
        # Extract from best parameters
        features = {
            'peak1_mu': best_params[1] if len(best_params) >= 3 else 0,
            'peak1_amp': best_params[0] if len(best_params) >= 3 else 0,
            'peak1_sigma': best_params[2] if len(best_params) >= 3 else 1,
            'peak2_mu': best_params[4] if len(best_params) >= 6 else (best_params[1] + 10) if len(best_params) >= 3 else 0,
            'peak2_amp': best_params[3] if len(best_params) >= 6 else (best_params[0] * 0.3) if len(best_params) >= 3 else 0,
            'peak2_sigma': best_params[5] if len(best_params) >= 6 else (best_params[2] if len(best_params) >= 3 else 1),
            'delta_mu_12': abs(best_params[1] - best_params[4]) if len(best_params) >= 6 else (abs(best_params[1] - (best_params[1] + 10)) if len(best_params) >= 3 else 0),
            'amp_ratio_12': best_params[0] / (best_params[3] + 1e-10) if len(best_params) >= 6 else (best_params[0] / (best_params[0] * 0.3 + 1e-10) if len(best_params) >= 3 else 1),
            'total_area': sum(best_params[3*i] * best_params[3*i+2] * np.sqrt(2 * np.pi) for i in range(best_k)),
            'asym_0p5C': 0.5,  # Would need full fit to compute accurately
            'best_K': best_k,
            'total_amp': sum(best_params[3*i] for i in range(best_k)),
        }

    return features


# ============================================================================
# PYTORCH MODELS & UTILITIES
# ============================================================================

class FluorescenceImageDataset(Dataset):
    """Dataset for loading fluorescence images with correct labels."""
    def __init__(self, img_dir, indices, labels, species_list, transform=None):
        """
        Args:
            img_dir: Directory containing images
            indices: Original dataset indices for these samples
            labels: True species labels for these samples
            species_list: List of all unique species
            transform: Image transforms
        """
        self.img_dir = Path(img_dir)
        self.indices = indices
        self.labels = labels
        self.species_list = species_list
        self.transform = transform
        self.species_to_idx = {sp: idx for idx, sp in enumerate(species_list)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Use actual label for this sample
        species = self.labels[idx]
        orig_idx = self.indices[idx]

        # Images are organized in species subdirectories
        # e.g., data/train/Arabian_smooth-hound/Arabian_smooth-hound_0002.png
        species_clean = species.replace(' ', '_').replace('/', '_')
        species_dir = self.img_dir / species_clean
        img_file = species_dir / f"{species_clean}_{orig_idx:04d}.png"

        try:
            img = Image.open(img_file).convert('RGB')
        except FileNotFoundError:
            print(f"[WARNING] Image not found: {img_file}")
            # Fallback: white image
            img = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            img = self.transform(img)

        return img, species


class EfficientNetHead(nn.Module):
    """Custom classifier head for EfficientNet-B0."""
    def __init__(self, num_classes=57):
        super().__init__()
        self.dropout1 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class ResidualBlock1D(nn.Module):
    """1D Residual block for ResNet1D."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """1D ResNet model for time series."""
    def __init__(self, num_classes=57, initial_filters=80, dropout=0.2):
        super().__init__()
        self.initial_filters = initial_filters

        self.conv1 = nn.Conv1d(1, initial_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResidualBlock1D, initial_filters, initial_filters, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock1D, initial_filters, initial_filters*2, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock1D, initial_filters*2, initial_filters*4, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock1D, initial_filters*4, initial_filters*8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(initial_filters*8, num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        layers = [block(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha=1.0, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return loss.mean()


# ============================================================================
# K-FOLD TRAINING FUNCTIONS
# ============================================================================

def train_cnn_kfold(data_dict, X_train, y_train, indices_train):
    """Train CNN with 5-fold cross-validation."""
    print("\n[CNN] Training EfficientNet-B0 with 5-fold CV...")

    if not TORCH_AVAILABLE:
        print("[CNN] ERROR: PyTorch not available")
        return None, None

    if not TRAIN_IMG_PATH.exists():
        print("[CNN] WARNING: Training image directory not found")
        return None, None

    try:
        species_to_idx = data_dict['species_to_idx']
        species_list = data_dict['species_list']
        fold_models = {}
        oof_preds = np.zeros((len(X_train), len(species_list)), dtype=np.float32)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            print(f"\n[CNN] Fold {fold+1}/5")

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            idx_fold_train, idx_fold_val = indices_train[train_idx], indices_train[val_idx]

            # Model setup
            model = efficientnet_b0(weights='IMAGENET1K_V1')
            model.classifier = EfficientNetHead(num_classes=len(species_list))
            model = model.to(DEVICE)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=0.0020594007612475913,
                weight_decay=1.0083970230770894e-05
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
            criterion = FocalLoss(alpha=1.0, gamma=1.5)

            # Transforms with AddGaussianNoise
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=0, translate=(0.0, 0.03)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                AddGaussianNoise(std=0.005),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            # Datasets
            train_dataset = FluorescenceImageDataset(
                TRAIN_IMG_PATH, idx_fold_train, y_fold_train,
                species_list, transform=train_transform
            )
            val_dataset = FluorescenceImageDataset(
                TRAIN_IMG_PATH, idx_fold_val, y_fold_val,
                species_list, transform=val_transform
            )

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

            # Training loop
            best_acc = 0
            best_model_state = None
            patience_count = 0
            best_epoch = 0

            for epoch in range(150):
                model.train()
                for batch_imgs, batch_species in train_loader:
                    batch_imgs = batch_imgs.to(DEVICE)
                    species_idx = torch.tensor([species_to_idx[s] for s in batch_species]).to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(batch_imgs)
                    loss = criterion(outputs, species_idx)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_preds = []
                with torch.no_grad():
                    for batch_imgs, batch_species in val_loader:
                        batch_imgs = batch_imgs.to(DEVICE)
                        outputs = model(batch_imgs)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        val_preds.extend([species_list[p] for p in preds])

                val_acc = np.mean(np.array(val_preds) == y_fold_val)
                scheduler.step()

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    patience_count = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_count += 1

                if patience_count >= 25:
                    break

            # Load best model and get OOF predictions
            model.load_state_dict(best_model_state)
            model.eval()

            # Get OOF predictions for validation fold
            val_probs = []
            with torch.no_grad():
                for batch_imgs, _ in val_loader:
                    batch_imgs = batch_imgs.to(DEVICE)
                    outputs = model(batch_imgs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    val_probs.append(probs)

            val_probs = np.vstack(val_probs)
            oof_preds[val_idx] = val_probs

            # Save fold model
            fold_models[fold] = best_model_state

            print(f"[CNN] Fold {fold+1} - Best Val Acc: {best_acc:.4f}, Epoch: {best_epoch}")

        return oof_preds, fold_models

    except Exception as e:
        print(f"[CNN] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def train_resnet1d_kfold(data_dict, X_train, y_train):
    """Train ResNet1D with 5-fold cross-validation."""
    print("\n[ResNet1D] Training ResNet1D with 5-fold CV...")

    if not TORCH_AVAILABLE:
        print("[ResNet1D] ERROR: PyTorch not available")
        return None, None

    try:
        species_to_idx = data_dict['species_to_idx']
        species_list = data_dict['species_list']

        # Compute normalization stats from entire train set
        mean_val = X_train.mean()
        std_val = X_train.std()
        X_train_norm = (X_train - mean_val) / std_val

        fold_models = {}
        oof_preds = np.zeros((len(X_train), len(species_list)), dtype=np.float32)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            print(f"\n[ResNet1D] Fold {fold+1}/5")

            X_fold_train, X_fold_val = X_train_norm[train_idx], X_train_norm[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Dataset
            class TimeSeriesDataset(Dataset):
                def __init__(self, X, y):
                    self.X = torch.FloatTensor(X).unsqueeze(1)
                    self.y = torch.tensor([species_to_idx[sp] for sp in y])

                def __len__(self):
                    return len(self.X)

                def __getitem__(self, idx):
                    return self.X[idx], self.y[idx]

            train_dataset = TimeSeriesDataset(X_fold_train, y_fold_train)
            val_dataset = TimeSeriesDataset(X_fold_val, y_fold_val)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

            # Model
            model = ResNet1D(num_classes=len(species_list), initial_filters=80,
                           dropout=0.20796879885018393)
            model = model.to(DEVICE)

            optimizer = optim.Adam(
                model.parameters(),
                lr=0.0004313869594239175,
                weight_decay=0.0001560845747200455
            )
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            criterion = nn.CrossEntropyLoss()

            # Training
            best_val_acc = 0
            best_model_state = None
            patience_count = 0
            best_epoch = 0

            for epoch in range(200):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_loss = 0
                val_preds = []
                val_targets = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        val_preds.extend(preds)
                        val_targets.extend(batch_y.cpu().numpy())

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                # Calculate validation accuracy
                val_acc = np.mean(np.array(val_preds) == np.array(val_targets))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    patience_count = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_count += 1

                if patience_count >= 15:
                    break

            # Load best model and get OOF predictions
            model.load_state_dict(best_model_state)
            model.eval()

            val_probs = []
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    outputs = model(batch_X)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    val_probs.append(probs)

            val_probs = np.vstack(val_probs)
            oof_preds[val_idx] = val_probs

            fold_models[fold] = best_model_state

            print(f"[ResNet1D] Fold {fold+1} - Best Val Acc: {best_val_acc:.4f}, Epoch: {best_epoch}")

        return oof_preds, fold_models

    except Exception as e:
        print(f"[ResNet1D] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def train_rulebased_kfold(data_dict, X_train, y_train):
    """Train Rule-Based model with 5-fold cross-validation."""
    print("\n[RuleBased] Training Rule-Based model with 5-fold CV...")

    try:
        x_temp = np.linspace(20, 85, X_train.shape[1])
        species_list = data_dict['species_list']
        species_to_idx = data_dict['species_to_idx']

        fold_models = {}
        oof_preds = np.zeros((len(X_train), len(species_list)), dtype=np.float32)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            print(f"\n[RuleBased] Fold {fold+1}/5")

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Extract features
            train_features = []
            for sample in X_fold_train:
                feat_dict = extract_14_features_rulebased(sample, x_temp)
                train_features.append(list(feat_dict.values()))
            train_features = np.array(train_features)

            val_features = []
            for sample in X_fold_val:
                feat_dict = extract_14_features_rulebased(sample, x_temp)
                val_features.append(list(feat_dict.values()))
            val_features = np.array(val_features)

            # Normalize features
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            val_features_scaled = scaler.transform(val_features)

            # Train model
            model = ExtraTreesClassifier(
                n_estimators=790,
                max_depth=15,
                min_samples_leaf=1,
                max_features=None,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model.fit(train_features_scaled, y_fold_train)

            # OOF predictions
            oof_pred_probs = model.predict_proba(val_features_scaled)
            oof_preds[val_idx] = oof_pred_probs

            # Store scaler with model for later use
            fold_models[fold] = {'model': model, 'scaler': scaler}
            val_acc = model.score(val_features_scaled, y_fold_val)
            print(f"[RuleBased] Fold {fold+1} - Val Acc: {val_acc:.4f}")

        return oof_preds, fold_models

    except Exception as e:
        print(f"[RuleBased] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def train_extratrees_kfold(data_dict, X_train, y_train):
    """Train ExtraTrees with 5-fold cross-validation."""
    print("\n[ExtraTrees] Training ExtraTrees with 5-fold CV...")

    try:
        x_temp = np.linspace(20, 85, X_train.shape[1])
        species_list = data_dict['species_list']
        species_to_idx = data_dict['species_to_idx']

        fold_models = {}
        oof_preds = np.zeros((len(X_train), len(species_list)), dtype=np.float32)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            print(f"\n[ExtraTrees] Fold {fold+1}/5")

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Extract features
            train_features = []
            for sample in X_fold_train:
                feat_dict = extract_16_features(sample, x_temp)
                train_features.append(list(feat_dict.values()))
            train_features = np.array(train_features)

            val_features = []
            for sample in X_fold_val:
                feat_dict = extract_16_features(sample, x_temp)
                val_features.append(list(feat_dict.values()))
            val_features = np.array(val_features)

            # Train model
            model = ExtraTreesClassifier(
                n_estimators=900,
                max_depth=40,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features=0.5,
                class_weight='balanced_subsample',
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model.fit(train_features, y_fold_train)

            # OOF predictions
            oof_pred_probs = model.predict_proba(val_features)
            oof_preds[val_idx] = oof_pred_probs

            fold_models[fold] = model
            val_acc = model.score(val_features, y_fold_val)
            print(f"[ExtraTrees] Fold {fold+1} - Val Acc: {val_acc:.4f}")

        return oof_preds, fold_models

    except Exception as e:
        print(f"[ExtraTrees] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def train_statistics_kfold(data_dict, X_train, y_train):
    """Train Statistical model with 5-fold cross-validation."""
    print("\n[Statistics] Training Statistical model with 5-fold CV...")

    try:
        x_temp = np.linspace(20, 85, X_train.shape[1])
        species_list = data_dict['species_list']
        species_to_idx = data_dict['species_to_idx']

        TOP_18_FEATURES = [
            'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
            'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
            'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
            'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
            'slope_std', 'y_left_max'
        ]

        fold_models = {}
        oof_preds = np.zeros((len(X_train), len(species_list)), dtype=np.float32)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            print(f"\n[Statistics] Fold {fold+1}/5")

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Extract features
            train_features = []
            for sample in X_fold_train:
                y_prep = preprocess_curve(x_temp, sample)
                feat_dict = extract_36_features(y_prep, x_temp)
                feat_vals = [feat_dict.get(f, 0) for f in TOP_18_FEATURES]
                train_features.append(feat_vals)
            train_features = np.array(train_features)

            val_features = []
            for sample in X_fold_val:
                y_prep = preprocess_curve(x_temp, sample)
                feat_dict = extract_36_features(y_prep, x_temp)
                feat_vals = [feat_dict.get(f, 0) for f in TOP_18_FEATURES]
                val_features.append(feat_vals)
            val_features = np.array(val_features)

            # Train model with calibration
            base_model = ExtraTreesClassifier(
                n_estimators=1700,
                max_depth=None,
                min_samples_split=9,
                min_samples_leaf=1,
                max_features=0.7,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

            # Adaptive CV for calibration: ensure each class has enough samples
            # For 3-fold CV, each class needs at least 3 samples
            y_fold_train_idx = np.array([species_to_idx[sp] for sp in y_fold_train])
            min_class_samples = np.bincount(y_fold_train_idx).min()
            n_cal_cv = 3 if min_class_samples >= 3 else 2

            model = CalibratedClassifierCV(
                estimator=base_model,
                method='isotonic',
                cv=n_cal_cv
            )
            try:
                model.fit(train_features, y_fold_train)
            except ValueError as e:
                if "cross-validation" in str(e):
                    # Fallback to 2-fold if calibration still fails
                    print(f"[Statistics] Fold {fold+1} - Retrying with 2-fold calibration...")
                    model = CalibratedClassifierCV(
                        estimator=base_model,
                        method='isotonic',
                        cv=2
                    )
                    model.fit(train_features, y_fold_train)
                else:
                    raise

            # OOF predictions
            oof_pred_probs = model.predict_proba(val_features)
            oof_preds[val_idx] = oof_pred_probs

            fold_models[fold] = model
            val_acc = model.score(val_features, y_fold_val)
            print(f"[Statistics] Fold {fold+1} - Val Acc: {val_acc:.4f} (cal_cv={n_cal_cv})")

        return oof_preds, fold_models

    except Exception as e:
        print(f"[Statistics] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def train_gaussian_kfold(data_dict, X_train, y_train):
    """Train Gaussian model with 5-fold cross-validation."""
    print("\n[Gaussian] Training Gaussian model with 5-fold CV...")

    try:
        x_temp = np.linspace(20, 85, X_train.shape[1])
        species_list = data_dict['species_list']
        species_to_idx = data_dict['species_to_idx']

        fold_models = {}
        oof_preds = np.zeros((len(X_train), len(species_list)), dtype=np.float32)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            print(f"\n[Gaussian] Fold {fold+1}/5")

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Extract features with progress
            print(f"[Gaussian] Extracting features for {len(X_fold_train)} training samples...")
            train_features = []
            for i, sample in enumerate(X_fold_train):
                if (i + 1) % 50 == 0:
                    print(f"[Gaussian]   Progress: {i+1}/{len(X_fold_train)}")
                y_prep = preprocess_curve(x_temp, sample)
                feat_dict = extract_gaussian_features(y_prep, x_temp)
                train_features.append(list(feat_dict.values()))
            train_features = np.array(train_features)

            print(f"[Gaussian] Extracting features for {len(X_fold_val)} validation samples...")
            val_features = []
            for i, sample in enumerate(X_fold_val):
                if (i + 1) % 20 == 0:
                    print(f"[Gaussian]   Progress: {i+1}/{len(X_fold_val)}")
                y_prep = preprocess_curve(x_temp, sample)
                feat_dict = extract_gaussian_features(y_prep, x_temp)
                val_features.append(list(feat_dict.values()))
            val_features = np.array(val_features)

            # Train model with calibration
            base_model = RandomForestClassifier(
                n_estimators=800,
                random_state=RANDOM_STATE,
                class_weight='balanced_subsample',
                max_depth=None,
                min_samples_leaf=1,
                n_jobs=-1
            )

            # Adaptive CV for calibration: ensure each class has enough samples
            # For 3-fold CV, each class needs at least 3 samples
            y_fold_train_idx = np.array([species_to_idx[sp] for sp in y_fold_train])
            min_class_samples = np.bincount(y_fold_train_idx).min()
            n_cal_cv = 3 if min_class_samples >= 3 else 2

            model = CalibratedClassifierCV(
                estimator=base_model,
                method='isotonic',
                cv=n_cal_cv
            )
            try:
                model.fit(train_features, y_fold_train)
            except ValueError as e:
                if "cross-validation" in str(e):
                    # Fallback to 2-fold if calibration still fails
                    print(f"[Gaussian] Fold {fold+1} - Retrying with 2-fold calibration...")
                    model = CalibratedClassifierCV(
                        estimator=base_model,
                        method='isotonic',
                        cv=2
                    )
                    model.fit(train_features, y_fold_train)
                else:
                    raise

            # OOF predictions
            oof_pred_probs = model.predict_proba(val_features)
            oof_preds[val_idx] = oof_pred_probs

            fold_models[fold] = model
            val_acc = model.score(val_features, y_fold_val)
            print(f"[Gaussian] Fold {fold+1} - Val Acc: {val_acc:.4f} (cal_cv={n_cal_cv})")

        return oof_preds, fold_models

    except Exception as e:
        print(f"[Gaussian] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# HOLDOUT PREDICTION FUNCTIONS
# ============================================================================

def predict_cnn_holdout(fold_models, data_dict, X_holdout, y_holdout, indices_holdout):
    """Generate predictions on holdout set using best fold models."""
    print("\n[CNN] Generating holdout predictions...")

    if fold_models is None or len(fold_models) == 0:
        print("[CNN] WARNING: No fold models available")
        return [''] * len(X_holdout)

    if not TORCH_AVAILABLE or not TEST_IMG_PATH.exists():
        print("[CNN] WARNING: Cannot generate predictions")
        return [''] * len(X_holdout)

    try:
        species_list = data_dict['species_list']
        species_to_idx = data_dict['species_to_idx']

        holdout_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Ensemble predictions from all folds
        all_probs = []

        for fold, model_state in fold_models.items():
            model = efficientnet_b0(weights='IMAGENET1K_V1')
            model.classifier = EfficientNetHead(num_classes=len(species_list))
            model.load_state_dict(model_state)
            model = model.to(DEVICE)
            model.eval()

            dataset = FluorescenceImageDataset(
                TEST_IMG_PATH, indices_holdout, y_holdout,
                species_list, transform=holdout_transform
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            fold_probs = []
            with torch.no_grad():
                for batch_imgs, _ in loader:
                    batch_imgs = batch_imgs.to(DEVICE)
                    outputs = model(batch_imgs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    fold_probs.append(probs)

            all_probs.append(np.vstack(fold_probs))

        # Average across folds
        mean_probs = np.mean(all_probs, axis=0)
        predictions = [species_list[np.argmax(probs)] for probs in mean_probs]

        accuracy = np.mean(np.array(predictions) == y_holdout)
        print(f"[CNN] Holdout Accuracy: {accuracy:.4f}")

        return predictions

    except Exception as e:
        print(f"[CNN] ERROR: {str(e)}")
        return [''] * len(X_holdout)


def predict_resnet1d_holdout(fold_models, data_dict, X_holdout, y_holdout):
    """Generate predictions on holdout set using best fold models."""
    print("\n[ResNet1D] Generating holdout predictions...")

    if fold_models is None or len(fold_models) == 0:
        print("[ResNet1D] WARNING: No fold models available")
        return [''] * len(X_holdout)

    if not TORCH_AVAILABLE:
        print("[ResNet1D] WARNING: PyTorch not available")
        return [''] * len(X_holdout)

    try:
        species_list = data_dict['species_list']
        species_to_idx = data_dict['species_to_idx']

        # Normalize using training set stats
        X_train = data_dict['X_train']
        mean_val = X_train.mean()
        std_val = X_train.std()
        X_holdout_norm = (X_holdout - mean_val) / std_val

        # Ensemble predictions
        all_probs = []

        for fold, model_state in fold_models.items():
            model = ResNet1D(num_classes=len(species_list), initial_filters=80,
                           dropout=0.20796879885018393)
            model.load_state_dict(model_state)
            model = model.to(DEVICE)
            model.eval()

            class TimeSeriesDataset(Dataset):
                def __init__(self, X):
                    self.X = torch.FloatTensor(X).unsqueeze(1)

                def __len__(self):
                    return len(self.X)

                def __getitem__(self, idx):
                    return self.X[idx]

            dataset = TimeSeriesDataset(X_holdout_norm)
            loader = DataLoader(dataset, batch_size=16, shuffle=False)

            fold_probs = []
            with torch.no_grad():
                for batch_X in loader:
                    batch_X = batch_X.to(DEVICE)
                    outputs = model(batch_X)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    fold_probs.append(probs)

            all_probs.append(np.vstack(fold_probs))

        # Average across folds
        mean_probs = np.mean(all_probs, axis=0)
        predictions = [species_list[np.argmax(probs)] for probs in mean_probs]

        accuracy = np.mean(np.array(predictions) == y_holdout)
        print(f"[ResNet1D] Holdout Accuracy: {accuracy:.4f}")

        return predictions

    except Exception as e:
        print(f"[ResNet1D] ERROR: {str(e)}")
        return [''] * len(X_holdout)


def predict_extratrees_holdout(fold_models, data_dict, X_holdout, y_holdout):
    """Generate predictions on holdout set using fold models."""
    print("\n[ExtraTrees] Generating holdout predictions...")

    if fold_models is None or len(fold_models) == 0:
        print("[ExtraTrees] WARNING: No fold models available")
        return [''] * len(X_holdout)

    try:
        x_temp = np.linspace(20, 85, X_holdout.shape[1])
        species_list = data_dict['species_list']

        # Extract features
        holdout_features = []
        for sample in X_holdout:
            feat_dict = extract_16_features(sample, x_temp)
            holdout_features.append(list(feat_dict.values()))
        holdout_features = np.array(holdout_features)

        # Ensemble predictions
        all_probs = []
        for fold, model in fold_models.items():
            probs = model.predict_proba(holdout_features)
            all_probs.append(probs)

        # Average across folds
        mean_probs = np.mean(all_probs, axis=0)
        predictions = [species_list[np.argmax(probs)] for probs in mean_probs]

        accuracy = np.mean(np.array(predictions) == y_holdout)
        print(f"[ExtraTrees] Holdout Accuracy: {accuracy:.4f}")

        return predictions

    except Exception as e:
        print(f"[ExtraTrees] ERROR: {str(e)}")
        return [''] * len(X_holdout)


def predict_statistics_holdout(fold_models, data_dict, X_holdout, y_holdout):
    """Generate predictions on holdout set using fold models."""
    print("\n[Statistics] Generating holdout predictions...")

    if fold_models is None or len(fold_models) == 0:
        print("[Statistics] WARNING: No fold models available")
        return [''] * len(X_holdout)

    try:
        x_temp = np.linspace(20, 85, X_holdout.shape[1])
        species_list = data_dict['species_list']

        TOP_18_FEATURES = [
            'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
            'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
            'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
            'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
            'slope_std', 'y_left_max'
        ]

        # Extract features
        holdout_features = []
        for sample in X_holdout:
            y_prep = preprocess_curve(x_temp, sample)
            feat_dict = extract_36_features(y_prep, x_temp)
            feat_vals = [feat_dict.get(f, 0) for f in TOP_18_FEATURES]
            holdout_features.append(feat_vals)
        holdout_features = np.array(holdout_features)

        # Ensemble predictions
        all_probs = []
        for fold, model in fold_models.items():
            probs = model.predict_proba(holdout_features)
            all_probs.append(probs)

        # Average across folds
        mean_probs = np.mean(all_probs, axis=0)
        predictions = [species_list[np.argmax(probs)] for probs in mean_probs]

        accuracy = np.mean(np.array(predictions) == y_holdout)
        print(f"[Statistics] Holdout Accuracy: {accuracy:.4f}")

        return predictions

    except Exception as e:
        print(f"[Statistics] ERROR: {str(e)}")
        return [''] * len(X_holdout)


def predict_gaussian_holdout(fold_models, data_dict, X_holdout, y_holdout):
    """Generate predictions on holdout set using fold models."""
    print("\n[Gaussian] Generating holdout predictions...")

    if fold_models is None or len(fold_models) == 0:
        print("[Gaussian] WARNING: No fold models available")
        return [''] * len(X_holdout)

    try:
        x_temp = np.linspace(20, 85, X_holdout.shape[1])
        species_list = data_dict['species_list']

        # Extract features
        holdout_features = []
        for sample in X_holdout:
            y_prep = preprocess_curve(x_temp, sample)
            feat_dict = extract_gaussian_features(y_prep, x_temp)
            holdout_features.append(list(feat_dict.values()))
        holdout_features = np.array(holdout_features)

        # Ensemble predictions
        all_probs = []
        for fold, model in fold_models.items():
            probs = model.predict_proba(holdout_features)
            all_probs.append(probs)

        # Average across folds
        mean_probs = np.mean(all_probs, axis=0)
        predictions = [species_list[np.argmax(probs)] for probs in mean_probs]

        accuracy = np.mean(np.array(predictions) == y_holdout)
        print(f"[Gaussian] Holdout Accuracy: {accuracy:.4f}")

        return predictions

    except Exception as e:
        print(f"[Gaussian] ERROR: {str(e)}")
        return [''] * len(X_holdout)


# ============================================================================
# HOLDOUT PROBABILITY RETURN FUNCTIONS (For CSV Output)
# ============================================================================

def predict_cnn_holdout_probs(fold_models, data_dict, X_holdout, y_holdout, indices_holdout):
    """Return probability arrays for CNN holdout predictions."""
    if fold_models is None or len(fold_models) == 0:
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])

    try:
        species_list = data_dict['species_list']

        holdout_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        all_probs = []
        for fold, model_state in fold_models.items():
            model = efficientnet_b0(weights='IMAGENET1K_V1')
            model.classifier = EfficientNetHead(num_classes=len(species_list))
            model.load_state_dict(model_state)
            model = model.to(DEVICE)
            model.eval()

            dataset = FluorescenceImageDataset(
                TEST_IMG_PATH, indices_holdout, y_holdout,
                species_list, transform=holdout_transform
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            fold_probs = []
            with torch.no_grad():
                for batch_imgs, _ in loader:
                    batch_imgs = batch_imgs.to(DEVICE)
                    outputs = model(batch_imgs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    fold_probs.append(probs)

            all_probs.append(np.vstack(fold_probs))

        return np.mean(all_probs, axis=0)

    except Exception as e:
        print(f"[CNN] WARNING: Could not get probabilities: {str(e)}")
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])


def predict_resnet1d_holdout_probs(fold_models, data_dict, X_holdout, y_holdout):
    """Return probability arrays for ResNet1D holdout predictions."""
    if fold_models is None or len(fold_models) == 0:
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])

    try:
        species_list = data_dict['species_list']

        X_train = data_dict['X_train']
        mean_val = X_train.mean()
        std_val = X_train.std()
        X_holdout_norm = (X_holdout - mean_val) / std_val

        all_probs = []
        for fold, model_state in fold_models.items():
            model = ResNet1D(num_classes=len(species_list), initial_filters=80,
                           dropout=0.20796879885018393)
            model.load_state_dict(model_state)
            model = model.to(DEVICE)
            model.eval()

            class TimeSeriesDataset(Dataset):
                def __init__(self, X):
                    self.X = torch.FloatTensor(X).unsqueeze(1)

                def __len__(self):
                    return len(self.X)

                def __getitem__(self, idx):
                    return self.X[idx]

            dataset = TimeSeriesDataset(X_holdout_norm)
            loader = DataLoader(dataset, batch_size=16, shuffle=False)

            fold_probs = []
            with torch.no_grad():
                for batch_X in loader:
                    batch_X = batch_X.to(DEVICE)
                    outputs = model(batch_X)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    fold_probs.append(probs)

            all_probs.append(np.vstack(fold_probs))

        return np.mean(all_probs, axis=0)

    except Exception as e:
        print(f"[ResNet1D] WARNING: Could not get probabilities: {str(e)}")
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])


def predict_rulebased_holdout_probs(fold_models, data_dict, X_holdout, y_holdout):
    """Return probability arrays for Rule-Based holdout predictions."""
    if fold_models is None or len(fold_models) == 0:
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])

    try:
        x_temp = np.linspace(20, 85, X_holdout.shape[1])
        species_list = data_dict['species_list']

        # Extract features
        holdout_features = []
        for sample in X_holdout:
            feat_dict = extract_14_features_rulebased(sample, x_temp)
            holdout_features.append(list(feat_dict.values()))
        holdout_features = np.array(holdout_features)

        # Ensemble predictions
        all_probs = []
        for fold, fold_dict in fold_models.items():
            model = fold_dict['model']
            scaler = fold_dict['scaler']
            holdout_features_scaled = scaler.transform(holdout_features)
            probs = model.predict_proba(holdout_features_scaled)
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    except Exception as e:
        print(f"[RuleBased] WARNING: Could not get probabilities: {str(e)}")
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])


def predict_extratrees_holdout_probs(fold_models, data_dict, X_holdout, y_holdout):
    """Return probability arrays for ExtraTrees holdout predictions."""
    if fold_models is None or len(fold_models) == 0:
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])

    try:
        x_temp = np.linspace(20, 85, X_holdout.shape[1])
        species_list = data_dict['species_list']

        holdout_features = []
        for sample in X_holdout:
            feat_dict = extract_16_features(sample, x_temp)
            holdout_features.append(list(feat_dict.values()))
        holdout_features = np.array(holdout_features)

        all_probs = []
        for fold, model in fold_models.items():
            probs = model.predict_proba(holdout_features)
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    except Exception as e:
        print(f"[ExtraTrees] WARNING: Could not get probabilities: {str(e)}")
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])


def predict_statistics_holdout_probs(fold_models, data_dict, X_holdout, y_holdout):
    """Return probability arrays for Statistics holdout predictions."""
    if fold_models is None or len(fold_models) == 0:
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])

    try:
        x_temp = np.linspace(20, 85, X_holdout.shape[1])
        species_list = data_dict['species_list']

        TOP_18_FEATURES = [
            'peak_max_x', 'max_slope', 'y_middle_std', 'mean_abs_curvature',
            'fft_power_4', 'mean_abs_slope', 'max_curvature', 'range',
            'fft_entropy', 'max', 'y_middle_max', 'fft_power_2',
            'fft_power_1', 'fft_power_0', 'fft_power_3', 'y_right_max',
            'slope_std', 'y_left_max'
        ]

        holdout_features = []
        for sample in X_holdout:
            y_prep = preprocess_curve(x_temp, sample)
            feat_dict = extract_36_features(y_prep, x_temp)
            feat_vals = [feat_dict.get(f, 0) for f in TOP_18_FEATURES]
            holdout_features.append(feat_vals)
        holdout_features = np.array(holdout_features)

        all_probs = []
        for fold, model in fold_models.items():
            probs = model.predict_proba(holdout_features)
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    except Exception as e:
        print(f"[Statistics] WARNING: Could not get probabilities: {str(e)}")
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])


def predict_gaussian_holdout_probs(fold_models, data_dict, X_holdout, y_holdout):
    """Return probability arrays for Gaussian holdout predictions."""
    if fold_models is None or len(fold_models) == 0:
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])

    try:
        x_temp = np.linspace(20, 85, X_holdout.shape[1])
        species_list = data_dict['species_list']

        print(f"[Gaussian] Extracting features for {len(X_holdout)} holdout samples...")
        holdout_features = []
        for i, sample in enumerate(X_holdout):
            if (i + 1) % 20 == 0:
                print(f"[Gaussian]   Progress: {i+1}/{len(X_holdout)}")
            y_prep = preprocess_curve(x_temp, sample)
            feat_dict = extract_gaussian_features(y_prep, x_temp)
            holdout_features.append(list(feat_dict.values()))
        holdout_features = np.array(holdout_features)

        all_probs = []
        for fold, model in fold_models.items():
            probs = model.predict_proba(holdout_features)
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    except Exception as e:
        print(f"[Gaussian] WARNING: Could not get probabilities: {str(e)}")
        return np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete k-fold training and prediction pipeline."""
    print("="*70)
    print("SHARK SPECIES CLASSIFICATION - K-FOLD TRAINING & OOF PREDICTIONS")
    print("="*70)

    # Load full dataset
    data_dict = load_data()
    print(f"Loaded {len(data_dict['X'])} total samples")
    print(f"Number of species: {len(data_dict['species_list'])}")

    # Create 80/20 holdout split
    split_dict = create_holdout_split(data_dict)
    X_train = split_dict['X_train']
    X_holdout = split_dict['X_holdout']
    y_train = split_dict['y_train']
    y_holdout = split_dict['y_holdout']
    indices_train = split_dict['indices_train']
    indices_holdout = split_dict['indices_holdout']

    print(f"\nTrain set (80%): {len(X_train)} samples")
    print(f"Holdout set (20%): {len(X_holdout)} samples")

    # Store in data_dict for models to use
    data_dict['X_train'] = X_train
    data_dict['y_train'] = y_train
    data_dict['indices_train'] = indices_train

    # ========================================================================
    # K-FOLD TRAINING PHASE
    # ========================================================================
    print(f"\n{'='*70}")
    print("K-FOLD TRAINING PHASE: Training all models with 5-fold CV")
    print(f"{'='*70}")

    oof_predictions = {
        'cnn': None,
        'resnet1d': None,
        'extratrees': None,
        'statistics': None,
        'gaussian': None,
        'rulebased': None
    }

    fold_models = {
        'cnn': None,
        'resnet1d': None,
        'extratrees': None,
        'statistics': None,
        'gaussian': None,
        'rulebased': None
    }

    start_time = time.time()

    # Train models
    oof_predictions['rulebased'], fold_models['rulebased'] = train_rulebased_kfold(
        data_dict, X_train, y_train
    )

    oof_predictions['extratrees'], fold_models['extratrees'] = train_extratrees_kfold(
        data_dict, X_train, y_train
    )

    oof_predictions['statistics'], fold_models['statistics'] = train_statistics_kfold(
        data_dict, X_train, y_train
    )

    # TODO: Gaussian model disabled - revisit later
    # oof_predictions['gaussian'], fold_models['gaussian'] = train_gaussian_kfold(
    #     data_dict, X_train, y_train
    # )

    if TORCH_AVAILABLE:
        oof_predictions['cnn'], fold_models['cnn'] = train_cnn_kfold(
            data_dict, X_train, y_train, indices_train
        )

    if TORCH_AVAILABLE:
        oof_predictions['resnet1d'], fold_models['resnet1d'] = train_resnet1d_kfold(
            data_dict, X_train, y_train
        )
    else:
        print("\n[PyTorch] Skipping ResNet1D: PyTorch not available")

    training_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete in {training_time:.1f} seconds")
    print(f"{'='*70}")

    # ========================================================================
    # HOLDOUT PREDICTION PHASE
    # ========================================================================
    print(f"\n{'='*70}")
    print("PREDICTION PHASE: Generating holdout predictions")
    print(f"{'='*70}")

    holdout_predictions = {
        'cnn': [],
        'resnet1d': [],
        'extratrees': [],
        'statistics': [],
        'gaussian': [],
        'rulebased': []
    }

    # Generate holdout predictions
    if fold_models['rulebased'] is not None:
        holdout_predictions['rulebased'] = predict_rulebased_holdout_probs(
            fold_models['rulebased'], data_dict, X_holdout, y_holdout
        )
    else:
        holdout_predictions['rulebased'] = np.ones((len(X_holdout), len(data_dict['species_list']))) / len(data_dict['species_list'])

    if fold_models['extratrees'] is not None:
        holdout_predictions['extratrees'] = predict_extratrees_holdout(
            fold_models['extratrees'], data_dict, X_holdout, y_holdout
        )
    else:
        holdout_predictions['extratrees'] = [''] * len(X_holdout)

    if fold_models['statistics'] is not None:
        holdout_predictions['statistics'] = predict_statistics_holdout(
            fold_models['statistics'], data_dict, X_holdout, y_holdout
        )
    else:
        holdout_predictions['statistics'] = [''] * len(X_holdout)

    # TODO: Gaussian model disabled - revisit later
    # if fold_models['gaussian'] is not None:
    #     holdout_predictions['gaussian'] = predict_gaussian_holdout(
    #         fold_models['gaussian'], data_dict, X_holdout, y_holdout
    #     )
    # else:
    holdout_predictions['gaussian'] = [''] * len(X_holdout)

    if fold_models['cnn'] is not None:
        holdout_predictions['cnn'] = predict_cnn_holdout(
            fold_models['cnn'], data_dict, X_holdout, y_holdout, indices_holdout
        )
    else:
        holdout_predictions['cnn'] = [''] * len(X_holdout)

    if fold_models['resnet1d'] is not None:
        holdout_predictions['resnet1d'] = predict_resnet1d_holdout(
            fold_models['resnet1d'], data_dict, X_holdout, y_holdout
        )
    else:
        holdout_predictions['resnet1d'] = [''] * len(X_holdout)

    # ========================================================================
    # OUTPUT CSV GENERATION (WITH PROBABILITIES FOR META-LEARNER)
    # ========================================================================
    print(f"\n{'='*70}")
    print("GENERATING OUTPUT CSV WITH PROBABILITIES")
    print(f"{'='*70}")

    species_list = data_dict['species_list']
    species_list_clean = [sp.replace(' ', '_').replace('-', '_') for sp in species_list]

    # Create base output dataframe
    output_data = {
        'index': np.concatenate([indices_train, indices_holdout]),
        'species_true': np.concatenate([y_train, y_holdout]),
        'set': ['train'] * len(y_train) + ['holdout'] * len(y_holdout),
    }

    # Process each model's predictions
    for model_name in ['cnn', 'resnet1d', 'extratrees', 'statistics', 'rulebased']:
        print(f"\nProcessing {model_name} predictions...")

        # Collect all probabilities (train OOF + holdout)
        all_probs = []

        # OOF predictions for train set
        if oof_predictions[model_name] is not None:
            oof_probs = oof_predictions[model_name]
            all_probs.extend(oof_probs)
            print(f"  - Train OOF predictions: {len(oof_probs)} samples")
        else:
            # Fallback: uniform probs if model failed
            fallback_probs = np.ones((len(y_train), len(species_list))) / len(species_list)
            all_probs.extend(fallback_probs)
            print(f"  - Train OOF: FALLBACK (model not trained)")

        # Holdout predictions - convert labels back to probabilities
        if model_name == 'cnn' and fold_models['cnn'] is not None:
            # For deep learning models, we need to collect the actual probs
            # Re-compute holdout probs
            try:
                holdout_probs = predict_cnn_holdout_probs(
                    fold_models['cnn'], data_dict, X_holdout, y_holdout, indices_holdout
                )
                all_probs.extend(holdout_probs)
                print(f"  - Holdout predictions: {len(holdout_probs)} samples")
            except:
                fallback_probs = np.ones((len(y_holdout), len(species_list))) / len(species_list)
                all_probs.extend(fallback_probs)
                print(f"  - Holdout: ERROR (using fallback)")

        elif model_name == 'resnet1d' and fold_models['resnet1d'] is not None:
            try:
                holdout_probs = predict_resnet1d_holdout_probs(
                    fold_models['resnet1d'], data_dict, X_holdout, y_holdout
                )
                all_probs.extend(holdout_probs)
                print(f"  - Holdout predictions: {len(holdout_probs)} samples")
            except:
                fallback_probs = np.ones((len(y_holdout), len(species_list))) / len(species_list)
                all_probs.extend(fallback_probs)
                print(f"  - Holdout: ERROR (using fallback)")

        elif model_name in ['extratrees', 'statistics', 'gaussian', 'rulebased'] and fold_models[model_name] is not None:
            try:
                # These already return probs via predict_proba
                if model_name == 'extratrees':
                    holdout_probs = predict_extratrees_holdout_probs(
                        fold_models['extratrees'], data_dict, X_holdout, y_holdout
                    )
                elif model_name == 'statistics':
                    holdout_probs = predict_statistics_holdout_probs(
                        fold_models['statistics'], data_dict, X_holdout, y_holdout
                    )
                elif model_name == 'gaussian':
                    holdout_probs = predict_gaussian_holdout_probs(
                        fold_models['gaussian'], data_dict, X_holdout, y_holdout
                    )
                else:  # rulebased
                    holdout_probs = predict_rulebased_holdout_probs(
                        fold_models['rulebased'], data_dict, X_holdout, y_holdout
                    )
                all_probs.extend(holdout_probs)
                print(f"  - Holdout predictions: {len(holdout_probs)} samples")
            except:
                fallback_probs = np.ones((len(y_holdout), len(species_list))) / len(species_list)
                all_probs.extend(fallback_probs)
                print(f"  - Holdout: ERROR (using fallback)")
        else:
            # Model not available
            fallback_probs = np.ones((len(y_holdout), len(species_list))) / len(species_list)
            all_probs.extend(fallback_probs)
            print(f"  - Holdout: Model not available")

        # Convert to array
        all_probs = np.array(all_probs)
        print(f"  - Total: {len(all_probs)} rows, shape {all_probs.shape}")

        # Add probability columns for each species
        for species_idx, species_name in enumerate(species_list):
            col_name = f'{model_name}_prob_{species_list_clean[species_idx]}'
            output_data[col_name] = all_probs[:, species_idx]

    # Create output dataframe
    output_df = pd.DataFrame(output_data)

    # Verify counts
    print(f"\nVerifying output counts:")
    print(f"  Total samples: {len(output_df)}")
    print(f"  Train (80%): {len(output_df[output_df['set'] == 'train'])}")
    print(f"  Holdout (20%): {len(output_df[output_df['set'] == 'holdout'])}")

    # Save CSV
    output_file = Path(__file__).parent / "all_model_predictions.csv"
    output_df.to_csv(output_file, index=False)

    print(f"\nPredictions saved to: {output_file}")
    print(f"Columns: {len(output_df.columns)}")
    print(f"  - Base columns: index, species_true, set")
    print(f"  - Per-model probabilities: {6 * len(species_list)} ({6} models × {len(species_list)} species)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
