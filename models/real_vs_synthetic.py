"""
Comprehensive comparison of ALL models trained on:
1. Real data only
2. Real + Synthetic data (synthetic only in training folds)

5-fold stratified CV, seed=8
Models: Random Forest, Gaussian, Rule-based, Statistics, CNN (EfficientNet), ResNet1D
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import io
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from scipy.integrate import simpson
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.stats import entropy
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
try:
    from PIL import Image
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch available! Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("PyTorch not available - skipping CNN and ResNet models")

# ===========================
# CONFIGURATION
# ===========================
RANDOM_STATE = 8
N_SPLITS = 5
REAL_DATA_PATH = "../data/shark_dataset.csv"
SYNTHETIC_DATA_PATH = "../syntheticDataGeneration/synthetic_only.csv"
SPECIES_COL = "Species"
RESULTS_DIR = Path("./comparison_results_all_models")
RESULTS_DIR.mkdir(exist_ok=True)

# Training params for deep learning
CNN_EPOCHS = 150
RESNET_EPOCHS = 100
CNN_BATCH_SIZE = 32
RESNET_BATCH_SIZE = 16
CNN_EARLY_STOPPING_PATIENCE = 25
RESNET_EARLY_STOPPING_PATIENCE = 25

print("="*80)
print("COMPLETE MODEL COMPARISON: REAL VS REAL+SYNTHETIC")
print("="*80)
print(f"Random seed: {RANDOM_STATE}")
print(f"CV folds: {N_SPLITS}")
print(f"Real data: {REAL_DATA_PATH}")
print(f"Synthetic data: {SYNTHETIC_DATA_PATH}")
print(f"Results directory: {RESULTS_DIR}")
print(f"PyTorch models: {'ENABLED' if TORCH_AVAILABLE else 'DISABLED'}")
print("="*80 + "\n")

# ===========================
# LOAD DATA
# ===========================
print("Loading datasets...")
real_data = pd.read_csv(REAL_DATA_PATH)
synthetic_data = pd.read_csv(SYNTHETIC_DATA_PATH)

print(f"  Real data: {len(real_data)} samples, {real_data[SPECIES_COL].nunique()} species")
print(f"  Synthetic data: {len(synthetic_data)} samples, {synthetic_data[SPECIES_COL].nunique()} species")

# Filter species with <2 samples (for stratification)
real_counts = real_data[SPECIES_COL].value_counts()
valid_species = real_counts[real_counts >= 2].index
real_data = real_data[real_data[SPECIES_COL].isin(valid_species)].reset_index(drop=True)
synthetic_data = synthetic_data[synthetic_data[SPECIES_COL].isin(valid_species)].reset_index(drop=True)

print(f"  After filtering: {len(real_data)} real samples, {len(synthetic_data)} synthetic samples")
print()

# ===========================
# SKLEARN FEATURE ENGINEERING
# ===========================

# Random Forest Features
def rf_feature_engineering(df):
    features = pd.DataFrame()
    temps = df.columns.astype(float)

    features['max'] = df.max(axis=1)
    features['min'] = df.min(axis=1)
    features['mean'] = df.mean(axis=1)
    features['std'] = df.std(axis=1)
    features['auc'] = df.apply(lambda row: simpson(row, temps), axis=1)
    features['centroid'] = df.apply(lambda row: np.sum(row*temps)/np.sum(row), axis=1)
    features['temp_peak'] = df.apply(lambda row: temps[np.argmax(row)], axis=1)
    features['fwhm'] = df.apply(lambda row: np.sum(row > 0.5*row.max()), axis=1)
    features['rise_time'] = df.apply(lambda row: np.argmax(row), axis=1)
    features['decay_time'] = df.apply(lambda row: len(row) - np.argmax(row[::-1]), axis=1)
    features['auc_left'] = df.apply(lambda row: simpson(row[:np.argmax(row)+1], temps[:np.argmax(row)+1]), axis=1)
    features['auc_right'] = df.apply(lambda row: simpson(row[np.argmax(row):], temps[np.argmax(row):]), axis=1)
    features['asymmetry'] = features['auc_left'] / (features['auc_right'] + 1e-8)
    features['fwhm_rise_ratio'] = features['fwhm'] / (features['rise_time'] + 1e-8)
    features['peak_temp_std'] = features['temp_peak'] * features['std']
    features['asymmetry_fwhm'] = features['asymmetry'] * features['fwhm']
    features['rise_decay_ratio'] = features['rise_time'] / (features['decay_time'] + 1e-8)

    return features

# [Gaussian, Rule-based, and Statistics feature functions - keeping them as before]
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)

def gaussian_sum(x, *p):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(p), 3):
        amp, mu, sigma = p[i:i+3]
        y += gaussian(x, float(amp), float(mu), abs(float(sigma)))
    return y

def preprocess_curve_gaussian(x, y):
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(1.5 / dx)) | 1)
    if win >= len(y):
        win = max(7, (len(y)//2)*2 - 1)
    y_s = savgol_filter(y, window_length=max(7, win), polyorder=3, mode="interp")
    q = np.quantile(y_s, 0.3)
    mask = y_s <= q
    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_s[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_b = y_s - baseline
    else:
        y_b = y_s - np.min(y_s)
    scale = np.quantile(y_b, 0.99)
    if scale > 0:
        y_b = y_b / scale
    y_b = np.maximum(y_b, 0.0)
    return y_b

def decimate(x, y, step=6):
    return x[::step], y[::step]

def seed_peaks(x, y, k):
    spread = max(np.quantile(y, 0.90) - np.quantile(y, 0.10), 1e-6)
    prom = spread * 0.15
    peaks, props = find_peaks(y, prominence=prom, distance=max(1, len(x)//150))
    if len(peaks) == 0:
        peaks = np.argsort(y)[::-1][:k]
        prominences = y[peaks] - np.min(y)
    else:
        prominences = props["prominences"]
    if len(peaks) > 0:
        w_idx = peak_widths(y, peaks, rel_height=0.5)[0]
        w_c = w_idx * (x[1] - x[0])
    else:
        w_c = np.array([(x[-1]-x[0])/(3*k)]*k)
    min_w = max(0.2, 4*(x[1]-x[0]))
    max_w = (x[-1]-x[0]) / 3.0
    if np.ndim(w_c) == 0: w_c = np.array([w_c])
    w_c = np.clip(w_c, min_w, max_w)
    order = np.argsort(prominences)[::-1] if len(peaks) else np.arange(len(peaks))
    peaks = peaks[order][:k]
    w_c = w_c[order][:k]
    sort_lr = np.argsort(peaks)
    return peaks[sort_lr], w_c[sort_lr]

def fit_k(x, y, k):
    peaks, w_c = seed_peaks(x, y, k)
    p0, lo, hi = [], [], []
    y_max = max(np.max(y), 1e-6)
    for j, pk in enumerate(peaks):
        mu0 = float(x[pk])
        amp0 = float(max(y[pk], 1e-6))
        sigma0 = float(max(w_c[j] / (2*np.sqrt(2*np.log(2))), (x[1]-x[0])*2))
        p0 += [amp0, mu0, sigma0]
        lo += [0.0, mu0 - 3*sigma0, (x[1]-x[0])*1e-3]
        hi += [y_max*5 + 1e-6, mu0 + 3*sigma0, (x[-1]-x[0])]
    try:
        popt, _ = curve_fit(gaussian_sum, x, y, p0=p0, bounds=(lo, hi), maxfev=15000)
        return popt
    except:
        return None

def BIC(n_params, rss, n):
    return np.log(n)*n_params + n*np.log(rss/n + 1e-12)

def fit_best_K(x, y, K=(1, 6)):
    best = None
    for k in range(K[0], K[1]+1):
        try:
            popt = fit_k(x, y, k)
            if popt is None:
                continue
            yhat = gaussian_sum(x, *popt)
            rss = float(np.sum((y - yhat)**2))
            bic = float(BIC(3*k, rss, len(x)))
            if best is None or bic < best["bic"]:
                best = {"k": k, "popt": popt, "bic": bic, "rss": rss}
        except Exception:
            pass
    return best

def extract_gaussian_features(row_vals, X_axis):
    y0 = preprocess_curve_gaussian(X_axis, np.asarray(row_vals, float))
    x, y = decimate(X_axis, y0, step=6)
    best = fit_best_K(x, y, K=(1, 6))
    if best is None:
        return {f"f{i}": 0.0 for i in range(30)}
    popt = best["popt"]
    peaks = [{"amp": float(popt[i]), "mu": float(popt[i+1]), "sigma": abs(float(popt[i+2]))}
             for i in range(0, len(popt), 3)]
    peaks.sort(key=lambda d: d["amp"], reverse=True)
    feats = {}
    for i in range(3):
        if i < len(peaks):
            feats[f"peak{i+1}_mu"] = peaks[i]["mu"]
            feats[f"peak{i+1}_amp"] = peaks[i]["amp"]
            feats[f"peak{i+1}_sigma"] = peaks[i]["sigma"]
        else:
            feats[f"peak{i+1}_mu"] = 0.0
            feats[f"peak{i+1}_amp"] = 0.0
            feats[f"peak{i+1}_sigma"] = 0.0
    feats["delta_mu_12"] = feats["peak1_mu"] - feats["peak2_mu"]
    feats["delta_mu_23"] = feats["peak2_mu"] - feats["peak3_mu"]
    feats["amp_ratio_12"] = (feats["peak1_amp"]+1e-9)/(feats["peak2_amp"]+1e-9)
    feats["amp_ratio_23"] = (feats["peak2_amp"]+1e-9)/(feats["peak3_amp"]+1e-9)
    feats["sigma_ratio_12"] = (feats["peak1_sigma"]+1e-9)/(feats["peak2_sigma"]+1e-9)
    feats["total_amp"] = sum(p["amp"] for p in peaks)
    feats["total_area"] = sum(p["amp"] * p["sigma"] * np.sqrt(2*np.pi) for p in peaks)
    feats["mean"] = float(np.mean(y0))
    feats["std"] = float(np.std(y0))
    feats["max"] = float(np.max(y0))
    feats["min"] = float(np.min(y0))
    feats["range"] = feats["max"] - feats["min"]
    feats["auc"] = float(np.trapz(y0))
    return feats

def gaussian_feature_engineering(df, X_axis):
    feat_list = []
    for i in range(len(df)):
        if i % 100 == 0:
            print(f"    Gaussian: {i}/{len(df)}")
        feats = extract_gaussian_features(df.iloc[i].values, X_axis)
        feat_list.append(feats)
    return pd.DataFrame(feat_list).fillna(0.0)

# Rule-based Features
def _curve_features(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    k = max(1, int(0.05 * len(y)))
    baseline = np.median(y[:k])
    yb = y - baseline
    yb = np.clip(yb, 0.0, None)
    idx_max = int(np.argmax(yb))
    ymax = float(yb[idx_max])
    tmax = float(t[idx_max])
    auc = float(np.trapezoid(yb, t))
    centroid = float(np.trapezoid(yb * t, t) / (auc + 1e-12))
    half = 0.5 * ymax
    above = np.where(yb >= half)[0]
    fwhm = float(t[above[-1]] - t[above[0]]) if above.size > 0 else 0.0
    def cross(level):
        idx = np.where(yb >= level)[0]
        return (int(idx[0]) if idx.size else idx_max), (int(idx[-1]) if idx.size else idx_max)
    lo, hi = 0.1 * ymax, 0.9 * ymax
    lo_i1, hi_i1 = cross(lo)[0], cross(hi)[0]
    lo_i2, hi_i2 = cross(lo)[1], cross(hi)[1]
    rise_time = float(t[max(hi_i1, lo_i1)] - t[min(hi_i1, lo_i1)]) if ymax > 0 else 0.0
    decay_time = float(t[max(lo_i2, hi_i2)] - t[min(lo_i2, hi_i2)]) if ymax > 0 else 0.0
    auc_left = float(np.trapezoid(yb[:idx_max + 1], t[:idx_max + 1]))
    auc_right = float(np.trapezoid(yb[idx_max:], t[idx_max:]))
    asymmetry = float((auc_right - auc_left) / (auc + 1e-12))
    mean_val = float(np.mean(y))
    std_val = float(np.std(y))
    max_val = float(np.max(y))
    min_val = float(np.min(y))
    return np.array([
        ymax, tmax, auc, centroid, fwhm, rise_time, decay_time,
        auc_left, auc_right, asymmetry, mean_val, std_val, max_val, min_val
    ], dtype=float)

def rb_feature_engineering(df):
    t = df.columns.astype(float).to_numpy()
    M = df.to_numpy(float)
    F = np.vstack([_curve_features(M[i, :], t) for i in range(M.shape[0])])
    names = [
        "ymax", "tmax", "auc", "centroid", "fwhm", "rise", "decay",
        "auc_left", "auc_right", "asym", "mean", "std", "max", "min"
    ]
    return pd.DataFrame(F, columns=names)

# Statistics Features
def preprocess_curve_stats(x, y):
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

def extract_stats_features(x, y):
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
    feat["cv"] = feat["std"] / (feat["mean"] + 1e-10)
    feat["peak_to_mean_ratio"] = feat["max"] / (feat["mean"] + 1e-10)
    return feat

def stats_feature_engineering(df, X_axis, x_proc):
    feat_list = []
    for i in range(len(df)):
        if i % 100 == 0:
            print(f"    Statistics: {i}/{len(df)}")
        f = extract_stats_features(X_axis, x_proc[i])
        feat_list.append(f)
    feat_df = pd.DataFrame(feat_list).fillna(0.0)
    top_18_features = [
        'peak_max_x', 'max_slope', 'y_middle_std', 'max', 'range', 'y_middle_max',
        'fft_power_1', 'fft_power_4', 'fft_power_0', 'fft_power_2', 'fft_entropy',
        'mean_abs_curvature', 'fft_power_3', 'y_middle_mean', 'y_right_max',
        'slope_std', 'mean_abs_slope', 'std'
    ]
    return feat_df[top_18_features]

# ===========================
# PYTORCH MODELS
# ===========================

if TORCH_AVAILABLE:
    # CNN/EfficientNet Model
    def _generate_image(temps: np.ndarray, values: np.ndarray) -> Image.Image:
        try:
            fig, ax = plt.subplots(figsize=(3, 2.25), dpi=96)
            ax.plot(temps, values, linewidth=1.5, color='steelblue')
            ax.set_xlim(temps.min(), temps.max())
            ax.set_xlabel('temperature')
            ax.set_ylabel('fluorescence')
            ax.grid(True, alpha=0.3)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=96)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            plt.close(fig)
            return img
        except Exception as e:
            print(f"Failed to generate image: {e}")
            return None

    class FluorescenceImageDataset(Dataset):
        def __init__(self, images: list, y_encoded: np.ndarray, transform=None):
            self.images = images
            self.y = y_encoded
            self.transform = transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            img = self.images[idx]
            label = self.y[idx]
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            return img, label

    class CNNModel(nn.Module):
        def __init__(self, num_classes, dropout1=0.5, dropout2=0.3, hidden_size=256):
            super(CNNModel, self).__init__()
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout1),
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout2),
                nn.Linear(hidden_size, num_classes)
            )

        def forward(self, x):
            return self.model(x)

    # ResNet1D Model
    class ResidualBlock1D(nn.Module):
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
        def __init__(self, num_classes, input_channels=1, initial_filters=80, dropout=0.2):
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
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

# ===========================
# TRAINING FUNCTIONS
# ===========================

def train_sklearn_model(model_name, X_real, y_real, X_synthetic=None, y_synthetic=None):
    """Train sklearn models"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")

    results = {
        "model": model_name,
        "real_only": {},
        "real_plus_synthetic": {}
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Get unique classes for confusion matrix
    unique_classes = sorted(np.unique(y_real))
    class_names = unique_classes

    # REAL ONLY
    print(f"\n[1/2] Training on REAL data only...")
    fold_acc_real = []
    fold_f1_real = []
    fold_conf_matrices_real = []
    all_y_true_real = []
    all_y_pred_real = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real, y_real), 1):
        X_train, X_test = X_real[train_idx], X_real[test_idx]
        y_train, y_test = y_real[train_idx], y_real[test_idx]

        model = create_sklearn_model(model_name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        fold_acc_real.append(acc)
        fold_f1_real.append(f1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        fold_conf_matrices_real.append(cm)

        # Collect all predictions for overall metrics
        all_y_true_real.extend(y_test)
        all_y_pred_real.extend(y_pred)

        print(f"  Fold {fold_idx}/{N_SPLITS} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # Aggregate confusion matrix
    agg_cm_real = np.sum(fold_conf_matrices_real, axis=0)

    # Per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_y_true_real, all_y_pred_real, labels=unique_classes, zero_division=0
    )

    results["real_only"]["accuracy_mean"] = float(np.mean(fold_acc_real))
    results["real_only"]["accuracy_std"] = float(np.std(fold_acc_real))
    results["real_only"]["f1_mean"] = float(np.mean(fold_f1_real))
    results["real_only"]["f1_std"] = float(np.std(fold_f1_real))
    results["real_only"]["fold_accuracies"] = [float(x) for x in fold_acc_real]
    results["real_only"]["fold_f1s"] = [float(x) for x in fold_f1_real]
    results["real_only"]["confusion_matrix"] = agg_cm_real.tolist()
    results["real_only"]["confusion_matrix_classes"] = [str(c) for c in class_names]

    # Per-species accuracy
    results["real_only"]["per_species_metrics"] = {}
    for i, cls in enumerate(class_names):
        per_class_acc = 0.0
        if agg_cm_real[i].sum() > 0:
            per_class_acc = agg_cm_real[i, i] / agg_cm_real[i].sum()

        results["real_only"]["per_species_metrics"][str(cls)] = {
            "accuracy": float(per_class_acc),
            "accuracy_pct": float(per_class_acc * 100),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_per_class[i]),
            "samples": int(support[i])
        }

    print(f"\n  Real Only Results:")
    print(f"    Accuracy: {results['real_only']['accuracy_mean']:.4f} ± {results['real_only']['accuracy_std']:.4f}")
    print(f"    F1 Score: {results['real_only']['f1_mean']:.4f} ± {results['real_only']['f1_std']:.4f}")
    print(f"  Per-Species Accuracy:")
    for cls, metrics in results["real_only"]["per_species_metrics"].items():
        print(f"    {cls}: {metrics['accuracy_pct']:.2f}% (n={metrics['samples']})")

    # REAL + SYNTHETIC
    if X_synthetic is not None:
        print(f"\n[2/2] Training on REAL + SYNTHETIC data...")
        fold_acc_synth = []
        fold_f1_synth = []
        fold_conf_matrices_synth = []
        all_y_true_synth = []
        all_y_pred_synth = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real, y_real), 1):
            X_train_real, X_test = X_real[train_idx], X_real[test_idx]
            y_train_real, y_test = y_real[train_idx], y_real[test_idx]

            X_train = np.vstack([X_train_real, X_synthetic])
            y_train = np.concatenate([y_train_real, y_synthetic])

            model = create_sklearn_model(model_name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            fold_acc_synth.append(acc)
            fold_f1_synth.append(f1)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
            fold_conf_matrices_synth.append(cm)

            # Collect all predictions
            all_y_true_synth.extend(y_test)
            all_y_pred_synth.extend(y_pred)

            print(f"  Fold {fold_idx}/{N_SPLITS} | Acc: {acc:.4f} | F1: {f1:.4f} | Train: {len(X_train)}")

        # Aggregate confusion matrix
        agg_cm_synth = np.sum(fold_conf_matrices_synth, axis=0)

        # Per-class metrics
        precision_s, recall_s, f1_per_class_s, support_s = precision_recall_fscore_support(
            all_y_true_synth, all_y_pred_synth, labels=unique_classes, zero_division=0
        )

        results["real_plus_synthetic"]["accuracy_mean"] = float(np.mean(fold_acc_synth))
        results["real_plus_synthetic"]["accuracy_std"] = float(np.std(fold_acc_synth))
        results["real_plus_synthetic"]["f1_mean"] = float(np.mean(fold_f1_synth))
        results["real_plus_synthetic"]["f1_std"] = float(np.std(fold_f1_synth))
        results["real_plus_synthetic"]["fold_accuracies"] = [float(x) for x in fold_acc_synth]
        results["real_plus_synthetic"]["fold_f1s"] = [float(x) for x in fold_f1_synth]
        results["real_plus_synthetic"]["confusion_matrix"] = agg_cm_synth.tolist()
        results["real_plus_synthetic"]["confusion_matrix_classes"] = [str(c) for c in class_names]

        # Per-species accuracy
        results["real_plus_synthetic"]["per_species_metrics"] = {}
        for i, cls in enumerate(class_names):
            per_class_acc = 0.0
            if agg_cm_synth[i].sum() > 0:
                per_class_acc = agg_cm_synth[i, i] / agg_cm_synth[i].sum()

            results["real_plus_synthetic"]["per_species_metrics"][str(cls)] = {
                "accuracy": float(per_class_acc),
                "accuracy_pct": float(per_class_acc * 100),
                "precision": float(precision_s[i]),
                "recall": float(recall_s[i]),
                "f1": float(f1_per_class_s[i]),
                "samples": int(support_s[i])
            }

        print(f"\n  Real + Synthetic Results:")
        print(f"    Accuracy: {results['real_plus_synthetic']['accuracy_mean']:.4f} ± {results['real_plus_synthetic']['accuracy_std']:.4f}")
        print(f"    F1 Score: {results['real_plus_synthetic']['f1_mean']:.4f} ± {results['real_plus_synthetic']['f1_std']:.4f}")
        print(f"  Per-Species Accuracy:")
        for cls, metrics in results["real_plus_synthetic"]["per_species_metrics"].items():
            print(f"    {cls}: {metrics['accuracy_pct']:.2f}% (n={metrics['samples']})")

        acc_diff = results["real_plus_synthetic"]["accuracy_mean"] - results["real_only"]["accuracy_mean"]
        f1_diff = results["real_plus_synthetic"]["f1_mean"] - results["real_only"]["f1_mean"]

        print(f"\n  Improvement:")
        print(f"    Accuracy: {acc_diff:+.4f} ({acc_diff/results['real_only']['accuracy_mean']*100:+.2f}%)")
        print(f"    F1 Score: {f1_diff:+.4f} ({f1_diff/results['real_only']['f1_mean']*100:+.2f}%)")

        results["improvement"] = {
            "accuracy_absolute": float(acc_diff),
            "accuracy_relative_pct": float(acc_diff/results['real_only']['accuracy_mean']*100),
            "f1_absolute": float(f1_diff),
            "f1_relative_pct": float(f1_diff/results['real_only']['f1_mean']*100)
        }

        # Per-species improvement
        results["per_species_improvement"] = {}
        for cls in class_names:
            cls_str = str(cls)
            real_acc = results["real_only"]["per_species_metrics"][cls_str]["accuracy_pct"]
            synth_acc = results["real_plus_synthetic"]["per_species_metrics"][cls_str]["accuracy_pct"]
            improvement = synth_acc - real_acc
            results["per_species_improvement"][cls_str] = float(improvement)

    return results

def create_sklearn_model(model_name):
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=900, max_depth=40, min_samples_split=3, min_samples_leaf=1,
            max_features=0.7, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == "Gaussian":
        return ExtraTreesClassifier(
            n_estimators=700, max_depth=20, min_samples_split=4, min_samples_leaf=1,
            max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == "Rule-based":
        return make_pipeline(
            StandardScaler(),
            ExtraTreesClassifier(
                n_estimators=790, min_samples_leaf=1, max_depth=15, max_features=None,
                random_state=RANDOM_STATE, n_jobs=-1
            )
        )
    elif model_name == "Statistics":
        return RandomForestClassifier(
            n_estimators=1500, max_depth=None, min_samples_split=6, min_samples_leaf=1,
            max_features=0.7, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        )

if TORCH_AVAILABLE:
    def train_pytorch_model(model_name, data_real, data_synthetic, num_classes, epochs, batch_size, lr=0.001):
        """Train PyTorch models (CNN or ResNet1D)"""
        print(f"\n{'='*80}")
        print(f"Training {model_name}")
        print(f"{'='*80}")

        results = {
            "model": model_name,
            "real_only": {},
            "real_plus_synthetic": {}
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        # Unpack data
        if model_name == "CNN (EfficientNet)":
            images_real, y_real = data_real
            images_synthetic, y_synthetic = data_synthetic
        else:  # ResNet1D
            X_real, y_real = data_real
            X_synthetic, y_synthetic = data_synthetic

        # Label encoding
        le = LabelEncoder()
        y_real_encoded = le.fit_transform(y_real)

        # REAL ONLY
        print(f"\n[1/2] Training on REAL data only...")
        fold_acc_real = []
        fold_f1_real = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(y_real_encoded, y_real_encoded), 1):
            if model_name == "CNN (EfficientNet)":
                train_images = [images_real[i] for i in train_idx]
                test_images = [images_real[i] for i in test_idx]
                train_labels = y_real_encoded[train_idx]
                test_labels = y_real_encoded[test_idx]

                # Training transform with augmentation
                transform_train = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                # Validation transform without augmentation
                transform_val = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                train_dataset = FluorescenceImageDataset(train_images, train_labels, transform=transform_train)
                test_dataset = FluorescenceImageDataset(test_images, test_labels, transform=transform_val)
            else:  # ResNet1D - Normalize data
                X_train_np = X_real[train_idx]
                X_test_np = X_real[test_idx]

                # Normalize: (X - mean) / (std + eps)
                X_train_mean = X_train_np.mean()
                X_train_std = X_train_np.std() + 1e-8
                X_train_norm = (X_train_np - X_train_mean) / X_train_std
                X_test_norm = (X_test_np - X_train_mean) / X_train_std

                X_train = torch.FloatTensor(X_train_norm).unsqueeze(1)
                X_test = torch.FloatTensor(X_test_norm).unsqueeze(1)
                y_train = torch.LongTensor(y_real_encoded[train_idx])
                y_test = torch.LongTensor(y_real_encoded[test_idx])

                train_dataset = TensorDataset(X_train, y_train)
                test_dataset = TensorDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Train
            if model_name == "CNN (EfficientNet)":
                model = CNNModel(num_classes, dropout1=0.5, dropout2=0.3, hidden_size=256).to(DEVICE)
            else:
                model = ResNet1D(num_classes, initial_filters=80, dropout=0.2).to(DEVICE)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01 if model_name == "CNN (EfficientNet)" else 0.0001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            # Early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            patience = CNN_EARLY_STOPPING_PATIENCE if model_name == "CNN (EfficientNet)" else RESNET_EARLY_STOPPING_PATIENCE

            for epoch in range(epochs):
                # Train
                model.train()
                train_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validate on training set (to monitor overfitting)
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()

                val_loss /= len(train_loader)
                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

            # Evaluate
            model.eval()
            y_pred_all = []
            y_true_all = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    y_pred_all.extend(predicted.cpu().numpy())
                    y_true_all.extend(labels.numpy())

            acc = accuracy_score(y_true_all, y_pred_all)
            f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

            fold_acc_real.append(acc)
            fold_f1_real.append(f1)

            print(f"  Fold {fold_idx}/{N_SPLITS} | Acc: {acc:.4f} | F1: {f1:.4f}")

        results["real_only"]["accuracy_mean"] = float(np.mean(fold_acc_real))
        results["real_only"]["accuracy_std"] = float(np.std(fold_acc_real))
        results["real_only"]["f1_mean"] = float(np.mean(fold_f1_real))
        results["real_only"]["f1_std"] = float(np.std(fold_f1_real))
        results["real_only"]["fold_accuracies"] = [float(x) for x in fold_acc_real]
        results["real_only"]["fold_f1s"] = [float(x) for x in fold_f1_real]

        print(f"\n  Real Only Results:")
        print(f"    Accuracy: {results['real_only']['accuracy_mean']:.4f} ± {results['real_only']['accuracy_std']:.4f}")
        print(f"    F1 Score: {results['real_only']['f1_mean']:.4f} ± {results['real_only']['f1_std']:.4f}")

        # REAL + SYNTHETIC
        print(f"\n[2/2] Training on REAL + SYNTHETIC data...")
        fold_acc_synth = []
        fold_f1_synth = []

        y_synthetic_encoded = le.transform(y_synthetic)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(y_real_encoded, y_real_encoded), 1):
            if model_name == "CNN (EfficientNet)":
                train_images_real = [images_real[i] for i in train_idx]
                test_images = [images_real[i] for i in test_idx]
                train_images = train_images_real + images_synthetic
                train_labels = np.concatenate([y_real_encoded[train_idx], y_synthetic_encoded])
                test_labels = y_real_encoded[test_idx]
                train_size = len(train_labels)

                # Training transform with augmentation
                transform_train = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                # Validation transform without augmentation
                transform_val = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                train_dataset = FluorescenceImageDataset(train_images, train_labels, transform=transform_train)
                test_dataset = FluorescenceImageDataset(test_images, test_labels, transform=transform_val)
            else:  # ResNet1D - Normalize data
                X_train_real = X_real[train_idx]
                X_train = np.vstack([X_train_real, X_synthetic])
                X_test = X_real[test_idx]

                # Normalize: (X - mean) / (std + eps)
                X_train_mean = X_train.mean()
                X_train_std = X_train.std() + 1e-8
                X_train_norm = (X_train - X_train_mean) / X_train_std
                X_test_norm = (X_test - X_train_mean) / X_train_std

                X_train_t = torch.FloatTensor(X_train_norm).unsqueeze(1)
                X_test_t = torch.FloatTensor(X_test_norm).unsqueeze(1)
                y_train = torch.LongTensor(np.concatenate([y_real_encoded[train_idx], y_synthetic_encoded]))
                y_test = torch.LongTensor(y_real_encoded[test_idx])
                train_size = len(y_train)

                train_dataset = TensorDataset(X_train_t, y_train)
                test_dataset = TensorDataset(X_test_t, y_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Train
            if model_name == "CNN (EfficientNet)":
                model = CNNModel(num_classes, dropout1=0.5, dropout2=0.3, hidden_size=256).to(DEVICE)
            else:
                model = ResNet1D(num_classes, initial_filters=80, dropout=0.2).to(DEVICE)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01 if model_name == "CNN (EfficientNet)" else 0.0001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            # Early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            patience = CNN_EARLY_STOPPING_PATIENCE if model_name == "CNN (EfficientNet)" else RESNET_EARLY_STOPPING_PATIENCE

            for epoch in range(epochs):
                # Train
                model.train()
                train_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validate on training set (to monitor overfitting)
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()

                val_loss /= len(train_loader)
                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

            # Evaluate
            model.eval()
            y_pred_all = []
            y_true_all = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    y_pred_all.extend(predicted.cpu().numpy())
                    y_true_all.extend(labels.numpy())

            acc = accuracy_score(y_true_all, y_pred_all)
            f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

            fold_acc_synth.append(acc)
            fold_f1_synth.append(f1)

            print(f"  Fold {fold_idx}/{N_SPLITS} | Acc: {acc:.4f} | F1: {f1:.4f} | Train: {train_size}")

        results["real_plus_synthetic"]["accuracy_mean"] = float(np.mean(fold_acc_synth))
        results["real_plus_synthetic"]["accuracy_std"] = float(np.std(fold_acc_synth))
        results["real_plus_synthetic"]["f1_mean"] = float(np.mean(fold_f1_synth))
        results["real_plus_synthetic"]["f1_std"] = float(np.std(fold_f1_synth))
        results["real_plus_synthetic"]["fold_accuracies"] = [float(x) for x in fold_acc_synth]
        results["real_plus_synthetic"]["fold_f1s"] = [float(x) for x in fold_f1_synth]

        print(f"\n  Real + Synthetic Results:")
        print(f"    Accuracy: {results['real_plus_synthetic']['accuracy_mean']:.4f} ± {results['real_plus_synthetic']['accuracy_std']:.4f}")
        print(f"    F1 Score: {results['real_plus_synthetic']['f1_mean']:.4f} ± {results['real_plus_synthetic']['f1_std']:.4f}")

        acc_diff = results["real_plus_synthetic"]["accuracy_mean"] - results["real_only"]["accuracy_mean"]
        f1_diff = results["real_plus_synthetic"]["f1_mean"] - results["real_only"]["f1_mean"]

        print(f"\n  Improvement:")
        print(f"    Accuracy: {acc_diff:+.4f} ({acc_diff/results['real_only']['accuracy_mean']*100:+.2f}%)")
        print(f"    F1 Score: {f1_diff:+.4f} ({f1_diff/results['real_only']['f1_mean']*100:+.2f}%)")

        results["improvement"] = {
            "accuracy_absolute": float(acc_diff),
            "accuracy_relative_pct": float(acc_diff/results['real_only']['accuracy_mean']*100),
            "f1_absolute": float(f1_diff),
            "f1_relative_pct": float(f1_diff/results['real_only']['f1_mean']*100)
        }

        return results

# ===========================
# VISUALIZATION
# ===========================

def plot_confusion_matrices(all_results):
    """Plot confusion matrices for each model"""
    num_models = len(all_results)

    for model_idx, result in enumerate(all_results):
        model_name = result["model"]

        # Skip models without confusion matrices (e.g., PyTorch models)
        if "confusion_matrix" not in result["real_only"]:
            print(f"  Skipped: {model_name} (no confusion matrix data)")
            continue

        # Real only
        cm_real = np.array(result["real_only"]["confusion_matrix"])
        classes = result["real_only"]["confusion_matrix_classes"]

        # Real + Synthetic
        cm_synth = np.array(result["real_plus_synthetic"]["confusion_matrix"])

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Confusion Matrices: {model_name}', fontsize=16, fontweight='bold')

        # Normalize for better visualization
        cm_real_norm = cm_real.astype('float') / (cm_real.sum(axis=1, keepdims=True) + 1e-8)
        cm_synth_norm = cm_synth.astype('float') / (cm_synth.sum(axis=1, keepdims=True) + 1e-8)

        # Real only
        ax = axes[0]
        im = ax.imshow(cm_real_norm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
        ax.set_title('Real Data Only (Normalized)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        plt.colorbar(im, ax=ax)

        # Add counts as text
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f'{int(cm_real[i, j])}\n({cm_real_norm[i, j]:.2%})',
                       ha='center', va='center',
                       color='white' if cm_real_norm[i, j] > 0.5 else 'black',
                       fontsize=8)

        # Real + Synthetic
        ax = axes[1]
        im = ax.imshow(cm_synth_norm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
        ax.set_title('Real + Synthetic Data (Normalized)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        plt.colorbar(im, ax=ax)

        # Add counts as text
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f'{int(cm_synth[i, j])}\n({cm_synth_norm[i, j]:.2%})',
                       ha='center', va='center',
                       color='white' if cm_synth_norm[i, j] > 0.5 else 'black',
                       fontsize=8)

        plt.tight_layout()
        filename = RESULTS_DIR / f'confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

def plot_per_species_metrics(all_results):
    """Plot per-species accuracy for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Per-Species Performance Analysis', fontsize=18, fontweight='bold')

    # Get all species
    first_result = all_results[0]
    all_species = list(first_result["real_only"]["per_species_metrics"].keys())

    # Plot 1: Real accuracy per species
    ax = axes[0, 0]
    for model_idx, result in enumerate(all_results):
        species_acc = [result["real_only"]["per_species_metrics"][sp]["accuracy_pct"] for sp in all_species]
        ax.plot(all_species, species_acc, marker='o', label=result["model"], alpha=0.7, linewidth=2)
    ax.set_title('Real Data Only: Accuracy per Species', fontsize=12, fontweight='bold')
    ax.set_xlabel('Species')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Real+Synthetic accuracy per species
    ax = axes[0, 1]
    for model_idx, result in enumerate(all_results):
        species_acc = [result["real_plus_synthetic"]["per_species_metrics"][sp]["accuracy_pct"] for sp in all_species]
        ax.plot(all_species, species_acc, marker='s', label=result["model"], alpha=0.7, linewidth=2)
    ax.set_title('Real + Synthetic Data: Accuracy per Species', fontsize=12, fontweight='bold')
    ax.set_xlabel('Species')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Improvement per species (first model only)
    ax = axes[1, 0]
    if "per_species_improvement" in all_results[0]:
        result = all_results[0]
        improvements = [result["per_species_improvement"][sp] for sp in all_species]
        colors = ['green' if x >= 0 else 'red' for x in improvements]
        ax.bar(all_species, improvements, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title(f'Improvement with Synthetic Data: {result["model"]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Species')
        ax.set_ylabel('Accuracy Change (%)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add value labels
        for i, (sp, val) in enumerate(zip(all_species, improvements)):
            ax.text(i, val, f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')

    # Plot 4: Sample distribution
    ax = axes[1, 1]
    x = np.arange(len(all_species))
    width = 0.35
    real_samples = [all_results[0]["real_only"]["per_species_metrics"][sp]["samples"] for sp in all_species]
    synth_samples = [all_results[0]["real_plus_synthetic"]["per_species_metrics"][sp]["samples"] for sp in all_species]
    ax.bar(x - width/2, real_samples, width, label='Real Data Only', alpha=0.8)
    ax.bar(x + width/2, synth_samples, width, label='Real + Synthetic', alpha=0.8)
    ax.set_title('Test Sample Distribution per Species', fontsize=12, fontweight='bold')
    ax.set_xlabel('Species')
    ax.set_ylabel('Number of Samples')
    ax.set_xticks(x)
    ax.set_xticklabels(all_species, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'per_species_metrics.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {RESULTS_DIR / 'per_species_metrics.png'}")
    plt.close()

def create_per_species_csv(all_results):
    """Create detailed per-species CSV for analysis"""
    # Get all species from first result
    first_result = all_results[0]
    all_species = list(first_result["real_only"]["per_species_metrics"].keys())

    rows = []
    for result in all_results:
        model_name = result["model"]

        for species in all_species:
            real_metrics = result["real_only"]["per_species_metrics"][species]
            synth_metrics = result["real_plus_synthetic"]["per_species_metrics"][species]

            acc_improvement = synth_metrics["accuracy_pct"] - real_metrics["accuracy_pct"]
            precision_improvement = synth_metrics["precision"] - real_metrics["precision"]
            recall_improvement = synth_metrics["recall"] - real_metrics["recall"]
            f1_improvement = synth_metrics["f1"] - real_metrics["f1"]

            rows.append({
                "Model": model_name,
                "Species": species,
                "Real_Accuracy_%": round(real_metrics["accuracy_pct"], 2),
                "Real_Precision": round(real_metrics["precision"], 4),
                "Real_Recall": round(real_metrics["recall"], 4),
                "Real_F1": round(real_metrics["f1"], 4),
                "Real_Samples": real_metrics["samples"],
                "Synth_Accuracy_%": round(synth_metrics["accuracy_pct"], 2),
                "Synth_Precision": round(synth_metrics["precision"], 4),
                "Synth_Recall": round(synth_metrics["recall"], 4),
                "Synth_F1": round(synth_metrics["f1"], 4),
                "Synth_Samples": synth_metrics["samples"],
                "Accuracy_Change_%": round(acc_improvement, 2),
                "Precision_Change": round(precision_improvement, 4),
                "Recall_Change": round(recall_improvement, 4),
                "F1_Change": round(f1_improvement, 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'per_species_detailed_metrics.csv', index=False)
    print(f"  Saved: {RESULTS_DIR / 'per_species_detailed_metrics.csv'}")

    return df

def plot_comparison_results(all_results):
    model_names = [r["model"] for r in all_results]
    real_acc = [r["real_only"]["accuracy_mean"] for r in all_results]
    real_f1 = [r["real_only"]["f1_mean"] for r in all_results]
    synth_acc = [r["real_plus_synthetic"]["accuracy_mean"] for r in all_results]
    synth_f1 = [r["real_plus_synthetic"]["f1_mean"] for r in all_results]

    real_acc_std = [r["real_only"]["accuracy_std"] for r in all_results]
    real_f1_std = [r["real_only"]["f1_std"] for r in all_results]
    synth_acc_std = [r["real_plus_synthetic"]["accuracy_std"] for r in all_results]
    synth_f1_std = [r["real_plus_synthetic"]["f1_std"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Real vs Real+Synthetic: All Models Comparison', fontsize=18, fontweight='bold')

    x = np.arange(len(model_names))
    width = 0.35

    # Accuracy
    ax = axes[0, 0]
    ax.bar(x - width/2, real_acc, width, label='Real Only', yerr=real_acc_std, capsize=5, alpha=0.8, color='steelblue')
    ax.bar(x + width/2, synth_acc, width, label='Real + Synthetic', yerr=synth_acc_std, capsize=5, alpha=0.8, color='coral')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # F1 Score
    ax = axes[0, 1]
    ax.bar(x - width/2, real_f1, width, label='Real Only', yerr=real_f1_std, capsize=5, alpha=0.8, color='steelblue')
    ax.bar(x + width/2, synth_f1, width, label='Real + Synthetic', yerr=synth_f1_std, capsize=5, alpha=0.8, color='coral')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Improvement heatmap
    ax = axes[1, 0]
    improvements = np.array([[r["improvement"]["accuracy_relative_pct"],
                              r["improvement"]["f1_relative_pct"]] for r in all_results])
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Accuracy', 'F1 Score'])
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_title('% Improvement with Synthetic Data', fontsize=14, fontweight='bold')

    for i in range(len(model_names)):
        for j in range(2):
            text = ax.text(j, i, f'{improvements[i, j]:.2f}%',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label='% Change')

    # Fold-wise performance
    ax = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for i, result in enumerate(all_results):
        real_folds = result["real_only"]["fold_accuracies"]
        synth_folds = result["real_plus_synthetic"]["fold_accuracies"]
        folds = range(1, N_SPLITS + 1)

        ax.plot(folds, real_folds, 'o-', label=f'{result["model"]} (Real)', alpha=0.7, color=colors[i])
        ax.plot(folds, synth_folds, 's--', label=f'{result["model"]} (R+S)', alpha=0.7, color=colors[i])

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Fold-wise Accuracy', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'comparison_all_models.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plots to {RESULTS_DIR / 'comparison_all_models.png'}")
    plt.close()

def create_summary_table(all_results):
    rows = []
    for result in all_results:
        rows.append({
            "Model": result["model"],
            "Real Acc": f"{result['real_only']['accuracy_mean']:.4f} ± {result['real_only']['accuracy_std']:.4f}",
            "Real F1": f"{result['real_only']['f1_mean']:.4f} ± {result['real_only']['f1_std']:.4f}",
            "Real+Synth Acc": f"{result['real_plus_synthetic']['accuracy_mean']:.4f} ± {result['real_plus_synthetic']['accuracy_std']:.4f}",
            "Real+Synth F1": f"{result['real_plus_synthetic']['f1_mean']:.4f} ± {result['real_plus_synthetic']['f1_std']:.4f}",
            "Acc Δ%": f"{result['improvement']['accuracy_relative_pct']:+.2f}%",
            "F1 Δ%": f"{result['improvement']['f1_relative_pct']:+.2f}%"
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'summary_all_models.csv', index=False)
    print(f"\nSaved summary to {RESULTS_DIR / 'summary_all_models.csv'}")

    print("\n" + "="*140)
    print("SUMMARY TABLE: ALL MODELS")
    print("="*140)
    print(df.to_string(index=False))
    print("="*140)

# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == "__main__":

    X_raw_real = real_data.drop(columns=[SPECIES_COL])
    y_real = real_data[SPECIES_COL].to_numpy()
    X_raw_synthetic = synthetic_data.drop(columns=[SPECIES_COL])
    y_synthetic = synthetic_data[SPECIES_COL].to_numpy()
    temp_cols = X_raw_real.columns.astype(float).to_numpy()

    all_results = []

    # === RANDOM FOREST ===
    print("\n" + "="*80)
    print("EXTRACTING FEATURES: Random Forest")
    print("="*80)
    X_rf_real = rf_feature_engineering(X_raw_real).to_numpy(float)
    X_rf_synthetic = rf_feature_engineering(X_raw_synthetic).to_numpy(float)
    rf_results = train_sklearn_model("Random Forest", X_rf_real, y_real, X_rf_synthetic, y_synthetic)
    all_results.append(rf_results)

    # === GAUSSIAN ===
    print("\n" + "="*80)
    print("EXTRACTING FEATURES: Gaussian")
    print("="*80)
    X_gauss_real_df = gaussian_feature_engineering(X_raw_real, temp_cols)
    X_gauss_real = StandardScaler().fit_transform(X_gauss_real_df.to_numpy(float))
    X_gauss_synthetic_df = gaussian_feature_engineering(X_raw_synthetic, temp_cols)
    X_gauss_synthetic = StandardScaler().fit_transform(X_gauss_synthetic_df.to_numpy(float))
    gauss_results = train_sklearn_model("Gaussian", X_gauss_real, y_real, X_gauss_synthetic, y_synthetic)
    all_results.append(gauss_results)

    # === RULE-BASED ===
    print("\n" + "="*80)
    print("EXTRACTING FEATURES: Rule-based")
    print("="*80)
    X_rb_real = rb_feature_engineering(X_raw_real).to_numpy(float)
    X_rb_synthetic = rb_feature_engineering(X_raw_synthetic).to_numpy(float)
    rb_results = train_sklearn_model("Rule-based", X_rb_real, y_real, X_rb_synthetic, y_synthetic)
    all_results.append(rb_results)

    # === STATISTICS ===
    print("\n" + "="*80)
    print("EXTRACTING FEATURES: Statistics")
    print("="*80)
    x_proc_real = np.array([preprocess_curve_stats(temp_cols, X_raw_real.iloc[i].values.astype(float))
                            for i in range(len(X_raw_real))])
    X_stats_real = stats_feature_engineering(X_raw_real, temp_cols, x_proc_real).to_numpy(float)
    x_proc_synthetic = np.array([preprocess_curve_stats(temp_cols, X_raw_synthetic.iloc[i].values.astype(float))
                                 for i in range(len(X_raw_synthetic))])
    X_stats_synthetic = stats_feature_engineering(X_raw_synthetic, temp_cols, x_proc_synthetic).to_numpy(float)
    stats_results = train_sklearn_model("Statistics", X_stats_real, y_real, X_stats_synthetic, y_synthetic)
    all_results.append(stats_results)

    # === CNN (EFFICIENTNET) ===
    if TORCH_AVAILABLE:
        print("\n" + "="*80)
        print("GENERATING IMAGES: CNN (EfficientNet)")
        print("="*80)
        print("  Generating real images...")
        images_real = []
        for idx, row in X_raw_real.iterrows():
            if idx % 100 == 0:
                print(f"    {idx}/{len(X_raw_real)}")
            img = _generate_image(temp_cols, row.values)
            images_real.append(img)

        print("  Generating synthetic images...")
        images_synthetic = []
        for idx, row in X_raw_synthetic.iterrows():
            if idx % 50 == 0:
                print(f"    {idx}/{len(X_raw_synthetic)}")
            img = _generate_image(temp_cols, row.values)
            images_synthetic.append(img)

        num_classes = len(np.unique(y_real))
        cnn_results = train_pytorch_model("CNN (EfficientNet)",
                                         (images_real, y_real),
                                         (images_synthetic, y_synthetic),
                                         num_classes, CNN_EPOCHS, CNN_BATCH_SIZE, lr=0.001)
        all_results.append(cnn_results)

        # === RESNET1D ===
        print("\n" + "="*80)
        print("PREPARING DATA: ResNet1D")
        print("="*80)
        X_resnet_real = X_raw_real.to_numpy(float)
        X_resnet_synthetic = X_raw_synthetic.to_numpy(float)
        resnet_results = train_pytorch_model("ResNet1D",
                                            (X_resnet_real, y_real),
                                            (X_resnet_synthetic, y_synthetic),
                                            num_classes, RESNET_EPOCHS, RESNET_BATCH_SIZE, lr=0.0004)
        all_results.append(resnet_results)

    # === SAVE RESULTS ===
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    with open(RESULTS_DIR / 'results_all_models.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {RESULTS_DIR / 'results_all_models.json'}")

    print(f"\nGenerating visualizations and detailed reports...")
    plot_comparison_results(all_results)
    create_summary_table(all_results)

    print(f"\nGenerating confusion matrices...")
    plot_confusion_matrices(all_results)

    print(f"\nGenerating per-species analysis...")
    plot_per_species_metrics(all_results)
    per_species_df = create_per_species_csv(all_results)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print("\nGenerated files:")
    print("  - results_all_models.json (full results with confusion matrices)")
    print("  - summary_all_models.csv (overall metrics)")
    print("  - per_species_detailed_metrics.csv (per-species breakdown)")
    print("  - comparison_all_models.png (overall comparison charts)")
    print("  - per_species_metrics.png (per-species analysis)")
    print("  - confusion_matrix_*.png (confusion matrices for each model)")
    print("="*80)
