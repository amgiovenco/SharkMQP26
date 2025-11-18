# Shark Species Classification Models - Complete Reproducibility Guide

This document provides comprehensive specifications for all shark species classification models, enabling exact reproduction with identical results.

---

## Overview

- **Dataset**: 651 total samples across 57 shark species
- **Source**: Melting curve fluorescence data (3475 temperature points per sample)
- **Data Location**: `../../data/shark_dataset.csv`
- **Random Seed**: 8 (used throughout all models)
- **Python Versions Tested**: PyTorch 2.9.0+cu130, scikit-learn, scipy

---

## 1. CNN (EfficientNet-B0) Model

**Location**: `cnn/` directory

### Data Specifications
- **Dataset Size**: 499 samples (subset of full dataset)
- **Number of Classes**: 57 shark species
- **Input Format**: Fluorescence curves converted to 2D images
- **Image Generation**: PNG images (288×216 pixels @ 96 DPI) created from temperature vs fluorescence plots
- **Image Dimensions**: 224×224 after resizing (ImageNet standard)

### Data Preprocessing & Augmentation

#### Training-Time Augmentation (enabled):
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=0, translate=(0.0, 0.03)),  # Small vertical shift only
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    AddGaussianNoise(std=0.005),  # Measurement noise
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])
```

#### Validation/Test-Time (no augmentation):
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Model Architecture
- **Base Model**: EfficientNet-B0 pretrained on ImageNet (IMAGENET1K_V1 weights)
- **Custom Classifier Head**:
  ```
  Input → Dropout(0.6217843386251581) → Linear(1280, 256) → ReLU → Dropout(0.19498440140497733) → Linear(256, 57) → Output
  ```
- **Dropout Rates**: 0.622 (first) and 0.195 (second) - optimized via Optuna
- **Input Channels**: 3 (RGB)
- **Output Classes**: 57 (shark species)

### Training Hyperparameters (Optimized)
| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.0004303702377686196 |
| Weight Decay | 4.572988042665251e-06 |
| Dropout 1 | 0.6217843386251581 |
| Dropout 2 | 0.19498440140497733 |
| Epochs | 150 (with early stopping) |
| Patience | 25 epochs |
| Optimizer | AdamW |
| Loss Function | Focal Loss |
| LR Scheduler | CosineAnnealingLR(T_max=150, eta_min=1e-6) |

### Focal Loss Parameters
- **Alpha**: 1.0
- **Gamma**: 1.2483412017424098 (focuses on hard examples)

### Training Procedure
1. **Random Seed**: Set torch.manual_seed(8), np.random.seed(8)
2. **Device**: CUDA if available, else CPU
3. **Cross-Validation**: 5-fold stratified cross-validation
4. **Data Split**: StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
5. **Model Training**: One model trained per fold
6. **Validation**: Every epoch on validation fold
7. **Best Model Selection**: Save model with highest validation accuracy
8. **Early Stopping**: Stop if no improvement for 25 epochs
9. **Learning Rate Schedule**: CosineAnnealingLR cooling to 1e-6

### Cross-Validation Results
| Fold | Val Accuracy | Test Accuracy | Epochs | Model File |
|------|--------------|---------------|--------|------------|
| 1 | 98.00% | 94.74% | 52 | efficientnet_b0_fold1_9800.pth |
| 2 | 99.00% | 95.39% | 78 | efficientnet_b0_fold2_9900.pth |
| 3 | 97.00% | 96.05% | 58 | efficientnet_b0_fold3_9700.pth |
| 4 | 99.00% | 96.05% | 102 | efficientnet_b0_fold4_9900.pth |
| 5 | 96.97% | 94.74% | 50 | efficientnet_b0_fold5_9697.pth |
| **Mean** | **97.99% ± 0.90%** | **95.39%** | **68** | — |

### Model Checkpoint Details
Each `.pth` checkpoint file contains:
```python
{
    'model_state_dict': model.state_dict(),
    'fold': fold_number,
    'val_acc': best_validation_accuracy,
    'val_metrics': {
        'accuracy': float,
        'f1': float,
        'precision': float,
        'recall': float
    },
    'history': {
        'train_loss': list,
        'train_acc': list,
        'val_loss': list,
        'val_acc': list,
        'val_f1': list,
        'val_precision': list,
        'val_recall': list,
        'learning_rates': list
    }
}
```

### Optimization Results (Latest)
After hyperparameter optimization via Optuna with macro_f1 as the optimization metric:

| Metric | Value |
|--------|-------|
| Optimization Metric | macro_f1 |
| Baseline CV Macro F1 | 96.39% |
| Best CV Macro F1 | 98.31% |
| Test Accuracy | 99.24% |
| Test Macro F1 Score | 99.40% |
| Test Loss | 0.0167 |
| **Improvement Percentage** | **191.86%** |

**Best Hyperparameters Found**:
- Batch Size: 16 (down from 32)
- Learning Rate: 0.0004303702377686196 (reduced)
- Weight Decay: 4.572988042665251e-06 (reduced)
- Dropout 1: 0.6217843386251581 (increased from 0.7)
- Dropout 2: 0.19498440140497733 (reduced from 0.5)
- Focal Gamma: 1.2483412017424098 (reduced from 1.5)

**Data Split**:
- Train/Validation: 80%
- Test Holdout: 20%
- Cross-Validation Folds: 5

### Performance Metrics (Optimized Model)
- **Mean Validation Accuracy**: 97.99% ± 0.90%
- **Mean F1-Score**: 97.73%
- **Mean Precision**: 97.87%
- **Mean Recall**: 97.99%
- **Best Fold (Test)**: Fold 3 with 96.05% accuracy
- **Test Accuracy (Optimized)**: 99.24%
- **Test Macro F1 (Optimized)**: 99.40%

### To Reproduce
1. Load image data from `../../data/train/` and `../../data/test/` directories
2. Set RANDOM_STATE=8 in all random operations
3. Apply exact augmentations listed above during training
4. Use AdamW optimizer with specified learning rate and weight decay
5. Train for up to 150 epochs with early stopping (patience=25)
6. Use CosineAnnealingLR scheduler
7. Save best model by validation accuracy per fold
8. Evaluate on test set

---

## 2. ResNet1D Model

**Location**: `resnet/` directory

### Data Specifications
- **Dataset Size**: 651 samples (full dataset)
- **Number of Classes**: 57 shark species
- **Input Format**: 1D time series (fluorescence vs temperature)
- **Sequence Length**: 3475 time steps (temperature points)
- **Input Channels**: 1

### Data Preprocessing

#### Normalization:
```python
mean = fluorescence_data.mean()  # 0.010646
std = fluorescence_data.std()    # 0.013126
fluorescence_normalized = (fluorescence_data - mean) / std
```
- Computed per dataset (applied before fold split)
- **Mean**: 0.010646
- **Std**: 0.013126

#### Data Augmentation (training only):
```python
if augment:
    # Small random noise
    noise = torch.randn_like(x) * 0.01
    x = x + noise

    # Random scaling
    scale = 1 + torch.FloatTensor([np.random.uniform(-0.05, 0.05)])
    x = x * scale

    # Random time shift (±5 steps)
    shift = np.random.randint(-5, 6)
    if shift != 0:
        x = torch.roll(x, shifts=shift, dims=1)
```
- **Augmentation Flag**: False (disabled in main training)
- Noise std: 0.01
- Scaling range: ±5%
- Time shift range: ±5 steps

### Model Architecture

**ResNet1D (Residual Neural Network for 1D sequences)**

```
Input (1, 3475)
    ↓
Conv1d(1, 80, kernel_size=7, stride=2, padding=3) + BatchNorm + ReLU + MaxPool
    ↓
Layer1: 2× ResidualBlock1D(80→80)
    ↓
Layer2: 2× ResidualBlock1D(80→160, stride=2)
    ↓
Layer3: 2× ResidualBlock1D(160→320, stride=2)
    ↓
Layer4: 2× ResidualBlock1D(320→640, stride=2)
    ↓
AdaptiveAvgPool1d(1) → Flatten
    ↓
Dropout(0.2080) → Linear(640, 57)
    ↓
Output (57 classes)
```

**Residual Block (1D)**:
- Conv1d(in_channels, out_channels, kernel_size=3, stride, padding=1) + BatchNorm + ReLU
- Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) + BatchNorm
- Skip connection with optional downsample for dimension changes

### Training Hyperparameters (Optimized)
| Parameter | Baseline | Optimized |
|-----------|----------|-----------|
| Initial Filters | 80 | 128 |
| Dropout | 0.208 | 0.2298 |
| Learning Rate | 0.000431 | 0.0001464 |
| Batch Size | 16 | 16 |
| Weight Decay | 0.000156 | 2.86e-05 |
| Epochs | 200 (with early stopping) | 200 (with early stopping) |
| Patience | 15 epochs | 15 epochs |
| Optimizer | Adam | Adam |
| Loss Function | CrossEntropyLoss | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |

### LR Scheduler Details (ReduceLROnPlateau)
```python
ReduceLROnPlateau(
    mode='min',
    factor=0.5,
    patience=5
)
```
- Monitors validation loss
- Reduces LR by 50% if no improvement for 5 epochs

### Cross-Validation Setup
- **CV Type**: StratifiedKFold
- **Number of Folds**: 5
- **Shuffle**: True
- **Random State**: 8
- **Stratification**: Preserves species distribution across folds

### Optimization Results
After hyperparameter optimization via Optuna with macro_f1 as the optimization metric:

| Metric | Value |
|--------|-------|
| Optimization Metric | macro_f1 |
| Baseline CV Macro F1 | 76.40% |
| Best CV Macro F1 | 85.08% |
| **Improvement Percentage** | **8.68%** |

**Best Hyperparameters Found**:
- Initial Filters: 128 (increased from 80)
- Dropout: 0.2297762256462723 (slightly increased from 0.208)
- Learning Rate: 0.0001464213993082976 (reduced from 0.000431)
- Batch Size: 16 (unchanged)
- Weight Decay: 2.8607285116257016e-05 (reduced from 0.000156)

### Cross-Validation Results (Baseline)
| Fold | Val Accuracy | Epochs | Train Set | Val Set | Model File |
|------|--------------|--------|-----------|---------|------------|
| 1 | 95.42% | 99 | 520 | 131 | resnet1d_fold0_9542.pth |
| 2 | 94.62% | 92 | 521 | 130 | resnet1d_fold1_9462.pth |
| 3 | 94.62% | 95 | 521 | 130 | resnet1d_fold2_9462.pth |
| 4 | 96.92% | 102 | 521 | 130 | resnet1d_fold3_9692.pth |
| 5 | 94.62% | 119 | 521 | 130 | resnet1d_fold4_9462.pth |
| **Mean** | **95.24% ± 0.90%** | **101** | — | — | — |

### Training Procedure Details
1. Create SubsetRandomSampler for each fold's train/val split
2. Normalize data using mean/std computed from entire dataset
3. Create fresh model instance for each fold
4. Initialize optimizer with specified LR and weight decay
5. Training loop:
   - Forward pass through model
   - Compute CrossEntropyLoss
   - Backward pass, optimizer step
   - Update scheduler with validation loss
   - Track best validation accuracy
   - Early stopping if no improvement for 15 epochs
6. Save best model state for each fold

### Model Checkpoint Details
Each `.pth` file contains just the model state dict:
```python
torch.save(best_model_state, filename)
```

### To Reproduce
1. Load shark_dataset.csv from `../../data/`
2. Extract all columns except "Species" as fluorescence values
3. Compute mean and std from entire dataset (0.010646, 0.013126)
4. Normalize all samples using these values
5. Create StratifiedKFold split with n_splits=5, shuffle=True, random_state=8
6. For each fold:
   - Initialize ResNet1D with initial_filters=80, dropout=0.208
   - Use Adam optimizer with lr=0.000431, weight_decay=0.000156
   - Use ReduceLROnPlateau scheduler
   - Train for up to 200 epochs with patience=15
   - Save best model by validation accuracy
7. Load and evaluate on test set

---

## 3. RandomForest (ExtraTrees) Model

**Location**: `randomforest/` directory

### Data Specifications
- **Dataset Size**: 651 samples
- **Number of Classes**: 57 shark species
- **Test/Train Split**: 80/20 (random_state=8, stratified)
- **Train Set Size**: 520 samples
- **Test Set Size**: 131 samples

### Feature Engineering

**16 Hand-Crafted Features** extracted from fluorescence curves:

1. **max**: Maximum fluorescence value
2. **min**: Minimum fluorescence value
3. **mean**: Mean fluorescence
4. **std**: Standard deviation of fluorescence
5. **auc**: Area under curve (Simpson integration)
6. **centroid**: Center of mass (weighted by temperature)
7. **temp_peak**: Temperature at peak fluorescence
8. **fwhm**: Full width at half maximum (count of points > 0.5*max)
9. **rise_time**: Number of indices to reach peak
10. **decay_time**: Number of indices from peak to end
11. **auc_left**: Area under curve from start to peak
12. **auc_right**: Area under curve from peak to end
13. **asymmetry**: auc_left / auc_right ratio
14. **fwhm_rise_ratio**: fwhm / rise_time (interaction feature)
15. **peak_temp_std**: temp_peak × std (interaction feature)
16. **asymmetry_fwhm**: asymmetry × fwhm (interaction feature)

Additional derived: **rise_decay_ratio** (rise_time / decay_time)

### Model Specifications

**Algorithm**: ExtraTreesClassifier (Extremely Randomized Trees)

| Parameter | Value |
|-----------|-------|
| n_estimators | 900 |
| max_depth | 40 |
| min_samples_split | 4 |
| min_samples_leaf | 2 |
| max_features | 0.5 |
| class_weight | 'balanced_subsample' |
| random_state | 8 |
| n_jobs | -1 (all CPU cores) |

### Training Procedure
1. Load CSV data
2. Drop species with < 2 samples for stratification validity
3. Engineer all 16 features from raw fluorescence curves
4. Train/test split with stratification (80/20, seed=8)
5. Train ExtraTreesClassifier with specified hyperparameters
6. Evaluate on test set
7. Perform 5-fold cross-validation on full engineered feature set

### Cross-Validation Results (5-Fold)
| Fold | Accuracy |
|------|----------|
| 1 | 87.79% |
| 2 | 85.38% |
| 3 | 90.00% |
| 4 | 87.69% |
| 5 | 88.46% |
| **Mean** | **87.86% ± 1.49%** |
| **Std Dev** | **1.49%** |

### Test Set Performance
- **Test Accuracy**: 87.02%
- **Macro Avg F1**: 0.82
- **Weighted Avg F1**: 0.86

### Optimization Results
The ExtraTreesClassifier achieved stable performance across all folds with excellent consistency:
- **Mean Accuracy**: 87.86% ± 1.49%
- **Standard Deviation**: 0.01490
- **Range**: 85.38% - 90.00%
- **Best Fold**: Fold 3 with 90.00%

### Feature Importance (Top 10)
1. fwhm (0.1177)
2. fwhm_rise_ratio (0.1127)
3. temp_peak (0.0852)
4. asymmetry_fwhm (0.0852)
5. decay_time (0.0839)
6. rise_time (0.0837)
7. asymmetry (0.0726)
8. rise_decay_ratio (0.0600)
9. max (0.0500)
10. std (0.0481)

### Model Files
- `extratrees_optimized_8702.pkl` (199MB) - Final trained model
- `extratrees_cv_results.json` - 5-fold CV results
- `optimization_results.json` - Hyperparameter optimization history

### To Reproduce
1. Load shark_dataset.csv
2. Drop species with < 2 samples
3. Engineer 16 features using scipy.integrate.simpson and numpy operations
4. Train/test split with stratify=y, test_size=0.2, random_state=8
5. Train ExtraTreesClassifier with exact hyperparameters
6. Evaluate on test set
7. Run 5-fold StratifiedKFold(n_splits=5, shuffle=True, random_state=8) for CV

---

## 4. Statistical Model (ExtraTrees + Feature Engineering)

**Location**: `statistics/` directory

### Data Specifications
- **Dataset Size**: 651 samples
- **Number of Classes**: 57 shark species
- **Test/Train Split**: 80/20 (random_state=8, stratified)

### Data Preprocessing

**Curve Smoothing & Normalization**:
```python
def preprocess_curve(x, y):
    # Savitzky-Golay smoothing
    win = max(7, int(round(1.5 / dx)) | 1)  # ~1.5°C window
    y_smooth = savgol_filter(y, window_length=win, polyorder=3, mode="interp")

    # Baseline removal: fit quadratic through low points (30th percentile)
    q = np.quantile(y_smooth, 0.3)
    mask = y_smooth <= q
    coeffs = np.polyfit(x[mask], y_smooth[mask], deg=2)
    baseline = np.polyval(coeffs, x)
    y_baseline = y_smooth - baseline

    # Normalize using 99th percentile
    scale = np.quantile(y_baseline, 0.99)
    y_norm = y_baseline / scale if scale > 0 else y_baseline
    y_norm = np.maximum(y_norm, 0.0)

    return y_norm
```

### Feature Engineering (36 Total Features)

#### Basic Statistics (7):
1. **mean**: E[Y]
2. **std**: σ = √E[(Y-μ)²]
3. **min**: inf{y_i}
4. **max**: sup{y_i}
5. **range**: max - min
6. **skewness**: γ₁ = E[(Y-μ)³]/σ³
7. **kurtosis**: E[(Y-μ)⁴]/σ⁴ - 3

#### Derivatives (5):
8. **max_slope**: max|∂y/∂x|
9. **mean_abs_slope**: E[|∂y/∂x|]
10. **slope_std**: σ(∂y/∂x)
11. **max_curvature**: max|∂²y/∂x²|
12. **mean_abs_curvature**: E[|∂²y/∂x²|]

#### Peaks (4):
13. **n_peaks**: Count of local maxima
14. **max_prominence**: Peak height above baseline
15. **mean_prominence**: E[prominence]
16. **peak_max_x**: x-coordinate of global maximum

#### Regional Stats: Left/Middle/Right (9):
17-19. **y_left_{mean,std,max}**: First third of domain
20-22. **y_middle_{mean,std,max}**: Middle third
23-25. **y_right_{mean,std,max}**: Last third

#### Quartiles (4):
26. **q25**: First quartile Q₁
27. **q50**: Median
28. **q75**: Third quartile Q₃
29. **iqr**: Q₃ - Q₁

#### FFT Frequency Domain (11):
30-34. **fft_power_{0-4}**: Top 5 frequency components
35. **fft_total_power**: Sum of all frequency powers (Parseval's theorem)
36. **fft_entropy**: Shannon entropy H = -Σ_k p_k log(p_k)

### Feature Selection
- **All Features Tested**: 36
- **Best k Features**: 18 (optimal by validation accuracy)
- **Selected Features**:
  1. peak_max_x
  2. max_slope
  3. y_middle_std
  4. mean_abs_curvature
  5. fft_power_4
  6. mean_abs_slope
  7. max_curvature
  8. range
  9. fft_entropy
  10. max
  11. y_middle_max
  12. fft_power_2
  13. fft_power_1
  14. fft_power_0
  15. fft_power_3
  16. y_right_max
  17. slope_std
  18. y_left_max

### Model Specifications

**Base Model**: ExtraTreesClassifier

| Parameter | Value |
|-----------|-------|
| n_estimators | 1700 |
| max_depth | None (unlimited) |
| min_samples_split | 9 |
| min_samples_leaf | 1 |
| max_features | 0.7 |
| class_weight | 'balanced' |
| random_state | 8 |
| n_jobs | -1 |

**Calibration**: CalibratedClassifierCV
- Method: 'isotonic'
- CV: 3-fold

### Cross-Validation Results (5-Fold)
| Fold | Accuracy |
|------|----------|
| 1 | 95.42% |
| 2 | 94.62% |
| 3 | 93.85% |
| 4 | 97.69% |
| 5 | 95.38% |
| **Mean** | **95.39% ± 1.29%** |

### Test Set Performance
- **Accuracy**: 96.18%
- **Macro Avg Precision**: 0.9719
- **Macro Avg Recall**: 0.9649
- **Macro Avg F1**: 0.9583

### Feature Importance (Top 10)
1. peak_max_x (0.1375)
2. y_middle_std (0.0528)
3. max_slope (0.0507)
4. n_peaks (0.0496)
5. y_middle_max (0.0450)
6. max (0.0448)
7. range (0.0441)
8. fft_entropy (0.0371)
9. fft_power_2 (0.0364)
10. fft_power_3 (0.0351)

### Model Files
- `trained_model.pkl` - Calibrated classifier
- `base_extratrees.pkl` - Base ExtraTree model
- `model_bundle.pkl` - Complete model bundle
- `model_metadata.pkl` - Model metadata and configuration
- `feature_importance.csv` - All 36 features and importances
- `per_class_metrics.csv` - Per-species precision/recall/F1

### To Reproduce
1. Load shark_dataset.csv
2. Preprocess all curves (smoothing, baseline removal, normalization)
3. Extract 36 features for each sample
4. Rank features by importance on full ExtraTree model
5. Select top 18 features
6. Train/test split with stratify=y, test_size=0.2, random_state=8
7. Train ExtraTreesClassifier on top 18 features
8. Wrap with CalibratedClassifierCV(method='isotonic', cv=3)
9. Evaluate on test set
10. Run 5-fold StratifiedKFold(n_splits=5, shuffle=True, random_state=8) for CV

---

## 5. Gaussian Curve Fitting Model

**Location**: `gaussian/` directory

### Data Specifications
- **Dataset Size**: 651 samples
- **Number of Classes**: 57 shark species
- **Number of Features**: 12 (extracted from Gaussian fits)

### Data Preprocessing

**Curve Preparation**:
```python
# Savitzky-Golay smoothing with ~1.5°C window
win = max(7, int(round(1.5 / dx)) | 1)
y_smooth = savgol_filter(y, window_length=win, polyorder=3, mode="interp")

# Quadratic baseline removal (fit through 30th percentile points)
q = np.quantile(y_smooth, 0.3)
mask = y_smooth <= q
baseline = np.polyval(np.polyfit(x[mask], y_smooth[mask], deg=2), x)
y_baseline = y_smooth - baseline

# Normalize using 99th percentile
scale = np.quantile(y_baseline, 0.99)
y_norm = y_baseline / scale if scale > 0 else y_baseline
y_norm = np.maximum(y_norm, 0.0)
```

**Decimation**: Downsample by factor of 6 for faster fitting (DECIMATE_STEP=6)

### Gaussian Curve Fitting

**Gaussian Model**:
```python
def gaussian(x, amp, mu, sigma):
    return amp * exp(-0.5 * ((x - mu) / sigma)²)

def gaussian_sum(x, *p):
    # Fit sum of k Gaussians (k parameters triplets: amp, mu, sigma)
    y = zeros_like(x)
    for i in range(0, len(p), 3):
        amp, mu, sigma = p[i:i+3]
        y += gaussian(x, amp, mu, abs(sigma))
    return y
```

**Optimal K Selection**:
- **K Range Tested**: 1 to 6 Gaussians
- **Selection Criterion**: Bayesian Information Criterion (BIC)
- **BIC Formula**: BIC = log(n)*n_params + n*log(RSS/n)

**Curve Fitting Details**:
- Algorithm: scipy.optimize.curve_fit
- Max iterations: 15,000
- Bounds: Determined from peak detection
- Peak seeding: Prominence-based detection with fallback

### Feature Extraction from Gaussian Fits

For each sample, extract 12 features from the best-fit Gaussian parameters:

**Peak 1 Features** (largest amplitude):
1. **peak1_mu**: Location of peak 1 (temperature °C)
2. **peak1_amp**: Amplitude of peak 1
3. **peak1_sigma**: Width (σ) of peak 1

**Peak 2 Features** (second largest):
4. **peak2_mu**: Location of peak 2
5. **peak2_amp**: Amplitude of peak 2
6. **peak2_sigma**: Width of peak 2

**Interaction Features**:
7. **delta_mu_12**: peak1_mu - peak2_mu (temperature difference)
8. **amp_ratio_12**: peak1_amp / peak2_amp (amplitude ratio)

**Integral Features**:
9. **total_area**: Sum of peak areas = Σ(amp_i × sigma_i × √(2π))

**Asymmetry**:
10. **asym_0p5C**: Asymmetry around main peak within ±0.5°C

**Model Selection**:
11. **best_K**: Number of Gaussians selected (1-6)

**Aggregation**:
12. **total_amp**: peak1_amp + peak2_amp

### Model Specifications

**Base Model**: RandomForestClassifier

| Parameter | Value |
|-----------|-------|
| n_estimators | 800 |
| random_state | 8 |
| class_weight | 'balanced_subsample' |
| max_depth | None |
| min_samples_leaf | 1 |
| n_jobs | -1 |

**Calibration**: CalibratedClassifierCV
- Method: 'isotonic'
- CV: 3-fold (or n_splits-1 for small folds)

### Cross-Validation Results (5-Fold)

| Fold | Accuracy | Model File |
|------|----------|------------|
| 1 | 90.08% | gaussian_fold1_9008.pkl |
| 2 | 89.23% | gaussian_fold2_8923.pkl |
| 3 | 89.23% | gaussian_fold3_8923.pkl |
| 4 | 88.46% | gaussian_fold4_8846.pkl |
| 5 | 87.69% | gaussian_fold5_8769.pkl |
| **Mean** | **88.94% ± 0.81%** | — |

### Test Set Performance (Best Fold: Fold 1)
- **Accuracy**: 90.08%
- **Top-3 Accuracy**: 95.4%
- **Abstain@0.60**: Coverage 84.73%, Accepted accuracy 95.5%
- **Macro Avg F1**: Varies by species

### Model Files
- `gaussian_fold1_9008.pkl` to `gaussian_fold5_8769.pkl` - Fold-specific models
- `gaussian_peak_features_all.csv` - Extracted Gaussian features for all samples
- `optimization_results.json` - Hyperparameter optimization results

### To Reproduce
1. Load shark_dataset.csv
2. Preprocess all curves (smoothing, baseline removal, normalization)
3. Decimate by factor of 6 (DECIMATE_STEP=6)
4. For each sample:
   - Fit Gaussians with k=1 to k=6
   - Select best k by BIC
   - Extract 12 features from parameters
5. Train 5 RandomForest models using StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
6. Wrap each with CalibratedClassifierCV(method='isotonic', cv=3)
7. Evaluate each fold on its validation set
8. Save all fold models

---

## 6. Rule-Based Model

**Location**: `rule_based/` directory

### Data Specifications
- **Dataset Size**: 651 samples (full dataset)
- **Number of Classes**: 57 shark species
- **Input Format**: 1D time series (fluorescence vs temperature)
- **Sequence Length**: 3475 time points
- **Input Channels**: 1

### Data Preprocessing

**Curve Features Extraction** (14 total features):

1. **ymax**: Maximum baseline-adjusted fluorescence
2. **tmax**: Temperature at peak fluorescence
3. **auc**: Area under curve (trapezoid integration)
4. **centroid**: Center of mass (weighted by temperature)
5. **fwhm**: Full width at half maximum
6. **rise**: Rise time (10%→90% of peak)
7. **decay**: Decay time (90%→10% of peak)
8. **auc_left**: Area from start to peak
9. **auc_right**: Area from peak to end
10. **asym**: Asymmetry ratio ((auc_right - auc_left) / auc)
11. **mean**: Mean fluorescence (raw)
12. **std**: Standard deviation of fluorescence (raw)
13. **max**: Maximum fluorescence (raw)
14. **min**: Minimum fluorescence (raw)

**Baseline Computation**:
- Robust baseline: median of first 5% of points
- Baseline-adjusted fluorescence: y_adjusted = y - baseline (clipped at 0)

### Model Architecture

**Base Model**: ExtraTreesClassifier (Extremely Randomized Trees)

| Parameter | Value |
|-----------|-------|
| n_estimators | 790 |
| max_depth | 15 |
| min_samples_leaf | 1 |
| max_features | None (all features) |
| random_state | 8 |
| n_jobs | -1 (all CPU cores) |

**Preprocessing Pipeline**:
- StandardScaler (fit on training data, applied to all splits)

### Training Procedure

**Cross-Validation Setup**:
- **CV Type**: StratifiedKFold
- **Number of Folds**: 5
- **Data Split**: 60% train, 20% validation, 20% test per fold
- **Random State**: 8
- **Stratification**: Preserves species distribution

**Training Steps for Each Fold**:
1. Create 5-fold stratified split on full dataset (651 samples)
2. For each fold:
   - 80% → further split to 60% train (419 samples) + 20% val (131 samples)
   - 20% → test set (131 samples)
3. Fit StandardScaler on training data
4. Train ExtraTreesClassifier on scaled training data
5. Evaluate on validation and test sets
6. Calculate per-class probability thresholds on validation set (F1-optimized)

### Cross-Validation Results
| Fold | Val Accuracy | Test Accuracy | Model File |
|------|--------------|---------------|------------|
| 1 | 98.08% | 96.18% | rulebased_9808.pkl |
| 2 | 97.14% | 97.14% | rulebased_9714.pkl |
| 3 | 95.24% | 95.24% | rulebased_9524.pkl |
| 4 | 96.19% | 96.19% | rulebased_9619.pkl |
| 5 | 96.92% | 96.92% | — |
| **Mean** | **96.71% ± 1.18%** | **96.33% ± 0.74%** | — |

### Per-Class Threshold Rule

**Threshold Mechanism**:
- For each class c, optimize threshold on validation set to maximize F1 (one-vs-rest)
- During prediction: accept class c only if P(c) ≥ threshold[c]
- Optional margin rule: accept only if top1 - top2 ≥ margin (default: 0.0)

### Model Files
- `rulebased_9808.pkl` to `rulebased_9619.pkl` - Fold-specific models (train_final_model.py)
- `models/rulebased_final.pkl` - Final model trained on 100% data
- `optimization_results.json` - Hyperparameter optimization history
- `rule_based.py` - Core implementation

### To Reproduce
1. Load shark_dataset.csv from `../../data/`
2. Extract all columns except "Species" as fluorescence values
3. Engineer 14 features from each curve:
   - Compute baseline as median of first 5%
   - Extract peak (ymax, tmax), width (fwhm), areas, and asymmetry
   - Calculate rise/decay times at 10%, 50%, 90% levels
4. Create StratifiedKFold split with n_splits=5, shuffle=True, random_state=8
5. For each fold:
   - Split 80% into 60% train + 20% val (further stratified split with test_size=0.25, random_state=8)
   - Fit StandardScaler on training data
   - Train ExtraTreesClassifier with exact hyperparameters
   - Compute per-class F1-optimized thresholds on validation set
   - Evaluate on test set
   - Save model with accuracy in filename
6. For final model: train on 100% data with same hyperparameters

---

## General Notes for All Models

### Common Parameters Across All Models
- **Random Seed**: 8 (for reproducibility)
- **Data Source**: `../../data/shark_dataset.csv`
- **Number of Classes**: 57 shark species
- **Total Dataset Size**: 651 samples (CNN uses 499)

### Reproducibility Checklist
When recreating any model:

- [ ] Set RANDOM_STATE/random_seed to 8 everywhere
- [ ] Use StratifiedKFold for all cross-validations
- [ ] Apply exact preprocessing steps in order
- [ ] Use exact hyperparameters from tables above
- [ ] Use exact train/test split ratio and random state
- [ ] Use exact data augmentation (if applicable)
- [ ] Use exact loss function and optimizer
- [ ] Use exact learning rate scheduler
- [ ] Save models with exact filename format
- [ ] Verify cross-validation fold structure matches

### Performance Summary Table

| Model | Architecture | Features | Val Accuracy | Test Accuracy | Optimization | Best Metric |
|-------|--------------|----------|--------------|---------------|--------------|------------|
| **CNN** (Optimized) | EfficientNet-B0 | Images (224×224) | 97.99% ± 0.90% | **99.24%** | Macro F1: 96.39% → 98.31% (+191.86%) | **Test: 99.24%** |
| **ResNet1D** (Optimized) | 1D ResNet-18 | Time series (3475) | 95.24% ± 0.90% | — | Macro F1: 76.40% → 85.08% (+8.68%) | Baseline fold 4: 96.92% val |
| **RandomForest** | ExtraTrees (900) | 16 engineered | 87.86% ± 1.49% | 87.02% | Stable CV | Fold 3: 90.00% |
| Statistics | ExtraTrees (1700) + cal | 18 selected/36 total | 95.39% ± 1.29% | 96.18% | — | Best fold: 97.69% |
| Gaussian | RandomForest (800) + cal | 12 from Gaussian fits | 88.94% ± 0.81% | 90.08% | — | Fold 1: 90.08% |
| Rule-Based | ExtraTrees (790) | 14 engineered | 96.71% ± 1.18% | 96.33% | — | Fold 1: 98.08% val |

---

## File Organization

```
models/
├── cnn/
│   ├── train_efficientnetb0.ipynb
│   ├── optimize_model.py
│   ├── efficientnet_b0_fold{1-5}_*.pth
│   ├── efficientnet_final_results.json
│   ├── scripts/
│   ├── results/
│   └── optuna_studies/
├── resnet/
│   ├── resnet.ipynb
│   ├── optimize_resnet_colab.ipynb
│   ├── resnet1d_fold{0-4}_*.pth
│   ├── results/
│   └── optuna_studies/
├── randomforest/
│   ├── randomforest.ipynb
│   ├── optimize_rf.py
│   ├── extratrees_optimized_8702.pkl
│   ├── extratrees_cv_results.json
│   ├── results/
│   └── optuna_studies/
├── statistics/
│   ├── StatisticalModel.ipynb
│   ├── optimize_stats.py
│   ├── trained_model.pkl
│   ├── model_bundle.pkl
│   ├── model_metadata.pkl
│   ├── results/
│   └── optuna_studies/
├── gaussian/
│   ├── GaussianCurve.ipynb
│   ├── gaussian_peak_features_all.csv
│   ├── gaussian_fold{1-5}_*.pkl
│   ├── results/
│   └── optuna_studies/
├── rule_based/
│   ├── rule_based.py
│   ├── train_final_model.py
│   ├── train_cv_model.py
│   ├── optimize_model.py
│   ├── rulebased_{fold}_*.pkl
│   ├── models/
│   │   └── rulebased_final.pkl
│   ├── optimization_results.json
│   ├── results/
│   └── optuna_studies/
└── MODEL_REPRODUCIBILITY_GUIDE.md (this file)
```

---

## Contact & Updates

This guide documents the exact specifications as of the last training run. If models are retrained with these specifications, identical results should be achieved.

All models use `random_state=8` for reproducibility. Any deviation from these specifications may result in different accuracy metrics.
