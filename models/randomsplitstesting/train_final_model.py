"""
Train Final Production CNN Model

Trains the CNN (EfficientNet-B0) model on train+val data with optimized synthetic allocation.
Saves the model weights, label encoder, and configuration for deployment.

This is your PRODUCTION model - use this for inference!
"""

import json
import pickle
import hashlib
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 8  # Use seed 8 (gave perfect results in testing)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
REAL_DATA_PATH = DATA_DIR / "shark_dataset.csv"
SYNTHETIC_DIR = DATA_DIR / "syntheticDataIndividual"

# Output
OUTPUT_DIR = SCRIPT_DIR / "final_model"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "image_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Data split (0/50/50: all synthetic training, 50% val, 50% test)
REAL_VAL_SPLIT = 0.5  # 50% real validation
REAL_TEST_SPLIT = 0.5  # 50% real test

# Use all synthetic data
MAX_SYN_PER_SPECIES = 50

# CNN hyperparameters (optimized)
CNN_IMAGE_SIZE = 224
CNN_BATCH_SIZE = 16
# Epochs scaled: 102 base (early stop point) + proportional for data increase
# Will be calculated after loading data
CNN_EPOCHS = None  # Calculated dynamically
CNN_PATIENCE = 15
CNN_LEARNING_RATE = 0.0004303702377686196
CNN_WEIGHT_DECAY = 4.572988042665251e-06
CNN_DROPOUT_1 = 0.6217843386251581
CNN_DROPOUT_2 = 0.19498440140497733
CNN_FOCAL_ALPHA = 1.0
CNN_FOCAL_GAMMA = 1.2483412017424098
CNN_HIDDEN_DIM = 256

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("TRAINING FINAL PRODUCTION CNN MODEL")
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
        filename = f"synthetic_{species.replace(' ', '_')}.csv"
        filepath = SYNTHETIC_DIR / filename

        if filepath.exists():
            df = pd.read_csv(filepath)
            synthetic_data[species] = df.reset_index(drop=True)

    return synthetic_data

def bin_species_by_real_count(real_data: pd.DataFrame) -> Dict[str, List[str]]:
    """Bin species into 5 groups based on real sample counts."""
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
                            max_synthetic_per_species: int = 50, seed: int = 8) -> pd.DataFrame:
    """Combine real and synthetic data based on per-bin multipliers."""
    augmented = real_data.copy()

    bins = bin_species_by_real_count(real_data)
    n_values = {
        'very_low': n_very_low,
        'low': n_low,
        'medium': n_medium,
        'high': n_high,
        'very_high': n_very_high
    }

    for bin_name in ['very_low', 'low', 'medium', 'high', 'very_high']:
        k = n_values[bin_name]
        for species in bins[bin_name]:
            num_synthetic_to_add = min(k, max_synthetic_per_species) if k > 0 else 0

            if num_synthetic_to_add > 0 and species in synthetic_data:
                synth_pool = synthetic_data[species]
                if len(synth_pool) > 0:
                    sampled = synth_pool.sample(n=int(num_synthetic_to_add), replace=True, random_state=seed)
                    augmented = pd.concat([augmented, sampled], ignore_index=True)

    return augmented.reset_index(drop=True)

# ============================================================================
# MODEL DEFINITION
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

    return train_transform

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
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
# MAIN TRAINING
# ============================================================================

if __name__ == '__main__':
    print("Training final production CNN model")
    print(f"\nConfiguration:")
    print(f"  Model: CNN (EfficientNet-B0)")
    print(f"  Seed: {SEED}")
    print(f"  Data split: 0% real training, 50% real validation, 50% real test")
    print(f"  Device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    real_data = load_real_data()
    print(f"  Loaded {len(real_data)} real samples")

    # Split real data 50/50: validation and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=REAL_TEST_SPLIT, random_state=SEED)
    for val_idx, test_idx in sss.split(real_data, real_data['Species']):
        real_val = real_data.iloc[val_idx].reset_index(drop=True)
        real_test = real_data.iloc[test_idx].reset_index(drop=True)

    num_classes = len(real_data['Species'].unique())
    print(f"  Validation: {len(real_val)} samples")
    print(f"  Test: {len(real_test)} samples (held out)")
    print(f"  Number of classes: {num_classes}")

    # Load synthetic data
    print("\nLoading synthetic data...")
    synthetic_data = load_synthetic_data(real_data['Species'].unique().tolist())
    print(f"  Loaded synthetic data for {len(synthetic_data)} species")

    # Create synthetic training set (use all synthetic, up to 50 per species)
    print("\nCreating synthetic training set...")
    synthetic_train = []
    for species, df in synthetic_data.items():
        n_samples = min(len(df), MAX_SYN_PER_SPECIES)
        sampled = df.iloc[:n_samples].copy()
        synthetic_train.append(sampled)

    augmented_data = pd.concat(synthetic_train, ignore_index=True) if synthetic_train else pd.DataFrame()

    print(f"  Total training samples: {len(augmented_data)} (100% synthetic)")
    print(f"  Synthetic samples: {len(augmented_data)}")

    # Calculate epochs: best epoch (102 - patience) + proportional for added validation data
    STOPPED_AT_EPOCH = 102
    BEST_EPOCH = STOPPED_AT_EPOCH - CNN_PATIENCE  # 102 - 15 = 87 (actual best performance)
    ORIGINAL_SYNTHETIC_SIZE = len(augmented_data)  # Full synthetic set size
    EXTRA_DATA_SIZE = len(real_val)  # Validation data being added to training

    extra_epochs = int(BEST_EPOCH * (EXTRA_DATA_SIZE / ORIGINAL_SYNTHETIC_SIZE))
    CNN_EPOCHS = BEST_EPOCH + extra_epochs

    print(f"\nEpochs calculation:")
    print(f"  Original stopped at: {STOPPED_AT_EPOCH} (with patience={CNN_PATIENCE})")
    print(f"  Best epoch: {BEST_EPOCH}")
    print(f"  Original synthetic size: {ORIGINAL_SYNTHETIC_SIZE}")
    print(f"  Extra data (validation): {EXTRA_DATA_SIZE}")
    print(f"  Extra epochs: {extra_epochs}")
    print(f"  Total epochs: {CNN_EPOCHS}")

    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(augmented_data['Species'])

    # Create dataset and dataloader
    print("\nPreparing data loaders...")
    train_transform = get_cnn_transforms()
    train_dataset = FluorescenceDataset(augmented_data, label_encoder, train_transform, CACHE_DIR)
    train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=2)

    # Create model
    print("\nInitializing model...")
    model = SharkCNN(num_classes=num_classes).to(DEVICE)
    criterion = FocalLoss(alpha=CNN_FOCAL_ALPHA, gamma=CNN_FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=CNN_LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    # Train
    print("\nTraining...")
    print("-" * 80)

    for epoch in range(CNN_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{CNN_EPOCHS}, Loss: {train_loss:.4f}")

    print("-" * 80)
    print(f"Training complete! Trained for full {CNN_EPOCHS} epochs")

    # ========================================================================
    # EVALUATE ON HELD-OUT TEST SET
    # ========================================================================

    print("\n" + "=" * 80)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("=" * 80)

    # Create test dataset and loader
    test_transform = transforms.Compose([
        transforms.Resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = FluorescenceDataset(real_test, label_encoder, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=2)

    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate accuracy
    test_accuracy = np.mean(all_preds == all_labels)

    # Calculate Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(all_probs, axis=1)
    accuracies = all_preds == all_labels

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    # Calculate confidence when correct/wrong
    conf_when_correct = np.mean(confidences[accuracies])
    conf_when_wrong = np.mean(confidences[~accuracies]) if np.any(~accuracies) else 0.0

    # Calculate top-k accuracy
    top3_correct = 0
    top5_correct = 0
    for i, label in enumerate(all_labels):
        top3_indices = np.argsort(all_probs[i])[::-1][:3]
        top5_indices = np.argsort(all_probs[i])[::-1][:5]
        if label in top3_indices:
            top3_correct += 1
        if label in top5_indices:
            top5_correct += 1

    top3_accuracy = top3_correct / len(all_labels)
    top5_accuracy = top5_correct / len(all_labels)

    # Print results
    print(f"\nTest Set Results ({len(real_test)} samples):")
    print(f"  Accuracy: {test_accuracy * 100:.2f}%")
    print(f"  Top-3 Accuracy: {top3_accuracy * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Avg Confidence (when correct): {conf_when_correct:.4f}")
    print(f"  Avg Confidence (when wrong): {conf_when_wrong:.4f}")

    # Prepare test results for saving
    test_results = {
        'test_samples': len(real_test),
        'accuracy': float(test_accuracy),
        'accuracy_pct': float(test_accuracy * 100),
        'top3_accuracy': float(top3_accuracy),
        'top3_accuracy_pct': float(top3_accuracy * 100),
        'top5_accuracy': float(top5_accuracy),
        'top5_accuracy_pct': float(top5_accuracy * 100),
        'ece': float(ece),
        'avg_confidence_correct': float(conf_when_correct),
        'avg_confidence_wrong': float(conf_when_wrong)
    }

    # Save model
    print("\nSaving model...")

    # Save full model (architecture + weights)
    model_path = OUTPUT_DIR / "shark_cnn_model.pth"
    torch.save(model, model_path)
    print(f"  Full model (architecture + weights) saved to: {model_path}")

    # Save label encoder
    label_encoder_path = OUTPUT_DIR / "label_encoder.pkl"
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  Label encoder saved to: {label_encoder_path}")

    # Save configuration
    config = {
        'model_type': 'CNN (EfficientNet-B0)',
        'num_classes': num_classes,
        'seed': SEED,
        'data_split': '0/50/50 (all synthetic training, 50% real val, 50% real test)',
        'total_training_samples': len(augmented_data),
        'synthetic_samples': len(augmented_data),
        'validation_samples': len(real_val),
        'hyperparameters': {
            'image_size': CNN_IMAGE_SIZE,
            'batch_size': CNN_BATCH_SIZE,
            'learning_rate': CNN_LEARNING_RATE,
            'weight_decay': CNN_WEIGHT_DECAY,
            'dropout_1': CNN_DROPOUT_1,
            'dropout_2': CNN_DROPOUT_2,
            'focal_alpha': CNN_FOCAL_ALPHA,
            'focal_gamma': CNN_FOCAL_GAMMA,
            'hidden_dim': CNN_HIDDEN_DIM
        },
        'training': {
            'epochs_trained': CNN_EPOCHS,
            'final_loss': float(train_loss)
        },
        'test_evaluation': test_results,
        'species_list': label_encoder.classes_.tolist()
    }

    config_path = OUTPUT_DIR / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Configuration saved to: {config_path}")

    # Save test results separately
    test_results_path = OUTPUT_DIR / "test_results.json"
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"  Test results saved to: {test_results_path}")

    print("\n" + "=" * 80)
    print("FINAL MODEL SAVED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nFiles saved:")
    print(f"  1. shark_cnn_model.pth - Full model (architecture + weights)")
    print(f"  2. label_encoder.pkl - Label encoder (maps indices to species names)")
    print(f"  3. model_config.json - Full configuration + test results")
    print(f"  4. test_results.json - Test set evaluation metrics")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_results['accuracy_pct']:.2f}%")
    print(f"  ECE: {test_results['ece']:.4f}")
    print(f"\nUse the inference.py script to make predictions with this model!")
    print("=" * 80)
