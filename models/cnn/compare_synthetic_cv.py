"""
Compare 5-fold CV with normal data vs. real + synthetic data.
Synthetic data is only added to training sets, NOT to validation/test sets.
Uses seed=8 and best hyperparameters from optimization.
"""
import numpy as np
import pandas as pd
import json
import io
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_score,
    recall_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    TORCH_AVAILABLE = False
    DEVICE = None
    print(f"Error importing required libraries: {e}")
    exit(1)

# Set random seeds for reproducibility
RANDOM_STATE = 8
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Config
REAL_DATA_PATH = "../../data/shark_dataset.csv"
SYNTHETIC_DATA_PATH = "../../data/synthetic_only.csv"
SPECIES_COL = "Species"
N_SPLITS = 5
NUM_EPOCHS = 150
PATIENCE = 25
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 1.5

# Best params from optimization_results.json
BEST_PARAMS = {
    "batch_size": 32,
    "learning_rate": 0.0020594007612475913,
    "weight_decay": 1.0083970230770894e-05,
    "dropout1": 0.7,
    "dropout2": 0.5,
}


class AddGaussianNoise(object):
    def __init__(self, std=0.005):
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


class FluorescenceImageDataset(Dataset):
    """Dataset for fluorescence curve images."""
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
    """EfficientNet-based CNN model."""
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def _generate_image(temps: np.ndarray, values: np.ndarray) -> Image.Image:
    """Generate PIL image from fluorescence curve, matching script 1's style."""
    try:
        DPI = 100
        IMAGE_SIZE = (224, 224)
        fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/DPI, IMAGE_SIZE[1]/DPI), dpi=DPI)

        # Plot the time series (matching color and linewidth)
        ax.plot(temps, values, linewidth=2, color='#2E86AB')

        # Set limits (matching ylim padding; xlim is the same)
        ax.set_xlim(temps.min(), temps.max())
        ax.set_ylim(values.min() - 0.001, values.max() + 0.001)

        # Remove axes, labels, and grid (axis off, no grid)
        ax.axis('off')

        # Remove all margins and padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save to buffer with tight bbox and no padding
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight', pad_inches=0,
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close(fig)
        return img
    except Exception as e:
        print(f"Failed to generate image: {e}")
        return None


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * accuracy_score(all_labels, all_preds)
    f1 = 100. * f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = 100. * precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = 100. * recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return running_loss / len(loader), accuracy, f1, precision, recall, all_preds, all_labels


def train_and_evaluate(train_images, y_train, val_images, y_val, num_classes, fold_idx, dataset_type):
    """Train model with validation and return metrics on validation set."""
    # Image transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.0, 0.03)),  # small vertical shift only
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.005),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    train_dataset = FluorescenceImageDataset(train_images, y_train, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=True, num_workers=0)

    val_dataset = FluorescenceImageDataset(val_images, y_val, transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=0)

    # Model
    model = CNNModel(
        num_classes=num_classes,
        dropout1=BEST_PARAMS['dropout1'],
        dropout2=BEST_PARAMS['dropout2'],
        hidden_size=256
    ).to(DEVICE)

    # Optimizer, criterion, scheduler
    optimizer = optim.AdamW(model.parameters(),
                           lr=BEST_PARAMS['learning_rate'],
                           weight_decay=BEST_PARAMS['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    # Training loop with early stopping on val_acc
    best_val_acc = 0.0
    best_val_metrics = None
    best_model_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, val_precision, val_recall, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            best_val_metrics = {
                'accuracy': val_acc / 100.0,  # Convert back to fraction for consistency
                'f1': val_f1 / 100.0,
                'precision': val_precision / 100.0,
                'recall': val_recall / 100.0
            }
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Report best validation metrics (converted to fraction)
    acc = best_val_metrics['accuracy']
    f1 = best_val_metrics['f1']
    precision = best_val_metrics['precision']
    recall = best_val_metrics['recall']

    print(f"  Fold {fold_idx:2d}/5 ({dataset_type:20s}) | Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

    # Clean up
    del model, optimizer, criterion, train_loader, val_loader

    return val_preds, acc, f1, precision, recall


# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
df_real = pd.read_csv(REAL_DATA_PATH)
df_synthetic = pd.read_csv(SYNTHETIC_DATA_PATH)

X_real = df_real.drop(columns=[SPECIES_COL])
y_real = df_real[SPECIES_COL].astype(str)

X_synthetic = df_synthetic.drop(columns=[SPECIES_COL])
y_synthetic = df_synthetic[SPECIES_COL].astype(str)

print(f"Real data shape: {X_real.shape}")
print(f"Synthetic data shape: {X_synthetic.shape}")
print(f"Classes: {len(np.unique(y_real))}")

# ============================================================================
# GENERATE IMAGES
# ============================================================================
print("\nGenerating fluorescence curve images...")
temps = X_real.columns.astype(float).values

# Generate images for real data
print("  Generating real data images...")
images_real = []
for idx, row in X_real.iterrows():
    if idx % 100 == 0:
        print(f"    {idx}/{len(X_real)}")
    img = _generate_image(temps, row.values)
    if img is None:
        print(f"Failed to generate image for real sample {idx}")
        exit(1)
    images_real.append(img)

# Generate images for synthetic data
print("  Generating synthetic data images...")
images_synthetic = []
for idx, row in X_synthetic.iterrows():
    if idx % 100 == 0:
        print(f"    {idx}/{len(X_synthetic)}")
    img = _generate_image(temps, row.values)
    if img is None:
        print(f"Failed to generate image for synthetic sample {idx}")
        exit(1)
    images_synthetic.append(img)

print(f"Generated {len(images_real)} real images and {len(images_synthetic)} synthetic images")

# ============================================================================
# ENCODE LABELS
# ============================================================================
le = LabelEncoder()
y_real_encoded = le.fit_transform(y_real)
y_synthetic_encoded = le.transform(y_synthetic)  # Use same encoder

class_names = le.classes_
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Num classes: {num_classes}")

# ============================================================================
# CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("STARTING 5-FOLD CROSS-VALIDATION (seed=8)")
print("="*80)

# Setup results directory
results_dir = Path("./results/synthetic_comparison")
results_dir.mkdir(parents=True, exist_ok=True)

# Setup CV
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Results containers
results_normal = {
    "fold_accuracies": [],
    "fold_f1_scores": [],
    "fold_precisions": [],
    "fold_recalls": [],
    "all_predictions": [],
    "all_true_labels": [],
    "fold_details": []
}

results_synthetic = {
    "fold_accuracies": [],
    "fold_f1_scores": [],
    "fold_precisions": [],
    "fold_recalls": [],
    "all_predictions": [],
    "all_true_labels": [],
    "fold_details": []
}

# ============================================================================
# TRAIN AND EVALUATE: NORMAL DATA ONLY
# ============================================================================
print("\n" + "-"*80)
print("EXPERIMENT 1: Normal Data Only (Baseline)")
print("-"*80)

fold_idx = 0
for train_idx, val_idx in skf.split(images_real, y_real_encoded):
    fold_idx += 1

    # Get subset of images (all from real data)
    train_images = [images_real[i] for i in train_idx]
    y_train = y_real_encoded[train_idx]
    val_images = [images_real[i] for i in val_idx]
    y_val = y_real_encoded[val_idx]

    # Train and evaluate
    y_pred, acc, f1, prec, rec = train_and_evaluate(
        train_images, y_train, val_images, y_val,
        num_classes, fold_idx, "Normal Data Only"
    )

    # Store metrics
    results_normal["fold_accuracies"].append(acc)
    results_normal["fold_f1_scores"].append(f1)
    results_normal["fold_precisions"].append(prec)
    results_normal["fold_recalls"].append(rec)
    results_normal["all_predictions"].extend(y_pred)
    results_normal["all_true_labels"].extend(y_val)
    results_normal["fold_details"].append({
        "fold": fold_idx,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "val_size": len(y_val)
    })

# ============================================================================
# TRAIN AND EVALUATE: REAL + SYNTHETIC DATA
# ============================================================================
print("\n" + "-"*80)
print("EXPERIMENT 2: Real + Synthetic Data (Synthetic Only in Training)")
print("-"*80)

fold_idx = 0
for train_idx, val_idx in skf.split(images_real, y_real_encoded):
    fold_idx += 1

    # Get subset of images from REAL data for train/val split
    train_images_real = [images_real[i] for i in train_idx]
    y_train_real = y_real_encoded[train_idx]
    val_images = [images_real[i] for i in val_idx]
    y_val = y_real_encoded[val_idx]

    # ADD synthetic data ONLY to training set
    train_images_combined = train_images_real + images_synthetic
    y_train_combined = np.concatenate([y_train_real, y_synthetic_encoded])

    # Train and evaluate
    y_pred, acc, f1, prec, rec = train_and_evaluate(
        train_images_combined, y_train_combined, val_images, y_val,
        num_classes, fold_idx, "Real + Synthetic (Train Only)"
    )

    # Store metrics
    results_synthetic["fold_accuracies"].append(acc)
    results_synthetic["fold_f1_scores"].append(f1)
    results_synthetic["fold_precisions"].append(prec)
    results_synthetic["fold_recalls"].append(rec)
    results_synthetic["all_predictions"].extend(y_pred)
    results_synthetic["all_true_labels"].extend(y_val)
    results_synthetic["fold_details"].append({
        "fold": fold_idx,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "val_size": len(y_val),
        "train_size_real": len(train_images_real),
        "train_size_synthetic": len(images_synthetic),
        "train_size_total": len(train_images_combined)
    })

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS SUMMARY")
print("="*80)

print("\n[NORMAL DATA ONLY]")
print(f"  Mean Accuracy: {np.mean(results_normal['fold_accuracies']):.4f} ± {np.std(results_normal['fold_accuracies']):.4f}")
print(f"  Mean F1:       {np.mean(results_normal['fold_f1_scores']):.4f} ± {np.std(results_normal['fold_f1_scores']):.4f}")
print(f"  Mean Precision: {np.mean(results_normal['fold_precisions']):.4f} ± {np.std(results_normal['fold_precisions']):.4f}")
print(f"  Mean Recall:   {np.mean(results_normal['fold_recalls']):.4f} ± {np.std(results_normal['fold_recalls']):.4f}")

print("\n[REAL + SYNTHETIC DATA]")
print(f"  Mean Accuracy: {np.mean(results_synthetic['fold_accuracies']):.4f} ± {np.std(results_synthetic['fold_accuracies']):.4f}")
print(f"  Mean F1:       {np.mean(results_synthetic['fold_f1_scores']):.4f} ± {np.std(results_synthetic['fold_f1_scores']):.4f}")
print(f"  Mean Precision: {np.mean(results_synthetic['fold_precisions']):.4f} ± {np.std(results_synthetic['fold_precisions']):.4f}")
print(f"  Mean Recall:   {np.mean(results_synthetic['fold_recalls']):.4f} ± {np.std(results_synthetic['fold_recalls']):.4f}")

# Calculate improvements
acc_improvement = np.mean(results_synthetic['fold_accuracies']) - np.mean(results_normal['fold_accuracies'])
f1_improvement = np.mean(results_synthetic['fold_f1_scores']) - np.mean(results_normal['fold_f1_scores'])

print("\n[IMPROVEMENTS (Synthetic vs Normal)]")
print(f"  Accuracy Δ: {acc_improvement:+.4f}")
print(f"  F1 Δ:       {f1_improvement:+.4f}")

# ============================================================================
# SAVE METRICS TO JSON
# ============================================================================
print("\n" + "="*80)
print("SAVING METRICS TO JSON")
print("="*80)

summary_data = {
    "configuration": {
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "num_epochs": NUM_EPOCHS,
        "best_params": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in BEST_PARAMS.items()},
        "class_names": list(class_names),
        "num_classes": num_classes
    },
    "normal_data_only": {
        "mean_accuracy": float(np.mean(results_normal['fold_accuracies'])),
        "std_accuracy": float(np.std(results_normal['fold_accuracies'])),
        "mean_f1": float(np.mean(results_normal['fold_f1_scores'])),
        "std_f1": float(np.std(results_normal['fold_f1_scores'])),
        "mean_precision": float(np.mean(results_normal['fold_precisions'])),
        "std_precision": float(np.std(results_normal['fold_precisions'])),
        "mean_recall": float(np.mean(results_normal['fold_recalls'])),
        "std_recall": float(np.std(results_normal['fold_recalls'])),
        "fold_details": results_normal['fold_details'],
        "all_true_labels": [int(x) for x in results_normal['all_true_labels']],
        "all_predictions": [int(x) for x in results_normal['all_predictions']]
    },
    "real_plus_synthetic": {
        "mean_accuracy": float(np.mean(results_synthetic['fold_accuracies'])),
        "std_accuracy": float(np.std(results_synthetic['fold_accuracies'])),
        "mean_f1": float(np.mean(results_synthetic['fold_f1_scores'])),
        "std_f1": float(np.std(results_synthetic['fold_f1_scores'])),
        "mean_precision": float(np.mean(results_synthetic['fold_precisions'])),
        "std_precision": float(np.std(results_synthetic['fold_precisions'])),
        "mean_recall": float(np.mean(results_synthetic['fold_recalls'])),
        "std_recall": float(np.std(results_synthetic['fold_recalls'])),
        "fold_details": results_synthetic['fold_details'],
        "all_true_labels": [int(x) for x in results_synthetic['all_true_labels']],
        "all_predictions": [int(x) for x in results_synthetic['all_predictions']]
    },
    "comparison": {
        "accuracy_improvement": float(acc_improvement),
        "f1_improvement": float(f1_improvement),
        "accuracy_improvement_percent": float(acc_improvement * 100),
        "f1_improvement_percent": float(f1_improvement * 100)
    }
}

summary_file = results_dir / "comparison_metrics.json"
with open(summary_file, 'w') as f:
    json.dump(summary_data, f, indent=2)

print(f"[OK] Metrics saved to {summary_file}")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Convert predictions to arrays for confusion matrices
y_true_normal = np.array(results_normal['all_true_labels'])
y_pred_normal = np.array(results_normal['all_predictions'])
y_true_synthetic = np.array(results_synthetic['all_true_labels'])
y_pred_synthetic = np.array(results_synthetic['all_predictions'])

# Helper function to generate confusion matrices with color only (no numbers)
def plot_confusion_matrices(y_true, y_pred, dataset_name, weighted=False):
    """Generate unweighted and optionally weighted confusion matrices without annotations."""
    cm = confusion_matrix(y_true, y_pred)

    if weighted:
        # Normalize by row (true label) to show percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm

    # Determine figure size and label size based on number of classes
    n_classes = len(class_names)
    if n_classes <= 10:
        figsize = (12, 10)
        label_size = 10
    elif n_classes <= 20:
        figsize = (16, 14)
        label_size = 9
    else:
        figsize = (20, 18)
        label_size = 8

    fig, ax = plt.subplots(figsize=figsize)

    # Generate heatmap without annotations
    sns.heatmap(cm_normalized, cmap='Blues' if 'Normal' in dataset_name else 'Greens',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={'label': 'Count' if not weighted else 'Normalized'},
                annot=False, fmt='d')

    matrix_type = "Normalized (Weighted)" if weighted else "Unweighted (Raw Counts)"
    ax.set_title(f'Confusion Matrix - {dataset_name}\n{matrix_type} (5-Fold CV, Seed=8)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # Rotate labels for readability
    ax.set_xticklabels(class_names, rotation=90, ha='right', fontsize=label_size)
    ax.set_yticklabels(class_names, rotation=0, fontsize=label_size)

    plt.tight_layout()
    return fig, ax

# 1. Confusion Matrix - Normal Data (Unweighted)
print("  Generating confusion matrices (Normal Data)...")
fig, ax = plot_confusion_matrices(y_true_normal, y_pred_normal, "Normal Data Only", weighted=False)
plt.savefig(results_dir / 'confusion_matrix_normal_unweighted.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: confusion_matrix_normal_unweighted.png")

# 2. Confusion Matrix - Normal Data (Weighted/Normalized)
fig, ax = plot_confusion_matrices(y_true_normal, y_pred_normal, "Normal Data Only", weighted=True)
plt.savefig(results_dir / 'confusion_matrix_normal_weighted.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: confusion_matrix_normal_weighted.png")

# 3. Confusion Matrix - Real + Synthetic (Unweighted)
print("  Generating confusion matrices (Real + Synthetic)...")
fig, ax = plot_confusion_matrices(y_true_synthetic, y_pred_synthetic, "Real + Synthetic Data", weighted=False)
plt.savefig(results_dir / 'confusion_matrix_synthetic_unweighted.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: confusion_matrix_synthetic_unweighted.png")

# 4. Confusion Matrix - Real + Synthetic (Weighted/Normalized)
fig, ax = plot_confusion_matrices(y_true_synthetic, y_pred_synthetic, "Real + Synthetic Data", weighted=True)
plt.savefig(results_dir / 'confusion_matrix_synthetic_weighted.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: confusion_matrix_synthetic_weighted.png")

# 3. Accuracy Comparison Chart
print("  Generating accuracy comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))
folds = list(range(1, N_SPLITS + 1))
x = np.arange(len(folds))
width = 0.35
bars1 = ax.bar(x - width/2, results_normal['fold_accuracies'], width, label='Normal Data Only', alpha=0.8, color='#3498db')
bars2 = ax.bar(x + width/2, results_synthetic['fold_accuracies'], width, label='Real + Synthetic', alpha=0.8, color='#2ecc71')

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy Comparison Across Folds', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(results_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: accuracy_comparison.png")

# 4. F1 Score Comparison Chart
print("  Generating F1 score comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, results_normal['fold_f1_scores'], width, label='Normal Data Only', alpha=0.8, color='#3498db')
bars2 = ax.bar(x + width/2, results_synthetic['fold_f1_scores'], width, label='Real + Synthetic', alpha=0.8, color='#2ecc71')

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('F1 Score (Macro)', fontsize=12)
ax.set_title('F1 Score Comparison Across Folds', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(results_dir / 'f1_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: f1_comparison.png")

# 5. Summary Metrics Comparison (Mean ± Std)
print("  Generating summary metrics comparison chart...")
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
normal_means = [
    np.mean(results_normal['fold_accuracies']),
    np.mean(results_normal['fold_f1_scores']),
    np.mean(results_normal['fold_precisions']),
    np.mean(results_normal['fold_recalls'])
]
normal_stds = [
    np.std(results_normal['fold_accuracies']),
    np.std(results_normal['fold_f1_scores']),
    np.std(results_normal['fold_precisions']),
    np.std(results_normal['fold_recalls'])
]
synthetic_means = [
    np.mean(results_synthetic['fold_accuracies']),
    np.mean(results_synthetic['fold_f1_scores']),
    np.mean(results_synthetic['fold_precisions']),
    np.mean(results_synthetic['fold_recalls'])
]
synthetic_stds = [
    np.std(results_synthetic['fold_accuracies']),
    np.std(results_synthetic['fold_f1_scores']),
    np.std(results_synthetic['fold_precisions']),
    np.std(results_synthetic['fold_recalls'])
]

x_pos = [0, 1]
for idx, (metric, ax) in enumerate(zip(metrics, axes)):
    ax.bar(x_pos, [normal_means[idx], synthetic_means[idx]],
           yerr=[normal_stds[idx], synthetic_stds[idx]],
           capsize=10, alpha=0.8, color=['#3498db', '#2ecc71'],
           edgecolor='black', linewidth=1.5)
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Normal', 'Real+Synthetic'], fontsize=11)
    ax.set_ylim([0.85, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    ax.text(0, normal_means[idx] + normal_stds[idx] + 0.01,
            f'{normal_means[idx]:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(1, synthetic_means[idx] + synthetic_stds[idx] + 0.01,
            f'{synthetic_means[idx]:.4f}', ha='center', fontsize=10, fontweight='bold')

fig.suptitle('Mean ± Std Metrics Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(results_dir / 'summary_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: summary_metrics_comparison.png")

# 6. Per-Class Performance Comparison
print("  Generating per-class performance chart...")
from sklearn.metrics import precision_recall_fscore_support

prec_normal, rec_normal, f1_normal, _ = precision_recall_fscore_support(
    y_true_normal, y_pred_normal, labels=np.arange(num_classes), zero_division=0
)
prec_synthetic, rec_synthetic, f1_synthetic, _ = precision_recall_fscore_support(
    y_true_synthetic, y_pred_synthetic, labels=np.arange(num_classes), zero_division=0
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(num_classes)
width = 0.35

# Precision
axes[0].bar(x - width/2, prec_normal, width, label='Normal', alpha=0.8, color='#3498db')
axes[0].bar(x + width/2, prec_synthetic, width, label='Real+Synthetic', alpha=0.8, color='#2ecc71')
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('Precision', fontsize=12)
axes[0].set_title('Per-Class Precision', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(class_names, rotation=90, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Recall
axes[1].bar(x - width/2, rec_normal, width, label='Normal', alpha=0.8, color='#3498db')
axes[1].bar(x + width/2, rec_synthetic, width, label='Real+Synthetic', alpha=0.8, color='#2ecc71')
axes[1].set_xlabel('Class', fontsize=12)
axes[1].set_ylabel('Recall', fontsize=12)
axes[1].set_title('Per-Class Recall', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(class_names, rotation=90, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# F1 Score
axes[2].bar(x - width/2, f1_normal, width, label='Normal', alpha=0.8, color='#3498db')
axes[2].bar(x + width/2, f1_synthetic, width, label='Real+Synthetic', alpha=0.8, color='#2ecc71')
axes[2].set_xlabel('Class', fontsize=12)
axes[2].set_ylabel('F1 Score', fontsize=12)
axes[2].set_title('Per-Class F1 Score', fontsize=12, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(class_names, rotation=90, ha='right')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: per_class_performance.png")

# 7. Box plot of fold accuracies
print("  Generating fold accuracy distribution chart...")
fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = [results_normal['fold_accuracies'], results_synthetic['fold_accuracies']]
bp = ax.boxplot(data_to_plot, labels=['Normal Data Only', 'Real + Synthetic'], patch_artist=True)

colors = ['#3498db', '#2ecc71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Fold Accuracy Distribution (5-Fold CV)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add individual points
for i, dataset in enumerate(data_to_plot, 1):
    y = dataset
    x = np.random.normal(i, 0.04, size=len(y))
    ax.scatter(x, y, alpha=0.4, s=50, color='black')

plt.tight_layout()
plt.savefig(results_dir / 'fold_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fold_distribution.png")

print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE")
print("="*80)
print(f"\nAll results saved to: {results_dir.absolute()}")
print("\nGenerated files:")
print(f"  - comparison_metrics.json")
print(f"  - confusion_matrix_normal_unweighted.png")
print(f"  - confusion_matrix_normal_weighted.png")
print(f"  - confusion_matrix_synthetic_unweighted.png")
print(f"  - confusion_matrix_synthetic_weighted.png")
print(f"  - accuracy_comparison.png")
print(f"  - f1_comparison.png")
print(f"  - summary_metrics_comparison.png")
print(f"  - per_class_performance.png")
print(f"  - fold_distribution.png")
print("\nDone!")