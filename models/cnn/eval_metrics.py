"""
Evaluate EfficientNet-B0 with best hyperparameters.
Protocol:
  1. 60/20/20 split (train / val / test)
  2. Train on 60% with early stopping on val → record best_epoch
  3. Scale epochs: scaled_epochs = round(best_epoch * (80/60))
  4. Retrain on 80% (train+val) for scaled_epochs, no early stopping
  5. Report metrics on held-out 20% test set
"""
import numpy as np
import pandas as pd
import joblib
import json
import io
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Config
DATA_DIR   = Path(__file__).parent.parent.parent / "data"
CSV_PATH   = DATA_DIR / "shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
IMAGE_SIZE   = 224
EPOCHS       = 150
PATIENCE     = 25
SCALE_FACTOR = 80 / 60  # ratio of (train+val) to train-only data

# Best hyperparameters from optimization
BEST_PARAMS = {
    'cnn_dropout1':      0.42013403301390473,
    'cnn_dropout2':      0.11110962092151311,
    'cnn_learning_rate': 0.000582146876968696,
    'cnn_batch_size':    16,
    'cnn_weight_decay':  1.1132387479263832e-06,
    'cnn_focal_alpha':   0.9423473333849446,
    'cnn_focal_gamma':   1.7916114932802345,
}


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


class AddGaussianNoise(object):
    def __init__(self, std=0.005):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomAffine(degrees=0, translate=(0.0, 0.03)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.005),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


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
    def __init__(self, num_classes, dropout1=0.7, dropout2=0.5):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            probs = torch.softmax(outputs, dim=1)
            confidence, preds = probs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(confidence.cpu().numpy())
    macro_f1 = 100. * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return running_loss / len(loader), macro_f1, all_preds, all_labels, all_probs


def build_model_and_optimizer(num_classes):
    m = CNNModel(
        num_classes=num_classes,
        dropout1=BEST_PARAMS['cnn_dropout1'],
        dropout2=BEST_PARAMS['cnn_dropout2']
    ).to(DEVICE)
    crit = FocalLoss(
        alpha=BEST_PARAMS['cnn_focal_alpha'],
        gamma=BEST_PARAMS['cnn_focal_gamma']
    )
    opt = optim.AdamW(
        m.parameters(),
        lr=BEST_PARAMS['cnn_learning_rate'],
        weight_decay=BEST_PARAMS['cnn_weight_decay']
    )
    return m, crit, opt


# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
X_raw = df.drop(columns=[SPECIES_COL])
y     = df[SPECIES_COL].astype(str)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
num_classes = len(class_names)

print(f"Data shape: {X_raw.shape}")
print(f"Classes ({num_classes}): {list(class_names)}")

# ─── 60 / 20 / 20 split ───────────────────────────────────────────────────────
# Step 1: carve out 20% test (holdout)
X_tv, X_test, y_tv, y_test, idx_tv, idx_test = train_test_split(
    X_raw, y_encoded, np.arange(len(y_encoded)),
    test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
)
# Step 2: split remaining 80% into 60% train / 20% val  (0.25 of 80% = 20% overall)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_tv, y_tv, np.arange(len(y_tv)),
    test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE
)

total = len(y_encoded)
print(f"\nTrain : {len(y_train):4d}  ({100*len(y_train)/total:.0f}%)")
print(f"Val   : {len(y_val):4d}  ({100*len(y_val)/total:.0f}%)")
print(f"Test  : {len(y_test):4d}  ({100*len(y_test)/total:.0f}%)")

# ─── Generate images once ─────────────────────────────────────────────────────
print("\nPre-generating images...")
temps = X_raw.columns.astype(float).values
all_images = []
for _, row in X_raw.iterrows():
    img = _generate_image(temps, row.values)
    all_images.append(img if img is not None else Image.new('RGB', (288, 216)))
print(f"Generated {len(all_images)} images")

# Map back to original indices
# idx_tv are indices into X_raw; idx_train/idx_val are indices into X_tv
orig_idx_train = idx_tv[idx_train]
orig_idx_val   = idx_tv[idx_val]

train_images = [all_images[i] for i in orig_idx_train]
val_images   = [all_images[i] for i in orig_idx_val]
tv_images    = [all_images[i] for i in idx_tv]          # train+val combined
test_images  = [all_images[i] for i in idx_test]

batch_size = BEST_PARAMS['cnn_batch_size']

# ─── Phase 1: train on 60%, early-stop on 20% val ────────────────────────────
print("\n" + "="*60)
print("PHASE 1: Train on 60%, early-stop on val (20%)")
print("="*60)

train_ds = FluorescenceImageDataset(train_images, y_train, get_transforms(True))
val_ds   = FluorescenceImageDataset(val_images,   y_val,   get_transforms(False))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

model, criterion, optimizer = build_model_and_optimizer(num_classes)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_f1  = 0.0
best_epoch   = 0
patience_cnt = 0

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion,
                             optimizer, DEVICE)
    _, val_f1, _, _, _ = validate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        best_epoch   = epoch + 1   # 1-based
        patience_cnt = 0
    else:
        patience_cnt += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | train_loss={train_loss:.4f} "
              f"| val_macro_f1={val_f1:.2f}% | best={best_val_f1:.2f}% @ ep{best_epoch}")

    if patience_cnt >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

scaled_epochs = round(best_epoch * SCALE_FACTOR)
print(f"\nBest epoch on 60% data : {best_epoch}")
print(f"Scale factor (80/60)   : {SCALE_FACTOR:.4f}")
print(f"Scaled epochs for 80%  : {scaled_epochs}")

# ─── Phase 2: retrain on 80% (train+val) for scaled_epochs ───────────────────
print("\n" + "="*60)
print(f"PHASE 2: Retrain on 80% (train+val) for {scaled_epochs} epochs")
print("="*60)

tv_ds     = FluorescenceImageDataset(tv_images, y_tv, get_transforms(True))
tv_loader = DataLoader(tv_ds, batch_size=batch_size, shuffle=True, num_workers=0)

model, criterion, optimizer = build_model_and_optimizer(num_classes)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scaled_epochs, eta_min=1e-6)

for epoch in range(scaled_epochs):
    train_loss = train_epoch(model, tv_loader, criterion,
                             optimizer, DEVICE)
    scheduler.step()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{scaled_epochs} | train_loss={train_loss:.4f}")

print("Training complete.")

# ─── Evaluate on held-out test set ────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATING ON TEST SET (20% holdout)")
print("="*60)

test_ds     = FluorescenceImageDataset(test_images, y_test, get_transforms(False))
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        confidence, preds = probs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(confidence.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

accuracy    = 100. * accuracy_score(all_labels, all_preds)
precision_w = 100. * precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall_w    = 100. * recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1_w        = 100. * f1_score(all_labels, all_preds, average='weighted', zero_division=0)
macro_f1    = 100. * f1_score(all_labels, all_preds, average='macro', zero_division=0)
avg_conf    = 100. * float(all_probs.mean())

print(f"\n{'Metric':<25} {'Value':>10}")
print("-" * 36)
print(f"{'Model':<25} {'EfficientNet-B0':>10}")
print(f"{'Accuracy':<25} {accuracy:>9.2f}%")
print(f"{'Precision (weighted)':<25} {precision_w:>9.2f}%")
print(f"{'Recall (weighted)':<25} {recall_w:>9.2f}%")
print(f"{'F1-Score (weighted)':<25} {f1_w:>9.2f}%")
print(f"{'Macro F1-Score':<25} {macro_f1:>9.2f}%")
print(f"{'Avg. Confidence':<25} {avg_conf:>9.2f}%")
print(f"\n{'Best epoch (60% data)':<25} {best_epoch:>10}")
print(f"{'Scaled epochs (80%)':<25} {scaled_epochs:>10}")

print("\nPer-class report:")
print(classification_report(all_labels, all_preds,
                             target_names=class_names, zero_division=0))

# ─── Confidence: correct vs incorrect ─────────────────────────────────────────
correct_mask   = all_preds == all_labels
correct_conf   = 100. * all_probs[correct_mask].mean()
incorrect_conf = 100. * all_probs[~correct_mask].mean() if (~correct_mask).any() else float('nan')

print("\n" + "="*60)
print("CONFIDENCE ANALYSIS")
print("="*60)
print(f"{'Correct predictions':<30} {correct_mask.sum():>5}  avg conf: {correct_conf:.2f}%")
print(f"{'Incorrect predictions':<30} {(~correct_mask).sum():>5}  avg conf: {incorrect_conf:.2f}%")

# ─── Per-class confidence ──────────────────────────────────────────────────────
print(f"\n{'Class':<40} {'N pred':>6}  {'Avg conf':>9}  {'Correct':>8}")
print("-" * 68)
for cls_idx, cls_name in enumerate(class_names):
    cls_mask = all_preds == cls_idx
    if cls_mask.sum() == 0:
        continue
    cls_conf    = 100. * all_probs[cls_mask].mean()
    cls_correct = (all_preds[cls_mask] == all_labels[cls_mask]).sum()
    print(f"{cls_name:<40} {cls_mask.sum():>6}  {cls_conf:>8.2f}%  {cls_correct:>6}/{cls_mask.sum()}")

# ─── Misclassification details ─────────────────────────────────────────────────
print("\n" + "="*60)
print("MISCLASSIFICATIONS")
print("="*60)
print(f"{'True label':<40} {'Predicted':<40} {'Conf':>6}")
print("-" * 90)
for i in np.where(~correct_mask)[0]:
    true_name = class_names[all_labels[i]]
    pred_name = class_names[all_preds[i]]
    print(f"{true_name:<40} {pred_name:<40} {all_probs[i]*100:>5.1f}%")

# ─── Confusion matrix ─────────────────────────────────────────────────────────
plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)
n_classes = len(class_names)
fig_size  = max(24, n_classes * 0.45)

fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

ax.set_xticks(np.arange(n_classes))
ax.set_yticks(np.arange(n_classes))
ax.set_xticklabels(class_names, rotation=90, fontsize=7)
ax.set_yticklabels(class_names, fontsize=7)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)
ax.set_title('EfficientNet-B0 — Confusion Matrix (test set)', fontsize=13, pad=12)

thresh = cm.max() / 2.0
for i in range(n_classes):
    for j in range(n_classes):
        if cm[i, j] > 0:
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=5, color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
cm_path = plots_dir / "confusion_matrix.png"
fig.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved confusion matrix to {cm_path}")

# ─── Confidence histogram ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, 1, 41)
ax.hist(all_probs[correct_mask],  bins=bins, alpha=0.6, color='steelblue',
        label=f'Correct (n={correct_mask.sum()})',   density=True)
ax.hist(all_probs[~correct_mask], bins=bins, alpha=0.6, color='tomato',
        label=f'Incorrect (n={(~correct_mask).sum()})', density=True)
ax.set_xlabel('Softmax confidence', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('EfficientNet-B0 — Confidence Distribution (test set)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
hist_path = plots_dir / "confidence_histogram.png"
fig.savefig(hist_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved confidence histogram to {hist_path}")

# ─── Save results ─────────────────────────────────────────────────────────────
results = {
    "model": "EfficientNet-B0",
    "split": {
        "train": "60%", "val": "20%", "test": "20%",
        "random_state": RANDOM_STATE
    },
    "epoch_scaling": {
        "best_epoch_on_60pct": best_epoch,
        "scale_factor":        round(SCALE_FACTOR, 4),
        "scaled_epochs_on_80pct": scaled_epochs,
    },
    "best_params": BEST_PARAMS,
    "metrics": {
        "accuracy":                round(accuracy,       4),
        "precision_weighted":      round(precision_w,    4),
        "recall_weighted":         round(recall_w,       4),
        "f1_score_weighted":       round(f1_w,           4),
        "macro_f1_score":          round(macro_f1,       4),
        "avg_confidence":          round(avg_conf,       4),
        "correct_avg_confidence":  round(correct_conf,   4),
        "incorrect_avg_confidence": round(incorrect_conf, 4),
    }
}

out_path = Path(__file__).parent / "eval_metrics_results.json"
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, cls=_NumpyEncoder)
print(f"\nSaved results to {out_path}")

# ─── Save bundle for cnn_inference.py ─────────────────────────────────────────
inference_dir = Path(__file__).parent.parent.parent / "backend" / "worker" / "efficientnet"
inference_dir.mkdir(parents=True, exist_ok=True)

bundle = {
    "model":         model,
    "label_encoder": le,
    "cv_accuracy":   macro_f1,
    "params":        BEST_PARAMS,
}
bundle_path = inference_dir / "cnn_bundle.pkl"
joblib.dump(bundle, bundle_path)
print(f"Saved inference bundle to {bundle_path}")

print("\nDone!")
