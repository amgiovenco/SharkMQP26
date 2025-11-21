"""
Optimize CNN/EfficientNet model hyperparameters with Optuna.
Tunes learning rate, batch size, dropout rates, hidden layer size, and weight decay.
Generates images from fluorescence curves and trains the model.
"""
import numpy as np
import pandas as pd
import joblib
import json
import io
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import optuna
from optuna.storages import RDBStorage
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid Tkinter issues
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Config
DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_TRIALS = 300
IMAGE_SIZE = 224
EPOCHS = 150
PATIENCE = 25
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 1.5
MIXUP_ALPHA = 0.4
USE_MIXUP = True

# Output directory for results
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Optuna persistent storage
STUDY_NAME = "cnn_efficientnet"
STORAGE_PATH = RESULTS_DIR / "optuna_studies"
STORAGE_PATH.mkdir(exist_ok=True)
STORAGE_URL = f"sqlite:///{STORAGE_PATH}/optuna_studies.db"


def _generate_image(temps: np.ndarray, values: np.ndarray) -> Image.Image:
    """Generate PIL image from fluorescence curve."""
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
    """Add Gaussian noise to tensor."""
    def __init__(self, std=0.005):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std
    def __repr__(self):
        return self.__class__.__name__ + f'(std={self.std})'


def get_transforms(is_training=True):
    """Get augmentation transforms matching the original notebook."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomAffine(degrees=0, translate=(0.0, 0.03)),   # vertical shift only
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
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FluorescenceImageDataset(Dataset):
    """Dataset for fluorescence curve images (pre-generated)."""
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
    """EfficientNet-based CNN model with fixed 256 hidden units."""
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


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Generate mixed-up batch."""
    if alpha == 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=False):
    """Train one epoch with optional mixup."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item() if not use_mixup else 0
    acc = 100. * correct / total if not use_mixup else float('nan')
    return running_loss / len(loader), acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = 100. * accuracy_score(all_labels, all_preds)
    f1  = 100. * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return running_loss / len(loader), acc, f1, all_preds, all_labels


def train_cnn(all_images, y_encoded, dropout1, dropout2, learning_rate, batch_size, weight_decay, focal_alpha, focal_gamma, trial=None):
    """5-fold CV, early stopping on validation MACRO F1, returns mean best val-f1."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(all_images, y_encoded)):
        # ---- datasets -------------------------------------------------
        train_imgs = [all_images[i] for i in tr_idx]
        val_imgs   = [all_images[i] for i in val_idx]
        y_tr = y_encoded[tr_idx]
        y_val = y_encoded[val_idx]

        train_ds = FluorescenceImageDataset(train_imgs, y_tr,
                                            transform=get_transforms(is_training=True))
        val_ds   = FluorescenceImageDataset(val_imgs,   y_val,
                                            transform=get_transforms(is_training=False))

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=0)

        # ---- model ----------------------------------------------------
        model = CNNModel(num_classes=len(np.unique(y_encoded)),
                         dropout1=dropout1, dropout2=dropout2).to(DEVICE)

        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        optimizer = optim.AdamW(model.parameters(),
                                lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=EPOCHS, eta_min=1e-6)

        # ---- training state -------------------------------------------
        best_val_f1 = 0.0
        patience_cnt = 0
        history = {'val_f1': []}

        for epoch in range(EPOCHS):
            train_loss, _ = train_epoch(model, train_loader, criterion,
                                        optimizer, DEVICE, use_mixup=USE_MIXUP)
            val_loss, val_acc, val_f1, _, _ = validate(model, val_loader,
                                                       criterion, DEVICE)

            scheduler.step()
            history['val_f1'].append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_cnt = 0
            else:
                patience_cnt += 1

            # Report intermediate value to Optuna for pruning
            if trial is not None:
                trial.report(val_f1, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if patience_cnt >= PATIENCE:
                # print(f"Fold {fold+1} early stop at epoch {epoch+1}")
                break

        cv_scores.append(best_val_f1)

    return np.mean(cv_scores)


# Load data
print("Loading data...")
df = pd.read_csv(CSV_PATH)
X_raw = df.drop(columns=[SPECIES_COL])
y = df[SPECIES_COL].astype(str)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Data shape: {X_raw.shape}")
print(f"Classes: {len(np.unique(y_encoded))}")

# Split into 80% train+val (for CV) and 20% test (holdout)
X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
    X_raw, y_encoded, np.arange(len(y_encoded)),
    test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
)

print(f"\nTrain+Val shape: {X_trainval.shape}")
print(f"Test (holdout) shape: {X_test.shape}")

# Pre-generate all images ONCE
print("\nPre-generating images (this takes ~1-2 minutes)...")
temps = X_raw.columns.astype(float).values
all_images = []
for idx, row in X_raw.iterrows():
    img = _generate_image(temps, row.values)
    if img is not None:
        all_images.append(img)
    else:
        all_images.append(Image.new('RGB', (288, 216)))
print(f"Generated {len(all_images)} images")

# Extract only trainval images for CV
trainval_images = [all_images[i] for i in idx_trainval]
test_images = [all_images[i] for i in idx_test]

# Baseline
print("\n" + "="*60)
print("BASELINE: CNN/EfficientNet (5-fold CV on 80% trainval)")
print("="*60)

base_score = train_cnn(trainval_images, y_trainval, dropout1=0.7, dropout2=0.5,
                       learning_rate=0.001, batch_size=32, weight_decay=0.0001, focal_alpha=1.0, focal_gamma=1.5)
print(f"Baseline CV Macro F1: {base_score:.2f}%")

best_overall_score = base_score
best_overall_params = {
    "cnn_dropout1": 0.7,
    "cnn_dropout2": 0.5,
    "cnn_learning_rate": 0.001,
    "cnn_batch_size": 32,
    "cnn_weight_decay": 0.0001,
    "cnn_focal_alpha": 1.0,
    "cnn_focal_gamma": 1.5
}

# Objective function
def objective(trial):
    dropout1     = trial.suggest_float('cnn_dropout1', 0.3, 0.8)
    dropout2     = trial.suggest_float('cnn_dropout2', 0.1, 0.6)
    lr           = trial.suggest_float('cnn_learning_rate', 1e-4, 1e-2, log=True)
    batch_size   = trial.suggest_categorical('cnn_batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('cnn_weight_decay', 1e-6, 1e-4, log=True)
    focal_alpha  = trial.suggest_float('cnn_focal_alpha', 0.5, 2.0)
    focal_gamma  = trial.suggest_float('cnn_focal_gamma', 1.0, 3.0)

    score = train_cnn(trainval_images, y_trainval,
                      dropout1=dropout1,
                      dropout2=dropout2,
                      learning_rate=lr,
                      batch_size=batch_size,
                      weight_decay=weight_decay,
                      focal_alpha=focal_alpha,
                      focal_gamma=focal_gamma,
                      trial=trial)
    return score


print("\n" + "="*60)
print("OPTIMIZING: CNN/EfficientNet hyperparameters (Macro F1)")
print("="*60)

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

storage = RDBStorage(STORAGE_URL)
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=pruner,
    storage=storage,
    study_name=STUDY_NAME,
    load_if_exists=True
)
print(f"Study: {STUDY_NAME} | Completed trials: {len(study.trials)}")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest CV Macro F1: {study.best_value:.2f}%")
print(f"Best params: {study.best_params}")

if study.best_value > best_overall_score:
    best_overall_score = study.best_value
    best_overall_params = study.best_params

# Evaluate best model on holdout test set
print("\n" + "="*60)
print("EVALUATING BEST MODEL ON HOLDOUT TEST SET (20%)")
print("="*60)

best = study.best_params if study.best_value > base_score else {
    "cnn_dropout1": 0.7, "cnn_dropout2": 0.5,
    "cnn_learning_rate": 0.001, "cnn_batch_size": 32,
    "cnn_weight_decay": 0.0001, "cnn_focal_alpha": 1.0, "cnn_focal_gamma": 1.5
}

# Train on full trainval set with best params
test_model = CNNModel(num_classes=len(np.unique(y_trainval)),
                      dropout1=best['cnn_dropout1'],
                      dropout2=best['cnn_dropout2']).to(DEVICE)

test_dataset = FluorescenceImageDataset(trainval_images, y_trainval,
                                        transform=get_transforms(is_training=True))
test_loader = DataLoader(test_dataset,
                         batch_size=best['cnn_batch_size'],
                         shuffle=True, num_workers=0)

optimizer = optim.AdamW(test_model.parameters(),
                        lr=best['cnn_learning_rate'],
                        weight_decay=best['cnn_weight_decay'])
criterion = FocalLoss(alpha=best['cnn_focal_alpha'], gamma=best['cnn_focal_gamma'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                        T_max=EPOCHS, eta_min=1e-6)

# Train on full trainval set
for epoch in range(EPOCHS):
    train_epoch(test_model, test_loader, criterion,
                optimizer, DEVICE, use_mixup=USE_MIXUP)
    scheduler.step()

# Evaluate on test set
test_dataset_eval = FluorescenceImageDataset(test_images, y_test,
                                             transform=get_transforms(is_training=False))
test_loader_eval = DataLoader(test_dataset_eval, batch_size=best['cnn_batch_size'],
                              shuffle=False, num_workers=0)

test_loss, test_acc, test_f1, test_preds, test_labels = validate(test_model, test_loader_eval,
                                                                  criterion, DEVICE)

print(f"\nTest Accuracy: {test_acc:.2f}%")
print(f"Test Macro F1 Score: {test_f1:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"\nBest CV Macro F1 (80% trainval): {best_overall_score:.2f}%")
print(f"Test Accuracy (20% holdout): {test_acc:.2f}%")
print(f"Test Macro F1 (20% holdout): {test_f1:.2f}%")
print(f"Best params: {best_overall_params}")
print(f"Improvement over baseline CV: {(best_overall_score - base_score)*100:.2f}%")

# Export results to JSON
results_dict = {
    "optimization_metric": "macro_f1",
    "baseline_cv_macro_f1": float(base_score),
    "best_model": "efficientnet_b0",
    "best_cv_macro_f1": float(best_overall_score),
    "test_accuracy": float(test_acc),
    "test_macro_f1_score": float(test_f1),
    "test_loss": float(test_loss),
    "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_overall_params.items()},
    "improvement_percentage": float((best_overall_score - base_score) * 100),
    "data_split": {
        "trainval": "80%",
        "test_holdout": "20%",
        "cv_folds": 5
    }
}

with open(RESULTS_DIR / "optimization_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nSaved optimization results to {RESULTS_DIR / 'optimization_results.json'}")

# Train final model on all data
print("\n" + "="*60)
print("Training final model on all data...")
print("="*60)

best = study.best_params

final_dataset = FluorescenceImageDataset(all_images, y_encoded,
                                         transform=get_transforms(is_training=True))
final_loader = DataLoader(final_dataset,
                          batch_size=best['cnn_batch_size'],
                          shuffle=True, num_workers=0)

final_model = CNNModel(num_classes=len(np.unique(y_encoded)),
                       dropout1=best['cnn_dropout1'],
                       dropout2=best['cnn_dropout2']).to(DEVICE)

optimizer = optim.AdamW(final_model.parameters(),
                        lr=best['cnn_learning_rate'],
                        weight_decay=best['cnn_weight_decay'])
criterion = FocalLoss(alpha=best['cnn_focal_alpha'], gamma=best['cnn_focal_gamma'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                        T_max=EPOCHS, eta_min=1e-6)

# train for full EPOCHS (no early stopping)
for epoch in range(EPOCHS):
    train_epoch(final_model, final_loader, criterion,
                optimizer, DEVICE, use_mixup=USE_MIXUP)
    scheduler.step()

# Save entire model (architecture + weights)
torch.save(final_model.state_dict(), RESULTS_DIR / "cnn_final.pth")

bundle = {
    "model": final_model,
    "label_encoder": le,
    "cv_accuracy": best_overall_score,
    "params": best_overall_params
}

joblib.dump(bundle, RESULTS_DIR / "cnn_bundle.pkl")
print(f"Saved optimized model to {RESULTS_DIR / 'cnn_final.pth'} and {RESULTS_DIR / 'cnn_bundle.pkl'}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Baseline CV Macro F1 (80% trainval): {base_score:.2f}%")
print(f"Best CV Macro F1 (80% trainval): {best_overall_score:.2f}%")
print(f"Test Accuracy (20% holdout): {test_acc:.2f}%")
print(f"Test Macro F1 Score (20% holdout): {test_f1:.2f}%")
print(f"Improvement: {(best_overall_score - base_score)*100:.2f}%")
print("\nDone!")
