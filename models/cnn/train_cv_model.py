"""
Cross-validation test for CNN/EfficientNet model (10-fold stratified, 90/10 split)
Sanity check to detect overfitting in optimize_model.py
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import io
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("PyTorch/torchvision/matplotlib/PIL not available!")

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_SPLITS = 10
NUM_EPOCHS = 150

# Best params from results/efficientnet_final_results.json
BEST_PARAMS = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "dropout1": 0.5,
    "dropout2": 0.3,
    "hidden_size": 256,
}


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


# Load data
print("Loading data...")
df = pd.read_csv(CSV_PATH)
X_raw = df.drop(columns=[SPECIES_COL])
y = df[SPECIES_COL].astype(str)

print(f"Data shape: {X_raw.shape}")
print(f"Classes: {len(np.unique(y))}")

# Generate images
print("Generating fluorescence curve images...")
temps = X_raw.columns.astype(float).values
images = []
for idx, row in X_raw.iterrows():
    if idx % 100 == 0:
        print(f"  {idx}/{len(X_raw)}")
    img = _generate_image(temps, row.values)
    if img is None:
        print(f"Failed to generate image for sample {idx}")
        exit(1)
    images.append(img)

print(f"Generated {len(images)} images")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\n" + "="*70)
print(f"CROSS-VALIDATION: CNN/EfficientNet (10-fold stratified, seed={RANDOM_STATE})")
print("="*70)
print(f"Best params: {BEST_PARAMS}")

# Create results directory
results_dir = Path("./cv_results")
results_dir.mkdir(exist_ok=True)
print(f"Saving fold models to: {results_dir}/")

# Image transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 10-fold stratified cross-validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_accuracies = []
fold_f1_scores = []
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(images, y_encoded), 1):
    # Get subset of images
    train_images = [images[i] for i in train_idx]
    y_train = y_encoded[train_idx]
    test_images = [images[i] for i in test_idx]
    y_test = y_encoded[test_idx]

    # Create datasets and dataloaders
    train_dataset = FluorescenceImageDataset(train_images, y_train, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=True, num_workers=0)

    test_dataset = FluorescenceImageDataset(test_images, y_test, transform=transform_val)
    test_loader = DataLoader(test_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=0)

    # Model
    model = CNNModel(
        num_classes=len(np.unique(y_encoded)),
        dropout1=BEST_PARAMS['dropout1'],
        dropout2=BEST_PARAMS['dropout2'],
        hidden_size=BEST_PARAMS['hidden_size']
    ).to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(model.parameters(),
                          lr=BEST_PARAMS['learning_rate'],
                          weight_decay=BEST_PARAMS['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()

        val_loss /= len(train_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Evaluate on test set
    model.eval()
    test_preds = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    y_pred = np.array(test_preds)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    fold_accuracies.append(acc)
    fold_f1_scores.append(f1)

    # Save fold model with accuracy in filename
    model_filename = f"fold_{fold_idx:02d}_acc_{acc:.4f}_f1_{f1:.4f}.pth"
    torch.save(model, results_dir / model_filename)

    fold_results.append({
        "fold": fold_idx,
        "accuracy": float(acc),
        "f1": float(f1),
        "test_size": len(y_test),
        "model_file": model_filename
    })

    print(f"  Fold {fold_idx:2d}/10 | Accuracy: {acc:.4f} | F1: {f1:.4f} | Test size: {len(y_test)} | Saved: {model_filename}")

    # Clean up
    del model, optimizer, criterion, train_loader, test_loader

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Mean F1:       {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
print(f"Min Accuracy:  {np.min(fold_accuracies):.4f}")
print(f"Max Accuracy:  {np.max(fold_accuracies):.4f}")
print(f"\nTraining data: 90% ({len(images) * 9 // 10} samples per fold)")
print(f"Test data:     10% ({len(images) // 10} samples per fold)")
print("\n[Note] optimize_model.py trains on 100% data, overfitting is expected.")
print("="*70)

# Save summary
summary = {
    "model": "CNN/EfficientNet",
    "seed": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "best_params": BEST_PARAMS,
    "mean_accuracy": float(np.mean(fold_accuracies)),
    "std_accuracy": float(np.std(fold_accuracies)),
    "mean_f1": float(np.mean(fold_f1_scores)),
    "std_f1": float(np.std(fold_f1_scores)),
    "min_accuracy": float(np.min(fold_accuracies)),
    "max_accuracy": float(np.max(fold_accuracies)),
    "fold_results": fold_results
}

summary_file = results_dir / "cv_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[OK] Summary saved to {summary_file}")
print(f"[OK] All fold models saved to {results_dir}/")
