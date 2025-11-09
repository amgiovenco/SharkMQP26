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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.storages import RDBStorage
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend to avoid Tkinter issues
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
N_TRIALS = 200
NUM_EPOCHS = 100
PATIENCE = 15

# Optuna persistent storage
STUDY_NAME = "cnn_efficientnet"
STORAGE_PATH = Path("./optuna_studies")
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


def train_cnn(all_images, y_encoded, dropout1, dropout2, hidden_size, learning_rate, batch_size, weight_decay):
    """Train CNN with early stopping using pre-generated images."""
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

    # Stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for train_idx, val_idx in skf.split(all_images, y_encoded):
        # Get subset of images
        train_images = [all_images[i] for i in train_idx]
        y_train = y_encoded[train_idx]
        val_images = [all_images[i] for i in val_idx]
        y_val = y_encoded[val_idx]

        # Create datasets and dataloaders with pre-generated images
        train_dataset = FluorescenceImageDataset(train_images, y_train, transform=transform_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_dataset = FluorescenceImageDataset(val_images, y_val, transform=transform_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Model
        model = CNNModel(
            num_classes=len(np.unique(y_encoded)),
            dropout1=dropout1,
            dropout2=dropout2,
            hidden_size=hidden_size
        ).to(DEVICE)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        criterion = nn.CrossEntropyLoss()

        # Early stopping
        best_val_acc = 0.0
        patience_counter = 0
        min_val_loss = float('inf')

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
            val_preds = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
                    val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            val_acc = np.mean(val_preds == y_val)
            val_loss /= len(val_loader)

            # Update learning rate based on validation loss
            scheduler.step(val_loss)

            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # Early stopping based on validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        fold_scores.append(best_val_acc)

    return np.mean(fold_scores)


# Load data
print("Loading data...")
df = pd.read_csv(CSV_PATH)
X_raw = df.drop(columns=[SPECIES_COL])
y = df[SPECIES_COL].astype(str)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Data shape: {X_raw.shape}")
print(f"Classes: {len(np.unique(y_encoded))}")

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

# Baseline
print("\n" + "="*60)
print("BASELINE: CNN/EfficientNet")
print("="*60)

base_score = train_cnn(all_images, y_encoded, dropout1=0.5, dropout2=0.3, hidden_size=256,
                       learning_rate=0.001, batch_size=32, weight_decay=0.0001)
print(f"Baseline CV accuracy: {base_score:.4f}")

best_overall_score = base_score
best_overall_params = {
    "dropout1": 0.5,
    "dropout2": 0.3,
    "hidden_size": 256,
    "learning_rate": 0.001,
    "batch_size": 32,
    "weight_decay": 0.0001
}

# Objective function
def objective(trial):
    dropout1 = trial.suggest_float('dropout1', 0.2, 0.7)
    dropout2 = trial.suggest_float('dropout2', 0.1, 0.5)
    hidden_size = trial.suggest_int('hidden_size', 128, 512, step=64)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    cv_score = train_cnn(all_images, y_encoded, dropout1, dropout2, hidden_size, learning_rate, batch_size, weight_decay)
    return cv_score


print("\n" + "="*60)
print("OPTIMIZING: CNN/EfficientNet hyperparameters")
print("="*60)

storage = RDBStorage(STORAGE_URL)
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    storage=storage,
    study_name=STUDY_NAME,
    load_if_exists=True
)
print(f"Study: {STUDY_NAME} | Completed trials: {len(study.trials)}")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest CV accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

if study.best_value > best_overall_score:
    best_overall_score = study.best_value
    best_overall_params = study.best_params

# Results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"\nBest CV accuracy: {best_overall_score:.4f}")
print(f"Best params: {best_overall_params}")
print(f"Improvement over baseline: {(best_overall_score - base_score)*100:.2f}%")

# Export results to JSON
results_dict = {
    "baseline_cv_accuracy": float(base_score),
    "best_model": "efficientnet_b0",
    "best_cv_accuracy": float(best_overall_score),
    "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_overall_params.items()},
    "improvement_percentage": float((best_overall_score - base_score) * 100)
}

with open("./optimization_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nSaved optimization results to ./optimization_results.json")

# Train final model on all data
print("\n" + "="*60)
print("Training final model on all data...")
print("="*60)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

final_dataset = FluorescenceImageDataset(all_images, y_encoded, transform=transform_train)
final_loader = DataLoader(final_dataset, batch_size=best_overall_params['batch_size'], shuffle=True, num_workers=0)

final_model = CNNModel(
    num_classes=len(np.unique(y_encoded)),
    dropout1=best_overall_params['dropout1'],
    dropout2=best_overall_params['dropout2'],
    hidden_size=best_overall_params['hidden_size']
).to(DEVICE)

optimizer = optim.Adam(final_model.parameters(), lr=best_overall_params['learning_rate'],
                       weight_decay=best_overall_params['weight_decay'])
criterion = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
    final_model.train()
    for X_batch, y_batch in final_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Save entire model (architecture + weights)
torch.save(final_model, "./cnn_final.pth")

bundle = {
    "model": final_model,
    "label_encoder": le,
    "cv_accuracy": best_overall_score,
    "params": best_overall_params
}

joblib.dump(bundle, "./cnn_bundle.pkl")
print(f"Saved optimized model to ./cnn_final.pth and ./cnn_bundle.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final CV accuracy: {best_overall_score:.4f}")
print(f"Improvement: {(best_overall_score - base_score)*100:.2f}%")
print("\nDone!")
