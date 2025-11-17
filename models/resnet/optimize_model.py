"""
Optimize ResNet1D model hyperparameters with Optuna.
Tunes learning rate, batch size, dropout, initial filters, and weight decay.
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import optuna
from optuna.storages import RDBStorage
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Config
DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
N_TRIALS = 300
NUM_EPOCHS = 150
PATIENCE = 15

# Optuna persistent storage
STUDY_NAME = "resnet1d"
STORAGE_PATH = Path("./optuna_studies")
STORAGE_PATH.mkdir(exist_ok=True)
STORAGE_URL = f"sqlite:///{STORAGE_PATH}/optuna_studies.db"

# Custom Dataset Class with Augmentation
class SharkFluorescenceDataset(Dataset):
    """Custom dataset for shark fluorescence data with optional augmentation."""
    def __init__(self, data_df, augment=False, mean=None, std=None):
        # Extract species names (labels) from first column
        self.species = data_df.iloc[:, 0].values

        # Extract fluorescence values (all columns except first)
        # Shape: (num_samples, num_time_steps)
        self.fluorescence = data_df.iloc[:, 1:].values.astype(np.float32)

        # Normalization
        if mean is None or std is None:
            # Calculate mean and std from this data (for training set)
            self.mean = self.fluorescence.mean()
            self.std = self.fluorescence.std()
        else:
            # Use provided mean and std (for val/test sets)
            self.mean = mean
            self.std = std

        # Apply normalization: (x - mean) / std
        self.fluorescence = (self.fluorescence - self.mean) / (self.std + 1e-8)

        # Encode species names to numeric labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.species)

        self.num_classes = len(self.label_encoder.classes_)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get fluorescence values and add channel dimension
        # Shape: (1, num_time_steps) - 1D conv expects (channels, length)
        x = torch.FloatTensor(self.fluorescence[idx]).unsqueeze(0)

        # Apply augmentation if enabled
        if self.augment:
            # Small random noise
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            # Random scaling
            scale = 1 + torch.FloatTensor([np.random.uniform(-0.05, 0.05)])
            x = x * scale
            # Random shift along time axis (small horizontal shift)
            shift = np.random.randint(-5, 6)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=1)

        # Get label
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y

# ResNet1D model (from predict_resnet.py)
class ResidualBlock1D(nn.Module):
    """Residual block for 1d convolution."""
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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)  # ReLU AFTER addition (critical)

        return out


class ResNet1D(nn.Module):
    """1d residual network for sequence processing."""
    def __init__(self, num_classes, input_channels=1, initial_filters=64, dropout=0.65):
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
        x = self.dropout(x)  # Dropout before FC layer (critical)
        x = self.fc(x)
        return x


def train_resnet(initial_filters, dropout, learning_rate, batch_size, weight_decay):
    """Train ResNet1D with early stopping using per-fold normalization."""
    # Load data
    df = pd.read_csv(CSV_PATH)
    y = df.iloc[:, 0].astype(str)
    X_raw = df.iloc[:, 1:]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # Create stratified k-fold splits for CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for train_idx, val_idx in skf.split(X_raw, y_encoded):
        # Split data by fold indices
        train_df = pd.concat([y.iloc[train_idx], X_raw.iloc[train_idx]], axis=1)
        val_df = pd.concat([y.iloc[val_idx], X_raw.iloc[val_idx]], axis=1)

        # Create datasets with per-fold normalization (no data leakage)
        train_dataset = SharkFluorescenceDataset(train_df, augment=True)
        val_dataset = SharkFluorescenceDataset(
            val_df,
            augment=False,
            mean=train_dataset.mean,
            std=train_dataset.std
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model
        model = ResNet1D(
            num_classes=num_classes,
            input_channels=1,
            initial_filters=initial_filters,
            dropout=dropout
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
        best_val_f1 = 0.0
        patience_counter = 0
        min_val_loss = float('inf')

        for _ in range(NUM_EPOCHS):
            # === TRAIN EPOCH ===
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_dataset)

            # === VALIDATION ===
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_dataset)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            scheduler.step(val_loss)

            # Track best validation F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

            # Early stopping based on validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        fold_scores.append(best_val_f1)

    return np.mean(fold_scores)


# Baseline
print("\n" + "="*60)
print("BASELINE: ResNet1D (Macro F1 Score)")
print("="*60)

base_score = train_resnet(
    initial_filters=64,
    dropout=0.65,
    learning_rate=0.001,
    batch_size=32,
    weight_decay=0.0001
)
print(f"Baseline CV macro F1: {base_score:.4f}")

best_overall_score = base_score
best_overall_params = {
    "initial_filters": 64,
    "dropout": 0.65,
    "learning_rate": 0.001,
    "batch_size": 32,
    "weight_decay": 0.0001
}

# Objective function
def objective(trial):
    initial_filters = trial.suggest_int('initial_filters', 32, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.1, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    cv_score = train_resnet(
        initial_filters=initial_filters,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=int(batch_size),
        weight_decay=weight_decay
    )
    return cv_score


print("\n" + "="*60)
print("OPTIMIZING: ResNet1D hyperparameters (Macro F1 Score)")
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

print(f"\nBest CV macro F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

if study.best_value > best_overall_score:
    best_overall_score = study.best_value
    best_overall_params = study.best_params

# Results
print("\n" + "="*60)
print("FINAL RESULTS (Macro F1 Score)")
print("="*60)
print(f"\nBest CV macro F1: {best_overall_score:.4f}")
print(f"Best params: {best_overall_params}")
print(f"Improvement over baseline: {(best_overall_score - base_score)*100:.2f}%")

# Export results to JSON
results_dict = {
    "metric": "macro_f1_score",
    "baseline_cv_macro_f1": float(base_score),
    "best_model": "resnet1d",
    "best_cv_macro_f1": float(best_overall_score),
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

# Load all data for final training
df_final = pd.read_csv(CSV_PATH)
y_all = df_final.iloc[:, 0].astype(str)
X_all = df_final.iloc[:, 1:]

# Create dataset with augmentation on full data
full_dataset = SharkFluorescenceDataset(df_final, augment=True)
train_loader_final = DataLoader(
    full_dataset,
    batch_size=int(best_overall_params['batch_size']),
    shuffle=True
)

# Create final model with best parameters
final_model = ResNet1D(
    num_classes=full_dataset.num_classes,
    input_channels=1,
    initial_filters=int(best_overall_params['initial_filters']),
    dropout=best_overall_params['dropout']
).to(DEVICE)

optimizer = optim.Adam(
    final_model.parameters(),
    lr=best_overall_params['learning_rate'],
    weight_decay=best_overall_params['weight_decay']
)
criterion = nn.CrossEntropyLoss()

# Train for 200 epochs on full data
for _ in range(200):
    final_model.train()
    for inputs, labels in train_loader_final:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save model
torch.save(final_model.state_dict(), "./resnet1d_optimized.pth")

bundle = {
    "model": final_model,
    "label_encoder": full_dataset.label_encoder,
    "cv_accuracy": best_overall_score,
    "params": best_overall_params
}

joblib.dump(bundle, "./optimized_resnet_model.pkl")
print(f"Saved optimized model to ./resnet1d_optimized.pth and ./optimized_resnet_model.pkl")

print("\n" + "="*60)
print("SUMMARY (Macro F1 Score)")
print("="*60)
print(f"Final CV macro F1: {best_overall_score:.4f}")
print(f"Improvement: {(best_overall_score - base_score)*100:.2f}%")
print("\nDone!")
