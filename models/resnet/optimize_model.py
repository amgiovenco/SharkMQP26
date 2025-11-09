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
import optuna
from optuna.storages import RDBStorage
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        # Optional: enable cuDNN benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("PyTorch not available!")

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_TRIALS = 200
NUM_EPOCHS = 100
PATIENCE = 15

# Optuna persistent storage
STUDY_NAME = "resnet1d"
STORAGE_PATH = Path("./optuna_studies")
STORAGE_PATH.mkdir(exist_ok=True)
STORAGE_URL = f"sqlite:///{STORAGE_PATH}/optuna_studies.db"

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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """1d residual network for sequence processing."""
    def __init__(self, num_classes, input_channels=1, initial_filters=64, dropout=0.5):
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
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_resnet(X, y, initial_filters, dropout, learning_rate, batch_size, weight_decay):
    """Train ResNet1D with early stopping."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Create stratified k-fold splits for CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    X_tensor = torch.FloatTensor(X).to(DEVICE)
    y_tensor = torch.LongTensor(y_encoded).to(DEVICE)

    for train_idx, val_idx in skf.split(X, y_encoded):
        X_train = X_tensor[train_idx]
        y_train = y_tensor[train_idx]
        X_val = X_tensor[val_idx]
        y_val = y_tensor[val_idx]

        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Model
        model = ResNet1D(
            num_classes=len(np.unique(y_encoded)),
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
        best_val_acc = 0.0
        patience_counter = 0
        min_val_loss = float('inf')

        for epoch in range(NUM_EPOCHS):
            # Train
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            with torch.no_grad():
                outputs = model(X_val)
                val_preds = torch.argmax(outputs, dim=1)
                val_acc = (val_preds == y_val).float().mean().item()
                val_loss = criterion(outputs, y_val).item()

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

# Normalize X
X = (X_raw - X_raw.mean().values) / (X_raw.std().values + 1e-8)
X = X.fillna(0).astype(np.float32).values
X = np.expand_dims(X, axis=1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Data shape: {X.shape}")
print(f"Classes: {len(np.unique(y_encoded))}")

# Baseline
print("\n" + "="*60)
print("BASELINE: ResNet1D")
print("="*60)

base_score = train_resnet(X, y, initial_filters=64, dropout=0.5, learning_rate=0.001, batch_size=32, weight_decay=0.0001)
print(f"Baseline CV accuracy: {base_score:.4f}")

best_overall_score = base_score
best_overall_params = {
    "initial_filters": 64,
    "dropout": 0.5,
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

    cv_score = train_resnet(X, y, initial_filters, dropout, learning_rate, batch_size, weight_decay)
    return cv_score


print("\n" + "="*60)
print("OPTIMIZING: ResNet1D hyperparameters")
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
    "best_model": "resnet1d",
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

X_tensor = torch.FloatTensor(X).to(DEVICE)
y_tensor = torch.LongTensor(y_encoded).to(DEVICE)

train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_overall_params['batch_size'], shuffle=True)

final_model = ResNet1D(
    num_classes=len(np.unique(y_encoded)),
    input_channels=1,
    initial_filters=best_overall_params['initial_filters'],
    dropout=best_overall_params['dropout']
).to(DEVICE)

optimizer = optim.Adam(final_model.parameters(), lr=best_overall_params['learning_rate'],
                       weight_decay=best_overall_params['weight_decay'])
criterion = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
    final_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Save model
torch.save(final_model.state_dict(), "./resnet1d_optimized.pth")

bundle = {
    "model": final_model,
    "label_encoder": le,
    "cv_accuracy": best_overall_score,
    "params": best_overall_params
}

joblib.dump(bundle, "./optimized_resnet_model.pkl")
print(f"Saved optimized model to ./resnet1d_optimized.pth and ./optimized_resnet_model.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final CV accuracy: {best_overall_score:.4f}")
print(f"Improvement: {(best_overall_score - base_score)*100:.2f}%")
print("\nDone!")
