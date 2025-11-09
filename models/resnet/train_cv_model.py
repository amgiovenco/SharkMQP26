"""
Cross-validation test for ResNet1D model (10-fold stratified, 90/10 split)
Sanity check to detect overfitting in train_final_model.py
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
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
        torch.backends.cudnn.benchmark = True
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("PyTorch not available!")

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
NUM_EPOCHS = 100
RANDOM_STATE = 8
N_SPLITS = 10

# Best hyperparameters from Trial 67
BEST_PARAMS = {
    "initial_filters": 80,
    "dropout": 0.20796879885018393,
    "learning_rate": 0.0004313869594239175,
    "batch_size": 16,
    "weight_decay": 0.0001560845747200455
}

# ResNet1D model (same as train_final_model.py)
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
        x = self.dropout(x)
        x = self.fc(x)
        return x


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

print("\n" + "="*70)
print(f"CROSS-VALIDATION: ResNet1D (10-fold stratified, seed={RANDOM_STATE})")
print("="*70)
print(f"Best params: {BEST_PARAMS}")

# Create results directory
results_dir = Path("./cv_results")
results_dir.mkdir(exist_ok=True)
print(f"Saving fold models to: {results_dir}/")

# 10-fold stratified cross-validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_accuracies = []
fold_f1_scores = []
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    y_test_tensor = torch.LongTensor(y_test).to(DEVICE)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=True)

    # Create model
    model = ResNet1D(
        num_classes=len(np.unique(y_encoded)),
        input_channels=1,
        initial_filters=BEST_PARAMS['initial_filters'],
        dropout=BEST_PARAMS['dropout']
    ).to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(model.parameters(),
                          lr=BEST_PARAMS['learning_rate'],
                          weight_decay=BEST_PARAMS['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred_tensor = torch.argmax(outputs, dim=1)
        y_pred = y_pred_tensor.cpu().numpy()

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
    del model, optimizer, criterion, train_loader

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Mean F1:       {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
print(f"Min Accuracy:  {np.min(fold_accuracies):.4f}")
print(f"Max Accuracy:  {np.max(fold_accuracies):.4f}")
print(f"\nTraining data: 90% ({len(X) * 9 // 10} samples per fold)")
print(f"Test data:     10% ({len(X) // 10} samples per fold)")
print("\n[Note] train_final_model.py trains on 100% data, overfitting is expected.")
print("="*70)

# Save summary
summary = {
    "model": "ResNet1D",
    "seed": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in BEST_PARAMS.items()},
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
