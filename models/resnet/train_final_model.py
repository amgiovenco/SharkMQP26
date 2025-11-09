"""
Train final ResNet1D model on 100% of data using best hyperparameters from Optuna.
Best Trial 67 parameters.
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
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

# Best hyperparameters from Trial 67
BEST_PARAMS = {
    "initial_filters": 80,
    "dropout": 0.20796879885018393,
    "learning_rate": 0.0004313869594239175,
    "batch_size": 16,
    "weight_decay": 0.0001560845747200455
}

# ResNet1D model
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

# Train final model on 100% of data
print("\n" + "="*60)
print("TRAINING: ResNet1D Final Model (100% Data)")
print("="*60)
print(f"\nBest Hyperparameters:")
print(f"  initial_filters: {BEST_PARAMS['initial_filters']}")
print(f"  dropout: {BEST_PARAMS['dropout']:.6f}")
print(f"  learning_rate: {BEST_PARAMS['learning_rate']:.6f}")
print(f"  batch_size: {BEST_PARAMS['batch_size']}")
print(f"  weight_decay: {BEST_PARAMS['weight_decay']:.6f}")

X_tensor = torch.FloatTensor(X).to(DEVICE)
y_tensor = torch.LongTensor(y_encoded).to(DEVICE)

train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=True)

# Model
final_model = ResNet1D(
    num_classes=len(np.unique(y_encoded)),
    input_channels=1,
    initial_filters=BEST_PARAMS['initial_filters'],
    dropout=BEST_PARAMS['dropout']
).to(DEVICE)

# Optimizer
optimizer = optim.Adam(final_model.parameters(),
                      lr=BEST_PARAMS['learning_rate'],
                      weight_decay=BEST_PARAMS['weight_decay'])
criterion = nn.CrossEntropyLoss()

# Training loop
print(f"\nTraining for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    final_model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")

# Save entire model (architecture + weights)
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

torch.save(final_model, "./resnet_final.pth")
print("Saved model to ./resnet_final.pth")

# Save bundle with label encoder and metadata
bundle = {
    "model": final_model,
    "label_encoder": le,
    "params": BEST_PARAMS
}
joblib.dump(bundle, "./resnet1d_final_bundle.pkl")
print("Saved bundle to ./resnet1d_final_bundle.pkl")

# Save results to JSON
results_dict = {
    "model_type": "resnet1d_final",
    "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in BEST_PARAMS.items()},
    "label_encoder_classes": [str(c) for c in le.classes_],
    "training_data_shape": list(X.shape),
    "num_classes": int(len(np.unique(y_encoded)))
}

with open("./resnet1d_training_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)
print("Saved results to ./resnet1d_training_results.json")

print("\n" + "="*60)
print("Done! Final model trained on 100% of data")
print("="*60)
