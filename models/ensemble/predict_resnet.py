"""resnet1d model prediction."""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except (ImportError, RuntimeError):
    TORCH_AVAILABLE = False
    DEVICE = None


class ResidualBlock1D(nn.Module):
    """residual block for 1d convolution."""
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


def get_resnet_predictions(X_raw: pd.DataFrame, num_classes: int, models_dir: str = "./models") -> np.ndarray:
    """get resnet1d predictions."""
    print("  resnet1d...", end=" ", flush=True)
    if not TORCH_AVAILABLE:
        print("[FAIL] pytorch not available")
        return None

    try:
        model_path = f"{models_dir}/resnet1d_2_9615.pth"
        if not Path(model_path).exists():
            print("[FAIL] not found")
            return None

        # Load the model - handle both state_dict and full model
        loaded = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if isinstance(loaded, dict) and 'state_dict' in loaded:
            # It's a checkpoint dict
            model = ResNet1D(num_classes=num_classes)
            model.load_state_dict(loaded['state_dict'])
        elif isinstance(loaded, dict) and not isinstance(next(iter(loaded.values()), None), torch.nn.Module):
            # It's a state_dict
            model = ResNet1D(num_classes=num_classes)
            model.load_state_dict(loaded)
        else:
            # It's already a model
            model = loaded

        model = model.to(DEVICE)
        model.eval()

        # Normalize using GLOBAL mean/std (same as training)
        # Training computed: self.mean = self.fluorescence.mean() (single value)
        #                    self.std = self.fluorescence.std()   (single value)
        X_values = X_raw.values.astype(np.float32)
        global_mean = X_values.mean()
        global_std = X_values.std() + 1e-8
        X_norm = (X_values - global_mean) / global_std
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        X_norm = np.expand_dims(X_norm, axis=1)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(DEVICE)
            outputs = model(X_tensor)
            proba = torch.softmax(outputs, dim=1)
            proba = proba.cpu().numpy()

        print(f"[OK] ({proba.shape})")
        return proba
    except Exception as e:
        print(f"[FAIL] error: {e}")
        return None
