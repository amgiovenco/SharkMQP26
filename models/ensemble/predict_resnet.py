import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from pathlib import Path


# --- Residual Block Definition ---
class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with two convolutional layers and a skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            bias=False
        )
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
        out = self.relu(out)

        return out


# --- ResNet1D Definition ---
class ResNet1D(nn.Module):
    """
    1D ResNet for time-series classification.
    """
    def __init__(self, num_classes, input_channels=1, initial_filters=64, dropout=0.65):
        super(ResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(
            input_channels,
            initial_filters,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
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
        """
        Create a layer consisting of multiple residual blocks.
        """
        downsample = None

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample))

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

        x = self.fc(x)

        return x


class ResNetPredictor:
    """
    Utility class for making predictions with a trained ResNet1D model.
    """

    def __init__(self, model_path, num_classes=57, initial_filters=80, dropout=0.2080, device=None):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path: Path to the saved .pth model file
            num_classes: Number of output classes (species)
            initial_filters: Initial number of filters in the ResNet
            dropout: Dropout rate used in the model
            device: torch device (cuda or cpu). If None, auto-detects.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load model
        self.model = ResNet1D(
            num_classes=num_classes,
            initial_filters=initial_filters,
            dropout=dropout
        ).to(self.device)

        # Load saved weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.num_classes = num_classes
        self.model_path = model_path

        print(f"Model loaded from: {model_path}")
        print(f"Device: {self.device}")
        print(f"Number of classes: {num_classes}")

    def normalize_data(self, fluorescence_data, mean=None, std=None):
        """
        Normalize fluorescence data using provided or calculated mean/std.

        Args:
            fluorescence_data: numpy array of shape (num_samples, num_time_steps)
            mean: Mean for normalization. If None, calculated from data.
            std: Std for normalization. If None, calculated from data.

        Returns:
            Normalized data, mean, std
        """
        if mean is None or std is None:
            mean = fluorescence_data.mean()
            std = fluorescence_data.std()

        normalized = (fluorescence_data - mean) / std
        return normalized, mean, std

    def predict(self, csv_path, return_probabilities=False, normalize_mean=None, normalize_std=None):
        """
        Make predictions on data from a CSV file.

        Args:
            csv_path: Path to CSV file with species names in first column and fluorescence data in remaining columns
            return_probabilities: If True, return softmax probabilities for all classes
            normalize_mean: Mean for normalization (optional, computed from data if not provided)
            normalize_std: Std for normalization (optional, computed from data if not provided)

        Returns:
            Dictionary containing:
                - 'species': List of species names from CSV
                - 'predictions': Predicted class indices
                - 'confidence': Confidence scores (max softmax probability)
                - 'probabilities': Full probability distributions (if return_probabilities=True)
        """
        # Load CSV
        data = pd.read_csv(csv_path)

        # Extract species names and fluorescence values
        species_names = data.iloc[:, 0].values
        fluorescence_values = data.iloc[:, 1:].values.astype(np.float32)

        # Normalize
        normalized_data, used_mean, used_std = self.normalize_data(
            fluorescence_values,
            normalize_mean,
            normalize_std
        )

        # Make predictions
        all_predictions = []
        all_confidences = []
        all_probabilities = []

        with torch.no_grad():
            for i in range(len(normalized_data)):
                # Convert to tensor and add channel dimension
                x = torch.FloatTensor(normalized_data[i]).unsqueeze(0).unsqueeze(0).to(self.device)

                # Forward pass
                output = self.model(x)

                # Get predictions and confidences
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction]

                all_predictions.append(prediction)
                all_confidences.append(confidence)
                all_probabilities.append(probabilities)

        results = {
            'species': species_names,
            'predictions': np.array(all_predictions),
            'confidence': np.array(all_confidences),
            'fluorescence_length': fluorescence_values.shape[1],
            'normalization_mean': used_mean,
            'normalization_std': used_std
        }

        if return_probabilities:
            results['probabilities'] = np.array(all_probabilities)

        return results

    def predict_with_labels(self, csv_path, label_encoder=None, return_probabilities=False,
                           normalize_mean=None, normalize_std=None):
        """
        Make predictions and return species names instead of class indices.

        Args:
            csv_path: Path to CSV file
            label_encoder: sklearn LabelEncoder object mapping class indices to species names.
                          If None, uses class indices as labels.
            return_probabilities: If True, return softmax probabilities
            normalize_mean: Mean for normalization
            normalize_std: Std for normalization

        Returns:
            Dictionary with additional 'predicted_species' field containing species names
        """
        results = self.predict(csv_path, return_probabilities, normalize_mean, normalize_std)

        if label_encoder is not None:
            results['predicted_species'] = label_encoder.inverse_transform(results['predictions'])
        else:
            results['predicted_species'] = results['predictions']

        return results


def main():
    """
    Main function for batch ensemble predictions.
    This script is designed to be called by precompute_predictions.py
    """
    print("ResNet module loaded. Use get_resnet_predictions() for batch predictions.")


# ============================================================================
# BATCH PREDICTION FUNCTION FOR ENSEMBLE
# ============================================================================

def get_resnet_predictions(X_data, num_classes, models_dir):
    """
    Get ResNet predictions for the ensemble stacking.

    Args:
        X_data: Input DataFrame with fluorescence data (rows=samples, cols=features)
        num_classes: Number of output classes
        models_dir: Directory containing model files

    Returns:
        numpy array of shape (n_samples, num_classes) with class probabilities,
        or None if model loading fails
    """
    # Try to find the ResNet model file
    model_files = list(Path(models_dir).glob("RESNET_*.pth"))
    if not model_files:
        print("[WARN] ResNet model not found in models_dir")
        return None

    model_path = str(model_files[0])

    try:
        # Initialize device
        device = torch.device('cuda' if torch.cuda.is_available() else
                            ('mps' if torch.backends.mps.is_available() else 'cpu'))

        # Load model
        model = ResNet1D(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Extract fluorescence data and normalize
        fluorescence_values = X_data.iloc[:, :].values.astype(np.float32)

        # Normalize
        mean = fluorescence_values.mean()
        std = fluorescence_values.std()
        normalized_data = (fluorescence_values - mean) / std

        # Get predictions for all samples
        all_probs = []

        with torch.no_grad():
            for i in range(len(normalized_data)):
                # Convert to tensor and add channel dimension
                x = torch.FloatTensor(normalized_data[i]).unsqueeze(0).unsqueeze(0).to(device)

                # Forward pass
                output = model(x)

                # Get probabilities
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                all_probs.append(probs)

        return np.array(all_probs)

    except Exception as e:
        print(f"[ERROR] Failed to get ResNet predictions: {e}")
        return None


if __name__ == '__main__':
    main()
