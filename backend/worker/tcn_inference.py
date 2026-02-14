"""
TCN-based shark species inference from fluorescence curves.
Loads trained TCN model and predicts species from temperature/fluorescence data.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# Constants
MODEL_DIR = Path(__file__).parent / "tcn"
BUNDLE_PATH = MODEL_DIR / "tcn_bundle.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- TCN Architecture ---
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                     padding=self.padding, dilation=dilation)
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.bn2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=3, dropout=0.2, reverse_dilation=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            if reverse_dilation:
                dilation_size = 2 ** (num_levels - i - 1)
            else:
                dilation_size = 2 ** i

            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size,
                            dilation_size, dropout)
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        y = self.network(x)
        y = torch.mean(y, dim=2)
        return self.fc(y)


class TCNInference:
    """Inference class for TCN-based shark species prediction."""

    def __init__(self, bundle_path: Path = BUNDLE_PATH):
        """
        Initialize inference with trained TCN model.

        Args:
            bundle_path: Path to the model bundle pickle file
        """
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path}. "
                f"Please ensure tcn_bundle.pkl is in {MODEL_DIR}/"
            )

        print(f"Loading TCN model from {bundle_path}...")

        # Load with CPU mapping if CUDA not available
        import functools
        import sys

        # Ensure TCN classes are available in the correct module namespace for unpickling
        current_module = sys.modules[__name__]
        if 'worker.worker' in sys.modules:
            sys.modules['worker.worker'].TemporalConvNet = TemporalConvNet
            sys.modules['worker.worker'].TemporalBlock = TemporalBlock
            sys.modules['worker.worker'].CausalConv1d = CausalConv1d

        if not torch.cuda.is_available():
            original_torch_load = torch.load
            torch.load = functools.partial(original_torch_load, map_location='cpu')

        bundle = joblib.load(bundle_path)

        # Restore original torch.load
        if not torch.cuda.is_available():
            torch.load = original_torch_load

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.label_encoder = bundle["label_encoder"]
        self.hyperparameters = bundle.get("hyperparameters", {})
        self.test_accuracy = bundle.get("test_accuracy", None)
        self.test_f1 = bundle.get("test_f1", None)

        # Move model to device and set to eval mode
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.num_classes = len(self.label_encoder.classes_)

        print(f"TCN model loaded successfully!")
        print(f"  Device: {DEVICE}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Species: {list(self.label_encoder.classes_)}")
        if self.test_accuracy:
            print(f"  Test Accuracy: {self.test_accuracy:.2f}%")
        if self.test_f1:
            print(f"  Test F1: {self.test_f1:.4f}")

    def predict(
        self,
        values: np.ndarray,
        return_probabilities: bool = False
    ) -> Dict:
        """
        Predict species from fluorescence curve.

        Args:
            values: Fluorescence values (1D array)
            return_probabilities: If True, return all class probabilities

        Returns:
            Dictionary with prediction results:
            {
                'species': str,
                'confidence': float,
                'probabilities': dict (if return_probabilities=True)
            }
        """
        # Scale the input
        values_scaled = self.scaler.transform(values.reshape(1, -1))

        # Reshape for TCN: (batch, channels, sequence_length)
        # TCN expects (batch_size, 1, sequence_length)
        x = torch.FloatTensor(values_scaled).unsqueeze(1).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        # Convert to numpy
        predicted_idx = predicted_idx.cpu().item()
        confidence = confidence.cpu().item()
        probabilities_np = probabilities.cpu().numpy()[0]

        # Get species name
        species = self.label_encoder.inverse_transform([predicted_idx])[0]

        result = {
            'species': species,
            'confidence': confidence
        }

        if return_probabilities:
            result['probabilities'] = {
                self.label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(probabilities_np)
            }

        return result

    def predict_batch(
        self,
        values_batch: np.ndarray,
        return_probabilities: bool = False
    ) -> List[Dict]:
        """
        Predict species for multiple fluorescence curves.

        Args:
            values_batch: Fluorescence values (2D array: [n_samples, n_features])
            return_probabilities: If True, return all class probabilities

        Returns:
            List of prediction dictionaries
        """
        results = []
        for values in values_batch:
            result = self.predict(values, return_probabilities)
            results.append(result)
        return results


def run_inference(filepath: str, sample_index: int = 0, device: str = None) -> Dict:
    """
    Run inference on a specific sample from a CSV file.

    Args:
        filepath: Path to the CSV file containing fluorescence data
        sample_index: Index of the sample to analyze (0-based)
        device: Device to use ('cuda' or 'cpu'), auto-detected if None

    Returns:
        Dictionary with:
        {
            'success': bool,
            'predictions': [
                {
                    'species': str,
                    'confidence': float,
                    'rank': int
                }
            ],
            'sample_index': int,
            'curve_data': {
                'frequencies': list,
                'signal': list
            }
        }
    """
    import traceback
    import logging

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading CSV from {filepath}")

        # Load CSV
        df = pd.read_csv(filepath)
        logger.info(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)[:5]}...")

        if sample_index < 0 or sample_index >= len(df):
            raise ValueError(f"sample_index {sample_index} out of range [0, {len(df)-1}]")

        # Extract temperature columns and values
        # Temperature columns are numeric column names
        temp_cols = []
        for col in df.columns:
            if col == 'Species':
                continue
            try:
                float(col)
                temp_cols.append(col)
            except (ValueError, TypeError):
                continue

        if not temp_cols:
            raise ValueError(f"No temperature columns found in CSV. Columns: {list(df.columns)}")

        temps = np.array([float(col) for col in temp_cols])
        logger.info(f"Found {len(temp_cols)} temperature columns, range: {temps.min():.2f} - {temps.max():.2f}")

        # Get the specific sample
        sample_values = df.iloc[sample_index][temp_cols].values.astype(float)
        logger.info(f"Extracted sample {sample_index}, value range: {sample_values.min():.2f} - {sample_values.max():.2f}")

        # Initialize inference model
        logger.info("Initializing TCNInference model...")
        inference = TCNInference()

        # Run prediction with all probabilities
        logger.info("Running prediction...")
        result = inference.predict(sample_values, return_probabilities=True)
        logger.info(f"Prediction completed: {result['species']} with confidence {result['confidence']:.4f}")

        # Format predictions in descending order of confidence
        predictions = []
        probs = result.get('probabilities', {})

        # Sort by confidence descending
        sorted_species = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        for rank, (species, confidence) in enumerate(sorted_species, start=1):
            predictions.append({
                'species': species,
                'confidence': float(confidence),
                'rank': rank
            })

        logger.info(f"Returning {len(predictions)} predictions")
        return {
            'success': True,
            'predictions': predictions,
            'sample_index': sample_index,
            'curve_data': {
                'frequencies': temps.tolist(),
                'signal': sample_values.tolist()
            }
        }

    except Exception as e:
        error_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(error_msg)  # Also print to stdout for immediate visibility
        return {
            'success': False,
            'error': str(e),
            'predictions': [],
            'sample_index': sample_index
        }


def test_inference(csv_path: Path = None, n_samples: int = 20):
    """
    Test inference on random samples from dataset.

    Args:
        csv_path: Path to shark_dataset.csv
        n_samples: Number of samples to test
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent.parent / "data" / "shark_dataset.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Test data not found at {csv_path}")

    print("="*70)
    print("TESTING TCN INFERENCE")
    print("="*70)

    # Load model
    inference = TCNInference()

    # Load test data
    print(f"\nLoading {n_samples} random samples from {csv_path.name}...")
    df = pd.read_csv(csv_path)

    # Sample randomly
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)

    # Extract temperature columns
    temp_cols = [col for col in df.columns if col != 'Species']
    temps = np.array([float(col) for col in temp_cols])

    # Extract values and true labels
    values_batch = sample_df[temp_cols].values
    true_species = sample_df['Species'].values

    print(f"  Temperature range: {temps.min():.2f} - {temps.max():.2f}")
    print(f"  Samples shape: {values_batch.shape}")

    # Run inference
    print(f"\nRunning inference on {len(values_batch)} samples...")
    correct = 0
    total = len(values_batch)

    print("\nResults:")
    print("-"*70)
    for i, (values, true_label) in enumerate(zip(values_batch, true_species)):
        result = inference.predict(values, return_probabilities=False)
        predicted = result['species']
        confidence = result['confidence']

        is_correct = predicted == true_label
        correct += int(is_correct)

        status = "✓" if is_correct else "✗"
        print(f"{i+1:2d}. {status} True: {true_label:30s} | Pred: {predicted:30s} | Conf: {confidence:.4f}")

    # Summary
    accuracy = 100.0 * correct / total
    print("-"*70)
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    # Run tests
    test_inference(n_samples=50)
