"""
CNN-based shark species inference from fluorescence curves.
Loads trained EfficientNet model and predicts species from temperature/fluorescence data.
"""
import numpy as np
import pandas as pd
import joblib
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# Constants
MODEL_DIR = Path(__file__).parent / "model"
BUNDLE_PATH = MODEL_DIR / "cnn_bundle.pkl"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalization from ImageNet (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CNNModel(nn.Module):
    """EfficientNet-based CNN model (must match training architecture)."""
    def __init__(self, num_classes, dropout1=0.7, dropout2=0.5):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)  # No pretrained weights
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


def generate_image(temps: np.ndarray, values: np.ndarray) -> Optional[Image.Image]:
    """
    Generate PIL image from fluorescence curve.

    Args:
        temps: Temperature values (1D array)
        values: Fluorescence values (1D array)

    Returns:
        PIL Image or None if generation fails
    """
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


def get_inference_transform():
    """Get transform pipeline for inference (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class SharkSpeciesInference:
    """Inference class for shark species prediction."""

    def __init__(self, bundle_path: Path = BUNDLE_PATH):
        """
        Initialize inference with trained model.

        Args:
            bundle_path: Path to the model bundle pickle file
        """
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path}. "
                f"Please ensure cnn_bundle.pkl is in {MODEL_DIR}/"
            )

        print(f"Loading model from {bundle_path}...")
        # Load with CPU mapping if CUDA not available
        import functools
        import sys

        # Ensure CNNModel is available in the correct module namespace for unpickling
        # This handles cases where the model was trained in a different module context
        current_module = sys.modules[__name__]
        if 'worker.worker' in sys.modules:
            sys.modules['worker.worker'].CNNModel = CNNModel

        if not torch.cuda.is_available():
            # Monkey-patch torch.load to always use CPU
            original_torch_load = torch.load
            torch.load = functools.partial(original_torch_load, map_location='cpu')

        bundle = joblib.load(bundle_path)

        # Restore original torch.load
        if not torch.cuda.is_available():
            torch.load = original_torch_load

        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.cv_accuracy = bundle.get("cv_accuracy", None)
        self.params = bundle.get("params", {})

        # Move model to device and set to eval mode
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.transform = get_inference_transform()
        self.num_classes = len(self.label_encoder.classes_)

        print(f"Model loaded successfully!")
        print(f"  Device: {DEVICE}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Species: {list(self.label_encoder.classes_)}")
        if self.cv_accuracy:
            print(f"  CV Accuracy: {self.cv_accuracy:.2f}%")

    def predict(
        self,
        temps: np.ndarray,
        values: np.ndarray,
        return_probabilities: bool = False
    ) -> Dict:
        """
        Predict species from fluorescence curve.

        Args:
            temps: Temperature values (1D array)
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
        # Generate image
        img = generate_image(temps, values)
        if img is None:
            raise ValueError("Failed to generate image from fluorescence curve")

        # Transform and prepare for model
        img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = self.model(img_tensor)
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
        temps: np.ndarray,
        values_batch: np.ndarray,
        return_probabilities: bool = False
    ) -> List[Dict]:
        """
        Predict species for multiple fluorescence curves.

        Args:
            temps: Temperature values (1D array, shared across batch)
            values_batch: Fluorescence values (2D array: [n_samples, n_temps])
            return_probabilities: If True, return all class probabilities

        Returns:
            List of prediction dictionaries
        """
        results = []
        for values in values_batch:
            result = self.predict(temps, values, return_probabilities)
            results.append(result)
        return results


def load_test_data(csv_path: Path, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test samples from CSV.

    Args:
        csv_path: Path to shark_dataset.csv
        n_samples: Number of random samples to load

    Returns:
        Tuple of (temps, values_batch, true_species)
    """
    df = pd.read_csv(csv_path)

    # Sample randomly
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)

    # Extract temperatures from column names
    temp_cols = [col for col in df.columns if col != 'Species']
    temps = np.array([float(col) for col in temp_cols])

    # Extract values
    values_batch = sample_df[temp_cols].values
    true_species = sample_df['Species'].values

    return temps, values_batch, true_species


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
            'sample_index': int
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
        logger.info("Initializing SharkSpeciesInference model...")
        inference = SharkSpeciesInference()

        # Run prediction with all probabilities
        logger.info("Running prediction...")
        result = inference.predict(temps, sample_values, return_probabilities=True)
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
    print("TESTING INFERENCE")
    print("="*70)

    # Load model
    inference = SharkSpeciesInference()

    # Load test data
    print(f"\nLoading {n_samples} random samples from {csv_path.name}...")
    temps, values_batch, true_species = load_test_data(csv_path, n_samples)

    print(f"  Temperature range: {temps.min():.2f} - {temps.max():.2f}")
    print(f"  Samples shape: {values_batch.shape}")

    # Run inference
    print(f"\nRunning inference on {len(values_batch)} samples...")
    correct = 0
    total = len(values_batch)

    print("\nResults:")
    print("-"*70)
    for i, (values, true_label) in enumerate(zip(values_batch, true_species)):
        result = inference.predict(temps, values, return_probabilities=False)
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
