"""
Inference Script for Shark Species Classification

Loads the trained CNN model and makes predictions on fluorescence curve data.
Returns top-k predictions with confidence scores for all 57 species.

Can predict from:
  1. Fluorescence curve values (list or numpy array)
  2. CSV row data (pandas Series)

Usage:
    from inference import SharkClassifier

    # Initialize classifier
    classifier = SharkClassifier()

    # Method 1: Predict from fluorescence curve (list or numpy array)
    predictions = classifier.predict(fluorescence_values, top_k=57)

    # Method 2: Predict from CSV row
    import pandas as pd
    df = pd.read_csv('shark_dataset.csv')
    predictions = classifier.predict_from_csv_row(df.iloc[0], top_k=57)

    # predictions will be a list of dicts:
    # [{'rank': 1, 'species': 'Great White Shark', 'confidence': 0.998}, ...]
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Union, Optional
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
import copy


# ============================================================================
# MODEL CLASSES (Required for loading the trained model)
# ============================================================================

class GaussianNoise(nn.Module):
    """Add Gaussian noise to tensor."""
    def __init__(self, std: float = 0.005):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class SharkCNN(nn.Module):
    """EfficientNet-B0 with custom classifier head."""

    def __init__(self, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)

        # Load pretrained weights
        base_model = efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone.load_state_dict(base_model.state_dict())

        in_features = self.backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.6217843386251581),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.19498440140497733),
            nn.Linear(hidden_dim, num_classes)
        )

        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha: float = 1.0, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============================================================================
# CLASSIFIER
# ============================================================================

class SharkClassifier:
    """
    Shark species classifier using CNN (EfficientNet-B0).

    Predicts shark species from fluorescence curve data.
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the classifier.

        Args:
            model_dir: Path to directory containing model files.
                      If None, uses ./final_model/
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / "final_model"
        else:
            model_dir = Path(model_dir)

        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        config_path = model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found at {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load label encoder
        label_encoder_path = model_dir / "label_encoder.pkl"
        if not label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")

        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load model (full model with architecture + weights)
        model_path = model_dir / "shark_cnn_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # PyTorch 2.6+ requires weights_only=False for full models (not just state_dict)
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

        # Setup image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Loaded SharkClassifier")
        print(f"  Model: {self.config['model_type']}")
        print(f"  Classes: {self.config['num_classes']}")
        print(f"  Device: {self.device}")
        print(f"  Trained on: {self.config['total_training_samples']} samples")

    def _fluorescence_to_image(self, fluorescence: Union[List[float], np.ndarray]) -> Image.Image:
        """
        Convert fluorescence curve to image.

        Args:
            fluorescence: Fluorescence values (length should match temperature points)

        Returns:
            PIL Image
        """
        fluorescence = np.array(fluorescence, dtype=float)
        temps = np.linspace(20, 95, len(fluorescence))

        fig, ax = plt.subplots(figsize=(3.0, 2.25), dpi=96)
        ax.plot(temps, fluorescence, 'b-', linewidth=2)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Fluorescence')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close(fig)

        return img

    def predict(self,
                fluorescence: Union[List[float], np.ndarray],
                top_k: int = 57,
                return_all: bool = False) -> List[Dict[str, Union[str, float]]]:
        """
        Predict shark species from fluorescence curve.

        Args:
            fluorescence: Fluorescence curve values (list or numpy array)
            top_k: Number of top predictions to return (default: 57 = all classes)
            return_all: If True, return all 57 classes regardless of top_k

        Returns:
            List of predictions, each a dict with:
                - 'species': Species name
                - 'confidence': Confidence score (0-1)
                - 'rank': Rank (1 = top prediction)

        Example:
            >>> predictions = classifier.predict(fluor_values, top_k=5)
            >>> print(predictions[0])
            {'species': 'Great White Shark', 'confidence': 0.998, 'rank': 1}
        """
        # Convert fluorescence to image
        img = self._fluorescence_to_image(fluorescence)

        # Transform and add batch dimension
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # Get top-k predictions
        if return_all:
            k = len(probabilities)
        else:
            k = min(top_k, len(probabilities))

        # Sort by confidence (descending)
        top_indices = np.argsort(probabilities)[::-1][:k]

        predictions = []
        for rank, idx in enumerate(top_indices, 1):
            species = self.label_encoder.classes_[idx]
            confidence = float(probabilities[idx])

            predictions.append({
                'rank': rank,
                'species': species,
                'confidence': confidence
            })

        return predictions

    def predict_batch(self,
                      fluorescence_list: List[Union[List[float], np.ndarray]],
                      top_k: int = 5) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Predict multiple samples at once.

        Args:
            fluorescence_list: List of fluorescence curves
            top_k: Number of top predictions per sample

        Returns:
            List of prediction lists (one per sample)
        """
        return [self.predict(fluor, top_k=top_k) for fluor in fluorescence_list]

    def get_species_list(self) -> List[str]:
        """Get list of all species the model can predict."""
        return self.label_encoder.classes_.tolist()

    def get_model_info(self) -> Dict:
        """Get model configuration and metadata."""
        return self.config

    def predict_from_csv_row(self, csv_row: Dict, top_k: int = 57) -> List[Dict[str, Union[str, float]]]:
        """
        Predict shark species from a single CSV row.

        Args:
            csv_row: Dictionary or pandas Series with columns for fluorescence values
                     (should have all columns except 'Species')
            top_k: Number of top predictions to return (default: 57 = all classes)

        Returns:
            List of predictions with rank, species, and confidence scores
        """
        # Extract fluorescence values (all columns except 'Species')
        fluorescence = {}
        for col in csv_row.index:
            if col != 'Species':
                fluorescence[col] = csv_row[col]

        # Convert to numpy array
        fluorescence_values = np.array(list(fluorescence.values()), dtype=float)

        # Use existing predict method
        return self.predict(fluorescence_values, top_k=top_k)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("SHARK SPECIES CLASSIFIER - INFERENCE DEMO")
    print("=" * 80)
    print()

    # Initialize classifier
    try:
        classifier = SharkClassifier()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run train_final_model.py first to train and save the model!")
        exit(1)

    print()

    # Example 1: Load from CSV and predict
    print("Example 1: Predict from real data")
    print("-" * 80)

    # Try to load a sample from the dataset
    data_path = Path(__file__).parent.parent / "data" / "shark_dataset.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)

        # Get a random sample
        sample_idx = 0
        sample_row = df.iloc[sample_idx]
        true_species = sample_row['Species']

        # Extract fluorescence values (all columns except 'Species')
        temp_cols = [col for col in df.columns if col != 'Species']
        fluorescence = sample_row[temp_cols].values.astype(float)

        print(f"Sample #{sample_idx + 1}")
        print(f"True species: {true_species}")
        print(f"Fluorescence curve: {len(fluorescence)} temperature points")
        print()

        # Get top-5 predictions
        print("Top-5 Predictions:")
        predictions = classifier.predict(fluorescence, top_k=5)

        for pred in predictions:
            confidence_pct = pred['confidence'] * 100
            marker = "✓" if pred['species'] == true_species else " "
            print(f"  {marker} {pred['rank']}. {pred['species']:<40} {confidence_pct:>6.2f}%")

        print()

        # Get all predictions
        print("Getting all 57 species predictions...")
        all_predictions = classifier.predict(fluorescence, top_k=57)
        print(f"Total predictions: {len(all_predictions)}")
        print(f"Sum of confidences: {sum(p['confidence'] for p in all_predictions):.4f} (should be ~1.0)")
        print()

        # Show bottom 5 (least likely)
        print("Bottom-5 Predictions (least likely):")
        for pred in all_predictions[-5:]:
            confidence_pct = pred['confidence'] * 100
            print(f"  {pred['rank']}. {pred['species']:<40} {confidence_pct:>6.2f}%")

    else:
        print(f"Dataset not found at {data_path}")
        print("Creating synthetic example instead...")

        # Create a dummy fluorescence curve
        temps = np.linspace(20, 95, 76)
        fluorescence = np.random.rand(76) * 100

        predictions = classifier.predict(fluorescence, top_k=5)

        print("\nTop-5 Predictions (on random data):")
        for pred in predictions:
            confidence_pct = pred['confidence'] * 100
            print(f"  {pred['rank']}. {pred['species']:<40} {confidence_pct:>6.2f}%")

    print()
    print("-" * 80)
    print("Example 2: Predict from CSV row")
    print("-" * 80)

    if data_path.exists():
        df = pd.read_csv(data_path)
        sample_row = df.iloc[0]

        print(f"Predicting from CSV row 0...")
        print(f"True species: {sample_row['Species']}")
        print()

        # Use the new predict_from_csv_row method
        predictions = classifier.predict_from_csv_row(sample_row, top_k=5)

        print("Top-5 Predictions from CSV row:")
        for pred in predictions:
            confidence_pct = pred['confidence'] * 100
            marker = "✓" if pred['species'] == sample_row['Species'] else " "
            print(f"  {marker} {pred['rank']}. {pred['species']:<40} {confidence_pct:>6.2f}%")

    print()
    print("=" * 80)
    print("INFERENCE DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Usage in your code:")
    print()
    print("  from inference import SharkClassifier")
    print()
    print("  classifier = SharkClassifier()")
    print()
    print("  # Method 1: Predict from fluorescence values")
    print("  predictions = classifier.predict(fluorescence_values, top_k=57)")
    print()
    print("  # Method 2: Predict from CSV row")
    print("  import pandas as pd")
    print("  df = pd.read_csv('shark_dataset.csv')")
    print("  row = df.iloc[0]  # Get first row")
    print("  predictions = classifier.predict_from_csv_row(row, top_k=57)")
    print()
    print("  # predictions[0] = top prediction")
    print("  # predictions = all 57 species sorted by confidence")
    print()
