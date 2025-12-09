"""
Shark Species Classification Inference Module

Loads trained CNN model and makes predictions on fluorescence curve data.
Returns top-k predictions with confidence scores for all 57 shark species.

INTERFACE CONTRACT:
  Implements InferenceInterface and provides run_inference() for worker.py.
  See inference_interface.py for the required interface that any replacement
  model inference module must implement.

PRODUCTION USE:
  worker.py imports and calls:
    from worker.inference import run_inference
    result = run_inference(filepath, sample_index, device)

DEVELOPMENT USE:
  from worker.inference import SharkClassifier

  classifier = SharkClassifier()

  # Method 1: Predict from fluorescence curve
  predictions = classifier.predict(fluorescence_values, top_k=57)

  # Method 2: Predict from CSV row
  import pandas as pd
  df = pd.read_csv('shark_dataset.csv')
  predictions = classifier.predict_from_csv_row(df.iloc[0], top_k=57)

TO SWAP THIS MODEL:
  1. Ensure your new inference.py implements InferenceInterface
  2. Implement run_inference(filepath, sample_index, device) function
  3. Place your model files in backend/worker/model/
  4. Replace this inference.py file
  5. No changes needed to worker.py - it will work automatically
"""

import pickle
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

from .inference_interface import InferenceInterface


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

class SharkClassifier(InferenceInterface):
    """
    Shark species classifier using CNN (EfficientNet-B0).

    Predicts shark species from fluorescence curve data.
    Implements InferenceInterface for compatibility with worker.py.
    """

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None, verbose: bool = False):
        """
        Initialize the classifier.

        Args:
            model_dir: Path to directory containing model files.
                      If None, uses ./model/
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
            verbose: Whether to print initialization messages.
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / "model"
        else:
            model_dir = Path(model_dir)

        self.model_dir = model_dir
        self.verbose = verbose

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load label encoder
        label_encoder_path = model_dir / "label_encoder.pkl"
        if not label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")

        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load model (full model with architecture + weights)
        model_path = model_dir / "model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # PyTorch 2.6+ requires weights_only=False for full models (not just state_dict)
        # Temporarily add this module to sys.modules under __main__ so pickle can find classes
        import sys
        current_module = sys.modules[__name__]
        sys.modules['__main__'] = current_module
        try:
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        finally:
            # Restore sys.modules
            if '__main__' not in sys.modules or sys.modules['__main__'] is current_module:
                del sys.modules['__main__']

        self.model.to(self.device)
        self.model.eval()

        # Setup image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.verbose:
            print(f"Loaded SharkClassifier")
            print(f"  Classes: {len(self.label_encoder.classes_)}")
            print(f"  Device: {self.device}")

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
            species = str(self.label_encoder.classes_[idx])  # Ensure string
            confidence = float(probabilities[idx])  # Ensure native Python float

            predictions.append({
                'rank': int(rank),  # Ensure int
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
# PRODUCTION API - FOR WORKER
# ============================================================================

# Global classifier instance (lazy-loaded)
_classifier = None

def run_inference(filepath: str, sample_index: int = 0, device: Optional[str] = None) -> Dict:
    """
    Production inference function for the worker.

    Loads a CSV file, extracts a specific sample, and runs prediction.

    Args:
        filepath: Path to CSV file with fluorescence data
        sample_index: Index of sample in CSV to predict on (default: 0)
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.

    Returns:
        Dict with keys:
            - 'predictions': List of top predictions with rank, species, confidence
            - 'true_species': The actual species from CSV (if Species column exists)
            - 'sample_index': The sample index that was predicted
            - 'success': Whether prediction was successful

    Raises:
        FileNotFoundError: If CSV file or model files not found
        ValueError: If sample_index out of range or invalid data
    """
    global _classifier

    # Initialize classifier once (lazy load)
    if _classifier is None:
        _classifier = SharkClassifier(device=device, verbose=False)

    # Load CSV
    if not Path(filepath).exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Get sample
    if sample_index >= len(df) or sample_index < 0:
        raise ValueError(f"Sample index {sample_index} out of range for CSV with {len(df)} rows")

    sample_row = df.iloc[sample_index]

    # Extract fluorescence values (all columns except 'Species')
    fluorescence_cols = [col for col in df.columns if col != 'Species']
    fluorescence_values = sample_row[fluorescence_cols].values.astype(float)

    # Get true species if available
    true_species = sample_row.get('Species', None) if 'Species' in sample_row.index else None

    # Make prediction
    predictions = _classifier.predict(fluorescence_values, top_k=57)

    # Ensure all predictions are JSON-serializable (native Python types)
    clean_predictions = []
    for i, pred in enumerate(predictions):
        rank = int(pred['rank'])
        species = str(pred['species'])
        confidence = float(pred['confidence'])

        # Log suspicious values
        if confidence != confidence:  # NaN check
            print(f"[WARNING] NaN confidence detected at rank {rank}: {pred['confidence']} (raw type: {type(pred['confidence'])})")
        elif confidence < 0 or confidence > 1:
            print(f"[WARNING] Out-of-range confidence at rank {rank}: {confidence}")

        clean_predictions.append({
            'rank': rank,
            'species': species,
            'confidence': confidence
        })

        if i < 3:  # Log first 3
            print(f"[inference] Prediction {i+1}: {species} = {confidence} (type: {type(confidence).__name__})")

    # Format result with proper types
    result = {
        'predictions': clean_predictions,
        'sample_index': int(sample_index),
        'success': True,
    }

    if true_species is not None:
        result['true_species'] = str(true_species)
        result['top_prediction_correct'] = bool(clean_predictions[0]['species'] == true_species)

    print(f"[inference] Result success: {result['success']}, num_predictions: {len(result['predictions'])}")
    print(f"[inference] First 3 confidences: {[p['confidence'] for p in result['predictions'][:3]]}")

    return result