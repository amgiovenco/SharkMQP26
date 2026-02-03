"""
Inference Interface - Define the contract for any inference module.

Any model implementation must provide these methods to be compatible with worker.py.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional


class InferenceInterface(ABC):
    """
    Abstract base class defining the interface that any inference module must implement.

    This ensures compatibility with worker.py and makes it easy to swap models.
    """

    @abstractmethod
    def predict(self, fluorescence: Union[List[float], object],
                top_k: int = 57) -> List[Dict]:
        """
        Predict shark species from fluorescence curve.

        Args:
            fluorescence: Fluorescence values (list, array, or format-specific data)
            top_k: Number of top predictions to return

        Returns:
            List of dicts with keys: 'rank', 'species', 'confidence'
            Example: [
                {'rank': 1, 'species': 'Great White Shark', 'confidence': 0.98},
                {'rank': 2, 'species': 'Bull Shark', 'confidence': 0.02},
            ]
        """
        pass

    @abstractmethod
    def predict_from_csv_row(self, csv_row: Dict, top_k: int = 57) -> List[Dict]:
        """
        Predict from a pandas Series or dict (CSV row).

        Args:
            csv_row: pandas Series or dict with fluorescence columns (excluding 'Species')
            top_k: Number of top predictions

        Returns:
            List of prediction dicts (same format as predict())
        """
        pass

    @abstractmethod
    def get_species_list(self) -> List[str]:
        """Get list of all species this model can predict."""
        pass


def run_inference(filepath: str, sample_index: int = 0,
                 device: Optional[str] = None) -> Dict:
    """
    REQUIRED: Production inference function that worker.py calls.

    This is the entry point that worker.py imports and uses.
    Must have this exact signature.

    Args:
        filepath: Path to CSV file with fluorescence data
        sample_index: Row index in CSV to predict on
        device: 'cuda' or 'cpu' (None = auto-detect)

    Returns:
        Dict with keys:
            - 'predictions': List of top predictions (rank, species, confidence)
            - 'sample_index': The sample index that was predicted
            - 'success': True if successful
            - 'true_species': Actual species from CSV (if Species column exists)
            - 'top_prediction_correct': Whether top prediction matches true species

    Example return value:
        {
            'predictions': [
                {'rank': 1, 'species': 'Arabian smooth-hound', 'confidence': 1.0},
                {'rank': 2, 'species': 'Spotted Eagleray', 'confidence': 0.0},
                ...
            ],
            'true_species': 'Arabian smooth-hound',
            'sample_index': 0,
            'success': True,
            'top_prediction_correct': True
        }
    """
    raise NotImplementedError(
        "run_inference() must be implemented in your inference module"
    )
