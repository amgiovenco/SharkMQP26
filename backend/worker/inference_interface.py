"""
Inference Interface - Required contract for plug-and-play model compatibility.

To make your model compatible with worker.py, implement the run_inference() function
with the exact signature and return format shown below.
"""

from typing import Dict, List, Optional, TypedDict

class PredictionDict(TypedDict):
    """Single prediction result."""
    rank: int
    species: str
    confidence: float


class CurveData(TypedDict):
    """Optional curve data for visualization."""
    frequencies: List[float]
    signal: List[float]


class InferenceResult(TypedDict, total=False):
    """Complete inference result returned by run_inference()."""
    success: bool
    predictions: List[PredictionDict]
    sample_index: int
    curve_data: CurveData
    error: str  # Only present if success=False


def run_inference(filepath: str, sample_index: int = 0,
                 device: Optional[str] = None) -> InferenceResult:
    """
    Implement this method to use your own model for inference.
    
    Your inference module must export this function with this exact signature.

    Args:
        filepath: Path to CSV file with fluorescence data
        sample_index: Row index in CSV to predict on (0-based)
        device: 'cuda' or 'cpu' (None = auto-detect)

    Returns:
        Dict with the following structure:

        On success:
        {
            'success': True,
            'predictions': [
                {'rank': 1, 'species': 'Arabian smooth-hound', 'confidence': 0.95},
                {'rank': 2, 'species': 'Bull Shark', 'confidence': 0.03},
                ...
            ],
            'sample_index': 0,
            'curve_data': {
                'frequencies': [60.0, 60.5, 61.0, ...],
                'signal': [0.123, 0.456, ...]
            }
        }

        On failure:
        {
            'success': False,
            'error': 'Error message here',
            'predictions': [],
            'sample_index': 0
        }

    Implementation Requirements:
        1. Load CSV from filepath
        2. Extract temperature columns (numeric column names, excluding 'Species')
        3. Get fluorescence values for the specified sample_index
        4. Run your model's prediction
        5. Return all predictions ranked by confidence (descending)
        6. Handle errors gracefully and return success=False with error message

    Example Usage in worker.py:
        # To use CNN model:
        from worker.cnn_inference import run_inference as ml_inference

        # To use TCN model:
        from worker.tcn_inference import run_inference as ml_inference

        # To use your custom model:
        from worker.your_model_inference import run_inference as ml_inference

        # Then just call it:
        result = ml_inference(filepath='data.csv', sample_index=0)
    """
    raise NotImplementedError(
        "run_inference() must be implemented in your inference module."
    )