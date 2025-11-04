import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
from PIL import Image
import io
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
WORKER_DIR = Path(__file__).parent

# Class names
CLASS_NAMES = [
    'Arabian smooth-hound',
    'Atlantic Sharpnose shark',
    'Blackchin guitarfish',
    'Blacknose shark',
    'Blackspotted smooth-hound',
    'Blacktip reef shark',
    'Blacktip shark',
    'Blue shark',
    'Bonnethead shark',
    'Bowmouth guitarfish',
    'Brownbanded bamboo shark',
    'Bull shark',
    'Caribbean reef shark',
    'Common thresher shark',
    'Copper shark',
    'Dusky shark',
    'Finetooth shark',
    'Great hammerhead shark',
    'Great white shark',
    'Grey reef shark',
    'Gulper shark',
    'Gummy shark',
    'Halavi guitarfish',
    'Hooktooth shark',
    'Japanese topeshark',
    'Java shark',
    'Lemon shark',
    'Longtail stingray',
    'Milk shark',
    'Narrownose smooth-hound',
    'Night shark',
    'Nurse shark',
    'Oceanic whitetip shark',
    'Pacific bonnethead shark',
    'Pacific guitarfish',
    'Pacific smalltail shark',
    'Pelagic thresher shark',
    'Porbeagle shark',
    'Roughskin dogfish',
    'Sandbar shark',
    'Sandtiger shark',
    'Scalloped bonnethead shark',
    'Scalloped hammerhead shark',
    'Shortfin mako',
    'Silky shark',
    'Silvertip shark',
    'Small tail shark',
    'Smooth hammerhead shark',
    'Spadenose stingray',
    'Spinner shark',
    'Spot-tail shark',
    'Spotted Eagleray',
    'Thornback ray',
    'Tiger shark',
    'Tope shark',
    'Whitecheeck shark',
    'Zebra shark',
]

def get_model(num_classes=57, device='cpu'):
    """Load EfficientNet-B0 model with the same architecture as training"""
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Replace classifier head to match training
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model.to(device)

def load_checkpoint(model_path, device='cpu'):
    """Load a trained model from checkpoint"""
    model = get_model(device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_transforms():
    """Get inference transforms (same as training validation transforms)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def downsample_lttb(frequencies, signal, threshold=150):
    """
    LTTB (Largest-Triangle-Three-Buckets) downsampling algorithm.
    Reduces data points while preserving the shape of the curve.

    Args:
        frequencies: List/array of frequency values
        signal: List/array of signal values
        threshold: Target number of points (default 150)

    Returns:
        Tuple of (downsampled_frequencies, downsampled_signal)
    """
    data_length = len(frequencies)

    # If already small enough, return as-is
    if data_length <= threshold:
        return frequencies, signal

    frequencies = np.array(frequencies)
    signal = np.array(signal)

    bucket_size = (data_length - 2) / (threshold - 2)
    downsampled_freq = [frequencies[0]]
    downsampled_signal = [signal[0]]

    a_index = 0

    for i in range(threshold - 2):
        avg_range_start = int(np.floor((i + 1) * bucket_size)) + 1
        avg_range_end = int(np.floor((i + 2) * bucket_size)) + 1
        avg_range_length = avg_range_end - avg_range_start

        avg_freq = np.mean(frequencies[avg_range_start:avg_range_end])
        avg_signal = np.mean(signal[avg_range_start:avg_range_end])

        range_start = int(np.floor(i * bucket_size)) + 1
        range_end = int(np.floor((i + 1) * bucket_size)) + 1

        max_area = -1
        max_area_index = -1

        for j in range(range_start, range_end):
            if j >= data_length:
                break
            area = abs(
                (downsampled_freq[-1] - frequencies[j]) * (signal[j] - downsampled_signal[-1]) -
                (downsampled_freq[-1] - avg_freq) * (downsampled_signal[-1] - avg_signal)
            ) / 2

            if area > max_area:
                max_area = area
                max_area_index = j

        if max_area_index >= 0:
            downsampled_freq.append(float(frequencies[max_area_index]))
            downsampled_signal.append(float(signal[max_area_index]))
            a_index = max_area_index

    downsampled_freq.append(float(frequencies[-1]))
    downsampled_signal.append(float(signal[-1]))

    return downsampled_freq, downsampled_signal

def csv_to_image(csv_path, sample_index=0, save_image_path=None):
    """
    Convert frequency spectrum CSV data to PIL Image using matplotlib line plot.
    Replicates the exact format used in training (generate_images.py).

    Args:
        csv_path: Path to CSV file
        sample_index: Which row to use (0-indexed). If CSV has multiple samples, selects this one
        save_image_path: Optional path to save the generated image

    Returns:
        PIL Image object
    """
    df = pd.read_csv(csv_path)

    # Get the frequency columns (all except 'Species' if it exists)
    frequency_cols = [col for col in df.columns if col != 'Species']

    # Convert column names to float (frequency values)
    # Handle both numeric column names and string column names
    try:
        time_values = np.array([float(col) for col in frequency_cols])
    except (ValueError, TypeError):
        # If columns can't be converted to float, assume they are already numeric indices
        time_values = np.array(frequency_cols, dtype=float)

    # Use the specified sample index (default to first if out of bounds)
    if sample_index >= len(df):
        sample_index = 0
    signal_values = df[frequency_cols].iloc[sample_index].values.astype(float)

    # Get min/max only from finite values
    finite_mask = np.isfinite(signal_values)
    if finite_mask.any():
        signal_min = np.min(signal_values[finite_mask])
        signal_max = np.max(signal_values[finite_mask])
    else:
        # No valid data - this should be caught by validation in run_inference
        signal_min = 0
        signal_max = 1

    # Handle case where all signal values are the same (flat line)
    if signal_min == signal_max:
        signal_min = signal_min - 0.5
        signal_max = signal_max + 0.5

    # Create matplotlib line plot (same as generate_images.py)
    IMAGE_SIZE = (224, 224)
    DPI = 100

    fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/DPI, IMAGE_SIZE[1]/DPI), dpi=DPI)

    # Plot the line (matplotlib handles NaN by not plotting those points)
    ax.plot(time_values, signal_values, linewidth=2, color='#2E86AB')

    # Remove axes and labels for clean image (exactly as in generate_images.py)
    ax.set_xlim(time_values.min(), time_values.max())
    ax.set_ylim(signal_min - 0.001, signal_max + 0.001)
    ax.axis('off')

    # Remove all margins and padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, dpi=DPI, bbox_inches='tight', pad_inches=0,
                facecolor='white', edgecolor='none', format='png')
    plt.close(fig)

    # Load image from buffer
    buf.seek(0)
    img = Image.open(buf).convert('RGB')

    # Save to file if requested
    if save_image_path:
        img.save(save_image_path)
        print(f"Saved visualization to: {save_image_path}")

    return img

def run_inference(csv_path, model_path=None, sample_index=0, device='cpu', save_image_path=None):
    """
    Run inference on a CSV file

    Args:
        csv_path: Path to the CSV file
        model_path: Path to the .pth model file (defaults to finding it in worker dir)
        sample_index: Which sample row to use (0-indexed). Default is 0
        device: Device to run inference on ('cpu' or 'cuda')
        save_image_path: Optional path to save the generated image visualization

    Returns:
        dict with inference results
    """
    # Find model if not provided
    if model_path is None:
        model_files = list(WORKER_DIR.glob('efficientnet_b0_fold*.pth'))
        if not model_files:
            raise FileNotFoundError(f"No .pth model files found in {WORKER_DIR}")
        # Use fold 1 as default
        model_files.sort()
        model_path = model_files[0]

    # Load model
    model = load_checkpoint(str(model_path), device=device)
    transform = get_transforms()

    # Get curve data for visualization
    df = pd.read_csv(csv_path)
    frequency_cols = [col for col in df.columns if col != 'Species']
    try:
        time_values = np.array([float(col) for col in frequency_cols])
    except (ValueError, TypeError):
        time_values = np.array(frequency_cols, dtype=float)

    if sample_index >= len(df):
        sample_index = 0
    signal_values = df[frequency_cols].iloc[sample_index].values.astype(float)

    # Validate signal data quality - reject only truly bad data
    if not np.any(np.isfinite(signal_values)):
        raise ValueError("Sample contains only NaN or Inf values - no valid measurement")

    # Check if there's at least some real signal (not all zeros or noise)
    finite_values = signal_values[np.isfinite(signal_values)]
    if len(finite_values) == 0:
        raise ValueError("Sample has no valid finite values")

    # Convert CSV to image (using specified sample index)
    img = csv_to_image(csv_path, sample_index=sample_index, save_image_path=save_image_path)

    # Apply transforms
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs[0]
        probs = torch.softmax(logits, dim=0)

        # Get top-k predictions
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)

        winner_idx = top_indices[0].item()
        winner_label = CLASS_NAMES[winner_idx]
        winner_prob = top_probs[0].item()

    # Format results
    topk_results = []
    for prob, idx in zip(top_probs, top_indices):
        label = CLASS_NAMES[idx.item()]
        topk_results.append({
            'label': label,
            'prob': float(prob.item())
        })

    # Prepare curve data - only include finite values for visualization
    # NaN values are skipped since matplotlib handles them by not plotting
    curve_freqs = []
    curve_signal = []
    for f, s in zip(time_values, signal_values):
        if np.isfinite(s):
            curve_freqs.append(float(f))
            curve_signal.append(float(s))

    # Downsample curve data for faster API response
    downsampled_freq, downsampled_signal = downsample_lttb(
        curve_freqs,
        curve_signal,
        threshold=150
    )

    return {
        'winner': winner_label,
        'confidence': float(winner_prob),
        'topk': topk_results,
        'source_file': str(csv_path),
        'curve_data': {
            'frequencies': downsampled_freq,
            'signal': downsampled_signal,
        }
    }


if __name__ == '__main__':
    # Test inference
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        # Save image to same directory as CSV file
        csv_path = Path(csv_file)
        save_path = csv_path.parent / f"{csv_path.stem}_input.png"

        result = run_inference(csv_file, save_image_path=str(save_path))
        print(f"\nWinner: {result['winner']} ({result['confidence']:.2%})")
        print(f"\nTop 5:")
        for i, pred in enumerate(result['topk'], 1):
            print(f"  {i}. {pred['label']:<30} {pred['prob']:.4f}")
    else:
        print("Usage: python inference.py <csv_file>")
