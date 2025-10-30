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

def csv_to_image(csv_path, save_image_path=None):
    """
    Convert frequency spectrum CSV data to PIL Image using matplotlib line plot.
    Replicates the exact format used in training (generate_images.py).

    Args:
        csv_path: Path to CSV file
        save_image_path: Optional path to save the generated image

    Returns:
        PIL Image object
    """
    df = pd.read_csv(csv_path)

    # Get the frequency columns (all except 'Species')
    frequency_cols = [col for col in df.columns if col != 'Species']

    # Convert column names to float (frequency values)
    time_values = np.array([float(col) for col in frequency_cols])

    # Get the first row of data (or average if multiple rows)
    if len(df) > 1:
        signal_values = df[frequency_cols].mean(axis=0).values.astype(float)
    else:
        signal_values = df[frequency_cols].iloc[0].values.astype(float)

    # Create matplotlib line plot (same as generate_images.py)
    IMAGE_SIZE = (224, 224)
    DPI = 100

    fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/DPI, IMAGE_SIZE[1]/DPI), dpi=DPI)

    # Plot the line
    ax.plot(time_values, signal_values, linewidth=2, color='#2E86AB')

    # Remove axes and labels for clean image (exactly as in generate_images.py)
    ax.set_xlim(time_values.min(), time_values.max())
    ax.set_ylim(signal_values.min() - 0.001, signal_values.max() + 0.001)
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

def run_inference(csv_path, model_path=None, device='cpu', save_image_path=None):
    """
    Run inference on a CSV file

    Args:
        csv_path: Path to the CSV file
        model_path: Path to the .pth model file (defaults to finding it in worker dir)
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
        # Use the best model (fold3 achieved 100% validation accuracy)
        model_files.sort()
        model_path = model_files[0]  # Use first fold as default

    # Load model
    model = load_checkpoint(str(model_path), device=device)
    transform = get_transforms()

    # Convert CSV to image
    img = csv_to_image(csv_path, save_image_path=save_image_path)

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

    return {
        'winner': winner_label,
        'confidence': float(winner_prob),
        'topk': topk_results,
        'source_file': str(csv_path),
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
