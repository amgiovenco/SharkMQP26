"""
Prediction script for EfficientNet-B0 shark species classifier.

Takes CSV data with melting curve data (temperature and fluorescence values),
converts it to an image the same way the training notebook does, and returns
inference results with class predictions and confidence scores.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import os
import hashlib


# Configuration matching the training notebook
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))


class AddGaussianNoise(object):
    """Add Gaussian noise to tensor (matching training transforms)"""
    def __init__(self, std=0.005):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


def get_model(num_classes, hyperparams=None):
    """
    Build EfficientNet-B0 model with custom classifier head.

    Args:
        num_classes: Number of output classes
        hyperparams: Dictionary with dropout values. If None, uses defaults.

    Returns:
        Model instance
    """
    if hyperparams is None:
        hyperparams = {
            'cnn_dropout1': 0.7,
            'cnn_dropout2': 0.5,
        }

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Get input features
    in_features = model.classifier[1].in_features

    # Build custom classifier head
    classifier_layers = nn.Sequential(
        nn.Dropout(hyperparams['cnn_dropout1']),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(hyperparams['cnn_dropout2']),
        nn.Linear(256, num_classes)
    )

    # Replace classifier
    model.classifier = classifier_layers

    return model


def csv_to_image(csv_data):
    """
    Convert CSV melting curve data to an image.

    Assumes CSV has two columns: 'Temperature' (or similar x-axis) and
    'Fluorescence' (or similar y-axis). Creates a simple 2D scatter/line plot
    image that the model was trained on.

    Args:
        csv_data: Either a pandas DataFrame or path to CSV file

    Returns:
        PIL Image object (224x224)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Load CSV if path is provided
    if isinstance(csv_data, str):
        df = pd.read_csv(csv_data)
    else:
        df = csv_data

    # Get the two columns (assume first is x-axis, second is y-axis)
    cols = df.columns.tolist()
    x_col = cols[0]  # Temperature or similar
    y_col = cols[1]  # Fluorescence or similar

    x_data = df[x_col].values
    y_data = df[y_col].values

    # Create figure and plot (matching training style)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=56)  # 224x224 at 56 dpi

    # Plot the melting curve
    ax.plot(x_data, y_data, 'b-', linewidth=1.5)
    ax.scatter(x_data, y_data, s=10, alpha=0.5)

    # Styling to match typical melting curve graphs
    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6)

    # Tight layout to avoid label cutoff
    plt.tight_layout()

    # Convert to PIL Image
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img.load()

    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img

    plt.close(fig)

    return img


def get_inference_transforms():
    """
    Get transforms for inference (no augmentation, just normalization).
    Matches the is_training=False transforms from the training notebook.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model_from_checkpoint(checkpoint_path, num_classes):
    """
    Load a trained model from a .pth checkpoint file.

    Args:
        checkpoint_path: Path to .pth file from training
        num_classes: Number of classes

    Returns:
        Loaded model on appropriate device
    """
    # Initialize model architecture
    model = get_model(num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)
    model.eval()

    return model, checkpoint


def get_class_names(checkpoint_path_or_classes):
    """
    Get class names from a classes.txt file or from checkpoint metadata.

    Args:
        checkpoint_path_or_classes: Path to checkpoint file or path to classes.txt

    Returns:
        List of class names
    """
    # If it's a checkpoint file, try to get classes from it
    if checkpoint_path_or_classes.endswith('.pth'):
        try:
            checkpoint = torch.load(checkpoint_path_or_classes, map_location='cpu')
            if 'class_names' in checkpoint:
                return checkpoint['class_names']
        except:
            pass

    # Otherwise try to load from classes.txt
    classes_file = Path(checkpoint_path_or_classes).parent / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    # Fallback: generic class names
    return None


def predict(csv_data, model, class_names=None):
    """
    Make prediction on CSV data.

    Args:
        csv_data: Path to CSV file or pandas DataFrame
        model: Trained model
        class_names: List of class names. If None, uses numeric indices.

    Returns:
        Dictionary with prediction results
    """
    # Convert CSV to image
    img = csv_to_image(csv_data)

    # Apply inference transforms
    transform = get_inference_transforms()
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        logits = output[0]
        probabilities = torch.softmax(logits, dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    # Get top-5 predictions
    top5_probs, top5_indices = torch.topk(probabilities, min(5, len(probabilities)))

    results = {
        'predicted_class_index': predicted_class,
        'predicted_class_name': class_names[predicted_class] if class_names else f"Class {predicted_class}",
        'confidence': float(confidence),
        'top_5_predictions': []
    }

    for rank, (prob, idx) in enumerate(zip(top5_probs, top5_indices), 1):
        class_name = class_names[idx.item()] if class_names else f"Class {idx.item()}"
        results['top_5_predictions'].append({
            'rank': rank,
            'class_index': idx.item(),
            'class_name': class_name,
            'probability': float(prob.item())
        })

    return results


def main():
    """
    Main function for batch ensemble predictions.
    This script is designed to be called by precompute_predictions.py
    """
    print("CNN module loaded. Use get_cnn_predictions() for batch predictions.")


# ============================================================================
# BATCH PREDICTION FUNCTION FOR ENSEMBLE
# ============================================================================

def load_image_from_disk(row_idx, species_name):
    """
    Load image from disk for a given species and row index.

    Args:
        row_idx: Row index in CSV (image filename suffix)
        species_name: Species name (may have spaces)

    Returns:
        PIL Image or None if not found
    """
    # Normalize species name: replace spaces with underscores
    species_normalized = species_name.replace(" ", "_")

    # Try train directory first, then test
    for data_dir in ["../../data/train", "../../data/test"]:
        img_path = Path(data_dir) / species_normalized / f"{species_normalized}_{row_idx:04d}.png"
        if img_path.exists():
            img = Image.open(img_path)
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])
                    img = rgb_img
                else:
                    img = img.convert('RGB')
            return img

    return None


def get_cnn_predictions(X_data, num_classes, models_dir, y_species=None):
    """
    Get CNN predictions for the ensemble stacking.

    Args:
        X_data: Input DataFrame with fluorescence data (rows=samples, cols=features)
        num_classes: Number of output classes
        models_dir: Directory containing model files
        y_species: Series with species labels for each sample (required for loading images)

    Returns:
        numpy array of shape (n_samples, num_classes) with class probabilities,
        or None if model loading fails
    """
    # Try to find the CNN model file
    model_files = list(Path(models_dir).glob("CNN_*.pth"))
    if not model_files:
        print("[WARN] CNN model not found in models_dir")
        return None

    model_path = str(model_files[0])

    try:
        # Load model
        model = get_model(num_classes)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()

        # Get predictions for all samples
        all_probs = []
        transform = get_inference_transforms()

        skipped = 0

        with torch.no_grad():
            for original_idx, (idx, row) in enumerate(X_data.iterrows()):
                # Get species name from y_species if provided
                # idx is the original CSV row index (0-indexed in the dataframe, but after header in CSV)
                species = y_species.loc[idx] if y_species is not None else None

                if species is None:
                    print(f"[WARN] No species label for sample {idx}, skipping")
                    skipped += 1
                    continue

                # Load image from disk using original index
                # The image suffix corresponds to the original pandas/CSV row index (0-based)
                img = load_image_from_disk(idx, species)

                if img is None:
                    print(f"[WARN] Image not found for {species} sample {idx}, skipping")
                    skipped += 1
                    continue

                # Preprocess image
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                # Get prediction
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs.cpu().numpy()[0])

        if skipped > 0:
            print(f"[WARN] Skipped {skipped} samples (images not found)")

        return np.array(all_probs) if all_probs else None

    except Exception as e:
        print(f"[ERROR] Failed to get CNN predictions: {e}")
        return None


if __name__ == '__main__':
    main()
