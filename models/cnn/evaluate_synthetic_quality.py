"""
Evaluate synthetic data quality using CNN embeddings and per-species nearest-neighbor distances.

This script:
1. Loads a trained EfficientNet-B0 CNN model
2. Removes the classification head to extract 256-dim embeddings
3. Extracts embeddings for real samples per species
4. Computes per-species nearest-neighbor (NN) distances from real-to-real samples
5. Extracts embeddings for synthetic samples
6. Computes per-species NN distance for each synthetic sample (to real samples of same species)
7. Generates per-species quality thresholds and classifies synthetic samples
8. Generates a detailed quality report (good/bad synthetic samples) per species

Reference: MODEL_REPRODUCIBILITY_GUIDE.md (Model 1: CNN with EfficientNet-B0)
Key improvement: Uses per-species thresholds instead of global thresholds to account for
varying intra-class density across species with different sample counts.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Data paths (resolve relative to this script's location)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REAL_DATA_PATH = PROJECT_ROOT / "data" / "shark_dataset.csv"
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "syntheticDataGeneration" / "syntheticDataIndividual"

# Model checkpoint to load (use best fold)
MODEL_CHECKPOINT = SCRIPT_DIR / "efficientnet_b0_fold3_9700.pth"  # Adjust fold as needed

# Model architecture parameters (from reproducibility guide)
NUM_CLASSES = 57
EMBEDDING_DIM = 256
DROPOUT1 = 0.7
DROPOUT2 = 0.5
HIDDEN_SIZE = 256

# Quality thresholds
PERCENTILE_GOOD = 75  # Samples within 75th percentile are "good"
PERCENTILE_BAD = 95   # Samples above 95th percentile are "bad"

# Output directory
OUTPUT_DIR = Path("./results/synthetic_quality_assessment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print(f"Output directory: {OUTPUT_DIR.absolute()}")


# ============================================================================
# UTILITIES
# ============================================================================

def generate_image(temps: np.ndarray, values: np.ndarray) -> Image.Image:
    """Generate PIL image from fluorescence curve."""
    try:
        DPI = 100
        IMAGE_SIZE = (224, 224)
        fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/DPI, IMAGE_SIZE[1]/DPI), dpi=DPI)

        ax.plot(temps, values, linewidth=2, color='#2E86AB')
        ax.set_xlim(temps.min(), temps.max())
        ax.set_ylim(values.min() - 0.001, values.max() + 0.001)
        ax.axis('off')

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight', pad_inches=0,
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close(fig)
        return img
    except Exception as e:
        print(f"Failed to generate image: {e}")
        return None


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class CNNModel(nn.Module):
    """EfficientNet-B0 model matching the training checkpoint format."""
    def __init__(self, num_classes, dropout1=0.7, dropout2=0.5, hidden_size=256):
        super(CNNModel, self).__init__()
        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class EmbeddingExtractor(nn.Module):
    """Wrapper to extract embeddings from trained CNN model (stops before final classification layer)."""
    def __init__(self, cnn_model):
        super(EmbeddingExtractor, self).__init__()
        self.cnn = cnn_model

    def forward(self, x):
        """Returns 256-dim embedding vector by stopping before final linear layer."""
        # Pass through features
        x = self.cnn.model.features(x)
        x = self.cnn.model.avgpool(x)
        x = torch.flatten(x, 1)

        # Pass through classifier but stop before the last layer
        # Classifier has 4 layers: [Dropout(0.7), Linear, ReLU, Dropout(0.5), Linear]
        # We want to stop after layer 3 (the last Dropout)
        for i, layer in enumerate(self.cnn.model.classifier):
            if i == 4:  # Skip the final Linear layer (index 4)
                break
            x = layer(x)

        return x


class FluorescenceImageDataset(Dataset):
    """Dataset for fluorescence curve images."""
    def __init__(self, images: list, species: list, transform=None):
        self.images = images
        self.species = species
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        species = self.species[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, species


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def load_checkpoint(checkpoint_path):
    """Load trained CNN checkpoint and return embedding extractor."""
    checkpoint_path = str(checkpoint_path)  # Convert Path to string
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Create the full CNN model
    cnn_model = CNNModel(
        num_classes=NUM_CLASSES,
        dropout1=DROPOUT1,
        dropout2=DROPOUT2,
        hidden_size=HIDDEN_SIZE
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # The checkpoint keys don't have "model." prefix, but our CNN model has them
    # Remap the keys to match our model structure
    remapped_state = {}
    for key, val in state_dict.items():
        new_key = f"model.{key}"
        remapped_state[new_key] = val

    cnn_model.load_state_dict(remapped_state)

    print("[OK] Checkpoint loaded successfully")

    # Wrap in embedding extractor
    extractor = EmbeddingExtractor(cnn_model)
    return extractor


def extract_embeddings(model, loader, device, max_samples=None):
    """Extract embeddings for all samples in dataloader."""
    model.to(device)
    model.eval()

    embeddings = []
    species_list = []

    with torch.no_grad():
        for i, (images, species) in enumerate(loader):
            images = images.to(device).float()
            emb = model(images)  # [batch_size, 256]

            embeddings.append(emb.cpu().numpy())
            species_list.extend(species)

            if max_samples and i * len(images) >= max_samples:
                break

    embeddings = np.vstack(embeddings)  # [N, 256]
    return embeddings, species_list


def compute_class_centroids(embeddings, species_list, unique_species):
    """Compute mean embedding for each class (species)."""
    centroids = {}

    for species in unique_species:
        mask = np.array([s == species for s in species_list])
        if mask.sum() > 0:
            centroids[species] = embeddings[mask].mean(axis=0)

    return centroids


# ============================================================================
# DISTANCE COMPUTATION
# ============================================================================

def compute_per_species_real_nn_distances(embeddings, species_list, unique_species):
    """
    Compute per-species nearest-neighbor (NN) distances for real samples.
    For each species, compute the distance from each real sample to its nearest
    real neighbor within the same species (excluding itself).

    Returns:
        real_nn_dists_per_species (dict): {species_name: [list of NN distances]}
        real_to_real_distances (np.array): Concatenated NN distances (for global stats)
    """
    from collections import defaultdict

    real_nn_dists_per_species = defaultdict(list)
    species_arr = np.array(species_list)

    for species in unique_species:
        mask = (species_arr == species)
        embs_sp = embeddings[mask]

        if len(embs_sp) > 1:
            # For each sample in this species, find its nearest neighbor in the same species
            for i in range(len(embs_sp)):
                dists = np.linalg.norm(embs_sp - embs_sp[i], axis=1)
                # Exclude self (distance ~ 0)
                dists_excluding_self = dists[dists > 1e-6]
                if len(dists_excluding_self) > 0:
                    real_nn_dists_per_species[species].append(np.min(dists_excluding_self))
        # If species has only 1 sample, we can't compute NN distance; skip it

    # Concatenate all NN distances for global statistics
    all_nn_distances = []
    for dists in real_nn_dists_per_species.values():
        all_nn_distances.extend(dists)
    real_to_real_distances = np.array(all_nn_distances) if all_nn_distances else np.array([])

    return real_nn_dists_per_species, real_to_real_distances


def compute_per_species_thresholds(real_nn_dists_per_species, real_to_real_distances,
                                    unique_species, percentile_good, percentile_bad):
    """
    Compute per-species quality thresholds based on per-species real NN distances.
    Falls back to global percentiles for species with n=1.

    Returns:
        thresholds_per_species (dict): {species_name: {'good': threshold, 'bad': threshold}}
    """
    thresholds_per_species = {}

    for species in unique_species:
        dists = np.array(real_nn_dists_per_species.get(species, []))

        if len(dists) > 0:
            # Use per-species percentiles
            thresholds_per_species[species] = {
                'good': np.nanpercentile(dists, percentile_good),
                'bad': np.nanpercentile(dists, percentile_bad),
                'n_reals': len(dists) + 1,  # +1 because we're counting pairs
                'source': 'per-species'
            }
        else:
            # Fallback to global percentiles if species has too few reals
            thresholds_per_species[species] = {
                'good': np.nanpercentile(real_to_real_distances, percentile_good) if len(real_to_real_distances) > 0 else np.nan,
                'bad': np.nanpercentile(real_to_real_distances, percentile_bad) if len(real_to_real_distances) > 0 else np.nan,
                'n_reals': 0,
                'source': 'fallback-global'
            }

    return thresholds_per_species


def compute_distances_to_centroid(embeddings, species_list, centroids):
    """Compute L2 distance from each sample to its class centroid."""
    distances = []

    for i, species in enumerate(species_list):
        if species in centroids:
            centroid = centroids[species]
            dist = np.linalg.norm(embeddings[i] - centroid)
            distances.append(dist)
        else:
            distances.append(np.nan)

    return np.array(distances)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("SYNTHETIC DATA QUALITY ASSESSMENT - EMBEDDING-BASED APPROACH")
    print("="*80)

    print(f"Real data: {REAL_DATA_PATH}")
    print(f"Synthetic data dir: {SYNTHETIC_DATA_DIR}")

    # ========================================================================
    # 1. LOAD REAL DATA
    # ========================================================================
    print("\n[1/5] Loading real data...")
    df_real = pd.read_csv(str(REAL_DATA_PATH))
    X_real = df_real.drop(columns=['Species'])
    y_real = df_real['Species'].astype(str).values

    print(f"  Real data shape: {X_real.shape}")
    print(f"  Unique species: {len(np.unique(y_real))}")
    temps = X_real.columns.astype(float).values

    # Generate images for real data
    print("  Generating images for real data...")
    images_real = []
    for idx, row in X_real.iterrows():
        if idx % 100 == 0:
            print(f"    {idx}/{len(X_real)}")
        img = generate_image(temps, row.values)
        if img is None:
            print(f"ERROR: Failed to generate image for real sample {idx}")
            return
        images_real.append(img)

    print(f"  [OK] Generated {len(images_real)} real images")

    # ========================================================================
    # 2. LOAD MODEL AND EXTRACT REAL EMBEDDINGS
    # ========================================================================
    print("\n[2/5] Loading model and extracting embeddings from real data...")

    # Load checkpoint and get embedding extractor
    model = load_checkpoint(MODEL_CHECKPOINT)

    # Create dataset and dataloader for real data
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_real = FluorescenceImageDataset(images_real, y_real.tolist(), transform=transform_test)
    loader_real = DataLoader(dataset_real, batch_size=32, shuffle=False, num_workers=0)

    # Extract embeddings
    embeddings_real, species_real = extract_embeddings(model, loader_real, DEVICE)
    print(f"  [OK] Extracted {embeddings_real.shape[0]} real embeddings (shape: {embeddings_real.shape})")

    # Compute class centroids
    unique_species = np.unique(y_real)
    centroids = compute_class_centroids(embeddings_real, species_real, unique_species)
    print(f"  [OK] Computed {len(centroids)} class centroids")

    # Compute per-species nearest neighbor distances for real data (real-to-real)
    # For each species, find distances from each real sample to its nearest real neighbor within the same species
    real_nn_dists_per_species, real_to_real_distances = compute_per_species_real_nn_distances(
        embeddings_real, species_real, unique_species
    )

    print(f"\n  REAL DATA Nearest Neighbor distances (real-to-real, per-species):")
    print(f"    Overall (pooled) statistics:")
    print(f"      Mean:   {np.nanmean(real_to_real_distances):.4f}")
    print(f"      Median: {np.nanmedian(real_to_real_distances):.4f}")
    print(f"      Std:    {np.nanstd(real_to_real_distances):.4f}")
    print(f"      Min:    {np.nanmin(real_to_real_distances):.4f}")
    print(f"      Max:    {np.nanmax(real_to_real_distances):.4f}")
    print(f"\n    Per-species statistics:")
    for sp in sorted(unique_species):
        dists_sp = np.array(real_nn_dists_per_species.get(sp, []))
        if len(dists_sp) > 0:
            print(f"      {sp} (n={len(dists_sp)+1}): mean={np.mean(dists_sp):.4f}, median={np.median(dists_sp):.4f}, std={np.std(dists_sp):.4f}")

    # ========================================================================
    # 3. LOAD SYNTHETIC DATA
    # ========================================================================
    print("\n[3/5] Loading synthetic data...")

    synthetic_files = list(SYNTHETIC_DATA_DIR.glob("synthetic_*.csv"))
    print(f"  Found {len(synthetic_files)} synthetic species files")

    all_synthetic_data = []
    all_synthetic_species = []

    for file in sorted(synthetic_files):
        # Extract species name from file (remove "synthetic_" prefix and ".csv")
        species_name = file.stem.replace("synthetic_", "")
        df = pd.read_csv(file)

        # The synthetic CSV should have a Species column; use it if available
        if 'Species' in df.columns:
            # Use the species name from the CSV
            species_name_list = df['Species'].astype(str).unique()
            print(f"  Loading from {species_name}: {len(df)} samples (species: {species_name_list})")
            X_synth = df.drop(columns=['Species'])
            all_synthetic_species.extend(df['Species'].astype(str).tolist())
        else:
            # Use filename as species name
            print(f"  Loading {species_name}: {len(df)} samples")
            X_synth = df
            all_synthetic_species.extend([species_name] * len(X_synth))

        all_synthetic_data.append(X_synth)

    if all_synthetic_data:
        X_synthetic = pd.concat(all_synthetic_data, ignore_index=True)
        y_synthetic = np.array(all_synthetic_species)
        print(f"  [OK] Total synthetic samples: {len(X_synthetic)}")
    else:
        print("  No synthetic data files found!")
        return

    # Generate images for synthetic data
    print("  Generating images for synthetic data...")
    images_synthetic = []
    for idx, row in X_synthetic.iterrows():
        if idx % 100 == 0:
            print(f"    {idx}/{len(X_synthetic)}")
        img = generate_image(temps, row.values)
        if img is None:
            print(f"ERROR: Failed to generate image for synthetic sample {idx}")
            images_synthetic.append(None)
            continue
        images_synthetic.append(img)

    # Filter out failed images
    valid_indices = [i for i, img in enumerate(images_synthetic) if img is not None]
    images_synthetic = [images_synthetic[i] for i in valid_indices]
    y_synthetic = y_synthetic[valid_indices]
    X_synthetic = X_synthetic.iloc[valid_indices].reset_index(drop=True)

    print(f"  [OK] Generated {len(images_synthetic)} synthetic images (failed: {len(all_synthetic_species) - len(images_synthetic)})")

    # ========================================================================
    # 4. EXTRACT SYNTHETIC EMBEDDINGS AND COMPUTE DISTANCES
    # ========================================================================
    print("\n[4/5] Extracting embeddings from synthetic data...")

    dataset_synthetic = FluorescenceImageDataset(images_synthetic, y_synthetic.tolist(), transform=transform_test)
    loader_synthetic = DataLoader(dataset_synthetic, batch_size=32, shuffle=False, num_workers=0)

    embeddings_synthetic, species_synthetic = extract_embeddings(model, loader_synthetic, DEVICE)
    print(f"  [OK] Extracted {embeddings_synthetic.shape[0]} synthetic embeddings")

    # Compute per-species nearest neighbor distances for synthetic data (synthetic-to-real)
    # For each synthetic sample, find distance to its nearest real sample of the SAME species
    synthetic_to_real_distances = []
    species_real_arr = np.array(species_real)

    for i in range(len(embeddings_synthetic)):
        sp = species_synthetic[i]
        # Get all real samples of the same species
        mask = (species_real_arr == sp)
        embs_real_sp = embeddings_real[mask]

        if len(embs_real_sp) > 0:
            # Compute distances to real samples of same species
            dists = np.linalg.norm(embs_real_sp - embeddings_synthetic[i], axis=1)
            synthetic_to_real_distances.append(np.min(dists))
        else:
            # No real samples of this species found (shouldn't happen if data is consistent)
            synthetic_to_real_distances.append(np.nan)

    synthetic_to_real_distances = np.array(synthetic_to_real_distances)

    print(f"\n  SYNTHETIC DATA Nearest Neighbor distances (synthetic-to-real, per-species):")
    print(f"    Overall (pooled) statistics:")
    print(f"      Mean:   {np.nanmean(synthetic_to_real_distances):.4f}")
    print(f"      Median: {np.nanmedian(synthetic_to_real_distances):.4f}")
    print(f"      Std:    {np.nanstd(synthetic_to_real_distances):.4f}")
    print(f"      Min:    {np.nanmin(synthetic_to_real_distances):.4f}")
    print(f"      Max:    {np.nanmax(synthetic_to_real_distances):.4f}")

    # Store for later use (renaming for consistency with rest of code)
    distances_synthetic = synthetic_to_real_distances
    distances_real = real_to_real_distances

    # ========================================================================
    # 5. GENERATE QUALITY REPORT
    # ========================================================================
    print("\n[5/5] Generating quality assessment report...")

    # Compute per-species quality thresholds
    thresholds_per_species = compute_per_species_thresholds(
        real_nn_dists_per_species,
        real_to_real_distances,
        unique_species,
        PERCENTILE_GOOD,
        PERCENTILE_BAD
    )

    print(f"\n  Quality thresholds (per-species based on real-to-real nearest neighbor distances):")
    for sp in sorted(thresholds_per_species.keys()):
        thresh = thresholds_per_species[sp]
        print(f"    {sp} (n_real={thresh['n_reals']}, {thresh['source']}):")
        print(f"      Good (p{PERCENTILE_GOOD}): {thresh['good']:.4f}, Bad (p{PERCENTILE_BAD}): {thresh['bad']:.4f}")

    # Classify samples using per-species thresholds
    quality = []
    for i, dist in enumerate(distances_synthetic):
        sp = species_synthetic[i]
        if sp not in thresholds_per_species:
            quality.append("unknown")
        elif np.isnan(dist):
            quality.append("unknown")
        else:
            good_thresh = thresholds_per_species[sp]['good']
            bad_thresh = thresholds_per_species[sp]['bad']
            if np.isnan(good_thresh) or np.isnan(bad_thresh):
                quality.append("unknown")
            elif dist <= good_thresh:
                quality.append("good")
            elif dist >= bad_thresh:
                quality.append("bad")
            else:
                quality.append("medium")

    # Create results dataframe
    results_df = pd.DataFrame({
        'species': species_synthetic,
        'distance_to_centroid': distances_synthetic,
        'quality': quality,
        'sample_index': range(len(species_synthetic))
    })

    # Save detailed results
    results_file = OUTPUT_DIR / "synthetic_quality_detailed.csv"
    results_df.to_csv(results_file, index=False)
    print(f"  [OK] Detailed results saved to {results_file}")

    # Generate summary statistics
    summary_stats = {
        'real_data': {
            'total_samples': int(len(distances_real)),
            'distance_stats': {
                'mean': float(np.nanmean(distances_real)),
                'median': float(np.nanmedian(distances_real)),
                'std': float(np.nanstd(distances_real)),
                'min': float(np.nanmin(distances_real)),
                'max': float(np.nanmax(distances_real)),
                'p25': float(np.nanpercentile(distances_real, 25)),
                'p50': float(np.nanpercentile(distances_real, 50)),
                'p75': float(np.nanpercentile(distances_real, 75)),
                'p95': float(np.nanpercentile(distances_real, 95))
            }
        },
        'synthetic_data': {
            'total_samples': int(len(distances_synthetic)),
            'good_count': int((np.array(quality) == 'good').sum()),
            'medium_count': int((np.array(quality) == 'medium').sum()),
            'bad_count': int((np.array(quality) == 'bad').sum()),
            'unknown_count': int((np.array(quality) == 'unknown').sum()),
            'distance_stats': {
                'mean': float(np.nanmean(distances_synthetic)),
                'median': float(np.nanmedian(distances_synthetic)),
                'std': float(np.nanstd(distances_synthetic)),
                'min': float(np.nanmin(distances_synthetic)),
                'max': float(np.nanmax(distances_synthetic)),
                'p25': float(np.nanpercentile(distances_synthetic, 25)),
                'p50': float(np.nanpercentile(distances_synthetic, 50)),
                'p75': float(np.nanpercentile(distances_synthetic, 75)),
                'p95': float(np.nanpercentile(distances_synthetic, 95))
            }
        },
        'thresholds_per_species': {
            str(sp): {
                'good': float(thresholds_per_species[sp]['good']),
                'bad': float(thresholds_per_species[sp]['bad']),
                'n_reals': int(thresholds_per_species[sp]['n_reals']),
                'source': thresholds_per_species[sp]['source']
            }
            for sp in sorted(thresholds_per_species.keys())
        },
        'config': {
            'percentile_good': PERCENTILE_GOOD,
            'percentile_bad': PERCENTILE_BAD,
            'model_checkpoint': str(MODEL_CHECKPOINT),
            'embedding_dim': EMBEDDING_DIM,
            'random_state': RANDOM_STATE,
            'evaluation_method': 'per-species nearest-neighbor distances'
        }
    }

    # Save summary
    summary_file = OUTPUT_DIR / "quality_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  [OK] Summary saved to {summary_file}")

    # Print summary
    print("\n" + "="*80)
    print("QUALITY ASSESSMENT SUMMARY")
    print("="*80)

    print("\nREAL DATA - Nearest Neighbor Statistics (how far apart real samples are from each other):")
    print(f"  Total samples:       {summary_stats['real_data']['total_samples']}")
    print(f"  Mean NN distance:    {summary_stats['real_data']['distance_stats']['mean']:.4f}")
    print(f"  Median NN distance:  {summary_stats['real_data']['distance_stats']['median']:.4f}")
    print(f"  Std deviation:       {summary_stats['real_data']['distance_stats']['std']:.4f}")
    print(f"  Min NN distance:     {summary_stats['real_data']['distance_stats']['min']:.4f}")
    print(f"  Max NN distance:     {summary_stats['real_data']['distance_stats']['max']:.4f}")
    print(f"  25th percentile:     {summary_stats['real_data']['distance_stats']['p25']:.4f}")
    print(f"  75th percentile:     {summary_stats['real_data']['distance_stats']['p75']:.4f}")
    print(f"  95th percentile:     {summary_stats['real_data']['distance_stats']['p95']:.4f}")

    print("\nSYNTHETIC DATA Quality Assessment:")
    print(f"  Total samples:       {summary_stats['synthetic_data']['total_samples']}")
    print(f"  Good samples:        {summary_stats['synthetic_data']['good_count']} ({100*summary_stats['synthetic_data']['good_count']/summary_stats['synthetic_data']['total_samples']:.1f}%)")
    print(f"  Medium samples:      {summary_stats['synthetic_data']['medium_count']} ({100*summary_stats['synthetic_data']['medium_count']/summary_stats['synthetic_data']['total_samples']:.1f}%)")
    print(f"  Bad samples:         {summary_stats['synthetic_data']['bad_count']} ({100*summary_stats['synthetic_data']['bad_count']/summary_stats['synthetic_data']['total_samples']:.1f}%)")

    print("\nSYNTHETIC DATA - Nearest Neighbor Statistics (distance to closest real sample):")
    print(f"  Mean NN distance:    {summary_stats['synthetic_data']['distance_stats']['mean']:.4f}")
    print(f"  Median NN distance:  {summary_stats['synthetic_data']['distance_stats']['median']:.4f}")
    print(f"  Std deviation:       {summary_stats['synthetic_data']['distance_stats']['std']:.4f}")
    print(f"  Min NN distance:     {summary_stats['synthetic_data']['distance_stats']['min']:.4f}")
    print(f"  Max NN distance:     {summary_stats['synthetic_data']['distance_stats']['max']:.4f}")
    print(f"  25th percentile:     {summary_stats['synthetic_data']['distance_stats']['p25']:.4f}")
    print(f"  75th percentile:     {summary_stats['synthetic_data']['distance_stats']['p75']:.4f}")
    print(f"  95th percentile:     {summary_stats['synthetic_data']['distance_stats']['p95']:.4f}")

    print(f"\nQuality Assessment (based on per-species real-to-real NN distances):")
    print(f"  Good samples (≤ p{PERCENTILE_GOOD} of species-specific real NN): {summary_stats['synthetic_data']['good_count']} samples")
    print(f"    -> These synthetic samples are close to at least one real sample of their species")
    print(f"  Medium samples: {summary_stats['synthetic_data']['medium_count']} samples")
    print(f"    -> Between good and bad thresholds; less common but still plausible")
    print(f"  Bad samples (≥ p{PERCENTILE_BAD} of species-specific real NN): {summary_stats['synthetic_data']['bad_count']} samples")
    print(f"    -> Too far from any real sample of their species")
    print(f"  Unknown samples: {summary_stats['synthetic_data'].get('unknown_count', 0)} samples")
    print(f"    -> Species not found in real data or distance could not be computed")

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\nGenerating visualizations...")

    # Helper: Import seaborn for KDE plots
    from scipy import stats

    # 1. Distance distribution histogram - SYNTHETIC ONLY
    fig, ax = plt.subplots(figsize=(12, 6))
    distances_valid = distances_synthetic[~np.isnan(distances_synthetic)]
    ax.hist(distances_valid, bins=50, alpha=0.7, edgecolor='black', color='#3498db', label='Synthetic Data')
    ax.set_xlabel('Distance to Nearest Real Sample', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Synthetic Sample Distances to Nearest Real Sample', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] distance_distribution.png")

    # 1b. Comparison histogram - REAL vs SYNTHETIC
    fig, ax = plt.subplots(figsize=(12, 6))
    distances_real_valid = distances_real[~np.isnan(distances_real)]
    distances_synth_valid = distances_synthetic[~np.isnan(distances_synthetic)]

    ax.hist(distances_real_valid, bins=50, alpha=0.6, edgecolor='black', color='#2ecc71', label='Real Data (NN distances)')
    ax.hist(distances_synth_valid, bins=50, alpha=0.6, edgecolor='black', color='#3498db', label='Synthetic Data (NN distances)')
    ax.axvline(np.nanmean(distances_real), color='#27ae60', linestyle='-', linewidth=2, label=f'Real Mean: {np.nanmean(distances_real):.4f}')
    ax.axvline(np.nanmean(distances_synthetic), color='#2980b9', linestyle='-', linewidth=2, label=f'Synthetic Mean: {np.nanmean(distances_synthetic):.4f}')
    ax.set_xlabel('Nearest Neighbor Distance', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Comparison: Real vs Synthetic Nearest Neighbor Distances', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distance_comparison_real_vs_synthetic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] distance_comparison_real_vs_synthetic.png")

    # 2. Quality breakdown pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    quality_counts = [summary_stats['synthetic_data']['good_count'], summary_stats['synthetic_data']['medium_count'], summary_stats['synthetic_data']['bad_count']]
    labels = [f"Good ({quality_counts[0]})", f"Medium ({quality_counts[1]})", f"Bad ({quality_counts[2]})"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax.pie(quality_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Synthetic Data Quality Breakdown', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'quality_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] quality_breakdown.png")

    # 3. Per-species quality distribution (boxplot)
    if len(np.unique(species_synthetic)) > 1:
        fig, ax = plt.subplots(figsize=(14, 6))
        species_unique = sorted(np.unique(species_synthetic))
        distances_by_species = [distances_synthetic[np.array(species_synthetic) == s] for s in species_unique]

        bp = ax.boxplot(distances_by_species, labels=species_unique, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.7)

        ax.set_xlabel('Species', fontsize=12)
        ax.set_ylabel('Nearest Neighbor Distance', fontsize=12)
        ax.set_title('Synthetic Data: Distance to Nearest Real Sample by Species', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'distance_by_species.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] distance_by_species.png")

    # 3b. Per-species comparison (REAL vs SYNTHETIC) - Side-by-side boxplots
    if len(np.unique(species_synthetic)) > 1:
        fig, ax = plt.subplots(figsize=(max(16, len(species_unique) * 1.2), 6))
        species_unique = sorted(np.unique(species_synthetic))

        # Create side-by-side boxplots for each species
        positions = np.arange(1, len(species_unique) + 1)
        offset = 0.2

        # Real data
        real_data = [distances_real[np.array(species_real) == s] for s in species_unique]
        bp_real = ax.boxplot(real_data, positions=positions - offset, widths=0.35, patch_artist=True,
                             boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                             whiskerprops=dict(color='#27ae60'),
                             capprops=dict(color='#27ae60'),
                             medianprops=dict(color='#1e8449', linewidth=2))

        # Synthetic data
        synth_data = [distances_synthetic[np.array(species_synthetic) == s] for s in species_unique]
        bp_synth = ax.boxplot(synth_data, positions=positions + offset, widths=0.35, patch_artist=True,
                              boxprops=dict(facecolor='#3498db', alpha=0.7),
                              whiskerprops=dict(color='#2980b9'),
                              capprops=dict(color='#2980b9'),
                              medianprops=dict(color='#1f618d', linewidth=2))

        ax.set_xticks(positions)
        ax.set_xticklabels(species_unique, rotation=45, ha='right')
        ax.set_ylabel('Nearest Neighbor Distance', fontsize=12)
        ax.set_xlabel('Species', fontsize=12)
        ax.set_title('Per-Species Comparison: Nearest Neighbor Distances (Real vs Synthetic)', fontsize=14, fontweight='bold')

        # Custom legend for the two datasets
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', alpha=0.7, label='Real Data'),
                          Patch(facecolor='#3498db', alpha=0.7, label='Synthetic Data')]
        ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'distance_by_species_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] distance_by_species_comparison.png")

    # 4. Violin plots - Real vs Synthetic distribution shapes
    fig, ax = plt.subplots(figsize=(10, 6))
    distances_real_valid = distances_real[~np.isnan(distances_real)]
    distances_synth_valid = distances_synthetic[~np.isnan(distances_synthetic)]

    parts = ax.violinplot([distances_real_valid, distances_synth_valid], positions=[1, 2],
                          showmeans=True, showmedians=True, widths=0.7)

    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.3)

    # Color the bodies differently
    parts['bodies'][0].set_facecolor('#2ecc71')
    parts['bodies'][0].set_alpha(0.6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real Data (NN)', 'Synthetic Data (NN)'])
    ax.set_ylabel('Nearest Neighbor Distance', fontsize=12)
    ax.set_title('Distribution Shape Comparison: Nearest Neighbor Distances', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'violin_plot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] violin_plot_comparison.png")

    # 5. CDF (Cumulative Distribution Function) - shows percentile coverage
    fig, ax = plt.subplots(figsize=(12, 6))

    # Real data CDF
    real_sorted = np.sort(distances_real_valid)
    real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    ax.plot(real_sorted, real_cdf, linewidth=2.5, color='#2ecc71', label='Real Data', marker='', alpha=0.8)

    # Synthetic data CDF
    synth_sorted = np.sort(distances_synth_valid)
    synth_cdf = np.arange(1, len(synth_sorted) + 1) / len(synth_sorted)
    ax.plot(synth_sorted, synth_cdf, linewidth=2.5, color='#3498db', label='Synthetic Data', marker='', alpha=0.8)

    ax.set_xlabel('Nearest Neighbor Distance', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution: Nearest Neighbor Distances', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cdf_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] cdf_comparison.png")

    # 6. Density plots (KDE) - overlaid distributions
    fig, ax = plt.subplots(figsize=(12, 6))

    # Real data density
    kde_real = stats.gaussian_kde(distances_real_valid)
    x_range = np.linspace(min(distances_real_valid.min(), distances_synth_valid.min()),
                          max(distances_real_valid.max(), distances_synth_valid.max()), 200)
    ax.plot(x_range, kde_real(x_range), linewidth=2.5, color='#2ecc71', label='Real Data (KDE)', alpha=0.8)
    ax.fill_between(x_range, kde_real(x_range), alpha=0.2, color='#2ecc71')

    # Synthetic data density
    kde_synth = stats.gaussian_kde(distances_synth_valid)
    ax.plot(x_range, kde_synth(x_range), linewidth=2.5, color='#3498db', label='Synthetic Data (KDE)', alpha=0.8)
    ax.fill_between(x_range, kde_synth(x_range), alpha=0.2, color='#3498db')

    ax.set_xlabel('Nearest Neighbor Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Kernel Density Estimation: Nearest Neighbor Distance Distributions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'density_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] density_comparison.png")

    # 7. Per-species quality breakdown bar chart
    if len(np.unique(species_synthetic)) > 1:
        fig, ax = plt.subplots(figsize=(max(12, len(species_unique) * 0.8), 6))
        species_unique = sorted(np.unique(species_synthetic))

        good_counts = []
        medium_counts = []
        bad_counts = []

        for species in species_unique:
            species_mask = np.array(species_synthetic) == species
            species_quality = np.array(quality)[species_mask]
            good_counts.append(int((species_quality == 'good').sum()))
            medium_counts.append(int((species_quality == 'medium').sum()))
            bad_counts.append(int((species_quality == 'bad').sum()))

        x = np.arange(len(species_unique))
        width = 0.6

        ax.bar(x, good_counts, width, label='Good', color='#2ecc71', alpha=0.8)
        ax.bar(x, medium_counts, width, bottom=good_counts, label='Medium', color='#f39c12', alpha=0.8)
        ax.bar(x, bad_counts, width, bottom=np.array(good_counts) + np.array(medium_counts),
               label='Bad', color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Species', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Per-Species Quality Breakdown (Stacked)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(species_unique, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'quality_by_species_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] quality_by_species_breakdown.png")

    # ========================================================================
    # PER-SPECIES REPORTS
    # ========================================================================
    print("\nGenerating per-species reports...")

    per_species_report = {}
    for species in sorted(np.unique(species_synthetic)):
        species_mask = np.array(species_synthetic) == species
        species_distances = distances_synthetic[species_mask]
        species_quality = np.array(quality)[species_mask]

        # Also get real data distances for this species
        real_species_mask = np.array(species_real) == species
        real_species_distances = distances_real[real_species_mask]

        per_species_report[str(species)] = {
            'thresholds': {
                'good': float(thresholds_per_species[species]['good']) if species in thresholds_per_species else float('nan'),
                'bad': float(thresholds_per_species[species]['bad']) if species in thresholds_per_species else float('nan'),
                'n_reals': int(thresholds_per_species[species]['n_reals']) if species in thresholds_per_species else 0,
                'source': thresholds_per_species[species]['source'] if species in thresholds_per_species else 'unknown'
            },
            'synthetic': {
                'total_samples': int(species_mask.sum()),
                'good_samples': int((species_quality == 'good').sum()),
                'medium_samples': int((species_quality == 'medium').sum()),
                'bad_samples': int((species_quality == 'bad').sum()),
                'unknown_samples': int((species_quality == 'unknown').sum()),
                'distance_mean': float(np.nanmean(species_distances)) if len(species_distances) > 0 else float('nan'),
                'distance_std': float(np.nanstd(species_distances)) if len(species_distances) > 0 else float('nan'),
                'distance_min': float(np.nanmin(species_distances)) if len(species_distances) > 0 else float('nan'),
                'distance_max': float(np.nanmax(species_distances)) if len(species_distances) > 0 else float('nan'),
                'distance_median': float(np.nanmedian(species_distances)) if len(species_distances) > 0 else float('nan')
            },
            'real': {
                'total_samples': int(real_species_mask.sum()),
                'distance_mean': float(np.nanmean(real_species_distances)) if len(real_species_distances) > 0 else float('nan'),
                'distance_std': float(np.nanstd(real_species_distances)) if len(real_species_distances) > 0 else float('nan'),
                'distance_min': float(np.nanmin(real_species_distances)) if len(real_species_distances) > 0 else float('nan'),
                'distance_max': float(np.nanmax(real_species_distances)) if len(real_species_distances) > 0 else float('nan'),
                'distance_median': float(np.nanmedian(real_species_distances)) if len(real_species_distances) > 0 else float('nan')
            }
        }

    per_species_file = OUTPUT_DIR / "per_species_quality.json"
    with open(per_species_file, 'w') as f:
        json.dump(per_species_report, f, indent=2)
    print(f"  [OK] per_species_quality.json")

    # Print per-species summary
    print("\n" + "-"*80)
    print("PER-SPECIES QUALITY SUMMARY")
    print("-"*80)
    for species in sorted(per_species_report.keys()):
        info = per_species_report[species]
        print(f"\n{species}:")
        print(f"  QUALITY THRESHOLDS (based on real NN distances of this species):")
        print(f"    Good threshold (p{PERCENTILE_GOOD}): {info['thresholds']['good']:.4f}")
        print(f"    Bad threshold (p{PERCENTILE_BAD}):  {info['thresholds']['bad']:.4f}")
        print(f"    Computed from n={info['thresholds']['n_reals']} real samples ({info['thresholds']['source']})")
        print(f"  REAL DATA (NN to other real samples of same species):")
        print(f"    Total samples: {info['real']['total_samples']}")
        print(f"    Mean NN distance: {info['real']['distance_mean']:.4f} ± {info['real']['distance_std']:.4f}")
        print(f"    Median NN distance: {info['real']['distance_median']:.4f}")
        print(f"  SYNTHETIC DATA (NN to real samples of same species):")
        print(f"    Total:  {info['synthetic']['total_samples']}")
        print(f"    Good:   {info['synthetic']['good_samples']} ({100*info['synthetic']['good_samples']/max(1, info['synthetic']['total_samples']):.1f}%)")
        print(f"    Medium: {info['synthetic']['medium_samples']} ({100*info['synthetic']['medium_samples']/max(1, info['synthetic']['total_samples']):.1f}%)")
        print(f"    Bad:    {info['synthetic']['bad_samples']} ({100*info['synthetic']['bad_samples']/max(1, info['synthetic']['total_samples']):.1f}%)")
        if info['synthetic'].get('unknown_samples', 0) > 0:
            print(f"    Unknown: {info['synthetic']['unknown_samples']} ({100*info['synthetic']['unknown_samples']/max(1, info['synthetic']['total_samples']):.1f}%)")
        print(f"    Mean NN distance: {info['synthetic']['distance_mean']:.4f} ± {info['synthetic']['distance_std']:.4f}")
        print(f"    Median NN distance: {info['synthetic']['distance_median']:.4f}")

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR SYNTHETIC DATA USAGE")
    print("="*80)
    print(f"\nNote: Quality based on per-species nearest-neighbor distances to real samples.")
    print(f"Thresholds are computed per-species based on how far apart real samples are from each other")
    print(f"within their respective species. This accounts for natural variation in intra-class density.\n")

    print(f"1. GOOD samples ({summary_stats['synthetic_data']['good_count']} samples):")
    print(f"   -> Distance to nearest real sample of same species ≤ p{PERCENTILE_GOOD} (per-species threshold)")
    print(f"   -> These synthetic samples closely match the distribution of their species")
    print(f"   -> USE IN TRAINING: They look like real data from their species")
    good_samples = results_df[results_df['quality'] == 'good']
    if len(good_samples) > 0:
        print(f"   -> Top species: {good_samples['species'].value_counts().head(3).to_dict()}")

    print(f"\n2. MEDIUM samples ({summary_stats['synthetic_data']['medium_count']} samples):")
    print(f"   -> Distance between p{PERCENTILE_GOOD} and p{PERCENTILE_BAD} (per-species thresholds)")
    print(f"   -> Moderately plausible but less common in real data of their species")
    print(f"   -> USE WITH CAUTION: Can be included but consider filtering or weighting down")
    medium_samples = results_df[results_df['quality'] == 'medium']
    if len(medium_samples) > 0:
        print(f"   -> Count: {len(medium_samples)} samples")

    print(f"\n3. BAD samples ({summary_stats['synthetic_data']['bad_count']} samples):")
    print(f"   -> Distance to nearest real sample of same species ≥ p{PERCENTILE_BAD} (per-species threshold)")
    print(f"   -> These don't match real samples of their species closely")
    print(f"   -> EXCLUDE FROM TRAINING: These are likely artifacts or mode collapses")
    bad_samples = results_df[results_df['quality'] == 'bad']
    if len(bad_samples) > 0:
        print(f"   -> Top species: {bad_samples['species'].value_counts().head(3).to_dict()}")

    if summary_stats['synthetic_data'].get('unknown_count', 0) > 0:
        print(f"\n4. UNKNOWN samples ({summary_stats['synthetic_data'].get('unknown_count', 0)} samples):")
        print(f"   -> Species not found in real data or distance computation failed")
        print(f"   -> REVIEW AND HANDLE: Check if species should be included in training data")

    # ========================================================================
    # FILTERED DATASETS
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING FILTERED DATASETS")
    print("="*80)

    # Save good samples only
    good_indices = results_df[results_df['quality'] == 'good']['sample_index'].values
    X_good = X_synthetic.iloc[good_indices]
    y_good = y_synthetic[good_indices]
    df_good = pd.DataFrame(X_good)
    df_good['Species'] = y_good
    good_file = OUTPUT_DIR / "synthetic_data_good_quality.csv"
    df_good.to_csv(good_file, index=False)
    print(f"  [OK] Saved {len(df_good)} good samples to {good_file}")

    # Save medium + good samples
    medium_indices = results_df[results_df['quality'].isin(['good', 'medium'])]['sample_index'].values
    X_good_medium = X_synthetic.iloc[medium_indices]
    y_good_medium = y_synthetic[medium_indices]
    df_good_medium = pd.DataFrame(X_good_medium)
    df_good_medium['Species'] = y_good_medium
    good_medium_file = OUTPUT_DIR / "synthetic_data_good_medium_quality.csv"
    df_good_medium.to_csv(good_medium_file, index=False)
    print(f"  [OK] Saved {len(df_good_medium)} good+medium samples to {good_medium_file}")

    # Save all with quality labels
    results_df.to_csv(OUTPUT_DIR / "synthetic_quality_detailed.csv", index=False)
    print(f"  [OK] Saved detailed quality assessment to synthetic_quality_detailed.csv")

    print("\n" + "="*80)
    print("QUALITY ASSESSMENT COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR.absolute()}\n")


if __name__ == "__main__":
    main()
