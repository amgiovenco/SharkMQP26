#!/usr/bin/env python3
"""
Regenerate fluorescence plot images with random_state=8 to match generate_predictions.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import shutil
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Configuration - MUST MATCH generate_predictions.py
CSV_FILE = '../../data/shark_dataset.csv'
OUTPUT_DIR = Path('../../data')
IMAGE_SIZE = (224, 224)
DPI = 100
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 8  # CRITICAL: Match random_state=8 in generate_predictions.py

def generate_line_plot(time_values, signal_values, species_name, output_path):
    """Generate a clean line plot for a shark time-series"""
    fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/DPI, IMAGE_SIZE[1]/DPI), dpi=DPI)

    # Plot the time series
    ax.plot(time_values, signal_values, linewidth=2, color='#2E86AB')

    # Remove axes and labels for clean image
    ax.set_xlim(time_values.min(), time_values.max())
    ax.set_ylim(signal_values.min() - 0.001, signal_values.max() + 0.001)
    ax.axis('off')

    # Remove all margins and padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save with tight bbox
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0,
                facecolor='white', edgecolor='none')
    plt.close(fig)

def main():
    """Generate images from CSV dataset and split into 80% train / 20% test."""
    print("Loading shark dataset...")

    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path.absolute()}\n"
            f"Expected dataset at {csv_path.absolute()}"
        )

    df = pd.read_csv(CSV_FILE)

    # Extract time values from column names (skip 'Species' column)
    time_columns = df.columns[1:]
    time_values = np.array([float(col) for col in time_columns])

    print(f"Dataset shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Time points: {len(time_values)}")

    # Get unique species
    species_list = sorted(df['Species'].unique())
    print(f"Number of unique species: {len(species_list)}")
    print(f"Using RANDOM_SEED={RANDOM_SEED} to match generate_predictions.py")

    # Split data using STRATIFIED split on entire dataset (same as generate_predictions.py)
    y = df['Species'].values
    indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=RANDOM_SEED  # Use same seed as generate_predictions.py
    )

    # Clear old directories
    train_dir = OUTPUT_DIR / 'train'
    test_dir = OUTPUT_DIR / 'test'

    if train_dir.exists():
        print(f"\nRemoving old train directory: {train_dir}")
        shutil.rmtree(train_dir)
    if test_dir.exists():
        print(f"Removing old test directory: {test_dir}")
        shutil.rmtree(test_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each species in both splits
    train_species_dirs = {}
    test_species_dirs = {}

    for species in species_list:
        train_species_dir = train_dir / species.replace(' ', '_').replace('/', '_')
        test_species_dir = test_dir / species.replace(' ', '_').replace('/', '_')
        train_species_dir.mkdir(parents=True, exist_ok=True)
        test_species_dir.mkdir(parents=True, exist_ok=True)
        train_species_dirs[species] = train_species_dir
        test_species_dirs[species] = test_species_dir

    print(f"\nGenerating images for train set ({TRAIN_RATIO*100:.0f}%)...")

    # Generate images for training set
    for idx in tqdm(train_indices, desc="Creating train images"):
        row = df.iloc[idx]
        species = row['Species']
        signal_values = row[time_columns].values.astype(float)

        # Create filename with 4-digit zero-padded index to match generate_predictions.py
        species_clean = species.replace(' ', '_').replace('/', '_')
        output_path = train_species_dirs[species] / f"{species_clean}_{idx:04d}.png"

        # Generate the plot
        generate_line_plot(time_values, signal_values, species, output_path)

    print(f"\nGenerating images for test set ({TEST_RATIO*100:.0f}%)...")

    # Generate images for test set
    for idx in tqdm(test_indices, desc="Creating test images"):
        row = df.iloc[idx]
        species = row['Species']
        signal_values = row[time_columns].values.astype(float)

        # Create filename with 4-digit zero-padded index
        species_clean = species.replace(' ', '_').replace('/', '_')
        output_path = test_species_dirs[species] / f"{species_clean}_{idx:04d}.png"

        # Generate the plot
        generate_line_plot(time_values, signal_values, species, output_path)

    print(f"\nImage generation complete!")
    print(f"Images saved to: {OUTPUT_DIR.absolute()}")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Train images: {len(train_indices)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Test images: {len(test_indices)} ({TEST_RATIO*100:.0f}%)")
    print(f"  Total images: {len(train_indices) + len(test_indices)}")

    # Verify counts
    actual_train = len(list(train_dir.rglob('*.png')))
    actual_test = len(list(test_dir.rglob('*.png')))
    print(f"\nVerification:")
    print(f"  Actual train images: {actual_train}")
    print(f"  Actual test images: {actual_test}")
    print(f"  Actual total: {actual_train + actual_test}")

if __name__ == '__main__':
    main()
