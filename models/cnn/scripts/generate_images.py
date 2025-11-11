import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE = '../../../data/shark_dataset.csv'
OUTPUT_DIR = Path('../../../data/shark_images')
IMAGE_SIZE = (224, 224)
DPI = 100

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
    """Generate images from CSV dataset for use in training."""
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
    print(f"Number of species: {len(df)}")
    print(f"Time points: {len(time_values)}")

    # Get unique species
    species_list = df['Species'].unique()
    print(f"Number of unique species: {len(species_list)}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for each species
    species_dirs = {}
    for species in species_list:
        species_dir = OUTPUT_DIR / species.replace(' ', '_').replace('/', '_')
        species_dir.mkdir(parents=True, exist_ok=True)
        species_dirs[species] = species_dir

    print(f"\nGenerating images...")

    # Generate images for each sample
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating images"):
        species = row['Species']
        signal_values = row[time_columns].values.astype(float)

        # Create filename
        species_clean = species.replace(' ', '_').replace('/', '_')
        output_path = species_dirs[species] / f"{species_clean}_{idx:04d}.png"

        # Generate the plot
        generate_line_plot(time_values, signal_values, species, output_path)

    print(f"\nImage generation complete!")
    print(f"Images saved to: {OUTPUT_DIR.absolute()}")

    # Print summary
    print(f"\nSummary:")
    for species in species_list[:10]:  # Show first 10
        species_dir = species_dirs[species]
        count = len(list(species_dir.glob('*.png')))
        print(f"  {species}: {count} images")
    if len(species_list) > 10:
        print(f"  ... and {len(species_list) - 10} more species")

if __name__ == '__main__':
    main()