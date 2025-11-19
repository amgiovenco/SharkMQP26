import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE = '../../../data/shark_dataset.csv'
OUTPUT_DIR = Path('../../../data/images')
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
    """Generate images from CSV dataset."""
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
    species_list = df['Species'].unique()
    print(f"Number of unique species: {len(species_list)}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating images...")

    # Generate images for all samples
    for idx in tqdm(range(len(df)), desc="Creating images"):
        row = df.iloc[idx]
        species = row['Species']
        signal_values = row[time_columns].values.astype(float)

        # Create filename
        species_clean = species.replace(' ', '_').replace('/', '_')
        output_path = OUTPUT_DIR / f"{species_clean}_{idx:04d}.png"

        # Generate the plot
        generate_line_plot(time_values, signal_values, species, output_path)

    print(f"\nImage generation complete!")
    print(f"Images saved to: {OUTPUT_DIR.absolute()}")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total images: {len(df)}")
    print(f"\nBreakdown by species:")
    for species in sorted(species_list):
        count = len(list(OUTPUT_DIR.glob(f"{species.replace(' ', '_').replace('/', '_')}*.png")))
        print(f"  {species}: {count} images")

if __name__ == '__main__':
    main()