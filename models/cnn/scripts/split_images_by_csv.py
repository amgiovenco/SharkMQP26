import shutil
from pathlib import Path
import pandas as pd
import json

# Configuration
IMAGES_DIR = Path('../../data/shark_images')
DATA_DIR = Path('../../data')
TRAIN_CSV = DATA_DIR / 'shark_training_data.csv'
VAL_CSV = DATA_DIR / 'shark_validation_data.csv'
TEST_CSV = DATA_DIR / 'shark_test_data.csv'

def split_images_by_csv():
    """
    Split images into train/val/test directories based on which CSV rows each species appears in.
    Since each species appears in all three CSVs, we need to count occurrences in each.

    Strategy: Assign images of a species to splits based on the ratio of rows in each CSV.
    """
    print("Organizing images by train/val/test split based on CSV row counts...")

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images directory not found: {IMAGES_DIR.absolute()}\n"
            f"Run generate_images.py first to create the images."
        )

    # Load all CSVs and count species in each
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Count rows per species in each CSV
    train_counts = train_df['Species'].value_counts().to_dict()
    val_counts = val_df['Species'].value_counts().to_dict()
    test_counts = test_df['Species'].value_counts().to_dict()

    # Create output directories
    for split in ['train', 'val', 'test']:
        (DATA_DIR / split).mkdir(parents=True, exist_ok=True)

    dataset_stats = {'train': {}, 'val': {}, 'test': {}}

    # Get all species directories in images folder
    species_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    print(f"\nFound {len(species_dirs)} species directories in shark_images/")

    # All unique species from all CSVs
    all_csv_species = set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())

    skipped_species = []

    for species_dir in species_dirs:
        # Convert image directory name to CSV format
        normalized_name = species_dir.name.replace('_', ' ')

        # Find the actual species name in CSVs (case-insensitive)
        actual_species_name = None
        for csv_name in sorted(all_csv_species):
            if csv_name.lower() == normalized_name.lower():
                actual_species_name = csv_name
                break

        if actual_species_name is None:
            skipped_species.append(species_dir.name)
            print(f"Warning: '{species_dir.name}' not found in any CSV")
            continue

        image_files = list(species_dir.glob('*.png'))

        if not image_files:
            print(f"Warning: No images found for {species_dir.name}")
            continue

        # Get row counts for this species in each CSV
        train_count = train_counts.get(actual_species_name, 0)
        val_count = val_counts.get(actual_species_name, 0)
        test_count = test_counts.get(actual_species_name, 0)
        total_count = train_count + val_count + test_count

        if total_count == 0:
            print(f"Warning: {actual_species_name} has 0 rows in all CSVs")
            continue

        # Calculate number of images to assign to each split
        num_images = len(image_files)
        train_images = round(num_images * train_count / total_count)
        val_images = round(num_images * val_count / total_count)
        test_images = num_images - train_images - val_images  # Remainder goes to test

        # Distribute images
        image_idx = 0
        for split, num_split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            if num_split_images > 0:
                split_species_dir = DATA_DIR / split / species_dir.name
                split_species_dir.mkdir(parents=True, exist_ok=True)

                for i in range(num_split_images):
                    if image_idx < len(image_files):
                        shutil.copy2(image_files[image_idx], split_species_dir / image_files[image_idx].name)
                        image_idx += 1

                dataset_stats[split][species_dir.name] = num_split_images

        print(f"  {species_dir.name:40s} -> Train: {train_images:3d}, Val: {val_images:3d}, Test: {test_images:3d}")

    # Create class mapping (sorted alphabetically)
    all_species = []
    for split in ['train', 'val', 'test']:
        all_species.extend(dataset_stats[split].keys())
    all_species = sorted(set(all_species))

    class_to_idx = {species: idx for idx, species in enumerate(all_species)}

    # Save metadata
    dataset_dir = DATA_DIR / 'shark_dataset_split'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / 'class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f, indent=2)

    with open(dataset_dir / 'split_stats.json', 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Dataset split complete!")
    print('='*60)
    train_total = sum(dataset_stats['train'].values())
    val_total = sum(dataset_stats['val'].values())
    test_total = sum(dataset_stats['test'].values())
    grand_total = train_total + val_total + test_total

    print(f"Train: {train_total:4d} images")
    print(f"Val:   {val_total:4d} images")
    print(f"Test:  {test_total:4d} images")
    print(f"Total: {grand_total:4d} images")
    print(f"Classes: {len(class_to_idx)}")

    if skipped_species:
        print(f"\nSkipped {len(skipped_species)} species not in CSVs: {', '.join(skipped_species)}")

    print(f"\nMetadata saved to {dataset_dir}")

if __name__ == '__main__':
    split_images_by_csv()
