from pathlib import Path
import json

# Configuration
DATA_DIR = Path('../../data')
DATASET_DIR = DATA_DIR / 'shark_dataset_split'

def prepare_dataset():
    """
    Load pre-split dataset from data/ directory.
    Expected structure:
    - data/train/
    - data/val/
    - data/test/
    Each containing species subdirectories with images.
    """
    print("Preparing dataset from pre-split data...")

    # Check if splits exist in data/
    splits_found = []
    for split in ['train', 'val', 'test']:
        split_dir = DATA_DIR / split
        if split_dir.exists() and split_dir.is_dir():
            splits_found.append(split)

    if not splits_found:
        raise FileNotFoundError(
            f"No train/val/test directories found in {DATA_DIR.absolute()}\n"
            f"Expected structure:\n"
            f"  {DATA_DIR}/train/\n"
            f"  {DATA_DIR}/val/\n"
            f"  {DATA_DIR}/test/\n"
            f"Each containing species subdirectories with images."
        )

    print(f"Found splits: {', '.join(splits_found)}")

    # Create symlinks in dataset_split directory (for compatibility)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    dataset_stats = {}
    class_to_idx = {}
    all_classes = set()

    # Gather all classes from all splits
    for split in ['train', 'val', 'test']:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            species_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            all_classes.update([d.name for d in species_dirs])

    # Create class mapping
    for idx, class_name in enumerate(sorted(all_classes)):
        class_to_idx[class_name] = idx

    # Collect statistics for each split
    for split in ['train', 'val', 'test']:
        split_dir = DATA_DIR / split
        dataset_stats[split] = {}

        if split_dir.exists():
            species_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            for species_dir in species_dirs:
                image_count = len(list(species_dir.glob('*.png')))
                dataset_stats[split][species_dir.name] = image_count

    # Save class mapping
    class_mapping_file = DATASET_DIR / 'class_to_idx.json'
    with open(class_mapping_file, 'w') as f:
        json.dump(class_to_idx, f, indent=2)

    # Save detailed stats
    stats_file = DATASET_DIR / 'split_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    # Print summary
    print(f"\nDataset preparation complete!")
    print(f"Class mapping saved to: {class_mapping_file.absolute()}")
    print(f"Split stats saved to: {stats_file.absolute()}")
    print(f"\nSplit summary:")
    for split in ['train', 'val', 'test']:
        if split in dataset_stats:
            total = sum(dataset_stats[split].values())
            print(f"  {split.capitalize()}: {total} images")

    grand_total = sum(
        sum(dataset_stats[s].values())
        for s in dataset_stats
        if s in dataset_stats
    )
    print(f"  Total: {grand_total} images")
    print(f"  Classes: {len(class_to_idx)}")

if __name__ == '__main__':
    prepare_dataset()
