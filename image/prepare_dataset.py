import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import json

# Configuration
IMAGES_DIR = Path('shark_images')
DATASET_DIR = Path('shark_dataset_split')
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def split_dataset():
    """Split images into train/val/test sets"""
    print("Preparing dataset splits...")

    # Create dataset directory structure
    for split in ['train', 'val', 'test']:
        (DATASET_DIR / split).mkdir(parents=True, exist_ok=True)

    # Get all species directories
    species_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(species_dirs)} species")

    dataset_stats = {'train': {}, 'val': {}, 'test': {}}
    class_to_idx = {}

    for idx, species_dir in enumerate(species_dirs):
        species_name = species_dir.name
        class_to_idx[species_name] = idx

        # Create species subdirectories in each split
        for split in ['train', 'val', 'test']:
            (DATASET_DIR / split / species_name).mkdir(exist_ok=True)

        # Get all images for this species
        image_files = list(species_dir.glob('*.png'))

        if len(image_files) == 0:
            print(f"Warning: No images found for {species_name}")
            continue

        # Split the data
        if len(image_files) == 1:
            # If only one image, put it in train
            train_files = image_files
            val_files = []
            test_files = []
        elif len(image_files) == 2:
            # If two images, one in train, one in val
            train_files = [image_files[0]]
            val_files = [image_files[1]]
            test_files = []
        else:
            # Normal split
            train_files, temp_files = train_test_split(
                image_files, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
            )

            if len(temp_files) == 1:
                val_files = temp_files
                test_files = []
            else:
                val_files, test_files = train_test_split(
                    temp_files, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
                )

        # Copy files to respective directories
        for file in train_files:
            shutil.copy2(file, DATASET_DIR / 'train' / species_name / file.name)

        for file in val_files:
            shutil.copy2(file, DATASET_DIR / 'val' / species_name / file.name)

        for file in test_files:
            shutil.copy2(file, DATASET_DIR / 'test' / species_name / file.name)

        # Update stats
        dataset_stats['train'][species_name] = len(train_files)
        dataset_stats['val'][species_name] = len(val_files)
        dataset_stats['test'][species_name] = len(test_files)

    # Save class mapping
    with open(DATASET_DIR / 'class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f, indent=2)

    # Print summary
    print(f"\nDataset split complete!")
    print(f"Dataset saved to: {DATASET_DIR.absolute()}")
    print(f"\nSplit summary:")
    print(f"  Train: {sum(dataset_stats['train'].values())} images")
    print(f"  Val:   {sum(dataset_stats['val'].values())} images")
    print(f"  Test:  {sum(dataset_stats['test'].values())} images")
    print(f"  Total: {sum(dataset_stats['train'].values()) + sum(dataset_stats['val'].values()) + sum(dataset_stats['test'].values())} images")

    # Save detailed stats
    with open(DATASET_DIR / 'split_stats.json', 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    print(f"\nClass mapping and stats saved to {DATASET_DIR}")

if __name__ == '__main__':
    split_dataset()
