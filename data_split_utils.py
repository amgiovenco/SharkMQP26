"""
Standardized Data Splitting Utility
All models should use this to ensure consistent 5-fold splits with seed 8.
Split: 60% train, 20% val, 20% test per fold.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split


SEED = 8
N_SPLITS = 5
DATA_PATH = "data/shark_dataset.csv"


def load_data(csv_path: str = DATA_PATH) -> tuple:
    """Load and prepare data."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Species"])
    y = df["Species"]

    return X, y


def get_fold_splits(X: pd.DataFrame, y: pd.Series, fold_idx: int = 0) -> dict:
    """
    Get train/val/test split for a specific fold using 5-fold stratified CV.

    Args:
        X: Features dataframe
        y: Labels series
        fold_idx: Which fold to return (0-4)

    Returns:
        Dict with 'train', 'val', 'test' indices and split info
    """
    if fold_idx < 0 or fold_idx >= N_SPLITS:
        raise ValueError(f"fold_idx must be 0-{N_SPLITS-1}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Get the fold we want
    for current_fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        if current_fold == fold_idx:
            # Further split train_val into train/val (60/20 of 80% = 12.5% val)
            X_train_val = X.iloc[train_val_idx]
            y_train_val = y.iloc[train_val_idx]

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.2,  # 20% of train_val
                random_state=SEED,
                stratify=y_train_val
            )

            return {
                'fold': fold_idx,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X.iloc[test_idx],
                'y_test': y.iloc[test_idx],
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X.iloc[test_idx]),
            }

    raise RuntimeError("Could not get fold splits")


def get_all_fold_splits(X: pd.DataFrame, y: pd.Series) -> list:
    """Get all 5 fold splits."""
    splits = []
    for fold_idx in range(N_SPLITS):
        splits.append(get_fold_splits(X, y, fold_idx))
    return splits


def print_fold_info(fold_split: dict):
    """Print info about a fold split."""
    print(f"\nFold {fold_split['fold'] + 1}/{N_SPLITS}:")
    print(f"  Train: {fold_split['train_size']} samples")
    print(f"  Val:   {fold_split['val_size']} samples")
    print(f"  Test:  {fold_split['test_size']} samples")
    print(f"  Total: {fold_split['train_size'] + fold_split['val_size'] + fold_split['test_size']} samples")


if __name__ == "__main__":
    # Test the utility
    print("Testing data split utility...")
    X, y = load_data()
    print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
    print(f"Seed: {SEED}, N_Splits: {N_SPLITS}")

    for fold_idx in range(N_SPLITS):
        fold_split = get_fold_splits(X, y, fold_idx)
        print_fold_info(fold_split)
