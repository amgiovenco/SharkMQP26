"""
Analyze per-species performance for each base model.

Identifies which species each model struggles with most,
useful for targeted synthetic data generation.

Usage:
    python analyze_model_performance.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

# ============================================================================
# CONFIG
# ============================================================================

DATA_PATH = "../../data/shark_dataset.csv"
PREDICTIONS_FILE = "base_predictions.npz"
OUTPUT_FILE = "model_performance_by_species.json"

# Model names - will be loaded from npz file
MODELS = {}

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Species"])
y = df["Species"].values

# Filter to valid classes (species with >= 2 samples)
counts = pd.Series(y).value_counts()
valid_classes = counts[counts >= 2].index.tolist()
mask = np.isin(y, valid_classes)
y_filtered = y[mask]

print(f"Total samples: {len(y)}")
print(f"Valid classes: {len(valid_classes)}")
print(f"Filtered samples: {len(y_filtered)}")

# ============================================================================
# LOAD PREDICTIONS
# ============================================================================

if not Path(PREDICTIONS_FILE).exists():
    print(f"\n[ERROR] {PREDICTIONS_FILE} not found!")
    print("Run precompute_predictions.py first")
    exit(1)

print(f"\nLoading predictions from {PREDICTIONS_FILE}...")
data = np.load(PREDICTIONS_FILE, allow_pickle=True)

# Check what's in the file
print(f"Available keys: {list(data.keys())}")

# Get predictions array
if 'predictions' in data:
    predictions_array = data['predictions']
elif 'base_predictions' in data:
    predictions_array = data['base_predictions']
else:
    # Try to get all arrays that look like predictions
    predictions_array = None
    for key in data.keys():
        arr = data[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            predictions_array = arr
            print(f"Using array '{key}' as predictions")
            break

if predictions_array is None:
    print("[ERROR] Could not find predictions array!")
    exit(1)

print(f"Predictions shape: {predictions_array.shape}")

# Get true labels from the npz file if available
if 'y_true' in data:
    y_true = data['y_true']
    print(f"Using y_true from npz file (shape: {y_true.shape})")
else:
    # Use filtered y from CSV
    y_true = y_filtered
    print(f"Using y from CSV after filtering (shape: {y_true.shape})")

# Ensure alignment
if predictions_array.shape[0] != len(y_true):
    print(f"[WARNING] Shape mismatch: predictions {predictions_array.shape[0]} vs labels {len(y_true)}")
    print("Truncating to match...")
    min_len = min(predictions_array.shape[0], len(y_true))
    predictions_array = predictions_array[:min_len]
    y_true = y_true[:min_len]

# Get class labels
if 'class_labels' in data:
    class_labels = np.array(list(data['class_labels']))
    print(f"Using class_labels from npz file: {len(class_labels)} classes")
else:
    class_labels = np.array(sorted(np.unique(y_true)))
    print(f"Using unique classes from labels: {len(class_labels)} classes")

print(f"Total predictions shape: {predictions_array.shape}")
num_models = predictions_array.shape[1] // len(class_labels)
print(f"Number of models in predictions: {num_models}")

# Get model names from npz file
if 'model_names' in data:
    model_names_list = data['model_names']
    if isinstance(model_names_list, np.ndarray):
        model_names_list = model_names_list.tolist()

    # Build MODELS dict with indices
    for idx, model_name in enumerate(model_names_list[:num_models]):
        if isinstance(model_name, bytes):
            model_name = model_name.decode('utf-8')
        MODELS[model_name] = idx

    print(f"Available models: {list(MODELS.keys())}")

# ============================================================================
# ANALYZE PER-SPECIES PERFORMANCE
# ============================================================================

results = {}

for model_name, model_idx in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name.upper()}")
    print(f"{'='*60}")

    # Extract predictions for this model
    # Each model produces len(class_labels) probability outputs
    start_idx = model_idx * len(class_labels)
    end_idx = start_idx + len(class_labels)

    if end_idx > predictions_array.shape[1]:
        print(f"[WARNING] Model index {model_idx} out of range. Skipping.")
        continue

    model_probs = predictions_array[:, start_idx:end_idx]
    y_pred = np.argmax(model_probs, axis=1).astype(int)

    # Convert to class labels
    y_pred_labels = class_labels[y_pred]

    # Calculate overall metrics
    overall_acc = accuracy_score(y_true, y_pred_labels)
    overall_f1 = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)

    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall Macro F1: {overall_f1:.4f}")

    # Per-species performance
    species_results = {}
    worst_species = []

    for species in sorted(class_labels):
        mask_species = y_true == species
        if mask_species.sum() == 0:
            continue

        y_true_species = y_true[mask_species]
        y_pred_species = y_pred_labels[mask_species]

        acc = accuracy_score(y_true_species, y_pred_species)

        # Handle cases where there's only one class
        if len(np.unique(y_true_species)) == 1:
            f1 = float(acc)  # If all labels are same, F1 equals accuracy
            precision = float(acc)
            recall = float(acc)
        else:
            f1 = f1_score(y_true_species, y_pred_species, average='weighted', zero_division=0)
            precision = precision_score(y_true_species, y_pred_species, average='weighted', zero_division=0)
            recall = recall_score(y_true_species, y_pred_species, average='weighted', zero_division=0)

        n_samples = mask_species.sum()

        species_results[species] = {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "n_samples": int(n_samples)
        }

        worst_species.append((species, acc, n_samples))

    # Sort by accuracy (worst first)
    worst_species.sort(key=lambda x: x[1])

    print(f"\nWorst 10 Species:")
    for i, (species, acc, n_samples) in enumerate(worst_species[:10], 1):
        print(f"  {i:2d}. {species:20s} - Acc: {acc:.4f} ({n_samples} samples)")

    results[model_name] = {
        "overall_accuracy": float(overall_acc),
        "overall_f1": float(overall_f1),
        "per_species": species_results
    }

# ============================================================================
# AGGREGATE ANALYSIS
# ============================================================================

print(f"\n{'='*60}")
print("AGGREGATE ANALYSIS - Worst Species Across All Models")
print(f"{'='*60}")

# Calculate average accuracy per species across models
species_avg_acc = {}

for model_name, model_results in results.items():
    for species, metrics in model_results["per_species"].items():
        if species not in species_avg_acc:
            species_avg_acc[species] = []
        species_avg_acc[species].append(metrics["accuracy"])

# Calculate statistics
species_stats = {}
for species in sorted(species_avg_acc.keys()):
    accs = species_avg_acc[species]
    species_stats[species] = {
        "mean_accuracy": float(np.mean(accs)),
        "min_accuracy": float(np.min(accs)),
        "max_accuracy": float(np.max(accs)),
        "std_accuracy": float(np.std(accs)),
        "worst_model": [
            m for m, m_results in results.items()
            if species in m_results["per_species"]
        ][np.argmin(accs)]
    }

# Sort by mean accuracy (worst first)
worst_species_overall = sorted(
    species_stats.items(),
    key=lambda x: x[1]["mean_accuracy"]
)

print(f"\nWorst 15 Species (by average accuracy across models):")
for i, (species, stats) in enumerate(worst_species_overall[:15], 1):
    worst_model = stats['worst_model']
    worst_model_acc = results[worst_model]["per_species"][species]["accuracy"]
    print(f"  {i:2d}. {species:20s} - Mean Acc: {stats['mean_accuracy']:.4f} "
          f"(min: {stats['min_accuracy']:.4f}, std: {stats['std_accuracy']:.4f}, "
          f"worst: {worst_model} @ {worst_model_acc:.4f})")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_data = {
    "per_model": results,
    "per_species_aggregate": species_stats,
    "worst_species_overall": [
        {
            "species": s,
            **stats
        }
        for s, stats in worst_species_overall
    ]
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n[OK] Results saved to {OUTPUT_FILE}")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print(f"\n{'='*60}")
print("SYNTHETIC DATA GENERATION RECOMMENDATIONS")
print(f"{'='*60}")

# Find species that are consistently weak across models
weak_across_models = [
    (s, stats) for s, stats in worst_species_overall
    if stats['mean_accuracy'] < 0.80  # Weak if < 80% across models
]

print(f"\nPriority 1: Weak Across All Models (Avg Acc < 0.80)")
print("These species need synthetic data:")
for species, stats in weak_across_models[:10]:
    print(f"  - {species:20s} (Avg: {stats['mean_accuracy']:.4f}, "
          f"Worst: {stats['worst_model']} @ {stats['min_accuracy']:.4f})")

# Find species with high variance (some models good, some bad)
high_variance = [
    (s, stats) for s, stats in worst_species_overall
    if stats['std_accuracy'] > 0.15  # High variance
]

print(f"\nPriority 2: High Variance Across Models (StdDev > 0.15)")
print("Model-specific tuning or targeted data may help:")
for species, stats in high_variance[:10]:
    print(f"  - {species:20s} (Std: {stats['std_accuracy']:.4f}, "
          f"Range: {stats['min_accuracy']:.4f} - {stats['max_accuracy']:.4f})")

print(f"\n[Done]")
