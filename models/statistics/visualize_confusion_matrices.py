"""
Generate confusion matrix visualizations from comparison_results.json

Creates both unweighted and weighted (normalized) confusion matrices
with color-only visualization (no annotations).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
results_dir = Path("results")
results_file = results_dir / "comparison_results.json"

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_results():
    """Load comparison results from JSON"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def plot_confusion_matrix_unweighted(cm_data, scenario_name, cmap='Blues'):
    """Plot unweighted confusion matrix with colors only (no annotations)"""
    classes = cm_data["classes"]
    cm = np.array(cm_data["matrix_unweighted"])

    # Dynamically size based on number of classes
    n_classes = len(classes)
    if n_classes <= 10:
        figsize = (12, 10)
        label_size = 10
    elif n_classes <= 20:
        figsize = (16, 14)
        label_size = 9
    else:
        figsize = (20, 18)
        label_size = 8

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap with no annotations
    sns.heatmap(
        cm,
        cmap=cmap,
        cbar=True,
        xticklabels=classes,
        yticklabels=classes,
        square=True,
        ax=ax,
        annot=False,
        fmt='d',
        cbar_kws={'label': 'Count'}
    )

    ax.set_title(f'{scenario_name} - Unweighted Confusion Matrix',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

    # Rotate labels for readability - use 90 for x-axis (vertical), 0 for y-axis
    ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=label_size)
    ax.set_yticklabels(classes, rotation=0, fontsize=label_size)

    plt.tight_layout()

    filename = f"confusion_matrix_{scenario_name.lower().replace(' ', '_').replace('+', 'plus')}_unweighted.png"
    filepath = results_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_confusion_matrix_weighted(cm_data, scenario_name, cmap='Blues'):
    """Plot weighted (normalized by true class) confusion matrix with colors only"""
    classes = cm_data["classes"]

    # Use precomputed weighted matrix from JSON
    if "matrix_weighted" in cm_data:
        cm_weighted = np.array(cm_data["matrix_weighted"], dtype=np.float64)
    else:
        # Fallback: compute from raw matrix
        cm = np.array(cm_data["matrix_unweighted"], dtype=np.float64)
        cm_weighted = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_weighted = np.nan_to_num(cm_weighted)

    # Dynamically size based on number of classes
    n_classes = len(classes)
    if n_classes <= 10:
        figsize = (12, 10)
        label_size = 10
    elif n_classes <= 20:
        figsize = (16, 14)
        label_size = 9
    else:
        figsize = (20, 18)
        label_size = 8

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap with no annotations
    sns.heatmap(
        cm_weighted,
        cmap=cmap,
        cbar=True,
        xticklabels=classes,
        yticklabels=classes,
        square=True,
        ax=ax,
        annot=False,
        fmt='.2f',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Proportion'}
    )

    ax.set_title(f'{scenario_name} - Weighted Confusion Matrix (Normalized by True Class)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

    # Rotate labels for readability - use 90 for x-axis (vertical), 0 for y-axis
    ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=label_size)
    ax.set_yticklabels(classes, rotation=0, fontsize=label_size)

    plt.tight_layout()

    filename = f"confusion_matrix_{scenario_name.lower().replace(' ', '_').replace('+', 'plus')}_weighted.png"
    filepath = results_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")
    return filepath


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Generating Confusion Matrix Visualizations")
    print("="*70)

    # Load results
    print("\nLoading comparison_results.json...")
    results = load_results()

    if "confusion_matrices" not in results:
        print("ERROR: No confusion matrix data found in comparison_results.json")
        print("Make sure to run compare_synthetic_data.py first")
        exit(1)

    cm_data = results["confusion_matrices"]

    # Scenario 1: Real Data Only (Blue)
    print("\nScenario 1: Real Data Only (Baseline)")
    print("-" * 70)
    plot_confusion_matrix_unweighted(cm_data["real_only"], "Real Data Only", cmap='Blues')
    plot_confusion_matrix_weighted(cm_data["real_only"], "Real Data Only", cmap='Blues')

    # Scenario 2: Real + Synthetic Training (Green)
    print("\nScenario 2: Real + Synthetic in Training")
    print("-" * 70)
    plot_confusion_matrix_unweighted(cm_data["real_synthetic"], "Real+Synthetic Training", cmap='Greens')
    plot_confusion_matrix_weighted(cm_data["real_synthetic"], "Real+Synthetic Training", cmap='Greens')

    print("\n" + "="*70)
    print("Output Files Generated:")
    print("="*70)
    print("  1. confusion_matrix_real_data_only_unweighted.png")
    print("  2. confusion_matrix_real_data_only_weighted.png")
    print("  3. confusion_matrix_real_plus_synthetic_training_unweighted.png")
    print("  4. confusion_matrix_real_plus_synthetic_training_weighted.png")
    print("\n✓ All confusion matrices generated successfully!")
    print("="*70 + "\n")
