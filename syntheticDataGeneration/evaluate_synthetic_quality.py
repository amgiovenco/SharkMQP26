"""
Evaluate synthetic data quality using raw Euclidean distance on fluorescence curves.

This script:
1. Loads real fluorescence curve data (raw samples from CSV)
2. Loads synthetic fluorescence curve data
3. Computes per-species nearest-neighbor (NN) distances from real-to-real samples
4. Generates per-species quality thresholds using multiple methods:
   - Mean + k*Std (statistical outlier detection)
   - Multiplier on Median/Max (simple absolute bounds)
5. Computes per-species NN distance for each synthetic sample (to real samples of same species)
6. Classifies synthetic samples using data-driven thresholds
7. Generates detailed quality reports and visualizations

Key advantage: No need for pre-trained models. Thresholds are derived from real data variability,
making them objective and adaptive per species.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 8
np.random.seed(RANDOM_STATE)

# Data paths (resolve relative to this script's location)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
REAL_DATA_PATH = PROJECT_ROOT / "data" / "shark_dataset.csv"
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "syntheticDataGeneration" / "syntheticDataIndividual"

# Threshold method: choose one of:
#   'mean_2std': mean + 2*std (lenient, covers ~95% of normal dist)
#   'mean_3std': mean + 3*std (stricter, covers ~99.7% of normal dist)
#   '1.5_median': 1.5 * median (simple, easy to interpret)
#   'max_nn': use max NN distance (conservative, keeps within observed range)
THRESHOLD_METHOD = 'mean_2std'

# Output directory
OUTPUT_DIR = Path("./results/synthetic_quality_assessment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Threshold method: {THRESHOLD_METHOD}")
print(f"Output directory: {OUTPUT_DIR.absolute()}")


# ============================================================================
# THRESHOLD COMPUTATION METHODS
# ============================================================================

def compute_threshold_mean_kstd(nn_dists, k=2):
    """
    Compute threshold as mean + k*std of NN distances.
    k=2 covers ~95% of normal distribution (lenient)
    k=3 covers ~99.7% (stricter)
    """
    mean = np.mean(nn_dists)
    std = np.std(nn_dists)
    return mean + k * std


def compute_threshold_median_multiplier(nn_dists, multiplier=1.5):
    """
    Compute threshold as multiplier * median.
    Simple, easy to interpret. E.g., 1.5x median or 2x median.
    """
    median = np.median(nn_dists)
    return multiplier * median


def compute_threshold_max_nn(nn_dists):
    """
    Use the max NN distance as threshold (conservative).
    Keeps synthetics within the range of observed real variation.
    """
    return np.max(nn_dists)


def get_threshold_for_species(nn_dists, method='mean_2std'):
    """
    Compute threshold for a species using specified method.

    Parameters:
    -----------
    nn_dists : array-like
        Nearest neighbor distances from real-to-real samples of this species
    method : str
        One of: 'mean_2std', 'mean_3std', '1.5_median', 'max_nn'

    Returns:
    --------
    threshold : float
    """
    if len(nn_dists) == 0:
        return np.nan

    if method == 'mean_2std':
        return compute_threshold_mean_kstd(nn_dists, k=2)
    elif method == 'mean_3std':
        return compute_threshold_mean_kstd(nn_dists, k=3)
    elif method == '1.5_median':
        return compute_threshold_median_multiplier(nn_dists, multiplier=1.5)
    elif method == 'max_nn':
        return compute_threshold_max_nn(nn_dists)
    else:
        raise ValueError(f"Unknown threshold method: {method}")


# ============================================================================
# DISTANCE COMPUTATION
# ============================================================================

def compute_per_species_real_nn_distances(X_real, species_list, unique_species):
    """
    Compute per-species nearest-neighbor (NN) distances for real samples using raw curves.
    For each species, compute the Euclidean distance from each real sample to its nearest
    real neighbor within the same species (excluding itself).

    Parameters:
    -----------
    X_real : np.ndarray or pd.DataFrame
        Raw fluorescence curve data (samples x features)
    species_list : list or array
        Species label for each sample
    unique_species : array
        Unique species names

    Returns:
    --------
    real_nn_dists_per_species (dict): {species_name: [list of NN distances]}
    real_to_real_distances (np.array): Concatenated NN distances (for global stats)
    """
    if isinstance(X_real, pd.DataFrame):
        X_real = X_real.values

    real_nn_dists_per_species = defaultdict(list)
    species_arr = np.array(species_list)

    for species in unique_species:
        mask = (species_arr == species)
        curves_sp = X_real[mask]

        if len(curves_sp) > 1:
            # Compute pairwise Euclidean distances for this species
            dists_matrix = cdist(curves_sp, curves_sp, metric='euclidean')

            # For each sample in this species, find its nearest neighbor
            for i in range(len(curves_sp)):
                # Exclude self (distance ~ 0)
                dists = dists_matrix[i]
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


def compute_per_species_thresholds(real_nn_dists_per_species, unique_species, method='mean_2std'):
    """
    Compute per-species quality thresholds based on per-species real NN distances.
    Uses data-driven statistical method (not percentile-based).

    Parameters:
    -----------
    real_nn_dists_per_species : dict
        {species_name: [list of NN distances]}
    unique_species : array
        Unique species names
    method : str
        Threshold method: 'mean_2std', 'mean_3std', '1.5_median', 'max_nn'

    Returns:
    --------
    thresholds_per_species (dict): {species_name: {'threshold': value, 'mean': ..., 'std': ..., etc.}}
    """
    thresholds_per_species = {}

    for species in unique_species:
        dists = np.array(real_nn_dists_per_species.get(species, []))

        if len(dists) > 0:
            threshold = get_threshold_for_species(dists, method=method)
            thresholds_per_species[species] = {
                'threshold': threshold,
                'n_reals': len(dists) + 1,  # +1 for the species itself
                'mean': float(np.mean(dists)),
                'std': float(np.std(dists)),
                'median': float(np.median(dists)),
                'max': float(np.max(dists)),
                'method': method
            }
        else:
            # If no real samples, set NaN threshold
            thresholds_per_species[species] = {
                'threshold': np.nan,
                'n_reals': 0,
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'max': np.nan,
                'method': method
            }

    return thresholds_per_species


def compute_synthetic_to_real_distances(X_synthetic, species_synthetic, X_real, species_real, unique_species):
    """
    Compute nearest-neighbor distances from each synthetic sample to real samples of the same species.

    Parameters:
    -----------
    X_synthetic : np.ndarray or pd.DataFrame
        Raw fluorescence curves for synthetic samples
    species_synthetic : array
        Species labels for synthetic samples
    X_real : np.ndarray or pd.DataFrame
        Raw fluorescence curves for real samples
    species_real : array
        Species labels for real samples
    unique_species : array
        Unique species names

    Returns:
    --------
    synthetic_to_real_distances : np.array
        Min distance from each synthetic sample to any real sample of same species
    """
    if isinstance(X_synthetic, pd.DataFrame):
        X_synthetic = X_synthetic.values
    if isinstance(X_real, pd.DataFrame):
        X_real = X_real.values

    synthetic_to_real_distances = []
    species_real_arr = np.array(species_real)

    for i in range(len(X_synthetic)):
        sp = species_synthetic[i]
        # Get all real samples of the same species
        mask = (species_real_arr == sp)
        curves_real_sp = X_real[mask]

        if len(curves_real_sp) > 0:
            # Compute distances to real samples of same species
            dists = cdist([X_synthetic[i]], curves_real_sp, metric='euclidean')[0]
            synthetic_to_real_distances.append(np.min(dists))
        else:
            # No real samples of this species found
            synthetic_to_real_distances.append(np.nan)

    return np.array(synthetic_to_real_distances)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("SYNTHETIC DATA QUALITY ASSESSMENT - RAW EUCLIDEAN DISTANCE APPROACH")
    print("="*80)

    print(f"Real data: {REAL_DATA_PATH}")
    print(f"Synthetic data dir: {SYNTHETIC_DATA_DIR}")

    # ========================================================================
    # 1. LOAD REAL DATA
    # ========================================================================
    print("\n[1/4] Loading real data...")
    df_real = pd.read_csv(str(REAL_DATA_PATH))
    X_real = df_real.drop(columns=['Species'])
    y_real = df_real['Species'].astype(str).values

    print(f"  Real data shape: {X_real.shape}")
    print(f"  Unique species: {len(np.unique(y_real))}")
    unique_species = np.unique(y_real)

    # Compute per-species nearest neighbor distances for real data (real-to-real)
    print("\n  Computing per-species nearest-neighbor distances from real data...")
    real_nn_dists_per_species, real_to_real_distances = compute_per_species_real_nn_distances(
        X_real, y_real, unique_species
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
    # 2. LOAD SYNTHETIC DATA
    # ========================================================================
    print("\n[2/4] Loading synthetic data...")

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

    # ========================================================================
    # 3. COMPUTE THRESHOLDS AND DISTANCES
    # ========================================================================
    print(f"\n[3/4] Computing quality thresholds using method: {THRESHOLD_METHOD}...")

    # Compute per-species quality thresholds based on real-to-real NN distances
    thresholds_per_species = compute_per_species_thresholds(
        real_nn_dists_per_species,
        unique_species,
        method=THRESHOLD_METHOD
    )

    print(f"\n  Quality thresholds (per-species based on real-to-real nearest neighbor distances):")
    for sp in sorted(thresholds_per_species.keys()):
        thresh = thresholds_per_species[sp]
        print(f"    {sp} (n_real={thresh['n_reals']}):")
        print(f"      Threshold: {thresh['threshold']:.4f}")
        print(f"      Method stats: mean={thresh['mean']:.4f}, std={thresh['std']:.4f}, median={thresh['median']:.4f}, max={thresh['max']:.4f}")

    # Compute synthetic-to-real distances
    print("\n  Computing synthetic sample distances to real samples...")
    synthetic_to_real_distances = compute_synthetic_to_real_distances(
        X_synthetic, y_synthetic,
        X_real, y_real,
        unique_species
    )

    print(f"\n  SYNTHETIC DATA Nearest Neighbor distances (synthetic-to-real, per-species):")
    print(f"    Overall (pooled) statistics:")
    print(f"      Mean:   {np.nanmean(synthetic_to_real_distances):.4f}")
    print(f"      Median: {np.nanmedian(synthetic_to_real_distances):.4f}")
    print(f"      Std:    {np.nanstd(synthetic_to_real_distances):.4f}")
    print(f"      Min:    {np.nanmin(synthetic_to_real_distances):.4f}")
    print(f"      Max:    {np.nanmax(synthetic_to_real_distances):.4f}")

    # Store for later use
    distances_synthetic = synthetic_to_real_distances
    distances_real = real_to_real_distances

    # ========================================================================
    # 4. GENERATE QUALITY REPORT
    # ========================================================================
    print("\n[4/4] Generating quality assessment report...")

    # Classify samples
    quality = []
    for i, dist in enumerate(distances_synthetic):
        sp = y_synthetic[i]
        if sp not in thresholds_per_species:
            quality.append("unknown")
        elif np.isnan(dist):
            quality.append("unknown")
        else:
            threshold = thresholds_per_species[sp]['threshold']
            if np.isnan(threshold):
                quality.append("unknown")
            else:
                # Single threshold approach: good if within threshold, bad if beyond
                if dist <= threshold:
                    quality.append("good")
                else:
                    quality.append("bad")

    # Create results dataframe
    results_df = pd.DataFrame({
        'species': y_synthetic,
        'distance_to_nearest_real': distances_synthetic,
        'quality': quality,
        'sample_index': range(len(y_synthetic))
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
                'threshold': float(thresholds_per_species[sp]['threshold']),
                'n_reals': int(thresholds_per_species[sp]['n_reals']),
                'mean': float(thresholds_per_species[sp]['mean']),
                'std': float(thresholds_per_species[sp]['std']),
                'median': float(thresholds_per_species[sp]['median']),
                'max': float(thresholds_per_species[sp]['max']),
                'method': thresholds_per_species[sp]['method']
            }
            for sp in sorted(thresholds_per_species.keys())
        },
        'config': {
            'threshold_method': THRESHOLD_METHOD,
            'random_state': RANDOM_STATE,
            'evaluation_method': 'raw euclidean distance on fluorescence curves'
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
    print(f"  Good samples:        {summary_stats['synthetic_data']['good_count']} ({100*summary_stats['synthetic_data']['good_count']/max(1, summary_stats['synthetic_data']['total_samples']):.1f}%)")
    print(f"  Bad samples:         {summary_stats['synthetic_data']['bad_count']} ({100*summary_stats['synthetic_data']['bad_count']/max(1, summary_stats['synthetic_data']['total_samples']):.1f}%)")

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
    print(f"  Good samples (≤ species threshold): {summary_stats['synthetic_data']['good_count']} samples")
    print(f"    -> These synthetic samples are close to at least one real sample of their species")
    print(f"  Bad samples (> species threshold): {summary_stats['synthetic_data']['bad_count']} samples")
    print(f"    -> Too far from any real sample of their species")
    print(f"  Unknown samples: {summary_stats['synthetic_data'].get('unknown_count', 0)} samples")
    print(f"    -> Species not found in real data or distance could not be computed")

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\nGenerating visualizations...")

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
    quality_counts = [summary_stats['synthetic_data']['good_count'], summary_stats['synthetic_data']['bad_count']]
    labels = [f"Good ({quality_counts[0]})", f"Bad ({quality_counts[1]})"]
    colors = ['#2ecc71', '#e74c3c']
    ax.pie(quality_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Synthetic Data Quality Breakdown', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'quality_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] quality_breakdown.png")

    # 3. Per-species quality distribution (boxplot)
    if len(np.unique(y_synthetic)) > 1:
        fig, ax = plt.subplots(figsize=(14, 6))
        species_unique = sorted(np.unique(y_synthetic))
        distances_by_species = [distances_synthetic[np.array(y_synthetic) == s] for s in species_unique]

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
    if len(np.unique(y_synthetic)) > 1:
        fig, ax = plt.subplots(figsize=(max(16, len(species_unique) * 1.2), 6))
        species_unique = sorted(np.unique(y_synthetic))

        # Create side-by-side boxplots for each species
        positions = np.arange(1, len(species_unique) + 1)
        offset = 0.2

        # Real data
        real_data = [distances_real[np.array(y_real) == s] for s in species_unique]
        bp_real = ax.boxplot(real_data, positions=positions - offset, widths=0.35, patch_artist=True,
                             boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                             whiskerprops=dict(color='#27ae60'),
                             capprops=dict(color='#27ae60'),
                             medianprops=dict(color='#1e8449', linewidth=2))

        # Synthetic data
        synth_data = [distances_synthetic[np.array(y_synthetic) == s] for s in species_unique]
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

    # 5. CDF (Cumulative Distribution Function)
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

    # 6. Density plots (KDE)
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
    if len(np.unique(y_synthetic)) > 1:
        fig, ax = plt.subplots(figsize=(max(12, len(species_unique) * 0.8), 6))
        species_unique = sorted(np.unique(y_synthetic))

        good_counts = []
        bad_counts = []

        for species in species_unique:
            species_mask = np.array(y_synthetic) == species
            species_quality = np.array(quality)[species_mask]
            good_counts.append(int((species_quality == 'good').sum()))
            bad_counts.append(int((species_quality == 'bad').sum()))

        x = np.arange(len(species_unique))
        width = 0.6

        ax.bar(x, good_counts, width, label='Good', color='#2ecc71', alpha=0.8)
        ax.bar(x, bad_counts, width, bottom=good_counts, label='Bad', color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Species', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Per-Species Quality Breakdown', fontsize=14, fontweight='bold')
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
    for species in sorted(np.unique(y_synthetic)):
        species_mask = np.array(y_synthetic) == species
        species_distances = distances_synthetic[species_mask]
        species_quality = np.array(quality)[species_mask]

        # Also get real data distances for this species
        real_species_mask = np.array(y_real) == species
        real_species_distances = distances_real[real_species_mask]

        per_species_report[str(species)] = {
            'threshold': {
                'value': float(thresholds_per_species[species]['threshold']) if species in thresholds_per_species else float('nan'),
                'n_reals': int(thresholds_per_species[species]['n_reals']) if species in thresholds_per_species else 0,
                'method': thresholds_per_species[species]['method'] if species in thresholds_per_species else 'unknown',
                'mean': float(thresholds_per_species[species]['mean']) if species in thresholds_per_species else float('nan'),
                'std': float(thresholds_per_species[species]['std']) if species in thresholds_per_species else float('nan'),
                'median': float(thresholds_per_species[species]['median']) if species in thresholds_per_species else float('nan'),
                'max': float(thresholds_per_species[species]['max']) if species in thresholds_per_species else float('nan')
            },
            'synthetic': {
                'total_samples': int(species_mask.sum()),
                'good_samples': int((species_quality == 'good').sum()),
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
        print(f"  QUALITY THRESHOLD (based on real NN distances of this species):")
        print(f"    Value: {info['threshold']['value']:.4f}")
        print(f"    Method: {info['threshold']['method']}")
        print(f"    Computed from n={info['threshold']['n_reals']} real samples")
        print(f"    Stats: mean={info['threshold']['mean']:.4f}, std={info['threshold']['std']:.4f}, median={info['threshold']['median']:.4f}")
        print(f"  REAL DATA (NN to other real samples of same species):")
        print(f"    Total samples: {info['real']['total_samples']}")
        print(f"    Mean NN distance: {info['real']['distance_mean']:.4f} ± {info['real']['distance_std']:.4f}")
        print(f"    Median NN distance: {info['real']['distance_median']:.4f}")
        print(f"  SYNTHETIC DATA (NN to real samples of same species):")
        print(f"    Total:  {info['synthetic']['total_samples']}")
        print(f"    Good:   {info['synthetic']['good_samples']} ({100*info['synthetic']['good_samples']/max(1, info['synthetic']['total_samples']):.1f}%)")
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
    print(f"Thresholds are computed per-species using method: {THRESHOLD_METHOD}")
    print(f"This accounts for natural variation in intra-class density across species.\n")

    print(f"1. GOOD samples ({summary_stats['synthetic_data']['good_count']} samples):")
    print(f"   -> Distance to nearest real sample of same species ≤ species-specific threshold")
    print(f"   -> These synthetic samples closely match the distribution of their species")
    print(f"   -> USE IN TRAINING: They look like real data from their species")
    good_samples = results_df[results_df['quality'] == 'good']
    if len(good_samples) > 0:
        print(f"   -> Top species: {good_samples['species'].value_counts().head(3).to_dict()}")

    print(f"\n2. BAD samples ({summary_stats['synthetic_data']['bad_count']} samples):")
    print(f"   -> Distance to nearest real sample of same species > species-specific threshold")
    print(f"   -> These don't match real samples of their species closely")
    print(f"   -> EXCLUDE FROM TRAINING: These are likely artifacts or mode collapses")
    bad_samples = results_df[results_df['quality'] == 'bad']
    if len(bad_samples) > 0:
        print(f"   -> Top species: {bad_samples['species'].value_counts().head(3).to_dict()}")

    if summary_stats['synthetic_data'].get('unknown_count', 0) > 0:
        print(f"\n3. UNKNOWN samples ({summary_stats['synthetic_data'].get('unknown_count', 0)} samples):")
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

    # Save all with quality labels
    results_df.to_csv(OUTPUT_DIR / "synthetic_quality_detailed.csv", index=False)
    print(f"  [OK] Saved detailed quality assessment to synthetic_quality_detailed.csv")

    print("\n" + "="*80)
    print("QUALITY ASSESSMENT COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR.absolute()}\n")


if __name__ == "__main__":
    main()
