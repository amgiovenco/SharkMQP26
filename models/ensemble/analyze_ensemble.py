#!/usr/bin/env python3
"""
Ensemble Analysis: Determine which models complement each other best.

Analyzes per-sample, per-class confidence scores to guide ensemble construction.
Methods: error correlation, KL-divergence, calibration, complementarity, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import pearsonr
from scipy.cluster import hierarchy
from sklearn.metrics import (
    brier_score_loss, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

def load_predictions(csv_path='all_model_predictions.csv'):
    """Load predictions CSV and extract model probabilities."""
    print(f"Loading predictions from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Extract true labels as strings
    true_labels_str = df['species_true'].values

    # Find model columns (format: {model}_prob_{species})
    model_cols = {}
    for col in df.columns:
        if '_prob_' in col:
            model_name = col.split('_prob_')[0]
            if model_name not in model_cols:
                model_cols[model_name] = []
            model_cols[model_name].append(col)

    # Sort species names for consistent ordering (with underscores)
    species_list_underscored = sorted(set(col.split('_prob_')[1] for col in df.columns if '_prob_' in col))

    # Get unique species from CSV (these are the ground truth names)
    species_list = sorted(df['species_true'].unique())

    print(f"Sample species from CSV: {species_list[:3]}")
    print(f"Sample species from columns: {species_list_underscored[:3]}")

    # Create mapping from CSV names to indices
    species_to_idx = {sp: idx for idx, sp in enumerate(species_list)}
    true_labels = np.array([species_to_idx[sp] for sp in true_labels_str], dtype=int)

    # Create reverse mapping for column lookup (underscored -> spaced)
    col_name_to_species = {}
    for col_name in species_list_underscored:
        # Try to find matching species in list by replacing underscores with spaces/hyphens
        for species in species_list:
            if col_name.replace('_', ' ').replace(' ', '_').lower() == species.replace(' ', '_').replace('-', '_').lower():
                col_name_to_species[col_name] = species
                break
        if col_name not in col_name_to_species:
            # Fallback: just replace underscores with spaces
            col_name_to_species[col_name] = col_name.replace('_', ' ')

    print(f"\nFound models: {list(model_cols.keys())}")
    print(f"Found {len(species_list)} classes")

    # Extract probability matrices for each model
    models = {}
    for model_name, cols in model_cols.items():
        # Sort columns to match species order (use underscored names)
        cols_sorted = [f"{model_name}_prob_{sp}" for sp in species_list_underscored]
        models[model_name] = df[cols_sorted].values.astype(np.float32)
        print(f"  {model_name}: {models[model_name].shape}")

    return df, models, species_list, true_labels

# ============================================================================
# 1. PAIRWISE ERROR CORRELATION
# ============================================================================

def compute_error_correlation(models, true_labels):
    """Measure correlation between models' error patterns."""
    print("\n" + "="*70)
    print("1. PAIRWISE ERROR CORRELATION")
    print("="*70)

    model_names = list(models.keys())
    n_models = len(model_names)

    # Get hard predictions (argmax)
    predictions = {}
    for name, probs in models.items():
        predictions[name] = np.argmax(probs, axis=1)

    # Compute error vectors
    errors = {}
    for name, preds in predictions.items():
        errors[name] = (preds != true_labels).astype(int)

    # Compute pairwise correlations
    correlation_matrix = np.zeros((n_models, n_models))

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
            elif i < j:
                try:
                    corr, _ = pearsonr(errors[name1], errors[name2])
                    if np.isnan(corr):
                        corr = 0.0  # Handle NaN correlation
                except:
                    corr = 0.0
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

    # Print results
    print("\nError Correlation Matrix:")
    corr_df = pd.DataFrame(correlation_matrix, index=model_names, columns=model_names)
    print(corr_df.round(3))

    # Identify good pairs
    print("\nModel Complementarity (Low error correlation = good ensemble):")
    good_pairs = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            corr = correlation_matrix[i, j]
            status = "[GOOD]" if corr < 0.5 else "[MODERATE]" if corr < 0.7 else "[REDUNDANT]"
            print(f"  {model_names[i]:15} + {model_names[j]:15}: {corr:6.3f} {status}")
            if corr < 0.5:
                good_pairs.append((model_names[i], model_names[j], corr))

    return correlation_matrix, model_names, good_pairs

# ============================================================================
# 2. KL-DIVERGENCE / JENSEN-SHANNON DISTANCE
# ============================================================================

def compute_js_divergence(models):
    """Measure distribution distance between models using Jensen-Shannon."""
    print("\n" + "="*70)
    print("2. JENSEN-SHANNON DIVERGENCE (distribution distance)")
    print("="*70)

    model_names = list(models.keys())
    n_models = len(model_names)

    # Compute pairwise JS-divergence
    js_matrix = np.zeros((n_models, n_models))

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                js_matrix[i, j] = 0.0
            elif i < j:
                # Average JS-divergence across all samples
                js_distances = []
                for sample_i in range(models[name1].shape[0]):
                    p = models[name1][sample_i] + 1e-10
                    q = models[name2][sample_i] + 1e-10
                    p /= p.sum()
                    q /= q.sum()
                    js = jensenshannon(p, q)
                    js_distances.append(js)

                avg_js = np.mean(js_distances)
                js_matrix[i, j] = avg_js
                js_matrix[j, i] = avg_js

    # Print results
    print("\nJensen-Shannon Distance Matrix:")
    js_df = pd.DataFrame(js_matrix, index=model_names, columns=model_names)
    print(js_df.round(3))

    print("\nModel Complementarity (High JS-divergence = different beliefs):")
    for i in range(n_models):
        for j in range(i+1, n_models):
            js = js_matrix[i, j]
            status = "[DIVERSE]" if js > 0.15 else "[SIMILAR]" if js > 0.08 else "[VERY SIMILAR]"
            print(f"  {model_names[i]:15} + {model_names[j]:15}: {js:6.3f} {status}")

    return js_matrix, model_names

# ============================================================================
# 3. CALIBRATION METRICS
# ============================================================================

def compute_calibration(models, true_labels, species_list):
    """Measure calibration quality of each model."""
    print("\n" + "="*70)
    print("3. CALIBRATION METRICS")
    print("="*70)

    calibration_scores = {}

    for model_name, probs in models.items():
        # Brier Score (lower is better)
        n_classes = len(species_list)
        true_one_hot = np.eye(n_classes)[true_labels]
        # Compute Brier score as mean squared error between one-hot and probabilities
        brier = np.mean((true_one_hot - probs) ** 2)

        # Expected Calibration Error (ECE)
        # Divide into bins, compute accuracy vs confidence
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        confidences = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        correct = (preds == true_labels).astype(float)

        ece = 0
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(confidences)

        calibration_scores[model_name] = {
            'brier': brier,
            'ece': ece,
        }

    print("\nCalibration Scores (lower = better calibrated):")
    for model_name in sorted(calibration_scores.keys()):
        scores = calibration_scores[model_name]
        print(f"  {model_name:15}: Brier={scores['brier']:.4f}, ECE={scores['ece']:.4f}")

    return calibration_scores

# ============================================================================
# 3.5. COMPREHENSIVE METRICS (Accuracy, Precision, Recall, F1 - weighted & macro)
# ============================================================================

def compute_comprehensive_metrics(predictions, true_labels):
    """
    Compute comprehensive classification metrics.

    Returns dict with:
    - accuracy
    - precision_macro, precision_weighted
    - recall_macro, recall_weighted
    - f1_macro, f1_weighted
    """
    metrics = {
        'accuracy': float(accuracy_score(true_labels, predictions)),
        'precision_macro': float(precision_score(true_labels, predictions, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(true_labels, predictions, average='weighted', zero_division=0)),
        'recall_macro': float(recall_score(true_labels, predictions, average='macro', zero_division=0)),
        'recall_weighted': float(recall_score(true_labels, predictions, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(true_labels, predictions, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(true_labels, predictions, average='weighted', zero_division=0)),
    }
    return metrics

def format_metrics_table(metrics_dict, model_names):
    """Format metrics into a nice table for display."""
    df = pd.DataFrame(metrics_dict).T
    df = df[['accuracy', 'precision_macro', 'precision_weighted',
             'recall_macro', 'recall_weighted', 'f1_macro', 'f1_weighted']]
    df.columns = ['Accuracy', 'Prec(M)', 'Prec(W)', 'Recall(M)', 'Recall(W)', 'F1(M)', 'F1(W)']
    return df

# ============================================================================
# 4. PER-CLASS SPECIALIZATION
# ============================================================================

def compute_class_specialization(models, true_labels, species_list):
    """Identify which models perform best on each class."""
    print("\n" + "="*70)
    print("4. PER-CLASS SPECIALIZATION")
    print("="*70)

    n_classes = len(species_list)
    model_names = list(models.keys())

    # Compute per-class accuracy
    per_class_acc = {}
    for model_name, probs in models.items():
        preds = np.argmax(probs, axis=1)
        acc_per_class = []
        for class_idx in range(n_classes):
            mask = (true_labels == class_idx)
            if mask.sum() > 0:
                acc = (preds[mask] == true_labels[mask]).mean()
            else:
                acc = 0
            acc_per_class.append(acc)
        per_class_acc[model_name] = acc_per_class

    # Identify best model per class
    print("\nTop model by class (first 20 classes shown):")
    for class_idx in range(min(20, n_classes)):
        class_name = species_list[class_idx]
        best_model = max(model_names, key=lambda m: per_class_acc[m][class_idx])
        best_acc = per_class_acc[best_model][class_idx]
        print(f"  {class_name:30}: {best_model:15} ({best_acc:.3f})")

    # Diversity check
    print("\nDiversity check: how many classes does each model specialize in?")
    for model_name in model_names:
        # Count how many classes this model is the best for
        specialist_count = 0
        for class_idx in range(n_classes):
            if per_class_acc[model_name][class_idx] == max(
                per_class_acc[m][class_idx] for m in model_names
            ):
                specialist_count += 1
        print(f"  {model_name:15}: specialist in {specialist_count:3}/{n_classes} classes")

    return per_class_acc

# ============================================================================
# 5. ENTROPY / CONFIDENCE DISTRIBUTION
# ============================================================================

def compute_entropy(models):
    """Analyze confidence distributions (entropy)."""
    print("\n" + "="*70)
    print("5. ENTROPY / CONFIDENCE DISTRIBUTION")
    print("="*70)

    entropy_stats = {}

    for model_name, probs in models.items():
        # Entropy
        epsilon = 1e-10
        entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)

        # Confidence (max probability)
        confidence = np.max(probs, axis=1)

        entropy_stats[model_name] = {
            'mean_entropy': entropy.mean(),
            'std_entropy': entropy.std(),
            'mean_confidence': confidence.mean(),
            'std_confidence': confidence.std(),
            'min_confidence': confidence.min(),
            'max_confidence': confidence.max(),
        }

    print("\nEntropy & Confidence Statistics:")
    print(f"{'Model':<15} {'Mean Entropy':<14} {'Mean Conf':<12} {'Conf Range':<20}")
    print("-" * 60)
    for model_name in sorted(entropy_stats.keys()):
        stats = entropy_stats[model_name]
        conf_range = f"[{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]"
        print(f"{model_name:<15} {stats['mean_entropy']:<14.4f} {stats['mean_confidence']:<12.4f} {conf_range:<20}")

    return entropy_stats

# ============================================================================
# 6. GREEDY FORWARD SELECTION
# ============================================================================

def greedy_forward_selection(models, true_labels, val_indices=None):
    """Greedily select best subset of models."""
    print("\n" + "="*70)
    print("6. GREEDY FORWARD SELECTION")
    print("="*70)

    if val_indices is None:
        val_indices = np.arange(len(true_labels))

    model_names = list(models.keys())
    selected = []
    remaining = set(model_names)

    all_metrics = {}

    # Find best single model
    best_model = None
    best_acc = 0
    for name in model_names:
        preds = np.argmax(models[name][val_indices], axis=1)
        acc = (preds == true_labels[val_indices]).mean()
        if acc > best_acc:
            best_acc = acc
            best_model = name

    selected.append(best_model)
    remaining.remove(best_model)

    # Compute detailed metrics for first model
    preds = np.argmax(models[best_model][val_indices], axis=1)
    metrics = compute_comprehensive_metrics(preds, true_labels[val_indices])
    all_metrics[best_model] = metrics

    print(f"Step 1: Selected {best_model}")
    print(f"        Accuracy: {metrics['accuracy']:.4f}, F1(M): {metrics['f1_macro']:.4f}")

    # Greedy addition
    step = 2
    while remaining and step <= len(model_names):
        best_new_model = None
        best_new_acc = 0
        best_new_metrics = None

        for candidate in remaining:
            # Ensemble: average probabilities
            ensemble_probs = np.zeros_like(models[selected[0]][val_indices])
            for model in selected + [candidate]:
                ensemble_probs += models[model][val_indices]
            ensemble_probs /= len(selected) + 1

            preds = np.argmax(ensemble_probs, axis=1)
            acc = (preds == true_labels[val_indices]).mean()

            if acc > best_new_acc:
                best_new_acc = acc
                best_new_model = candidate
                best_new_metrics = compute_comprehensive_metrics(preds, true_labels[val_indices])

        if best_new_model:
            selected.append(best_new_model)
            remaining.remove(best_new_model)
            ensemble_name = f"{'+'.join(selected)}"
            all_metrics[ensemble_name] = best_new_metrics
            print(f"Step {step}: Added {best_new_model}")
            print(f"        Ensemble: {'+'.join(selected)}")
            print(f"        Accuracy: {best_new_metrics['accuracy']:.4f}, F1(M): {best_new_metrics['f1_macro']:.4f}")
            step += 1
        else:
            break

    print(f"\n[SUCCESS] Final ensemble: {selected}")
    return selected, all_metrics

# ============================================================================
# 6.5. JACCARD SIMILARITY (correctness overlap)
# ============================================================================

def compute_jaccard_correctness(models, true_labels):
    """Measure overlap in correct predictions (Jaccard index)."""
    print("\n" + "="*70)
    print("6.5. JACCARD SIMILARITY (correctness overlap)")
    print("="*70)

    model_names = list(models.keys())
    n_models = len(model_names)

    # Get correct predictions for each model
    correct_sets = {}
    for model_name, probs in models.items():
        preds = np.argmax(probs, axis=1)
        correct = np.where(preds == true_labels)[0]
        correct_sets[model_name] = set(correct)

    # Compute Jaccard for each pair
    jaccard_matrix = np.zeros((n_models, n_models))

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                jaccard_matrix[i, j] = 1.0
            else:
                intersection = len(correct_sets[name1] & correct_sets[name2])
                union = len(correct_sets[name1] | correct_sets[name2])
                if union > 0:
                    jaccard_matrix[i, j] = intersection / union
                else:
                    jaccard_matrix[i, j] = 0.0

    # Print results
    print("\nJaccard Similarity Matrix (correctness overlap):")
    print("Low Jaccard = models correct on different samples = GOOD for ensemble")
    print("High Jaccard = models correct on same samples = REDUNDANT\n")

    jaccard_df = pd.DataFrame(jaccard_matrix, index=model_names, columns=model_names)
    print(jaccard_df.round(3))

    # Identify complementary pairs
    print("\nModel Complementarity (by correctness overlap):")
    complementary_pairs = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            j_score = jaccard_matrix[i, j]
            if j_score < 0.3:
                status = "[EXCELLENT]"
                complementary_pairs.append((model_names[i], model_names[j], j_score))
            elif j_score < 0.5:
                status = "[GOOD]"
                complementary_pairs.append((model_names[i], model_names[j], j_score))
            elif j_score < 0.7:
                status = "[MODERATE]"
            else:
                status = "[REDUNDANT]"

            print(f"  {model_names[i]:15} + {model_names[j]:15}: {j_score:6.3f} {status}")

    return jaccard_matrix, model_names, complementary_pairs

# ============================================================================
# 6.75. SPECIES COVERAGE (set cover problem)
# ============================================================================

def compute_species_coverage(models, true_labels, species_list):
    """Find which model combinations cover all species with ≥1 correct prediction."""
    print("\n" + "="*70)
    print("6.75. SPECIES COVERAGE ANALYSIS")
    print("="*70)

    model_names = list(models.keys())
    n_models = len(model_names)
    n_species = len(species_list)

    # For each model, find which species it gets correct
    model_coverage = {}
    for model_name, probs in models.items():
        preds = np.argmax(probs, axis=1)
        correct = (preds == true_labels)

        species_correct = set()
        for species_idx in range(n_species):
            # Check if this model got at least one sample of this species correct
            species_samples = (true_labels == species_idx)
            if np.any(correct[species_samples]):
                species_correct.add(species_idx)

        model_coverage[model_name] = species_correct

    # Print individual coverage
    print(f"\nIndividual model coverage:")
    for model in sorted(model_coverage.keys()):
        covered = len(model_coverage[model])
        print(f"  {model:15}: {covered:2}/{n_species} species")
        if covered < n_species:
            missing = [species_list[i] for i in range(n_species) if i not in model_coverage[model]]
            print(f"    Missing: {missing[:3]}{'...' if len(missing) > 3 else ''}")

    # Find minimum set cover (greedy algorithm)
    print(f"\nFinding minimum ensemble for full species coverage...")
    uncovered = set(range(n_species))
    selected_models = []

    while uncovered:
        # Find model that covers most uncovered species
        best_model = None
        best_new_covered = 0

        for model in model_names:
            if model not in selected_models:
                new_covered = len(model_coverage[model] & uncovered)
                if new_covered > best_new_covered:
                    best_new_covered = new_covered
                    best_model = model

        if best_model is None:
            break

        selected_models.append(best_model)
        uncovered -= model_coverage[best_model]
        print(f"  Step {len(selected_models)}: Added {best_model:15} > covers {best_new_covered} more species")

    if not uncovered:
        print(f"\n[SUCCESS] FULL COVERAGE POSSIBLE with {len(selected_models)} models:")
        for model in selected_models:
            print(f"  - {model}")
    else:
        print(f"\n[ALERT] Cannot achieve full coverage. Missing {len(uncovered)} species:")
        missing_species = [species_list[i] for i in uncovered]
        print(f"  {missing_species}")

    return model_coverage, selected_models

# ============================================================================
# 6.9. COMPREHENSIVE COMBINATION ANALYSIS
# ============================================================================

def analyze_all_combinations(models, true_labels, val_indices=None):
    """Analyze performance metrics for different model combinations."""
    print("\n" + "="*70)
    print("6.9. COMPREHENSIVE COMBINATION ANALYSIS")
    print("="*70)

    if val_indices is None:
        val_indices = np.arange(len(true_labels))

    model_names = list(models.keys())
    combination_metrics = {}

    # Test all single models
    print("\n--- INDIVIDUAL MODELS ---")
    single_model_metrics = {}
    for model_name in sorted(model_names):
        preds = np.argmax(models[model_name][val_indices], axis=1)
        metrics = compute_comprehensive_metrics(preds, true_labels[val_indices])
        single_model_metrics[model_name] = metrics
        combination_metrics[model_name] = metrics

    df_single = format_metrics_table(single_model_metrics, model_names)
    print(df_single.to_string())

    # Test all pairs
    print("\n--- ALL PAIRS ---")
    pair_metrics = {}
    for m1, m2 in combinations(sorted(model_names), 2):
        ensemble_probs = (models[m1][val_indices] + models[m2][val_indices]) / 2
        preds = np.argmax(ensemble_probs, axis=1)
        metrics = compute_comprehensive_metrics(preds, true_labels[val_indices])
        pair_name = f"{m1}+{m2}"
        pair_metrics[pair_name] = metrics
        combination_metrics[pair_name] = metrics

    df_pairs = format_metrics_table(pair_metrics, model_names)
    print(df_pairs.to_string())

    # Test triples (if not too many)
    if len(model_names) <= 6:
        print("\n--- TRIPLES ---")
        triple_metrics = {}
        for m1, m2, m3 in combinations(sorted(model_names), 3):
            ensemble_probs = (models[m1][val_indices] + models[m2][val_indices] + models[m3][val_indices]) / 3
            preds = np.argmax(ensemble_probs, axis=1)
            metrics = compute_comprehensive_metrics(preds, true_labels[val_indices])
            triple_name = f"{m1}+{m2}+{m3}"
            triple_metrics[triple_name] = metrics
            combination_metrics[triple_name] = metrics

        df_triples = format_metrics_table(triple_metrics, model_names)
        print(df_triples.to_string())

    # Test all models
    print("\n--- ALL MODELS ---")
    ensemble_probs = np.zeros_like(models[model_names[0]][val_indices])
    for model_name in model_names:
        ensemble_probs += models[model_name][val_indices]
    ensemble_probs /= len(model_names)
    preds = np.argmax(ensemble_probs, axis=1)
    metrics = compute_comprehensive_metrics(preds, true_labels[val_indices])
    all_models_name = '+'.join(sorted(model_names))
    combination_metrics[all_models_name] = metrics
    print(f"{all_models_name}")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")

    # Find and show top 5 by different metrics
    print("\n--- TOP 5 BY METRIC ---")
    sorted_by_acc = sorted(combination_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    sorted_by_f1m = sorted(combination_metrics.items(), key=lambda x: x[1]['f1_macro'], reverse=True)

    print("\nTop 5 by Accuracy:")
    for i, (name, metrics) in enumerate(sorted_by_acc[:5], 1):
        print(f"  {i}. {name}: {metrics['accuracy']:.4f} (F1-M: {metrics['f1_macro']:.4f})")

    print("\nTop 5 by F1-Macro (handles class imbalance):")
    for i, (name, metrics) in enumerate(sorted_by_f1m[:5], 1):
        print(f"  {i}. {name}: {metrics['f1_macro']:.4f} (Acc: {metrics['accuracy']:.4f})")

    return combination_metrics

# ============================================================================
# 7. MODEL CLUSTERING
# ============================================================================

def cluster_models(models):
    """Cluster models by probability distribution similarity."""
    print("\n" + "="*70)
    print("7. MODEL CLUSTERING (by distribution similarity)")
    print("="*70)

    model_names = list(models.keys())
    n_models = len(model_names)

    # Flatten each model's predictions
    flattened = []
    for name in model_names:
        flat = models[name].flatten()
        flattened.append(flat)

    flattened = np.array(flattened)

    # Compute distance matrix (using JS-divergence)
    distances = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Average JS-divergence between models
            p = flattened[i] + 1e-10
            q = flattened[j] + 1e-10
            p /= p.sum()
            q /= q.sum()
            js = jensenshannon(p, q)
            distances.append(js)

    # Hierarchical clustering
    condensed_dist = squareform(pdist(flattened, metric='euclidean'))
    linkage_matrix = hierarchy.linkage(condensed_dist[np.triu_indices_from(condensed_dist, k=1)], method='ward')
    clusters = hierarchy.fcluster(linkage_matrix, t=n_models//2 + 1, criterion='maxclust')

    print("\nModel Clusters:")
    for cluster_id in sorted(set(clusters)):
        cluster_models = [model_names[i] for i in range(n_models) if clusters[i] == cluster_id]
        print(f"  Cluster {cluster_id}: {cluster_models}")

    return clusters, model_names

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("SHARK ENSEMBLE ANALYSIS")
    print("="*70)

    # Load data
    csv_path = Path('all_model_predictions.csv')
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    df, models, species_list, true_labels = load_predictions(str(csv_path))

    # Run all analyses
    corr_matrix, model_names_1, good_pairs = compute_error_correlation(models, true_labels)
    js_matrix, model_names_2 = compute_js_divergence(models)
    calibration = compute_calibration(models, true_labels, species_list)
    per_class_acc = compute_class_specialization(models, true_labels, species_list)
    entropy = compute_entropy(models)
    selected, greedy_metrics = greedy_forward_selection(models, true_labels)
    combination_metrics = analyze_all_combinations(models, true_labels)
    jaccard_matrix, model_names_65, jaccard_pairs = compute_jaccard_correctness(models, true_labels)
    model_coverage, coverage_selected = compute_species_coverage(models, true_labels, species_list)
    clusters, model_names_7 = cluster_models(models)

    # ========================================================================
    # SUMMARY & RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*70)
    print("ENSEMBLE RECOMMENDATIONS")
    print("="*70)

    print(f"\n[REC] Recommended base models to use:")
    for model in selected:
        print(f"  - {model}")

    print(f"\n[REC] Best complementary pairs (by error correlation):")
    for m1, m2, corr in good_pairs[:3]:
        print(f"  - {m1} + {m2} (error correlation: {corr:.3f})")

    print(f"\n[REC] Best complementary pairs (by Jaccard correctness):")
    if jaccard_pairs:
        for m1, m2, j_score in jaccard_pairs[:3]:
            print(f"  - {m1} + {m2} (Jaccard: {j_score:.3f})")
    else:
        print("  - None found with low Jaccard")

    print(f"\n[REC] Most confident/stable model:")
    best_calib = min(calibration.keys(), key=lambda m: calibration[m]['ece'])
    print(f"  - {best_calib} (ECE: {calibration[best_calib]['ece']:.4f})")

    print(f"\n[REC] Best pair by multiple metrics (low error corr + low Jaccard + high JSD):")
    # Find pairs that score well on multiple metrics
    js_dict = {(model_names_2[i], model_names_2[j]): js_matrix[i][j]
               for i in range(len(model_names_2)) for j in range(i+1, len(model_names_2))}
    jaccard_dict = {(model_names_65[i], model_names_65[j]): jaccard_matrix[i][j]
                    for i in range(len(model_names_65)) for j in range(i+1, len(model_names_65))}

    best_combos = []
    for m1, m2, j_score in jaccard_pairs[:5]:
        key = tuple(sorted([m1, m2]))
        j_div = js_dict.get(key, 0)
        best_combos.append((m1, m2, j_score, j_div))

    if best_combos:
        best_combos.sort(key=lambda x: (x[2], -x[3]))  # Sort by low Jaccard, high JSD
        m1, m2, j_score, j_div = best_combos[0]
        print(f"  - {m1} + {m2}")
        print(f"    Jaccard: {j_score:.3f} (low = good), JSD: {j_div:.3f} (high = good)")

    print(f"\n[REC] For full species coverage (>=1 correct per species):")
    if not coverage_selected:
        print("  - All models together do NOT cover all 57 species")
    else:
        print(f"  - Use {len(coverage_selected)} models: {coverage_selected}")
        # Check if this is achievable with fewer models
        if len(coverage_selected) < len(models):
            print(f"    (you can drop: {[m for m in models.keys() if m not in coverage_selected]})")

    # Save results
    results = {
        'selected_models': selected,
        'coverage_required_models': coverage_selected,
        'model_species_coverage': {model: len(species) for model, species in model_coverage.items()},
        'good_pairs_by_error_correlation': [(m1, m2, float(corr)) for m1, m2, corr in good_pairs],
        'good_pairs_by_jaccard': [(m1, m2, float(j_score)) for m1, m2, j_score in jaccard_pairs],
        'calibration': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in calibration.items()},
        'entropy': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in entropy.items()},
        'combination_metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in combination_metrics.items()},
    }

    with open('ensemble_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Results saved to ensemble_analysis.json")

if __name__ == '__main__':
    main()
