#!/usr/bin/env python3
"""
Show per-species performance for the best ensemble combination.
Best combo: cnn + resnet1d + statistics (98.16% accuracy, 98.19% F1-weighted)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_predictions(csv_path='all_model_predictions.csv'):
    """Load predictions CSV and extract model probabilities."""
    print(f"Loading predictions from {csv_path}...")
    df = pd.read_csv(csv_path)

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

    # Create mapping from CSV names to indices
    species_to_idx = {sp: idx for idx, sp in enumerate(species_list)}
    true_labels = np.array([species_to_idx[sp] for sp in true_labels_str], dtype=int)

    # Extract probability matrices for each model
    models = {}
    for model_name, cols in model_cols.items():
        # Sort columns to match species order (use underscored names)
        cols_sorted = [f"{model_name}_prob_{sp}" for sp in species_list_underscored]
        models[model_name] = df[cols_sorted].values.astype(np.float32)

    return df, models, species_list, true_labels

def analyze_best_ensemble():
    """Analyze per-species performance of best ensemble."""
    print("\n" + "="*80)
    print("BEST ENSEMBLE PERFORMANCE BY SPECIES")
    print("Ensemble: cnn + resnet1d + statistics")
    print("="*80)

    # Load data
    csv_path = Path('all_model_predictions.csv')
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    df, models, species_list, true_labels = load_predictions(str(csv_path))

    # Create ensemble predictions (average the 3 best models)
    best_models = ['cnn', 'resnet1d', 'statistics']
    ensemble_probs = np.zeros_like(models['cnn'])

    for model_name in best_models:
        ensemble_probs += models[model_name]
    ensemble_probs /= len(best_models)

    # Get hard predictions
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    # Overall metrics
    overall_correct = (ensemble_preds == true_labels).sum()
    overall_total = len(true_labels)
    overall_acc = overall_correct / overall_total

    print(f"\n[OVERALL PERFORMANCE]")
    print(f"  Correct: {overall_correct}/{overall_total} ({overall_acc*100:.2f}%)")

    # Per-species analysis
    print(f"\n[PER-SPECIES BREAKDOWN]")
    print(f"{'Species':<40} {'Correct':<15} {'Accuracy':<15}")
    print("-" * 70)

    species_results = []

    for species_idx, species_name in enumerate(species_list):
        # Find all samples of this species
        mask = (true_labels == species_idx)
        total_samples = mask.sum()

        if total_samples == 0:
            continue

        # Count correct predictions
        correct_samples = (ensemble_preds[mask] == species_idx).sum()
        accuracy = correct_samples / total_samples

        species_results.append({
            'species': species_name,
            'correct': int(correct_samples),
            'total': int(total_samples),
            'accuracy': accuracy
        })

        print(f"{species_name:<40} {correct_samples:>3}/{total_samples:<3}       {accuracy*100:>6.1f}%")

    # Sort by accuracy
    species_results_sorted = sorted(species_results, key=lambda x: x['accuracy'])

    print("\n" + "="*80)
    print("[WORST PERFORMING SPECIES (Bottom 10)]")
    print("="*80)
    print(f"{'Species':<40} {'Correct':<15} {'Accuracy':<15}")
    print("-" * 70)

    for result in species_results_sorted[:10]:
        print(f"{result['species']:<40} {result['correct']:>3}/{result['total']:<3}       {result['accuracy']*100:>6.1f}%")

    print("\n" + "="*80)
    print("[BEST PERFORMING SPECIES (Top 10)]")
    print("="*80)
    print(f"{'Species':<40} {'Correct':<15} {'Accuracy':<15}")
    print("-" * 70)

    for result in species_results_sorted[-10:]:
        print(f"{result['species']:<40} {result['correct']:>3}/{result['total']:<3}       {result['accuracy']*100:>6.1f}%")

    # Statistics
    accuracies = [r['accuracy'] for r in species_results]
    print("\n" + "="*80)
    print("[STATISTICS ACROSS ALL SPECIES]")
    print("="*80)
    print(f"  Mean per-species accuracy:  {np.mean(accuracies)*100:.2f}%")
    print(f"  Median per-species accuracy: {np.median(accuracies)*100:.2f}%")
    print(f"  Min per-species accuracy:    {np.min(accuracies)*100:.2f}%")
    print(f"  Max per-species accuracy:    {np.max(accuracies)*100:.2f}%")
    print(f"  Std dev:                    {np.std(accuracies)*100:.2f}%")

    # Count perfect predictions
    perfect_count = sum(1 for r in species_results if r['correct'] == r['total'])
    print(f"\n  Species with 100% accuracy: {perfect_count}/{len(species_results)}")

    # Comparison with individual models
    print("\n" + "="*80)
    print("[COMPARISON WITH INDIVIDUAL MODELS]")
    print("="*80)

    model_accuracies = {}
    for model_name in best_models + ['extratrees', 'rulebased']:
        if model_name in models:
            preds = np.argmax(models[model_name], axis=1)
            acc = (preds == true_labels).mean()
            model_accuracies[model_name] = acc

    print(f"{'Model':<20} {'Accuracy':<15}")
    print("-" * 35)
    for model_name in ['cnn', 'resnet1d', 'statistics', 'extratrees', 'rulebased']:
        if model_name in model_accuracies:
            acc = model_accuracies[model_name]
            marker = " [BEST COMBO]" if model_name in best_models else ""
            print(f"{model_name:<20} {acc*100:>6.2f}%{marker}")

    print(f"{'cnn+resnet1d+stats':<20} {overall_acc*100:>6.2f}%            [ENSEMBLE]")

if __name__ == '__main__':
    analyze_best_ensemble()
