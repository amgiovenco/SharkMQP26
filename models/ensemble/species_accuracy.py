#!/usr/bin/env python3
"""
Per-species accuracy analysis.
Shows how many times each species was classified correctly vs incorrectly.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_species_accuracy(csv_path='all_model_predictions.csv'):
    """Analyze per-species accuracy across all models."""
    print(f"Loading predictions from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Get true labels
    true_labels = df['species_true'].values

    # Get all model columns
    model_cols = {}
    for col in df.columns:
        if '_prob_' in col:
            model_name = col.split('_prob_')[0]
            if model_name not in model_cols:
                model_cols[model_name] = []
            model_cols[model_name].append(col)

    # Get species list from columns
    species_list_underscored = sorted(set(col.split('_prob_')[1] for col in df.columns if '_prob_' in col))
    species_list = [sp.replace('_', ' ') for sp in species_list_underscored]

    print(f"\n{'='*70}")
    print("PER-SPECIES CLASSIFICATION ACCURACY")
    print(f"{'='*70}\n")

    # For each model, compute per-species accuracy
    all_species_stats = {sp: {'correct': 0, 'total': 0} for sp in species_list}

    for model_name, cols in model_cols.items():
        # Get probabilities
        cols_sorted = [f"{model_name}_prob_{sp}" for sp in species_list_underscored]
        probs = df[cols_sorted].values
        preds = np.argmax(probs, axis=1)

        # Convert predictions back to species names
        pred_labels = np.array([species_list[p] for p in preds])

        # Compute per-species stats
        for species in species_list:
            species_mask = (true_labels == species)
            if species_mask.sum() > 0:
                correct = (pred_labels[species_mask] == species).sum()
                total = species_mask.sum()
                all_species_stats[species]['correct'] += correct
                all_species_stats[species]['total'] += total

    # Compute accuracy per species (aggregated across all models)
    species_accuracy = []
    for species in species_list:
        stats = all_species_stats[species]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            err = stats['total'] - stats['correct']
            species_accuracy.append({
                'species': species,
                'correct': stats['correct'],
                'incorrect': err,
                'total': stats['total'],
                'accuracy': acc,
            })

    # Sort by accuracy (worst first)
    species_accuracy.sort(key=lambda x: x['accuracy'])

    # Display results
    print(f"{'Species':<40} {'Correct':>8} {'Incorrect':>10} {'Total':>6} {'Accuracy':>10}")
    print("-" * 80)

    for item in species_accuracy:
        acc_pct = f"{item['accuracy']*100:.1f}%"
        print(f"{item['species']:<40} {item['correct']:>8} {item['incorrect']:>10} {item['total']:>6} {acc_pct:>10}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    worst_species = species_accuracy[:5]
    best_species = species_accuracy[-5:]

    print(f"\nWorst 5 species (FOCUS HERE):")
    for item in worst_species:
        print(f"  {item['species']:40} {item['accuracy']*100:5.1f}% ({item['correct']}/{item['total']})")

    print(f"\nBest 5 species:")
    for item in reversed(best_species):
        print(f"  {item['species']:40} {item['accuracy']*100:5.1f}% ({item['correct']}/{item['total']})")

    # Overall stats
    total_correct = sum(item['correct'] for item in species_accuracy)
    total_samples = sum(item['total'] for item in species_accuracy)
    overall_acc = total_correct / total_samples if total_samples > 0 else 0

    print(f"\nOverall accuracy: {overall_acc*100:.2f}% ({total_correct}/{total_samples})")
    print(f"Number of species: {len(species_accuracy)}")

    # Export to CSV for detailed analysis
    results_df = pd.DataFrame(species_accuracy)
    results_df.to_csv('species_accuracy.csv', index=False)
    print(f"\n✓ Detailed results saved to species_accuracy.csv")

if __name__ == '__main__':
    analyze_species_accuracy()
