"""Quick test of the inference.py module"""

import pandas as pd
import numpy as np
from inference import SharkClassifier, SharkCNN, GaussianNoise, FocalLoss

print("=" * 80)
print("TESTING INFERENCE")
print("=" * 80)

# Load classifier
print("\n1. Loading classifier...")
try:
    classifier = SharkClassifier()
    print("✅ Classifier loaded successfully!")
except Exception as e:
    print(f"❌ Error loading classifier: {e}")
    exit(1)

# Load real data and get a sample
print("\n2. Loading sample data...")
try:
    real_data = pd.read_csv("../data/shark_dataset.csv")
    sample_row = real_data.iloc[0]
    sample_species = sample_row['Species']
    sample_fluorescence = sample_row.drop('Species').values.astype(float)
    print(f"✅ Loaded sample: {sample_species} with {len(sample_fluorescence)} temperature points")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Test prediction from fluorescence values
print("\n3. Testing prediction from fluorescence array...")
try:
    predictions = classifier.predict(sample_fluorescence, top_k=5)
    print(f"✅ Got predictions!")
    print(f"\n   Top 5 predictions:")
    for pred in predictions:
        print(f"     {pred['rank']}. {pred['species']}: {pred['confidence']:.4f}")

    # Check if top prediction matches
    top_pred = predictions[0]['species']
    if top_pred == sample_species:
        print(f"\n   ✅ TOP PREDICTION MATCHES! ({sample_species})")
    else:
        print(f"\n   ⚠️  Top prediction: {top_pred}, Actual: {sample_species}")
except Exception as e:
    print(f"❌ Error making prediction: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test prediction from CSV row
print("\n4. Testing prediction from CSV row...")
try:
    predictions_csv = classifier.predict_from_csv_row(sample_row, top_k=3)
    print(f"✅ CSV prediction works!")
    print(f"\n   Top 3 predictions:")
    for pred in predictions_csv:
        print(f"     {pred['rank']}. {pred['species']}: {pred['confidence']:.4f}")
except Exception as e:
    print(f"❌ Error with CSV prediction: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - INFERENCE IS WORKING!")
print("=" * 80)
