"""
Simple inference test script.

To test a different model, change the import below.
"""
import random
import pandas as pd

from cnn_inference import ml_inference, CNNModel # CHANGE

# Test configuration
CSV_PATH = "../../data/shark_dataset.csv"
NUM_SAMPLES = 5

def main():
    print(f"Testing inference on: {CSV_PATH}")

    # Load CSV to get random samples
    df = pd.read_csv(CSV_PATH)
    total_samples = len(df)

    # Pick random sample indices
    sample_indices = random.sample(range(total_samples), min(NUM_SAMPLES, total_samples))

    successful = 0
    failed = 0

    for i, sample_idx in enumerate(sample_indices, 1):
        print(f"\nTest {i}/{len(sample_indices)} - Sample index {sample_idx}")
        print("-" * 80)

        # Run inference
        result = ml_inference(CSV_PATH, sample_index=sample_idx)

        # Print result
        if result['success']:
            successful += 1
            print("SUCCESS")

            print(f"\nTop 3:")
            for pred in result['predictions'][:3]:
                print(f"  {pred['rank']}. {pred['species']:40s} {pred['confidence']:.4f}")
        else:
            failed += 1
            print("FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
