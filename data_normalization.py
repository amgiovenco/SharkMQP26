import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler

# === 1. Load the dataset ===
file_path = "species_14_features.csv"   # Change this if your file is elsewhere
df = pd.read_csv(file_path)

print("✅ Loaded dataset successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# === 2. Define transformations ===
cols_logscale = ['rise', 'std', 'auc_left', 'auc_right']
cols_zscore = ['min', 'mean']
shift_values = {}

# === 3. Apply Z-score normalization to 'min' and 'mean' ===
for col in cols_zscore:
    if col in df.columns:
        scaler = StandardScaler()
        df[f'{col}_z'] = scaler.fit_transform(df[[col]])
        print(f"✅ Applied Z-score normalization to '{col}' → new column: '{col}_z'")
    else:
        print(f"⚠️ Skipping '{col}' (column not found).")

# === 4. Apply log-scaling to 'rise', 'std', 'auc_left', 'auc_right' ===
for col in cols_logscale:
    if col not in df.columns:
        print(f"⚠️ Skipping '{col}' (column not found).")
        continue

    if (df[col] <= 0).any():
        shift = abs(df[col].min()) + 1
        print(f"⚠️ Detected non-positive values in '{col}'. Applying shift of {shift:.6f}")
        df[f"{col}_log"] = np.log(df[col] + shift)
        shift_values[col] = shift
    else:
        df[f"{col}_log"] = np.log1p(df[col])

    print(f"✅ Applied log-scaling to '{col}' → new column: '{col}_log'")

# === 5. Drop the original columns ===
cols_to_drop = cols_logscale + cols_zscore
df = df.drop(columns=cols_to_drop)
print(f"🗑️  Dropped original columns: {cols_to_drop}")

# === 6. Save the transformed dataset ===
output_path = "species_14_features_normalized4.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Saved transformed dataset to: {output_path}")

# === 7. Optional: Store and display shift metadata ===
if shift_values:
    print("\nℹ️  Shift values used for log-scaling:")
    for col, shift in shift_values.items():
        print(f"   {col}: {shift}")

print("\n✨ Normalization complete! Hooray! ✨")
