import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
file_path = "cleaned_data_but_in_rows.csv"
df = pd.read_csv(file_path)

# Target column
target_column = "Species"

# Drop species with fewer than 5 instances
species_counts = df[target_column].value_counts()
valid_species = species_counts[species_counts >= 5].index
df_filtered = df[df[target_column].isin(valid_species)]

# Stratified split: 60% train, 20% validation, 20% test
train_df, temp_df = train_test_split(
    df_filtered,
    test_size=0.40,
    stratify=df_filtered[target_column],
    random_state=8
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df[target_column],
    random_state=8
)

# Save output CSVs
train_df.to_csv("shark_training_data.csv", index=False)
val_df.to_csv("shark_validation_data.csv", index=False)
test_df.to_csv("shark_test_data.csv", index=False)

print("✅ Done! Files exported:")
print("  shark_training_data.csv")
print("  shark_validation_data.csv")
print("  shark_test_data.csv")
