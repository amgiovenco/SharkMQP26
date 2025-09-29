import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
# Example format:
# Columns: Species, temp_1, temp_2,..., temp_n
# Rows: Arabian smooth-hound, 0.01527, 0.01518,...
data = pd.read_csv("shark_dataset.csv")

# Features = Fluorescence values, Label = Species
X = data.drop(columns=["Species"])
y = data["Species"]

# Train/test split with stratification (keeps species balanced across sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,   # number of trees
    max_depth=None,     # fully expanded trees
    random_state=42,
    n_jobs=-1           # use all CPU cores
)

rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix for species-level performance
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 10 Important Temperatures:\n", importances.nlargest(10))
