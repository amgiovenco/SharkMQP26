import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load dataset ---
# Example format:
# Columns: Species, temp_1, temp_2,..., temp_n
# Rows: Arabian smooth-hound, 0.01527, 0.01518,...
data = pd.read_csv("shark_dataset.csv")

# --- Features = Fluorescence values, Label = Species ---
X = data.drop(columns=["Species"])
y = data["Species"]

# --- Train/test split with stratification (keeps species balanced across sets) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,   # number of trees
    max_depth=None,     # fully expanded trees
    random_state=8,
    n_jobs=-1           # use all CPU cores
)

rf.fit(X_train, y_train)

# --- Predictions ---
y_pred = rf.predict(X_test)

y_proba = rf.predict_proba(X_test)

# Pick 5 random test indices
sample_indices = np.random.choice(len(y_test), size=5, replace=False)

for i in sample_indices:
    row_idx = y_test.index[i]
    true_label = y_test.iloc[i]
    pred_label = y_pred[i]
    pred_conf = np.max(y_proba[i]) * 100

    # Top 5 probabilities
    top5_idx = np.argsort(y_proba[i])[::-1][:5]
    top5_labels = rf.classes_[top5_idx]
    top5_probs = y_proba[i][top5_idx] * 100

    # Print header
    print(f"Row {row_idx} | True: {true_label} | Pred: {pred_label} ({pred_conf:.1f}%)")

    # Print top 5 breakdown
    for lbl, prob in zip(top5_labels, top5_probs):
        print(f"  {lbl:<30} {prob:6.3f}%")

    print()

# --- Evaluation ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Confusion matrix for species-level performance ---
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)

# Plot heatmap
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

# --- Feature importance ---
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 10 Important Temperatures:\n", importances.nlargest(10))

# --- Stratified K-Fold Cross-Validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

print("Cross-validation accuracies:", scores)
print("Mean accuracy:", scores.mean())

plt.figure(figsize=(8,6))
plt.bar(range(1, len(scores)+1), scores, color="skyblue")
plt.axhline(np.mean(scores), color="red", linestyle="--", label=f"Mean = {scores.mean():.2f}")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Accuracies")
plt.legend()
plt.show()

# --- Per-class precision, recall, and F1 ---
report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose().drop(["accuracy", "macro avg", "weighted avg"])
precision = df["precision"]
recall = df["recall"]
f1 = df["f1-score"]

# Plot precision
plt.figure(figsize=(12,6))
precision.plot(kind="bar", color="skyblue")
plt.title("Per-Class Precision")
plt.ylabel("Precision")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plot recall
plt.figure(figsize=(12,6))
recall.plot(kind="bar", color="lightgreen")
plt.title("Per-Class Recall")
plt.ylabel("Recall")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plot F1 score
plt.figure(figsize=(12,6))
f1.plot(kind="bar", color="salmon")
plt.title("Per-Class F1 Score")
plt.ylabel("F1 Score")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# --- Hyperparameter tuning ---
# param_grid = {
#     "n_estimators": [50, 100, 200],
#     "max_depth": [None, 10, 20],
#     "min_samples_split": [2, 5, 10]
# }

# grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
#                            param_grid, cv=5, scoring="accuracy", n_jobs=-1)
# grid_search.fit(X, y)

# print("Best parameters:", grid_search.best_params_)
# print("Best cross-val score:", grid_search.best_score_)
