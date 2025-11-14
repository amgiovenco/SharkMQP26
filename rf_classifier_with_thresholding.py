# ============================================
# Hybrid (Rules + RF) with Confidence Rejection — FIXED
# ============================================
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "species_14_features_normalized2.csv"
DROP_RARE_SPECIES = True
TEST_SIZE = 0.20
RANDOM_STATE = 8
CONF_THRESHOLD = 0.30
REJECT_LABEL = "REJECT"

RF_PARAMS = dict(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found at {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

if DROP_RARE_SPECIES:
    df = df.groupby("Species").filter(lambda x: len(x) >= 5).reset_index(drop=True)

X = df.drop(columns=["Species"])
feature_cols = list(X.columns)
y_text = df["Species"].astype(str)

le = LabelEncoder()
y = le.fit_transform(y_text)
all_class_names = le.inverse_transform(np.arange(len(le.classes_)))

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Fit scaler on DataFrame to keep column names; transform with DataFrame to avoid warnings
scaler = StandardScaler()
scaler.fit(X_train_raw)  # fit on DF with feature names
X_train = pd.DataFrame(scaler.transform(X_train_raw), columns=feature_cols, index=X_train_raw.index)
X_test  = pd.DataFrame(scaler.transform(X_test_raw),  columns=feature_cols, index=X_test_raw.index)

# -----------------------------
# Train Random Forest
# -----------------------------
rf = RandomForestClassifier(**RF_PARAMS)
rf.fit(X_train, y_train)

# -----------------------------
# Optional manual rules (domain thresholds) — edit as needed
# -----------------------------
def rule_based_threshold(row: pd.Series) -> str | None:
    # Example rules (disabled):
    # if (row["tmax"] > 79.5) and (row["fwhm"] < 1.25):
    #     return "Arabian smooth-hound"
    # if (row["rise_log"] > 1.46) and (row["auc_right"] < 0.05):
    #     return "Blacktip shark"
    return None

# -----------------------------
# Predict with rejection
# -----------------------------
def predict_with_rejection(row_raw: pd.Series) -> str:
    # 1) manual rules first (raw values)
    r = rule_based_threshold(row_raw)
    if r is not None:
        return r

    # 2) RF with probability threshold
    row_df = pd.DataFrame([row_raw[feature_cols]], columns=feature_cols)  # preserve names
    row_scaled = pd.DataFrame(scaler.transform(row_df), columns=feature_cols)
    probs = rf.predict_proba(row_scaled.iloc[[0]])[0]
    max_p = probs.max()
    if max_p < CONF_THRESHOLD:
        return REJECT_LABEL
    pred_idx = probs.argmax()
    return le.inverse_transform([pred_idx])[0]

pred_text = [predict_with_rejection(row) for _, row in X_test_raw.iterrows()]
pred_array = np.array(pred_text, dtype=object)

# -----------------------------
# Metrics with rejection
# -----------------------------
is_rejected = (pred_array == REJECT_LABEL)
rejection_rate = is_rejected.mean()

accepted_mask = ~is_rejected
if accepted_mask.sum() == 0:
    print(f"\nConfidence threshold: {CONF_THRESHOLD:.2f}")
    print("All predictions were rejected. Lower CONF_THRESHOLD.")
    sys.exit(0)

y_acc_true = y_test[accepted_mask]
pred_acc_text = pred_array[accepted_mask]
y_acc_pred = le.transform(pred_acc_text)

print(f"\nConfidence threshold: {CONF_THRESHOLD:.2f}")
print(f"Rejection rate: {rejection_rate*100:.1f}% ({is_rejected.sum()} of {len(pred_array)})")

sel_acc = accuracy_score(y_acc_true, y_acc_pred)
print(f"Selective accuracy (accepted only): {sel_acc:.3f}")

# >>> FIX: compute labels present and matching target_names <<<
present_labels = np.unique(np.concatenate([y_acc_true, y_acc_pred]))
present_names  = le.inverse_transform(present_labels)

print("\nClassification Report (accepted only):")
print(classification_report(
    y_acc_true, y_acc_pred,
    labels=present_labels,              # ensure sizes match
    target_names=present_names,
    zero_division=0
))

# Confusion matrix (accepted only), normalized by true rows
cm = confusion_matrix(y_acc_true, y_acc_pred, labels=present_labels, normalize="true")
plt.figure(figsize=(12, 10))
im = plt.imshow(cm, interpolation="nearest", aspect="auto")
plt.title(f"Normalized Confusion Matrix — Accepted Only\n"
          f"thr={CONF_THRESHOLD:.2f} | coverage={(~is_rejected).mean():.2f} | sel-acc={sel_acc:.2f}")
plt.colorbar(im, fraction=0.046, pad=0.04, label="Proportion of True Class")
ticks = np.arange(len(present_names))
plt.xticks(ticks, present_names, rotation=90)
plt.yticks(ticks, present_names)
plt.xlabel("Predicted Species")
plt.ylabel("True Species")
plt.tight_layout()
plt.show()

# Optional: confidence distribution on the test set (no rules applied here)
test_probs = rf.predict_proba(X_test)
test_conf = test_probs.max(axis=1)
print("\nConfidence summary (RF on test set):")
print(pd.Series(test_conf).describe().round(3))
