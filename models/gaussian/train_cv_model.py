"""
Cross-validation test for Gaussian model (10-fold stratified, 90/10 split)
Sanity check to detect overfitting in optimize_model.py
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Config
CSV_PATH = "../../data/shark_dataset.csv"
SPECIES_COL = "Species"
RANDOM_STATE = 8
N_SPLITS = 10
K_RANGE = (1, 6)
DECIMATE_STEP = 6

# ============== Feature Extraction ==============

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)

def gaussian_sum(x, *p):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(p), 3):
        amp, mu, sigma = p[i:i+3]
        y += gaussian(x, float(amp), float(mu), abs(float(sigma)))
    return y

def preprocess_curve(x, y):
    """Smooth + quadratic baseline removal + normalization."""
    y = np.asarray(y, float)
    dx = x[1] - x[0]
    win = max(7, int(round(1.5 / dx)) | 1)
    if win >= len(y):
        win = max(7, (len(y)//2)*2 - 1)
    y_s = savgol_filter(y, window_length=max(7, win), polyorder=3, mode="interp")

    q = np.quantile(y_s, 0.3)
    mask = y_s <= q
    if mask.sum() >= 10:
        coeffs = np.polyfit(x[mask], y_s[mask], deg=2)
        baseline = np.polyval(coeffs, x)
        y_b = y_s - baseline
    else:
        y_b = y_s - np.min(y_s)

    scale = np.quantile(y_b, 0.99)
    if scale > 0:
        y_b = y_b / scale
    y_b = np.maximum(y_b, 0.0)
    return y_b

def decimate(x, y, step=8):
    return x[::step], y[::step]

def seed_peaks(x, y, k):
    spread = max(np.quantile(y, 0.90) - np.quantile(y, 0.10), 1e-6)
    prom = spread * 0.15
    peaks, props = find_peaks(y, prominence=prom, distance=max(1, len(x)//150))
    if len(peaks) == 0:
        peaks = np.argsort(y)[::-1][:k]
        prominences = y[peaks] - np.min(y)
    else:
        prominences = props["prominences"]

    if len(peaks) > 0:
        w_idx = peak_widths(y, peaks, rel_height=0.5)[0]
        w_c = w_idx * (x[1] - x[0])
    else:
        w_c = np.array([(x[-1]-x[0])/(3*k)]*k)

    min_w = max(0.2, 4*(x[1]-x[0]))
    max_w = (x[-1]-x[0]) / 3.0
    if np.ndim(w_c) == 0: w_c = np.array([w_c])
    w_c = np.clip(w_c, min_w, max_w)

    order = np.argsort(prominences)[::-1] if len(peaks) else np.arange(len(peaks))
    peaks = peaks[order][:k]
    w_c = w_c[order][:k]

    sort_lr = np.argsort(peaks)
    return peaks[sort_lr], w_c[sort_lr]

def fit_k(x, y, k):
    peaks, w_c = seed_peaks(x, y, k)
    p0, lo, hi = [], [], []
    y_max = max(np.max(y), 1e-6)
    for j, pk in enumerate(peaks):
        mu0 = float(x[pk])
        amp0 = float(max(y[pk], 1e-6))
        sigma0 = float(max(w_c[j] / (2*np.sqrt(2*np.log(2))), (x[1]-x[0])*2))
        p0 += [amp0, mu0, sigma0]
        lo += [0.0, mu0 - 3*sigma0, (x[1]-x[0])*1e-3]
        hi += [y_max*5 + 1e-6, mu0 + 3*sigma0, (x[-1]-x[0])]
    try:
        popt, _ = curve_fit(gaussian_sum, x, y, p0=p0, bounds=(lo, hi), maxfev=15000)
        return popt
    except:
        return None

def BIC(n_params, rss, n):
    return np.log(n)*n_params + n*np.log(rss/n + 1e-12)

def fit_best_K(x, y, K=(1, 5)):
    best = None
    for k in range(K[0], K[1]+1):
        try:
            popt = fit_k(x, y, k)
            if popt is None:
                continue
            yhat = gaussian_sum(x, *popt)
            rss = float(np.sum((y - yhat)**2))
            bic = float(BIC(3*k, rss, len(x)))
            if best is None or bic < best["bic"]:
                best = {"k": k, "popt": popt, "bic": bic, "rss": rss}
        except Exception:
            pass
    return best

def extract_enhanced_features(row_vals, X_axis):
    """Enhanced feature extraction with more curve statistics"""
    y0 = preprocess_curve(X_axis, np.asarray(row_vals, float))
    x, y = decimate(X_axis, y0, step=DECIMATE_STEP)
    best = fit_best_K(x, y, K=K_RANGE)

    if best is None:
        return {f"f{i}": 0.0 for i in range(30)}  # Return zeros if fitting fails

    popt = best["popt"]

    # Original peak features (top 3 peaks)
    peaks = [{"amp": float(popt[i]), "mu": float(popt[i+1]), "sigma": abs(float(popt[i+2]))}
             for i in range(0, len(popt), 3)]
    peaks.sort(key=lambda d: d["amp"], reverse=True)

    feats = {}
    for i in range(3):  # Top 3 peaks
        if i < len(peaks):
            feats[f"peak{i+1}_mu"] = peaks[i]["mu"]
            feats[f"peak{i+1}_amp"] = peaks[i]["amp"]
            feats[f"peak{i+1}_sigma"] = peaks[i]["sigma"]
        else:
            feats[f"peak{i+1}_mu"] = 0.0
            feats[f"peak{i+1}_amp"] = 0.0
            feats[f"peak{i+1}_sigma"] = 0.0

    # Peak ratios and differences
    feats["delta_mu_12"] = feats["peak1_mu"] - feats["peak2_mu"]
    feats["delta_mu_23"] = feats["peak2_mu"] - feats["peak3_mu"]
    feats["amp_ratio_12"] = (feats["peak1_amp"]+1e-9)/(feats["peak2_amp"]+1e-9)
    feats["amp_ratio_23"] = (feats["peak2_amp"]+1e-9)/(feats["peak3_amp"]+1e-9)
    feats["sigma_ratio_12"] = (feats["peak1_sigma"]+1e-9)/(feats["peak2_sigma"]+1e-9)

    # Total features
    feats["total_amp"] = sum(p["amp"] for p in peaks)
    feats["total_area"] = sum(p["amp"] * p["sigma"] * np.sqrt(2*np.pi) for p in peaks)

    # Basic stats
    feats["mean"] = float(np.mean(y0))
    feats["std"] = float(np.std(y0))
    feats["max"] = float(np.max(y0))
    feats["min"] = float(np.min(y0))
    feats["range"] = feats["max"] - feats["min"]
    feats["auc"] = float(np.trapz(y0))

    return feats

# ============== Main ==============

print("Loading data...")
df = pd.read_csv(CSV_PATH)
X_raw = df.drop(columns=[SPECIES_COL])
y = df[SPECIES_COL].astype(str)

# Filter species with <2 samples
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
mask = y.isin(valid_classes)
X_raw = X_raw[mask].reset_index(drop=True)
y = y[mask].to_numpy()

print(f"Data shape: {X_raw.shape}")
print(f"Classes: {len(np.unique(y))}")

# Extract features
print("Extracting Gaussian features...")
X_axis = X_raw.columns.astype(float).values
feat_list = []
for i in range(len(X_raw)):
    if i % 100 == 0:
        print(f"  {i}/{len(X_raw)}")
    feats = extract_enhanced_features(X_raw.iloc[i].values, X_axis)
    feat_list.append(feats)

feat_df = pd.DataFrame(feat_list).fillna(0.0)
X = feat_df.to_numpy(float)

print(f"Feature matrix: {X.shape}")

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Best parameters (using ExtraTreesClassifier as it's commonly used)
best_params = {
    'n_estimators': 700,
    'max_depth': 20,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

print("\n" + "="*70)
print(f"CROSS-VALIDATION: Gaussian Model (10-fold stratified, seed={RANDOM_STATE})")
print("="*70)
print(f"Best params: {best_params}")

# Create results directory
results_dir = Path("./cv_results")
results_dir.mkdir(exist_ok=True)
print(f"Saving fold models to: {results_dir}/")

# 10-fold stratified cross-validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_accuracies = []
fold_f1_scores = []
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train on 90%
    model = ExtraTreesClassifier(**best_params)
    model.fit(X_train, y_train)

    # Test on 10%
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    fold_accuracies.append(acc)
    fold_f1_scores.append(f1)

    # Save fold model with accuracy in filename
    model_filename = f"fold_{fold_idx:02d}_acc_{acc:.4f}_f1_{f1:.4f}.pkl"
    joblib.dump(model, results_dir / model_filename)

    fold_results.append({
        "fold": fold_idx,
        "accuracy": float(acc),
        "f1": float(f1),
        "test_size": len(y_test),
        "model_file": model_filename
    })

    print(f"  Fold {fold_idx:2d}/10 | Accuracy: {acc:.4f} | F1: {f1:.4f} | Test size: {len(y_test)} | Saved: {model_filename}")

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Mean F1:       {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
print(f"Min Accuracy:  {np.min(fold_accuracies):.4f}")
print(f"Max Accuracy:  {np.max(fold_accuracies):.4f}")
print(f"\nTraining data: 90% ({len(X) * 9 // 10} samples per fold)")
print(f"Test data:     10% ({len(X) // 10} samples per fold)")
print("\n[Note] optimize_model.py trains on 100% data, overfitting is expected.")
print("="*70)

# Save summary
summary = {
    "model": "Gaussian",
    "seed": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "best_params": best_params,
    "mean_accuracy": float(np.mean(fold_accuracies)),
    "std_accuracy": float(np.std(fold_accuracies)),
    "mean_f1": float(np.mean(fold_f1_scores)),
    "std_f1": float(np.std(fold_f1_scores)),
    "min_accuracy": float(np.min(fold_accuracies)),
    "max_accuracy": float(np.max(fold_accuracies)),
    "fold_results": fold_results
}

summary_file = results_dir / "cv_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[OK] Summary saved to {summary_file}")
print(f"[OK] All fold models saved to {results_dir}/")
