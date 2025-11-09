"""rule-based model prediction."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def engineer_features_rulebased(X_raw: pd.DataFrame) -> pd.DataFrame:
    """engineer features for rule-based model."""
    features_list = []
    temps = X_raw.columns.astype(float).values

    for idx, row in X_raw.iterrows():
        y = np.asarray(row.values, float)
        t = temps

        baseline = np.median(y[:max(1, int(0.05 * len(y)))])
        yb = np.clip(y - baseline, 0.0, None)

        idx_max = int(np.argmax(yb))
        ymax = float(yb[idx_max])
        tmax = float(t[idx_max])

        auc = float(np.trapezoid(yb, t))
        centroid = float(np.trapezoid(yb * t, t) / (auc + 1e-12))

        half = 0.5 * ymax
        above = np.where(yb >= half)[0]
        fwhm = float(t[above[-1]] - t[above[0]]) if above.size > 0 else 0.0

        def cross(level):
            idx_arr = np.where(yb >= level)[0]
            return (int(idx_arr[0]) if idx_arr.size else idx_max), (int(idx_arr[-1]) if idx_arr.size else idx_max)

        lo, hi = 0.1 * ymax, 0.9 * ymax
        lo_i1, hi_i1 = cross(lo)[0], cross(hi)[0]
        lo_i2, hi_i2 = cross(lo)[1], cross(hi)[1]
        rise_time = float(t[max(hi_i1, lo_i1)] - t[min(hi_i1, lo_i1)]) if ymax > 0 else 0.0
        decay_time = float(t[max(lo_i2, hi_i2)] - t[min(lo_i2, hi_i2)]) if ymax > 0 else 0.0

        auc_left = float(np.trapezoid(yb[:idx_max + 1], t[:idx_max + 1]))
        auc_right = float(np.trapezoid(yb[idx_max:], t[idx_max:]))
        asymmetry = float((auc_right - auc_left) / (auc + 1e-12))

        mean_val = float(np.mean(y))
        std_val = float(np.std(y))
        max_val = float(np.max(y))
        min_val = float(np.min(y))

        feat_array = [ymax, tmax, auc, centroid, fwhm, rise_time, decay_time,
                      auc_left, auc_right, asymmetry, mean_val, std_val, max_val, min_val]
        features_list.append(feat_array)

    features = pd.DataFrame(features_list, columns=["ymax", "tmax", "auc", "centroid", "fwhm", "rise", "decay",
                                                      "auc_left", "auc_right", "asym", "mean", "std", "max", "min"])
    return features


def get_rb_predictions(X_raw: pd.DataFrame, models_dir: str = "./models") -> np.ndarray:
    """get rule-based predictions."""
    try:
        model_path = f"{models_dir}/rulebased_9519.pkl"
        if not Path(model_path).exists():
            print("  rule-based...[FAIL] not found")
            return None
        bundle = joblib.load(model_path)

        X_eng = engineer_features_rulebased(X_raw)

        # Extract components from bundle
        if isinstance(bundle, dict) and 'model' in bundle:
            model_obj = bundle['model']
            scaler = bundle.get('scaler', None)
            feature_names = bundle.get('feature_names', None)
        else:
            # Old format - model is not a bundle
            model_obj = bundle
            scaler = None
            feature_names = None

        # Apply scaler if available
        if scaler is not None:
            X_eng_scaled = scaler.transform(X_eng)
        else:
            X_eng_scaled = X_eng

        # Get predictions
        proba = model_obj.predict_proba(X_eng_scaled)

        print(f"  rule-based...[OK] ({proba.shape})")
        return proba
    except Exception as e:
        print(f"  rule-based...[FAIL] error: {e}")
        return None
