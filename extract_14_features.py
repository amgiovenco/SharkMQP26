
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

## Run Command: python extract_14_features.py --input cleaned_data_but_in_rows.csv --output species_14_features.csv

def _interp_row(vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Row-wise linear interpolation with edge fill."""
    y = vals.astype(float).copy()
    mask = np.isfinite(y)
    if mask.sum() == 0:
        return np.zeros_like(y)
    # interpolate internal gaps
    y[~mask] = np.interp(x[~mask], x[mask], y[mask])
    # forward/backward fill edges
    first, last = np.argmax(mask), len(y) - np.argmax(mask[::-1]) - 1
    y[:first] = y[first]
    y[last+1:] = y[last]
    return y

def _fwhm(t, yb):
    if yb.size == 0:
        return np.nan
    peak = yb.max()
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    half = 0.5 * peak
    above = yb >= half
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    return t[idx[-1]] - t[idx[0]]

def _cross_time(t, y, level, start, end):
    """Continuous-time crossing via linear interp between nearest points in [start,end]."""
    if end <= start:
        return np.nan
    seg_t = t[start:end+1]
    seg_y = y[start:end+1]
    # Determine trend for choosing inequality direction
    if seg_y[0] <= seg_y[-1]:
        cond = seg_y >= level
    else:
        cond = seg_y <= level
    idx = np.argmax(cond)
    if not cond[idx]:
        return np.nan
    if idx == 0:
        return seg_t[0]
    t0, t1 = seg_t[idx-1], seg_t[idx]
    y0, y1 = seg_y[idx-1], seg_y[idx]
    if y1 == y0:
        return t1
    return t0 + (level - y0) * (t1 - t0) / (y1 - y0)

def _curve_features(y: np.ndarray, t: np.ndarray) -> dict:
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    # stats on original (un-baselined) signal
    orig_mean = float(np.nanmean(y))
    orig_std  = float(np.nanstd(y, ddof=0))
    orig_max  = float(np.nanmax(y))
    orig_min  = float(np.nanmin(y))

    # Clean/infill
    finite_mask = np.isfinite(y)
    if not finite_mask.all():
        y = _interp_row(y, t)

    # Robust baseline: median of first 5% of points
    n = len(y)
    k = max(1, int(round(0.05 * n)))
    base = float(np.median(y[:k]))
    yb = y - base
    yb = np.clip(yb, 0.0, None)

    # Peak features
    i_max = int(np.argmax(yb))
    ymax = float(yb[i_max])
    tmax = float(t[i_max])

    # Area features
    auc = float(np.trapz(yb, t))
    centroid = float(np.trapz(yb * t, t) / auc) if auc > 0 else np.nan

    # FWHM
    fwhm = float(_fwhm(t, yb))

    # Rise (10% -> 90%) up to peak
    if ymax > 0 and i_max > 0:
        t10 = _cross_time(t, yb, 0.1 * ymax, 0, i_max)
        t90 = _cross_time(t, yb, 0.9 * ymax, 0, i_max)
        rise = (t90 - t10) if (np.isfinite(t10) and np.isfinite(t90)) else np.nan
    else:
        rise = np.nan

    # Decay (90% -> 10%) after peak
    if ymax > 0 and i_max < len(yb) - 1:
        t90d = _cross_time(t, yb, 0.9 * ymax, i_max, len(yb) - 1)
        t10d = _cross_time(t, yb, 0.1 * ymax, i_max, len(yb) - 1)
        decay = (t10d - t90d) if (np.isfinite(t90d) and np.isfinite(t10d)) else np.nan
    else:
        decay = np.nan

    # AUC split at peak
    auc_left = float(np.trapz(yb[:i_max+1], t[:i_max+1])) if i_max > 0 else 0.0
    auc_right = float(np.trapz(yb[i_max:], t[i_max:])) if i_max < len(yb)-1 else 0.0
    asym = float((auc_right - auc_left) / auc) if auc > 0 else np.nan

    return {
        "ymax": ymax,
        "tmax": tmax,
        "auc": auc,
        "centroid": centroid,
        "fwhm": fwhm,
        "rise": float(rise) if np.isfinite(rise) else np.nan,
        "decay": float(decay) if np.isfinite(decay) else np.nan,
        "auc_left": auc_left,
        "auc_right": auc_right,
        "asym": asym,
        "mean": orig_mean,
        "std": orig_std,
        "max": orig_max,
        "min": orig_min,
    }

def main(input_csv: str, output_csv: str, species_col: str = "Species"):
    df = pd.read_csv(input_csv)

    if species_col not in df.columns:
        raise ValueError(f"Expected a '{species_col}' column with species names. Found: {list(df.columns)[:6]} ...")

    # Identify sequential columns and axis from headers
    feature_cols = [c for c in df.columns if c != species_col]
    t_vals, kept_cols = [], []
    for c in feature_cols:
        try:
            t = float(c)
            t_vals.append(t)
            kept_cols.append(c)
        except Exception:
            # drop non-numeric headers
            pass

    if not kept_cols:
        raise ValueError("No numeric sequential columns found. Column headers should be numeric (e.g., temperatures).")

    t = np.array(t_vals, dtype=float)
    order = np.argsort(t)
    t = t[order]
    kept_cols = [kept_cols[i] for i in order]

    # Coerce to numeric and clean
    mat = df[kept_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    # Row-wise interpolation & edge fill
    for i in range(mat.shape[0]):
        row = mat[i]
        mask = np.isfinite(row)
        if not mask.all():
            row = np.interp(t, t[mask], row[mask])
            first, last = np.argmax(mask), len(row) - np.argmax(mask[::-1]) - 1
            row[:first] = row[first]
            row[last+1:] = row[last]
            mat[i] = row

    # Compute features per row
    out_rows = []
    for i in range(mat.shape[0]):
        y = mat[i]
        feats = _curve_features(y, t)
        out_rows.append(feats)

    out_df = pd.DataFrame(out_rows)
    out_df.insert(0, "Species", df[species_col].values)

    # Column order: Species + 14 features
    col_order = ["Species",
                 "ymax","tmax",
                 "auc","centroid","fwhm","rise","decay","auc_left","auc_right","asym",
                 "mean","std","max","min"]
    out_df = out_df[col_order]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df)} rows × {len(out_df.columns)} cols to {output_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract 14 curve features from sequential data.")
    ap.add_argument("--input", required=True, help="Path to input CSV (rows = sequences, one column 'Species' + numeric headers for axis).")
    ap.add_argument("--output", required=True, help="Path to output CSV (Species + 14 features).")
    ap.add_argument("--species_col", default="Species", help="Name of the species label column (default: 'Species').")
    args = ap.parse_args()
    main(args.input, args.output, args.species_col)
