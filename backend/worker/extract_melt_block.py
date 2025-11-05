from __future__ import annotations
import argparse, csv, io, re, sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

START_DEFAULT = "Start Worksheet - Analysis - Melt DNAID-Elasmo Data"
END_DEFAULT   = "End Worksheet"



# RUN COMMAND: python extract_melt_block.py "test_output_file.csv" "test_transposed_demo.csv"




# ---------- helpers to clean "temperature" + numeric IDs ----------

_INT_RE = re.compile(r"^\s*[+-]?\d+\s*$")
def _is_int_str(x) -> bool:
    if x is None:
        return False
    s = str(x).strip()
    return bool(_INT_RE.match(s))

def _norm_text(x: object) -> str:
    s = str(x if x is not None else "").strip()
    # normalize common encoding artifacts
    s = s.replace("Â", "").replace("℃", "°C").replace("° C", "°C")
    return s

_TEMP_HEADER_RE = re.compile(r"^\s*temperature\b", re.IGNORECASE)

def _is_temperature_header(x) -> bool:
    s = _norm_text(x)
    return bool(_TEMP_HEADER_RE.search(s))

def clean_temperature_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Remove any cell that looks like a Temperature header (e.g., 'Temperature', 'Temperature (°C)', 'Temperature (Â°C)').
    - Preserve the first row (it contains temperature *values* like 60, 60.01).
    - Remove the ID column on the far left if it looks like IDs (majority plain integers).
      Otherwise, blank only the integer cells in that first column.
    """
    out = df.copy()

    # 1) Clear any 'Temperature...' cell anywhere
    out = out.map(lambda v: "" if _is_temperature_header(v) else v)

    # 2) Detect & remove the ID column (first column) rather than touching the first row.
    if out.shape[1] > 0:
        col0 = out.iloc[:, 0].astype(str).str.strip()
        non_empty = col0 != ""
        is_int = col0.apply(_is_int_str)

        # Ratio of plain integers among non-empty values in column 0
        non_empty_count = int(non_empty.sum())
        ratio = (is_int & non_empty).sum() / non_empty_count if non_empty_count else 0.0

        # Heuristic: if the first column is mostly integers, treat it as the ID column and drop it.
        # (Keeps decimals like 60.01 in the first *row*, which we are no longer modifying.)
        if ratio >= 0.6:
            out = out.iloc[:, 1:].reset_index(drop=True)
        else:
            # Not clearly an ID column—just blank integer-only cells in that first column.
            out.iloc[:, 0] = out.iloc[:, 0].mask(is_int, "")

    return out

# ---------- Text/CSV mode (robust to messy CSVs) ----------

def read_text_file_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return f.read().splitlines()

def find_all_marker_lines(lines: List[str], marker: str) -> List[int]:
    return [i for i, line in enumerate(lines) if marker in line.strip()]

def pair_runs_text(lines: List[str], start_marker: str, end_marker: str) -> List[Tuple[int,int]]:
    starts = find_all_marker_lines(lines, start_marker)
    ends   = sorted(find_all_marker_lines(lines, end_marker))
    pairs = []
    for s in starts:
        e = next((x for x in ends if x > s), None)
        if e is not None:
            pairs.append((s, e))
    return pairs

def sniff_delimiter_from_block(lines: List[str]) -> str:
    sample = "\n".join(lines[:200])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        candidates = [",", "\t", ";", "|"]
        counts = {c: sample.count(c) for c in candidates}
        return max(counts, key=counts.get) if any(counts.values()) else ","

def parse_rows(lines: List[str], delimiter: str) -> List[List[str]]:
    text = "\n".join(lines)
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = [list(r) for r in reader]
        if rows and all(len(r) <= 1 for r in rows):
            raise ValueError("Only one field detected; trying regex split.")
        return rows
    except Exception:
        splitter = re.compile(r"[,\t;|]")
        return [splitter.split(line) for line in lines]

def block_to_dataframe(lines: List[str]) -> pd.DataFrame:
    delim = sniff_delimiter_from_block(lines)
    rows  = parse_rows(lines, delim)
    return pd.DataFrame.from_records(rows)

def process_text_file_multi(input_path: Path,
                            start_marker: str,
                            end_marker: str) -> List[pd.DataFrame]:
    lines = read_text_file_lines(input_path)
    pairs = pair_runs_text(lines, start_marker, end_marker)
    if not pairs:
        raise ValueError("No Start/End pairs found in text file.")

    outputs: List[pd.DataFrame] = []
    for s, e in pairs:
        block = lines[s+1:e]
        if not block:
            continue
        # Drop the first line (Temperature header + sample numbers before transpose)
        block = block[1:] if len(block) >= 1 else block
        df = block_to_dataframe(block).transpose().reset_index(drop=True)
        df = clean_temperature_and_ids(df)   # <-- updated cleaner
        outputs.append(df)

    if not outputs:
        raise ValueError("Runs were found, but all blocks were empty after dropping the first line.")
    return outputs

# ---------- Excel mode ----------

def find_all_marker_rows_df(df: pd.DataFrame, marker: str) -> List[int]:
    mask = df.map(lambda x: (str(x).strip() if x is not None else ""))
    hit = (mask.map(lambda s: marker in s)).values
    loc = np.argwhere(hit)
    return sorted(set(int(r) for r, _ in loc)) if loc.size else []

def pair_runs_excel(df: pd.DataFrame, start_marker: str, end_marker: str) -> List[Tuple[int,int]]:
    starts = find_all_marker_rows_df(df, start_marker)
    ends   = find_all_marker_rows_df(df, end_marker)
    pairs = []
    for s in starts:
        e = next((x for x in ends if x > s), None)
        if e is not None:
            pairs.append((s, e))
    return pairs

def process_excel_file_multi(input_path: Path,
                             start_marker: str,
                             end_marker: str) -> List[pd.DataFrame]:
    sheets = pd.read_excel(input_path, sheet_name=None, header=None, engine=None)
    outputs: List[pd.DataFrame] = []

    for _, df in sheets.items():
        pairs = pair_runs_excel(df, start_marker, end_marker)
        for s, e in pairs:
            block = df.iloc[s+1:e, :].copy()
            if len(block) == 0:
                continue
            block = block.iloc[1:, :] if len(block) >= 1 else block
            out = block.transpose().reset_index(drop=True)
            out = clean_temperature_and_ids(out)  # <-- updated cleaner
            outputs.append(out)

    if not outputs:
        raise ValueError("No Start/End pairs found in any Excel sheet.")
    return outputs

# ---------- Orchestration & CLI ----------

def write_outputs(dfs: List[pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(dfs) == 1:
        df = dfs[0].copy()
        # Set the first row as column headers (temperature values)
        headers = [str(h).strip() for h in df.iloc[0]]
        # Remove empty headers and corresponding columns
        valid_headers = [(i, h) for i, h in enumerate(headers) if h and h != '']
        if valid_headers:
            valid_indices = [i for i, _ in valid_headers]
            df = df.iloc[1:, valid_indices].reset_index(drop=True)
            df.columns = [h for _, h in valid_headers]
        else:
            df = df.iloc[1:].reset_index(drop=True)
        df.to_csv(output_path, index=False, header=True)
    else:
        stem = output_path.stem
        suffix = output_path.suffix or ".csv"
        folder = output_path.parent
        for i, df in enumerate(dfs, start=1):
            df_copy = df.copy()
            # Set the first row as column headers (temperature values)
            headers = [str(h).strip() for h in df_copy.iloc[0]]
            # Remove empty headers and corresponding columns
            valid_headers = [(j, h) for j, h in enumerate(headers) if h and h != '']
            if valid_headers:
                valid_indices = [j for j, _ in valid_headers]
                df_copy = df_copy.iloc[1:, valid_indices].reset_index(drop=True)
                df_copy.columns = [h for _, h in valid_headers]
            else:
                df_copy = df_copy.iloc[1:].reset_index(drop=True)
            out = folder / f"{stem}_run{i}{suffix}"
            df_copy.to_csv(out, index=False, header=True)

def process_file(input_path: str,
                 output_path: str,
                 start_marker: str = START_DEFAULT,
                 end_marker: str = END_DEFAULT) -> None:
    in_path  = Path(input_path)
    out_path = Path(output_path)
    suf = in_path.suffix.lower()

    if suf in (".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"):
        dfs = process_excel_file_multi(in_path, start_marker, end_marker)
    else:
        dfs = process_text_file_multi(in_path, start_marker, end_marker)

    write_outputs(dfs, out_path)

def main():
    p = argparse.ArgumentParser(description="Extract runs, drop first line, remove Temperature header and ID column, transpose, save CSV(s).")
    p.add_argument("input_path")
    p.add_argument("output_path")
    p.add_argument("--start", default=START_DEFAULT)
    p.add_argument("--end",   default=END_DEFAULT)
    args = p.parse_args()
    process_file(args.input_path, args.output_path, args.start, args.end)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
