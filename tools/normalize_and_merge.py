import os
import sys
import pandas as pd
import numpy as np

# Use 1 Hz since your timestamps are per-second
TARGET_RATE_HZ = 1.0
TARGET_RULE = f"{int(1000 / TARGET_RATE_HZ)}ms"   # 1000 ms bins

INPUT_ROOT_DEFAULT  = r"C:\Users\mtomori3\Downloads\Filter files"
OUTPUT_ROOT_DEFAULT = r"C:\Users\mtomori3\Downloads\Merged same row"

META_COLS = {"experiment_date", "experiment_time", "unix_us",
             "unix_timestamp", "Date/Time", "Column1", "timestamp_local"}

TS_CANDIDATES = ["Column1", "Date/Time", "timestamp_local", "datetime", "date_time", "timestamp", "time"]

def minmax_0_1(col: pd.Series) -> pd.Series:
    vmin = col.min()
    vmax = col.max()
    if pd.isna(vmin) or pd.isna(vmax):
        return col
    rng = vmax - vmin
    if rng == 0:
        return pd.Series(np.zeros(len(col)), index=col.index)
    return (col - vmin) / rng

def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts_col = None
    for cand in TS_CANDIDATES:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column found. Expected one of: {TS_CANDIDATES}")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)
    df.index.name = "timestamp_local"
    return df

def value_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]

def load_sensor_csv(path: str, suffix: str, prefer_value_cols=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = to_datetime_index(df)

    keep = value_columns(df)

    # Use preferred sensor column name if present (e.g., "temperature", "eda", "bvp")
    if prefer_value_cols:
        prefer_present = [c for c in prefer_value_cols if c in df.columns]
        if prefer_present:
            keep = prefer_present

    if not keep:
        raise ValueError(f"No value columns found in {path}. Columns: {list(df.columns)}")

    # force numeric
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # resample to 1 Hz (or whatever TARGET_RATE_HZ is)
    df_resamp = df[keep].resample(TARGET_RULE).mean()
    df_resamp = df_resamp.add_suffix(f"_{suffix}")
    return df_resamp

def combine_three(eda_path: str, bvp_path: str, temp_path: str) -> pd.DataFrame:
    eda = load_sensor_csv(eda_path,  "eda",  prefer_value_cols=["eda", "EDA"])
    bvp = load_sensor_csv(bvp_path,  "bvp",  prefer_value_cols=["bvp", "BVP"])
    tmp = load_sensor_csv(temp_path, "temp", prefer_value_cols=["temperature", "temp", "TEMP"])

    combined = eda.join(bvp, how="outer").join(tmp, how="outer")
    combined = combined.sort_index().interpolate(method="time").ffill().bfill()
    return combined

def normalize_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if np.issubdtype(out[c].dtype, np.number):
            out[c] = minmax_0_1(out[c])
    return out

def main():
    if len(sys.argv) >= 3:
        input_root = sys.argv[1]
        output_root = sys.argv[2]
    else:
        input_root = INPUT_ROOT_DEFAULT
        output_root = OUTPUT_ROOT_DEFAULT

    if not os.path.isdir(input_root):
        print(f"Input root not found: {input_root}")
        sys.exit(1)

    os.makedirs(output_root, exist_ok=True)

    # participant folders under input_root
    for pid in sorted(os.listdir(input_root)):
        pid_in = os.path.join(input_root, pid)
        if not os.path.isdir(pid_in):
            continue

        eda_path  = os.path.join(pid_in, "eda.csv")
        bvp_path  = os.path.join(pid_in, "bvp.csv")
        temp_path = os.path.join(pid_in, "temperature.csv")  # change if your file is temp.csv

        if not (os.path.exists(eda_path) and os.path.exists(bvp_path) and os.path.exists(temp_path)):
            print(f"Skipping {pid_in} (missing eda/bvp/temperature)")
            continue

        try:
            combined = combine_three(eda_path, bvp_path, temp_path)
            combined_norm = normalize_all_columns(combined)

            pid_out = os.path.join(output_root, pid)
            os.makedirs(pid_out, exist_ok=True)

            out_csv = os.path.join(pid_out, "combined_same_row.csv")
            combined_norm.reset_index().to_csv(out_csv, index=False)
            print(f"Saved: {out_csv} (rows={len(combined_norm)})")

        except Exception as e:
            print(f"Error combining {pid_in}: {e}")

if __name__ == "__main__":
    main()