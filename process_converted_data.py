"""
Process 'Converted Data' folder: add Date/Time column from unix_timestamp, then filter by
time ranges into passive and active outputs.

Time ranges (1/27/2024):
  Passive: 1:19:57 PM - 1:27:30 PM
  Active:  2:18:32 PM - 2:24:59 PM
"""

import os
import re
import glob
import pandas as pd
from openpyxl import load_workbook

# Paths (relative to script directory)
DATA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
CONVERTED_DATA_DIR = os.path.join(DATA_BASE, "Converted Data")
OUTPUT_BASE = os.path.join(DATA_BASE, "Converted_Data_Output")
PASSIVE_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "passive")
ACTIVE_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "active")

# Time ranges: 1/27/2024
PASSIVE_START = "1/27/2024 1:19:57PM"
PASSIVE_END = "1/27/2024 1:27:30PM"
ACTIVE_START = "1/27/2024 2:18:32PM"
ACTIVE_END = "1/27/2024 2:24:59PM"

# Date formula for column B (unix micros -> Excel date). Format: m/d/yyyy h:mm:ss
DATE_TIME_FORMULA = "=(A{row}/86400/1000000)+25569+(-4/24)"
DATE_TIME_FORMAT = "m/d/yyyy h:mm:ss"

SENSOR_PATTERNS = ("eda", "bvp", "temperature")


def _parse_time_utc(s):
    """Parse time string as US/Eastern, return timezone-aware UTC for comparison with unix."""
    from datetime import datetime
    s = s.strip()
    for fmt in ("%m/%d/%Y %I:%M:%S%p", "%m/%d/%Y %I:%M:%S%p".lower()):
        try:
            naive = datetime.strptime(s, fmt)
            ts = pd.Timestamp(naive).tz_localize("US/Eastern")
            return ts.tz_convert("UTC")
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time: {s!r}")


def _glob_sensor_csvs(sensor_name):
    """Return paths of CSVs in Converted Data that match this sensor (e.g. eda (4).csv, temperature (20).csv)."""
    # Match "eda (2).csv", "temperature (20).csv", etc.
    all_csvs = glob.glob(os.path.join(CONVERTED_DATA_DIR, "*.csv"))
    paths = [
        p for p in all_csvs
        if re.match(rf"^{re.escape(sensor_name)}\s*\(\d+\)\.csv$", os.path.basename(p), re.IGNORECASE)
        or os.path.basename(p).lower() == f"{sensor_name}.csv"
    ]
    return sorted(paths)


def _load_and_concat_csvs(paths):
    """Load all CSVs and concatenate. Assumes unix_timestamp and one value column."""
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if "unix_timestamp" in df.columns and not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"Skip {p}: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["unix_timestamp"]).sort_values("unix_timestamp")


def run():
    if not os.path.isdir(CONVERTED_DATA_DIR):
        print(f"Converted Data folder not found: {CONVERTED_DATA_DIR}")
        return

    time_ranges = [
        ("passive", _parse_time_utc(PASSIVE_START), _parse_time_utc(PASSIVE_END), PASSIVE_OUTPUT_DIR),
        ("active", _parse_time_utc(ACTIVE_START), _parse_time_utc(ACTIVE_END), ACTIVE_OUTPUT_DIR),
    ]

    for sensor in SENSOR_PATTERNS:
        paths = _glob_sensor_csvs(sensor)
        if not paths:
            print(f"No CSVs found for {sensor}, skipping.")
            continue

        df_all = _load_and_concat_csvs(paths)
        if df_all is None or df_all.empty:
            print(f"No data for {sensor}, skipping.")
            continue

        # Add UTC datetime for filtering
        df_all["_dt"] = pd.to_datetime(df_all["unix_timestamp"] / 1e6, unit="s", errors="coerce", utc=True)
        df_all = df_all.dropna(subset=["_dt"])

        for mode, t_start, t_end, out_dir in time_ranges:
            mask = (df_all["_dt"] >= t_start) & (df_all["_dt"] <= t_end)
            subset = df_all.loc[mask].drop(columns=["_dt"]).copy()
            if subset.empty:
                print(f"{sensor} {mode}: no rows in range, skipping.")
                continue

            # Column order: unix_timestamp, Date/Time (computed for pandas), value column(s)
            dt_utc = pd.to_datetime(subset["unix_timestamp"] / 1e6, unit="s", utc=True)
            subset["Date/Time"] = dt_utc.dt.tz_convert("US/Eastern").dt.strftime("%m/%d/%Y %H:%M:%S")
            value_cols = [c for c in subset.columns if c not in ("unix_timestamp", "Date/Time")]
            subset = subset[["unix_timestamp", "Date/Time"] + value_cols]

            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{mode}_{sensor}.xlsx")
            subset.to_excel(out_path, index=False)

            # Replace column B with Excel formula and set number format
            wb = load_workbook(out_path)
            ws = wb.active
            for row in range(2, ws.max_row + 1):
                ws.cell(row=row, column=2).value = DATE_TIME_FORMULA.format(row=row)
                ws.cell(row=row, column=2).number_format = DATE_TIME_FORMAT
            wb.save(out_path)
            print(f"Wrote {len(subset)} rows -> {out_path}")


if __name__ == "__main__":
    run()
