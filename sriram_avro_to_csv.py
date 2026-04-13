"""
Task 1 (Sriram only): Convert all Sriram avro files to CSV using the original conversion logic.
Each avro produces one folder with 7 CSVs (accelerometer, eda, temperature, tags, bvp, systolic_peaks, steps).
Each CSV has unix_timestamp (or sensor timestamp), Date/Time column (m/d/yyyy h:mm:ss Eastern), then signal column(s).
"""
import os
import glob
import csv

import pandas as pd

# Use existing avro paths and extraction logic
from tools.avro_to_csv_example_script import (
    SRIRAM_AVRO_DIR,
    DATA_BASE,
    read_avro,
    extract_sensor_data,
)

OUTPUT_ROOT = os.path.join(DATA_BASE, "Sriram_CSV")


def _ts_to_date_time(ts, divisor=1e6):
    """Convert timestamp to Eastern date/time string m/d/yyyy h:mm:ss. ts in micros (divisor 1e6) or nanos (1e9)."""
    sec = ts / divisor
    return pd.to_datetime(sec, unit="s", utc=True).tz_convert("US/Eastern").strftime("%m/%d/%Y %H:%M:%S")


def write_sensor_csv(headers, rows, sheet_name, out_path, timestamp_divisor=1e6):
    """Write one CSV with Date/Time as second column (inserted after first column)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_headers = [headers[0], "Date/Time"] + list(headers[1:])
    new_rows = []
    for row in rows:
        ts = row[0]
        try:
            dt_str = _ts_to_date_time(ts, timestamp_divisor)
        except Exception:
            dt_str = ""
        new_rows.append([ts, dt_str] + list(row[1:]))
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)
        writer.writerows(new_rows)


def main():
    if not os.path.isdir(SRIRAM_AVRO_DIR):
        print(f"Sriram avro folder not found: {SRIRAM_AVRO_DIR}")
        return
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for avro_path in sorted(glob.glob(os.path.join(SRIRAM_AVRO_DIR, "*.avro"))):
        stem = os.path.splitext(os.path.basename(avro_path))[0]
        out_dir = os.path.join(OUTPUT_ROOT, stem)
        print(f"Converting {os.path.basename(avro_path)} -> {out_dir}/")
        data = read_avro(avro_path)
        sheets = extract_sensor_data(data)
        for sheet_name, (headers, rows) in sheets.items():
            # systolic_peaks uses nanoseconds; tags and rest use microseconds
            divisor = 1e9 if sheet_name == "systolic_peaks" else 1e6
            out_path = os.path.join(out_dir, f"{sheet_name}.csv")
            write_sensor_csv(headers, rows, sheet_name, out_path, timestamp_divisor=divisor)
            print(f"  {sheet_name}.csv ({len(rows)} rows)")
    print(f"Done. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
