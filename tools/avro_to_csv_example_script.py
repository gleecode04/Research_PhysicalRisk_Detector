from avro.datafile import DataFileReader
from avro.io import DatumReader
import json
import csv
import os
import glob
from datetime import datetime

import pandas as pd
from openpyxl import load_workbook

## Define the location of the Avro file and output folder.
# macOS example:
# avro_file_path = "/Users/timmytommy/Data/Avros/1-1-100_1707311064.avro"
# output_dir = "/Users/timmytommy/Data/Output/"
#
# Windows example:
# avro_file_path = "C:/Data/Avros/1-1-100_1707311064.avro"
# output_dir = "C:/Data/Output/"

# Base path for Data folder (repo root / Data; this file lives in tools/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_BASE = os.path.join(_REPO_ROOT, "Data")
# Avro input folders (each can contain multiple .avro files).
SRIRAM_AVRO_DIR = os.path.join(DATA_BASE, "sriram", "2024-02-06_SriRam")
JOSEPH_AVRO_DIR = os.path.join(DATA_BASE, "2024-02-09_Joseph")
# Result folders: one Excel workbook per avro file
SRIRAM_RESULTS_DIR = os.path.join(DATA_BASE, "sriram_results")
JOSEPH_RESULTS_DIR = os.path.join(DATA_BASE, "joseph_results")
# Merged output by time range (passive / active)
PASSIVE_DIR = os.path.join(DATA_BASE, "passive")
ACTIVE_DIR = os.path.join(DATA_BASE, "active")

# Time ranges for merge (use second sriram passive window; fix typos)
sriram_passive_start = "2/6/2024 12:20:34PM"
sriram_passive_end = "2/6/2024 12:26:36PM"
sriram_active_start = "2/6/2024 12:55:11PM"
sriram_active_end = "2/6/2024 1:01:53PM"
joseph_PASSIVE_start = "2/9/2024 9:33:09AM"
joseph_PASSIVE_end = "2/9/2024 9:37:58AM"
joseph_ACTIVE_start = "2/9/2024 9:53:22am"
joseph_ACTIVE_end = "2/9/2024 9:57:44am"

# Formula for column B: unix timestamp (microseconds) to Excel date/time. Format: m/d/yyyy h:mm:ss
DATE_TIME_FORMULA = "=(A{row}/86400/1000000)+25569+(-4/24)"
DATE_TIME_FORMAT = "m/d/yyyy h:mm:ss"
MERGE_SENSORS = ("temperature", "eda", "bvp")  # only these for final passive/active deliverable


def _parse_time(s):
    """Parse time string like '2/6/2024 12:55:11PM' or '2/9/2024 9:53:22am'."""
    s = s.strip()
    for fmt in ("%m/%d/%Y %I:%M:%S%p", "%m/%d/%Y %I:%M:%S%p".replace("%p", "%p").lower()):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time: {s!r}")


def _parse_time_utc(s):
    """Parse time string as US/Eastern and return as timezone-aware UTC for comparison with unix timestamps."""
    naive = _parse_time(s)
    ts = pd.Timestamp(naive).tz_localize("US/Eastern")
    return ts.tz_convert("UTC")


# ---------- Core logic: read one avro and extract sensor data (unchanged from original) ----------

def read_avro(avro_path):
    """Read one Avro file and return the single-record data dict."""
    with open(avro_path, "rb") as f:
        reader = DataFileReader(f, DatumReader())
        schema = json.loads(reader.meta.get("avro.schema").decode("utf-8"))
        data = next(reader)
    return data


def extract_sensor_data(data):
    """
    Extract sensor data from avro `data` dict. Returns dict: sheet_name -> (headers, rows).
    Core logic is identical to the original script; only output is returned instead of written to CSV.
    """
    avro_version = (
        (data["schemaVersion"]["major"]),
        (data["schemaVersion"]["minor"]),
        (data["schemaVersion"]["patch"]),
    )
    out = {}

    # Accelerometer
    acc = data["rawData"]["accelerometer"]
    timestamp = [
        round(acc["timestampStart"] + i * (1e6 / acc["samplingFrequency"]))
        for i in range(len(acc["x"]))
    ]
    if avro_version < (6, 5, 0):
        delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
        delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
        x_g = [val * delta_physical / delta_digital for val in acc["x"]]
        y_g = [val * delta_physical / delta_digital for val in acc["y"]]
        z_g = [val * delta_physical / delta_digital for val in acc["z"]]
    else:
        conversion_factor = acc["imuParams"]["conversionFactor"]
        x_g = [val * conversion_factor for val in acc["x"]]
        y_g = [val * conversion_factor for val in acc["y"]]
        z_g = [val * conversion_factor for val in acc["z"]]
    out["accelerometer"] = (["unix_timestamp", "x", "y", "z"], [[ts, x, y, z] for ts, x, y, z in zip(timestamp, x_g, y_g, z_g)])

    # Eda
    eda = data["rawData"]["eda"]
    timestamp = [
        round(eda["timestampStart"] + i * (1e6 / eda["samplingFrequency"]))
        for i in range(len(eda["values"]))
    ]
    out["eda"] = (["unix_timestamp", "eda"], [[ts, v] for ts, v in zip(timestamp, eda["values"])])

    # Temperature
    tmp = data["rawData"]["temperature"]
    timestamp = [
        round(tmp["timestampStart"] + i * (1e6 / tmp["samplingFrequency"]))
        for i in range(len(tmp["values"]))
    ]
    out["temperature"] = (["unix_timestamp", "temperature"], [[ts, v] for ts, v in zip(timestamp, tmp["values"])])

    # Tags
    tags = data["rawData"]["tags"]
    out["tags"] = (["tags_timestamp"], [[tag] for tag in tags["tagsTimeMicros"]])

    # BVP
    bvp = data["rawData"]["bvp"]
    timestamp = [
        round(bvp["timestampStart"] + i * (1e6 / bvp["samplingFrequency"]))
        for i in range(len(bvp["values"]))
    ]
    out["bvp"] = (["unix_timestamp", "bvp"], [[ts, v] for ts, v in zip(timestamp, bvp["values"])])

    # Systolic peaks
    sps = data["rawData"]["systolicPeaks"]
    out["systolic_peaks"] = (["systolic_peak_timestamp"], [[sp] for sp in sps["peaksTimeNanos"]])

    # Steps
    steps = data["rawData"]["steps"]
    timestamp = [
        round(steps["timestampStart"] + i * (1e6 / steps["samplingFrequency"]))
        for i in range(len(steps["values"]))
    ]
    out["steps"] = (["unix_timestamp", "steps"], [[ts, step] for ts, step in zip(timestamp, steps["values"])])

    return out


def write_excel_with_column_b(sheets_dict, output_xlsx_path):
    """
    Write one Excel workbook with one sheet per sensor. In each sheet:
    insert column B with formula =(Arow/86400/1000000)+25569+(-4/24) and format m/d/yyyy h:mm:ss;
    original column B becomes C, etc.
    """
    os.makedirs(os.path.dirname(output_xlsx_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
        for sheet_name, (headers, rows) in sheets_dict.items():
            df = pd.DataFrame(rows, columns=headers)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Insert column B (formula + format) in each sheet
    wb = load_workbook(output_xlsx_path)
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        # Insert column B (index 2)
        ws.insert_cols(2)
        ws.cell(row=1, column=2, value="Date/Time")
        nrows = ws.max_row
        for r in range(2, nrows + 1):
            ws.cell(row=r, column=2, value=DATE_TIME_FORMULA.format(row=r))
            cell_b = ws.cell(row=r, column=2)
            cell_b.number_format = DATE_TIME_FORMAT
    wb.save(output_xlsx_path)


def process_one_avro(avro_path, output_xlsx_path):
    """Read one avro, extract sensor data, write one Excel workbook with column B formula and format."""
    data = read_avro(avro_path)
    sheets = extract_sensor_data(data)
    write_excel_with_column_b(sheets, output_xlsx_path)


def run_batch():
    """Process all .avro files in SRIRAM_AVRO_DIR and JOSEPH_AVRO_DIR; write to sriram_results and joseph_results."""
    for avro_dir, results_dir in [
        (SRIRAM_AVRO_DIR, SRIRAM_RESULTS_DIR),
        (JOSEPH_AVRO_DIR, JOSEPH_RESULTS_DIR),
    ]:
        if not os.path.isdir(avro_dir):
            print(f"Skip (folder not found): {avro_dir}")
            continue
        os.makedirs(results_dir, exist_ok=True)
        for avro_path in sorted(glob.glob(os.path.join(avro_dir, "*.avro"))):
            stem = os.path.splitext(os.path.basename(avro_path))[0]
            out_path = os.path.join(results_dir, stem + ".xlsx")
            print(f"Processing: {avro_path} -> {out_path}")
            process_one_avro(avro_path, out_path)


def merge_by_time_ranges():
    """
    Merge Excel files from sriram_results and joseph_results by time ranges.
    Output: passive/ and active/ with files like sriram_passive_eda.xlsx, joseph_active_bvp.xlsx (Temperature, EDA, BVP only).
    """
    time_ranges = [
        ("sriram", "passive", _parse_time_utc(sriram_passive_start), _parse_time_utc(sriram_passive_end)),
        ("sriram", "active", _parse_time_utc(sriram_active_start), _parse_time_utc(sriram_active_end)),
        ("joseph", "passive", _parse_time_utc(joseph_PASSIVE_start), _parse_time_utc(joseph_PASSIVE_end)),
        ("joseph", "active", _parse_time_utc(joseph_ACTIVE_start), _parse_time_utc(joseph_ACTIVE_end)),
    ]
    os.makedirs(PASSIVE_DIR, exist_ok=True)
    os.makedirs(ACTIVE_DIR, exist_ok=True)

    # Accumulate: (person, mode, sensor) -> list of DataFrames to concat
    merged = {}

    for person, results_dir in [("sriram", SRIRAM_RESULTS_DIR), ("joseph", JOSEPH_RESULTS_DIR)]:
        if not os.path.isdir(results_dir):
            continue
        for xlsx_path in glob.glob(os.path.join(results_dir, "*.xlsx")):
            try:
                xl = pd.ExcelFile(xlsx_path)
            except Exception as e:
                print(f"Skip {xlsx_path}: {e}")
                continue
            for sheet in xl.sheet_names:
                if sheet.lower() not in [s.lower() for s in MERGE_SENSORS]:
                    continue
                df = pd.read_excel(xlsx_path, sheet_name=sheet)
                if df.empty:
                    continue
                # Column A is unix_timestamp (microseconds). Convert to UTC for comparison with time ranges (parsed as Eastern then UTC).
                if "unix_timestamp" in df.columns:
                    df["_dt"] = pd.to_datetime(df["unix_timestamp"] / 1e6, unit="s", errors="coerce", utc=True)
                else:
                    first_col = df.columns[0]
                    df["_dt"] = pd.to_datetime(df[first_col] / 1e6, unit="s", errors="coerce", utc=True)
                df = df.dropna(subset=["_dt"])

                for (p, mode, t_start, t_end) in time_ranges:
                    if p != person:
                        continue
                    mask = (df["_dt"] >= t_start) & (df["_dt"] <= t_end)
                    subset = df.loc[mask].drop(columns=["_dt"])
                    if subset.empty:
                        continue
                    key = (person, mode, sheet)
                    merged.setdefault(key, []).append(subset)
            xl.close()

    for (person, mode, sheet), frames in merged.items():
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(combined.columns[0])
        # Recompute Date/Time from unix_timestamp: show Eastern time, 24-hour format m/d/yyyy H:mm:ss (no AM/PM)
        ts_col = "unix_timestamp" if "unix_timestamp" in combined.columns else combined.columns[0]
        dt_utc = pd.to_datetime(combined[ts_col] / 1e6, unit="s", utc=True)
        combined["Date/Time"] = dt_utc.dt.tz_convert("US/Eastern").dt.strftime("%m/%d/%Y %H:%M:%S")
        # Column order: unix_timestamp, Date/Time, then value column(s)
        value_cols = [c for c in combined.columns if c not in (ts_col, "Date/Time")]
        combined = combined[[ts_col, "Date/Time"] + value_cols]
        out_dir = PASSIVE_DIR if mode == "passive" else ACTIVE_DIR
        out_path = os.path.join(out_dir, f"{person}_{mode}_{sheet}.xlsx")
        combined.to_excel(out_path, index=False)
        print(f"Merged {len(combined)} rows -> {out_path}")


if __name__ == "__main__":
    print("Batch: converting all avro files to Excel (with column B formula and format)...")
    run_batch()
    print("Merge: combining rows by time ranges into passive/ and active/...")
    merge_by_time_ranges()
    print("Done.")
