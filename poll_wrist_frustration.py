#!/usr/bin/env python3
"""
Poll a folder of Empatica minute CSVs (eda.csv, temperature.csv, pulse-rate.csv) every N seconds,
merge, and print final_ml frustration prediction.

Env:
  POLL_INTERVAL_SEC   default 300
  WRIST_MINUTE_DIR    folder with the three CSVs (or pass --dir)
  VIP_MODEL_DIR       default Data/outputs/ml_results_final_ml
"""

from __future__ import annotations

import os
import sys
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import warnings

from wrist_minute_preprocess import merge_emphatica_minute_csvs
from train import (
    predict_frustration_final_df,
    default_inference_window_rows,
    OUTPUT_DIR_FINAL_ML,
    FINAL_ML_VERSIONS_DIR,
)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Periodic merge + predict from minute biomarker CSV folder.")
    p.add_argument("--dir", default=os.environ.get("WRIST_MINUTE_DIR", "."), help="Folder with eda/temp/pulse-rate CSVs")
    p.add_argument("--output-dir", default=os.environ.get("VIP_MODEL_DIR", OUTPUT_DIR_FINAL_ML))
    p.add_argument(
        "--model-version",
        default=os.environ.get("VIP_MODEL_VERSION"),
        metavar="NAME",
        help=f"If set, use {FINAL_ML_VERSIONS_DIR}/<NAME> (overrides --output-dir). Env: VIP_MODEL_VERSION.",
    )
    p.add_argument("--interval", type=int, default=int(os.environ.get("POLL_INTERVAL_SEC", "300")))
    p.add_argument("--once", action="store_true", help="Run a single tick and exit.")
    p.add_argument("--assume-one-row-per-minute", action="store_true", default=True)
    p.add_argument("--no-assume-one-row-per-minute", action="store_false", dest="assume_one_row_per_minute")
    args = p.parse_args()

    out_dir = args.output_dir
    if args.model_version and str(args.model_version).strip():
        out_dir = os.path.join(FINAL_ML_VERSIONS_DIR, str(args.model_version).strip())

    win = default_inference_window_rows(args.assume_one_row_per_minute, None)
    d = args.dir
    eda = os.path.join(d, "eda.csv")
    temp = os.path.join(d, "temperature.csv")
    pulse = os.path.join(d, "pulse-rate.csv")

    def tick():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            merged = merge_emphatica_minute_csvs(eda, temp, pulse, ffill=True, interpolate_numeric=True)
            for x in w:
                if issubclass(x.category, UserWarning):
                    print(x.message, file=sys.stderr)
        try:
            out = predict_frustration_final_df(merged, out_dir, win)
        except ValueError as e:
            print(json.dumps({"error": str(e)}))
            return
        out["inference_window_rows"] = win
        print(json.dumps(out))

    while True:
        tick()
        if args.once:
            break
        print(f"Sleeping {args.interval}s …", file=sys.stderr)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
