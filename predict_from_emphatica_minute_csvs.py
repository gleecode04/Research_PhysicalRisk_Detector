#!/usr/bin/env python3
"""
Merge TA/Empatica-style minute CSVs (eda, temperature, pulse-rate) and run final_ml frustration prediction.

Example:
  python predict_from_emphatica_minute_csvs.py --dir Data/examples/empatica_minute --output-dir Data/outputs/ml_results_final_ml

    python predict_from_emphatica_minute_csvs.py \\
    --eda path/eda.csv --temperature path/temperature.csv --pulse path/pulse-rate.csv \\
    --output-dir Data/outputs/ml_results_final_ml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from wrist_minute_preprocess import merge_emphatica_minute_csvs
from train import predict_frustration_final_df, OUTPUT_DIR_FINAL_ML, FINAL_ML_VERSIONS_DIR


def main():
    p = argparse.ArgumentParser(
        description="Merge Empatica minute biomarker CSVs → final_ml frustration prediction."
    )
    p.add_argument(
        "--dir",
        default=None,
        help="Folder containing eda.csv, temperature.csv, pulse-rate.csv (TA layout).",
    )
    p.add_argument("--eda", default=None, help="Path to eda.csv")
    p.add_argument("--temperature", default=None, help="Path to temperature.csv")
    p.add_argument("--pulse", default=None, help="Path to pulse-rate.csv (used as BVP proxy)")
    p.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_FINAL_ML,
        help="Directory with pipeline_frustration_finalml.joblib and model_meta_finalml.json",
    )
    p.add_argument(
        "--model-version",
        default=None,
        metavar="NAME",
        help=f"If set, use {FINAL_ML_VERSIONS_DIR}/<NAME> (overrides --output-dir).",
    )
    p.add_argument(
        "--inference-window-rows",
        type=int,
        default=None,
        help="Tail rows for scoring (default 5 with --assume-one-row-per-minute else 300).",
    )
    p.add_argument(
        "--hz-wrist-window",
        action="store_true",
        help="Use last 300 rows (~5 min at 1 Hz). Default is 5 rows (Empatica aggregated_per_minute).",
    )
    p.add_argument(
        "--save-merged",
        default=None,
        metavar="PATH",
        help="If set, write merged wrist-format CSV here (for debugging).",
    )
    p.add_argument("--json", action="store_true", help="Print result as JSON only.")
    args = p.parse_args()

    out_dir = args.output_dir
    if args.model_version and str(args.model_version).strip():
        out_dir = os.path.join(FINAL_ML_VERSIONS_DIR, str(args.model_version).strip())

    if args.dir:
        eda = os.path.join(args.dir, "eda.csv")
        temp = os.path.join(args.dir, "temperature.csv")
        pulse = os.path.join(args.dir, "pulse-rate.csv")
    else:
        if not (args.eda and args.temperature and args.pulse):
            p.error("Provide --dir or all of --eda, --temperature, --pulse")
        eda, temp, pulse = args.eda, args.temperature, args.pulse

    if args.inference_window_rows is not None:
        win = int(args.inference_window_rows)
    else:
        win = 300 if args.hz_wrist_window else 5

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        merged = merge_emphatica_minute_csvs(eda, temp, pulse, ffill=True, interpolate_numeric=True)
        for x in w:
            if issubclass(x.category, UserWarning):
                print(x.message, file=sys.stderr)

    if args.save_merged:
        merged.to_csv(args.save_merged, index=False)
        if not args.json:
            print(f"Wrote merged CSV: {args.save_merged}", file=sys.stderr)

    try:
        result = predict_frustration_final_df(merged, out_dir, win)
    except ValueError as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        sys.exit(2)

    result["inference_window_rows"] = win
    result["note"] = (
        "pulse_rate_bpm was mapped to bvp_bvp (not true BVP). "
        "Use real BVP minute exports for best alignment with training."
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
