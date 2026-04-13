#!/usr/bin/env python3
"""
Empatica / TA **minute-wide** CSVs → filter / optional normalize → final_ml frustration prediction.

Expected layout (minute CSVs under ``Data/examples/empatica_minute/`` or your own folder):
  eda.csv           — timestamp_unix, eda_scl_usiemens, optional missing_value_reason
  temperature.csv   — timestamp_unix, temperature_celsius, ...
  pulse-rate.csv    — timestamp_unix, pulse_rate_bpm (used as bvp_bvp proxy), ...

Default preprocessing matches training distribution: **raw sensor scale** (no min–max).
Use --minmax-01 only if you retrained the model on min–max data; otherwise predictions
will be wrong.

Dependencies: train.py (predict_frustration_final_df + model dir), pandas, numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train import (
    FINAL_ML_VERSIONS_DIR,
    OUTPUT_DIR_FINAL_ML,
    default_inference_window_rows,
    predict_frustration_final_df,
)

SENSOR_COLS = ("eda_eda", "bvp_bvp", "temperature_temp")
DEFAULT_EDA_VALUE = "eda_scl_usiemens"
DEFAULT_TEMP_VALUE = "temperature_celsius"
DEFAULT_PULSE_VALUE = "pulse_rate_bpm"
REASON_COL = "missing_value_reason"


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _prep_stream(
    df: pd.DataFrame,
    value_col: str,
    stream_name: str,
    *,
    mask_bad_reason: bool,
) -> pd.DataFrame:
    if "timestamp_unix" not in df.columns:
        raise ValueError(f"{stream_name}: missing timestamp_unix")
    if value_col not in df.columns:
        raise ValueError(f"{stream_name}: missing value column {value_col!r}")
    out = df[["timestamp_unix", value_col]].copy()
    vals = pd.to_numeric(out[value_col], errors="coerce")
    if mask_bad_reason and REASON_COL in df.columns:
        bad = df[REASON_COL].notna() & (df[REASON_COL].astype(str).str.strip() != "")
        vals = vals.where(~bad, np.nan)
    out[value_col] = vals
    return out.rename(columns={value_col: f"_{stream_name}_raw"})


def merge_minute_wide(
    eda_path: str,
    temperature_path: str,
    pulse_path: str,
    *,
    eda_col: str = DEFAULT_EDA_VALUE,
    temp_col: str = DEFAULT_TEMP_VALUE,
    pulse_col: str = DEFAULT_PULSE_VALUE,
    mask_bad_reason: bool = True,
    interpolate: bool = True,
    ffill_bfill: bool = True,
) -> pd.DataFrame:
    eda = _read_csv(eda_path)
    tmp = _read_csv(temperature_path)
    pul = _read_csv(pulse_path)

    e = _prep_stream(eda, eda_col, "eda", mask_bad_reason=mask_bad_reason)
    t = _prep_stream(tmp, temp_col, "temp", mask_bad_reason=mask_bad_reason)
    p = _prep_stream(pul, pulse_col, "pulse", mask_bad_reason=mask_bad_reason)

    merged = e.merge(t, on="timestamp_unix", how="outer").merge(p, on="timestamp_unix", how="outer")
    merged = merged.sort_values("timestamp_unix").reset_index(drop=True)

    if interpolate and len(merged) > 1:
        for c in ("_eda_raw", "_temp_raw", "_pulse_raw"):
            merged[c] = merged[c].interpolate(limit_direction="both")

    if ffill_bfill:
        merged = merged.ffill().bfill()

    ts_ms = pd.to_datetime(merged["timestamp_unix"], unit="ms", utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "timestamp_unix": merged["timestamp_unix"],
            "timestamp_local": ts_ms.dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "eda_eda": merged["_eda_raw"],
            "bvp_bvp": merged["_pulse_raw"],
            "temperature_temp": merged["_temp_raw"],
        }
    )
    return out


def resample_one_minute_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to a regular 1-minute grid (mean), then interpolate gaps."""
    if df.empty:
        return df
    if "timestamp_unix" not in df.columns:
        raise ValueError("resample_one_minute_mean needs timestamp_unix on the frame")
    ts = pd.to_datetime(df["timestamp_unix"], unit="ms", utc=True, errors="coerce")
    work = df.copy()
    work["_ts"] = ts
    work = work.dropna(subset=["_ts"])
    sub = work.set_index("_ts")[list(SENSOR_COLS)].sort_index()
    sub = sub[~sub.index.duplicated(keep="last")]
    sub = sub.resample("1min").mean()
    sub = sub.interpolate(limit_direction="both").ffill().bfill()
    sub = sub.reset_index()
    tcol = sub.columns[0]
    sub["timestamp_unix"] = (sub[tcol].astype("int64") // 1_000_000).astype(np.int64)
    sub["timestamp_local"] = (
        pd.to_datetime(sub[tcol], utc=True).dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    return sub.drop(columns=[tcol])


def filter_rows(
    df: pd.DataFrame,
    *,
    drop_all_nan: bool,
    require_all_channels: bool,
) -> pd.DataFrame:
    d = df.copy()
    if drop_all_nan:
        d = d.loc[d[list(SENSOR_COLS)].notna().any(axis=1)]
    if require_all_channels:
        d = d.loc[d[list(SENSOR_COLS)].notna().all(axis=1)]
    return d.reset_index(drop=True)


def minmax_01(df: pd.DataFrame, cols: tuple[str, ...] = SENSOR_COLS) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or pd.isna(hi) or float(hi) == float(lo):
            out[c] = s
            continue
        out[c] = (s - lo) / (hi - lo)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge Empatica minute-wide CSVs, filter/normalize, run final_ml frustration prediction."
    )
    p.add_argument(
        "--dir",
        default=".",
        help="Folder containing eda.csv, temperature.csv, pulse-rate.csv",
    )
    p.add_argument("--eda", default=None)
    p.add_argument("--temperature", default=None)
    p.add_argument("--pulse", default=None, help="Usually pulse-rate.csv")
    p.add_argument("--output-dir", default=os.environ.get("VIP_MODEL_DIR", OUTPUT_DIR_FINAL_ML))
    p.add_argument(
        "--model-version",
        default=os.environ.get("VIP_MODEL_VERSION"),
        metavar="NAME",
        help=f"Load model from {FINAL_ML_VERSIONS_DIR}/<NAME>",
    )
    p.add_argument(
        "--no-mask-missing-reason",
        action="store_true",
        help="Do not force NaN where missing_value_reason is non-empty",
    )
    p.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Skip time interpolation between merged timestamps",
    )
    p.add_argument(
        "--no-ffill",
        action="store_true",
        help="Skip forward/back fill after merge",
    )
    p.add_argument(
        "--resample-1min",
        action="store_true",
        help="Resample merged series to 1-minute buckets (mean), like normalize_and_merge cadence",
    )
    p.add_argument(
        "--drop-all-nan-rows",
        action="store_true",
        default=True,
        help="Remove rows where all three sensors are NaN (default: on)",
    )
    p.add_argument(
        "--no-drop-all-nan-rows",
        action="store_false",
        dest="drop_all_nan_rows",
    )
    p.add_argument(
        "--require-all-channels",
        action="store_true",
        help="Keep only rows where eda, bvp, temp are all finite (stricter)",
    )
    p.add_argument(
        "--minmax-01",
        action="store_true",
        help="Min–max scale each sensor column to [0,1] (WARNING: mismatches default final_ml training)",
    )
    p.add_argument(
        "--inference-window-rows",
        type=int,
        default=None,
        help="Tail rows for scoring (default: 5 if minute data, else 300)",
    )
    p.add_argument(
        "--assume-one-row-per-minute",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-assume-one-row-per-minute",
        action="store_false",
        dest="assume_one_row_per_minute",
    )
    p.add_argument(
        "--save-preprocessed",
        default=None,
        metavar="CSV",
        help="Write merged+filtered table for debugging",
    )
    p.add_argument("--json-out", action="store_true", help="Print only JSON line")
    args = p.parse_args()

    if args.eda and args.temperature and args.pulse:
        eda_p, temp_p, pulse_p = args.eda, args.temperature, args.pulse
    else:
        d = args.dir
        eda_p = os.path.join(d, "eda.csv")
        temp_p = os.path.join(d, "temperature.csv")
        pulse_p = os.path.join(d, "pulse-rate.csv")

    out_dir = args.output_dir
    if args.model_version and str(args.model_version).strip():
        out_dir = os.path.join(FINAL_ML_VERSIONS_DIR, str(args.model_version).strip())

    win = default_inference_window_rows(args.assume_one_row_per_minute, args.inference_window_rows)

    if args.minmax_01:
        warnings.warn(
            "minmax-01 does not match default final_ml training (raw sensor scale). "
            "Use only if your checkpoint was trained on min–max data.",
            UserWarning,
            stacklevel=1,
        )

    merged = merge_minute_wide(
        eda_p,
        temp_p,
        pulse_p,
        mask_bad_reason=not args.no_mask_missing_reason,
        interpolate=not args.no_interpolate,
        ffill_bfill=not args.no_ffill,
    )

    if args.resample_1min:
        merged = resample_one_minute_mean(merged)

    merged = filter_rows(
        merged,
        drop_all_nan=args.drop_all_nan_rows,
        require_all_channels=args.require_all_channels,
    )

    if args.minmax_01:
        merged = minmax_01(merged)

    if args.save_preprocessed:
        merged.to_csv(args.save_preprocessed, index=False)
        if not args.json_out:
            print(f"Wrote preprocessed: {args.save_preprocessed}", file=sys.stderr)

    if merged.empty or merged[list(SENSOR_COLS)].isna().all().all():
        msg = "No usable rows after filter; check CSVs and device wear / missing_value_reason."
        print(json.dumps({"error": msg}))
        sys.exit(2)

    try:
        pred_in = merged[list(SENSOR_COLS)]
        pred = predict_frustration_final_df(pred_in, out_dir, win)
    except (ValueError, FileNotFoundError) as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(2)

    pred["inference_window_rows"] = win
    pred["preprocessed_rows"] = len(merged)
    if args.json_out:
        print(json.dumps(pred))
    else:
        print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
