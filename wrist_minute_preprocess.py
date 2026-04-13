"""
Merge Empatica-style per-minute biomarker CSVs (TA sample schema) into one DataFrame
with columns expected by final_ml: eda_eda, bvp_bvp, temperature_temp.

Note: pulse_rate_bpm is used as a stand-in for bvp_bvp (different physiology) — documented caveat.
"""

from __future__ import annotations

import os
import warnings

import pandas as pd

# TA / S3 download column names
DEFAULT_EDA_VALUE = "eda_scl_usiemens"
DEFAULT_TEMP_VALUE = "temperature_celsius"
DEFAULT_PULSE_VALUE = "pulse_rate_bpm"


def _read_minute_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def merge_emphatica_minute_csvs(
    eda_path: str,
    temperature_path: str,
    pulse_path: str,
    *,
    eda_col: str = DEFAULT_EDA_VALUE,
    temp_col: str = DEFAULT_TEMP_VALUE,
    pulse_col: str = DEFAULT_PULSE_VALUE,
    ffill: bool = True,
    interpolate_numeric: bool = True,
) -> pd.DataFrame:
    """
    Outer-join three minute CSVs on timestamp_unix (epoch ms).
    Output columns: timestamp_local, eda_eda, bvp_bvp, temperature_temp
    """
    eda = _read_minute_csv(eda_path)
    tmp = _read_minute_csv(temperature_path)
    pul = _read_minute_csv(pulse_path)

    def prep(df: pd.DataFrame, value_col: str, name: str) -> pd.DataFrame:
        if "timestamp_unix" not in df.columns:
            raise ValueError(f"{name}: need timestamp_unix column")
        if value_col not in df.columns:
            raise ValueError(f"{name}: missing value column {value_col!r}")
        out = df[["timestamp_unix", value_col]].copy()
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
        # Drop rows where missing_value_reason indicates bad wear (optional; keep NaN for merge)
        return out.rename(columns={value_col: f"_{name}_raw"})

    e = prep(eda, eda_col, "eda")
    t = prep(tmp, temp_col, "temp")
    p = prep(pul, pulse_col, "pulse")

    merged = e.merge(t, on="timestamp_unix", how="outer").merge(p, on="timestamp_unix", how="outer")
    merged = merged.sort_values("timestamp_unix").reset_index(drop=True)

    if interpolate_numeric and len(merged) > 1:
        for c in ("_eda_raw", "_temp_raw", "_pulse_raw"):
            if c in merged.columns:
                merged[c] = merged[c].interpolate(limit_direction="both")

    if ffill:
        merged = merged.ffill().bfill()

    ts_ms = pd.to_datetime(merged["timestamp_unix"], unit="ms", utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "timestamp_local": ts_ms.dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "eda_eda": merged["_eda_raw"],
            "bvp_bvp": merged["_pulse_raw"],
            "temperature_temp": merged["_temp_raw"],
        }
    )

    if out[["eda_eda", "bvp_bvp", "temperature_temp"]].isna().all().all():
        warnings.warn(
            "All sensor values are NaN after merge (e.g. device_not_worn_correctly in source CSVs). "
            "Predictions will not be meaningful until rows contain numeric data.",
            UserWarning,
            stacklevel=2,
        )

    return out
