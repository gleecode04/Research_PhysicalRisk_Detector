"""
Read-only ingest of Empatica digital_biomarkers aggregated_per_minute CSVs.
Merge on timestamp_unix (ms), last N rows/minutes, aggregate numeric features for ML.

Never modifies source CSVs; use --out for derived files only.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TS = "timestamp_unix"


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_biomarker_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TS not in df.columns:
        raise ValueError(f"{path}: missing {TS} (ms)")
    stem = _stem(path)
    renames = {}
    for c in df.columns:
        if c == TS:
            continue
        renames[c] = f"{stem}__{c}"
    return df.rename(columns=renames)


def merge_on_time(dfs: Sequence[pd.DataFrame], on: str = TS, how: str = "outer") -> pd.DataFrame:
    if not dfs:
        raise ValueError("No dataframes to merge")
    out = dfs[0]
    for nxt in dfs[1:]:
        out = out.merge(nxt, on=on, how=how)
    return out.sort_values(on).reset_index(drop=True)


def load_biomarker_csvs(paths: Iterable[str]) -> list[pd.DataFrame]:
    return [load_biomarker_csv(p) for p in sorted(paths)]


def last_n_minutes(df: pd.DataFrame, n: int = 5, ref_time_ms: int | None = None) -> pd.DataFrame:
    if df.empty or n < 1:
        return df.iloc[0:0].copy()
    ts = df[TS].astype(np.int64)
    if ref_time_ms is None:
        ref_time_ms = int(ts.max())
    cutoff = ref_time_ms - n * 60 * 1000
    mask = (ts > cutoff) & (ts <= ref_time_ms)
    sub = df.loc[mask].copy()
    if len(sub) > n:
        sub = sub.tail(n)
    return sub.reset_index(drop=True)


def last_n_rows(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if df.empty or n < 1:
        return df.iloc[0:0].copy()
    return df.sort_values(TS).tail(n).reset_index(drop=True)


def columns_matching_suffixes(df: pd.DataFrame, suffixes: Sequence[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c == TS or "__" not in c:
            continue
        _, rest = c.split("__", 1)
        if rest in suffixes:
            cols.append(c)
    return sorted(set(cols))


def aggregate_window_numeric(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.Series:
    out: dict[str, float] = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        out[f"{col}_mean"] = float(s.mean())
        out[f"{col}_std"] = float(s.std(ddof=0))
        out[f"{col}_min"] = float(s.min())
        out[f"{col}_max"] = float(s.max())
        out[f"{col}_median"] = float(s.median())
    return pd.Series(out)


def to_inference_frame(
    df_window: pd.DataFrame,
    suffixes: Sequence[str],
    mode: str = "agg",
) -> pd.DataFrame:
    numeric_cols = columns_matching_suffixes(df_window, suffixes)
    if mode == "window":
        return df_window.copy()
    if mode == "agg":
        ser = aggregate_window_numeric(df_window, numeric_cols)
        return ser.to_frame().T
    raise ValueError("mode must be 'window' or 'agg'")


def ordered_feature_names(suffixes: Sequence[str], stems: Sequence[str]) -> list[str]:
    """Stable feature order: for each stem x suffix (both sorted), emit stat names."""
    names = []
    for stem in sorted(stems):
        for suf in suffixes:
            base = f"{stem}__{suf}"
            for stat in ("mean", "std", "min", "max", "median"):
                names.append(f"{base}_{stat}")
    return names


def build_session_merged(session_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(session_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {session_dir}")
    dfs = load_biomarker_csvs(paths)
    return merge_on_time(dfs)


def session_aggregate_row(session_dir: str, suffixes: Sequence[str], window_minutes: int) -> tuple[pd.Series, list[str]]:
    merged = build_session_merged(session_dir)
    win = last_n_rows(merged, window_minutes)
    if win.empty:
        return pd.Series(dtype=float), []
    ser = aggregate_window_numeric(win, columns_matching_suffixes(win, suffixes)).sort_index()
    return ser, list(ser.index)


def main():
    parser = argparse.ArgumentParser(description="Merge minute biomarker CSVs; 5-min window (read-only inputs).")
    parser.add_argument("session_dir", help="Directory with aggregated_per_minute CSVs for one session")
    parser.add_argument("--window", type=int, default=5, help="Rows/minutes in window (default 5)")
    parser.add_argument("--mode", choices=("window", "agg"), default="agg")
    parser.add_argument(
        "--suffixes",
        default="eda_scl_usiemens,temperature_celsius,pulse_rate_bpm",
        help="Comma-separated Empatica column suffixes (after __ in merged names)",
    )
    parser.add_argument("--out", default="", help="Write derived CSV here only")
    args = parser.parse_args()
    merged = build_session_merged(args.session_dir)
    win = last_n_rows(merged, args.window)
    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    out_df = to_inference_frame(win, suffixes, mode=args.mode)
    print(out_df.to_string())
    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print("Wrote", args.out)


if __name__ == "__main__":
    main()
