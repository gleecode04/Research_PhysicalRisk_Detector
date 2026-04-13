"""
Load final_files_ML/Frustration CSVs with 3-class frustration labels from labels.txt (markdown table).
Participant ID from filename only (e.g. P13); labels are the source of truth, not filename tokens.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

FEATURE_COLS = ["eda_eda", "bvp_bvp", "temperature_temp"]

# Optional rolling features (not used by default final_ml training path).
ROLLING_WINDOW = 300

_ROLL_STAT_NAMES = ("roll_mean", "roll_std", "roll_min", "roll_max", "roll_median")


def final_ml_feature_names() -> list:
    """Default training: three raw wrist channels only."""
    return list(FEATURE_COLS)


def add_rolling_features(part: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Append rolling stats per base column (for extended models / inference parity)."""
    out = part.copy()
    for col in FEATURE_COLS:
        if col not in out.columns:
            raise ValueError(f"Missing column {col!r}")
        s = out[col].astype(float)
        r = s.rolling(window=window, min_periods=1)
        out[f"{col}_roll_mean"] = r.mean()
        out[f"{col}_roll_std"] = r.std(ddof=0)
        out[f"{col}_roll_min"] = r.min()
        out[f"{col}_roll_max"] = r.max()
        out[f"{col}_roll_median"] = r.median()
    return out


def final_ml_feature_names_with_rolling() -> list:
    names = list(FEATURE_COLS)
    for col in FEATURE_COLS:
        for suf in _ROLL_STAT_NAMES:
            names.append(f"{col}_{suf}")
    return names


# labels.txt uses "Medium" / "Low" / "High"
LEVEL_TO_INT = {
    "low": 0,
    "medium": 1,
    "med": 1,
    "high": 2,
}

PID_RE = re.compile(r"^P(\d+)", re.IGNORECASE)


def parse_labels_txt(path: str) -> dict[str, int]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("|---") or "Participant" in line and "Level" in line:
                continue
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) < 3:
                continue
            pid_raw = parts[0].strip()
            m_pid = PID_RE.match(pid_raw)
            if not m_pid:
                continue
            pid = "P" + m_pid.group(1)
            level = parts[2]
            key = level.lower().replace(".", "").strip()
            if key not in LEVEL_TO_INT and key != "medium":
                if "high" in key:
                    key = "high"
                elif "low" in key:
                    key = "low"
                elif "med" in key:
                    key = "medium"
            if key == "medium":
                y = 1
            elif key in LEVEL_TO_INT:
                y = LEVEL_TO_INT[key]
            else:
                continue
            out[pid] = y
    return out


def participant_from_filename(basename: str):
    m = PID_RE.match(os.path.basename(basename))
    if not m:
        return None
    return "P" + m.group(1)


def load_final_ml_frustration(root: str):
    """
    Returns X DataFrame (row-level, 3 features), y_frustration (int 0/1/2), participant_ids, label_map.
    Read-only on CSVs under root/Frustration.
    """
    labels_path = os.path.join(root, "labels.txt")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Missing {labels_path}")
    label_map = parse_labels_txt(labels_path)
    if not label_map:
        raise ValueError(f"No labels parsed from {labels_path}")

    fr_dir = os.path.join(root, "Frustration")
    if not os.path.isdir(fr_dir):
        raise FileNotFoundError(f"Missing folder {fr_dir}")

    rows_x = []
    rows_y = []
    rows_pid = []
    for path in sorted(glob.glob(os.path.join(fr_dir, "*.csv"))):
        pid = participant_from_filename(path)
        if pid is None:
            print(f"Warning: skip (no P<id> in name): {path}", file=sys.stderr)
            continue
        if pid not in label_map:
            print(f"Warning: no label for {pid}, skip {path}", file=sys.stderr)
            continue
        y = label_map[pid]
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: skip {path}: {e}", file=sys.stderr)
            continue
        miss = [c for c in FEATURE_COLS if c not in df.columns]
        if miss:
            print(f"Warning: skip {path}, missing {miss}", file=sys.stderr)
            continue
        part = df[FEATURE_COLS].astype(float)
        n = len(part)
        rows_x.append(part)
        rows_y.extend([y] * n)
        rows_pid.extend([pid] * n)

    if not rows_x:
        raise ValueError("No rows loaded from final_files_ML/Frustration; check CSVs and labels.")

    X = pd.concat(rows_x, ignore_index=True)
    y_frustration = np.array(rows_y, dtype=int)
    participant_ids = np.array(rows_pid)
    return X, y_frustration, participant_ids, label_map
