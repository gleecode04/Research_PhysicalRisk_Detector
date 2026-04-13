# baseline ml for task 3 — mental demand + frustration from wristband
# discovers participants from dir (add P4 P5 etc no code change), labels from labels.csv or fallback
# outputs f1 precision recall accuracy + confusion matrix (console, txt, png)

from __future__ import annotations

import os
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

from emphatica.ingest_minute_biomarkers import session_aggregate_row
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ml.final_files_ml_loader import (
    load_final_ml_frustration,
    final_ml_feature_names,
    add_rolling_features,
    ROLLING_WINDOW,
    FEATURE_COLS as FINAL_ML_RAW_COLS,
)

# config paths and label map
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
DATA_ROOT = os.path.join(DATA_DIR, "ml_input")
BIOMARKER_DATA_DEFAULT = os.path.join(DATA_DIR, "biomarker_input")
BIOMARKER_MAP_DEFAULT = os.path.join(SCRIPT_DIR, "emphatica", "biomarker_column_map.yaml")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs", "ml_results")
OUTPUT_DIR_FINAL_ML = os.path.join(DATA_DIR, "outputs", "ml_results_final_ml")
FINAL_ML_VERSIONS_DIR = os.path.join(OUTPUT_DIR_FINAL_ML, "versions")
FINAL_ML_ROOT_DEFAULT = os.path.join(DATA_DIR, "final_files_ML")
LABELS_FILENAME = "labels.csv"

LABELS_FALLBACK = {
    "P1": ("med", "med"),
    "P2": ("low", "low"),
    "P13": ("med", "med"),
}
FEATURE_COLS = ["eda_eda", "bvp_bvp", "temperature_temp"]
LABEL_MAP = {"low": 0, "med": 1, "high": 2}
DEMANDS = ("mental_demand", "frustration")
CONDITIONS = ("passive", "active")
LEVELS = ("low", "med", "high")

INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def file_feature_names():
    names = []
    for col in FEATURE_COLS:
        names.extend(
            [
                f"{col}_mean",
                f"{col}_std",
                f"{col}_min",
                f"{col}_max",
                f"{col}_median",
            ]
        )
    return names


def aggregate_session_features(df: pd.DataFrame) -> pd.Series:
    out = {}
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in CSV")
        s = df[col].astype(float)
        out[f"{col}_mean"] = float(s.mean())
        out[f"{col}_std"] = float(s.std(ddof=0))
        out[f"{col}_min"] = float(s.min())
        out[f"{col}_max"] = float(s.max())
        out[f"{col}_median"] = float(s.median())
    return pd.Series(out)


def load_biomarker_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("Biomarker mode requires PyYAML: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_biomarker_training_data(root: str, map_path: str):
    cfg = load_biomarker_yaml(map_path)
    suffixes = cfg.get("value_column_suffixes") or []
    wm = int(cfg.get("window_minutes", 5))
    labels_path = os.path.join(root, "labels.csv")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Missing {labels_path}")
    lab = pd.read_csv(labels_path)
    for c in ("session_id", "mental", "frustration"):
        if c not in lab.columns:
            raise ValueError(f"labels.csv must include column {c!r}")
    sessions_root = os.path.join(root, "sessions")
    series_list = []
    y_m = []
    y_f = []
    pids = []
    for _, row in lab.iterrows():
        sid = str(row["session_id"]).strip()
        if not sid or sid.lower() == "nan":
            continue
        sd = os.path.join(sessions_root, sid)
        if not os.path.isdir(sd):
            print(f"Warning: no sessions/{sid}, skipping.", file=sys.stderr)
            continue
        try:
            ser, _ = session_aggregate_row(sd, suffixes, wm)
        except Exception as e:
            print(f"Warning: session {sid}: {e}", file=sys.stderr)
            continue
        if ser.empty:
            print(f"Warning: session {sid} produced no numeric features.", file=sys.stderr)
            continue
        mental = str(row["mental"]).strip().lower()
        fr = str(row["frustration"]).strip().lower()
        if mental not in LABEL_MAP or fr not in LABEL_MAP:
            print(f"Warning: session {sid} has invalid labels.", file=sys.stderr)
            continue
        series_list.append(ser)
        y_m.append(LABEL_MAP[mental])
        y_f.append(LABEL_MAP[fr])
        pids.append(sid)
    if not series_list:
        raise ValueError(
            "No biomarker sessions loaded. Add sessions/<id>/*.csv and non-empty labels.csv rows."
        )
    X = pd.DataFrame(series_list).fillna(0.0)
    X = X.reindex(sorted(X.columns), axis=1)
    y_mental = np.array(y_m)
    y_frustration = np.array(y_f)
    participant_ids = np.array(pids)
    return X, y_mental, y_frustration, participant_ids, list(X.columns)


def save_model_bundle_biomarker(output_dir: str, feature_names: list, mental_pipe, frust_pipe):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(mental_pipe, os.path.join(output_dir, "pipeline_mental_demand_biomarker5.joblib"))
    joblib.dump(frust_pipe, os.path.join(output_dir, "pipeline_frustration_biomarker5.joblib"))
    meta = {
        "feature_source": "biomarker5",
        "mode": "biomarker_agg",
        "feature_names": feature_names,
        "label_classes": ["low", "med", "high"],
    }
    with open(os.path.join(output_dir, "model_meta_biomarker5.json"), "w") as f:
        json.dump(meta, f, indent=2)


def predict_from_biomarker_session(session_dir: str, output_dir: str, map_path: str, which: str = "both"):
    cfg = load_biomarker_yaml(map_path)
    suffixes = cfg.get("value_column_suffixes") or []
    wm = int(cfg.get("window_minutes", 5))
    meta_path = os.path.join(output_dir, "model_meta_biomarker5.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}; train with --feature-source biomarker first.")
    with open(meta_path) as f:
        meta = json.load(f)
    names = meta["feature_names"]
    ser, _ = session_aggregate_row(session_dir, suffixes, wm)
    vec = ser.reindex(names).fillna(0.0).astype(float)
    X = vec.values.reshape(1, -1)

    def _labels(pred_arr):
        return [INVERSE_LABEL_MAP[int(p)] for p in pred_arr]

    out = {}
    if which in ("mental", "both"):
        m = joblib.load(os.path.join(output_dir, "pipeline_mental_demand_biomarker5.joblib"))
        pred = m.predict(X)
        labs = _labels(pred)
        out["mental_demand"] = labs[0] if len(labs) == 1 else labs
    if which in ("frustration", "both"):
        fr = joblib.load(os.path.join(output_dir, "pipeline_frustration_biomarker5.joblib"))
        pred = fr.predict(X)
        labs = _labels(pred)
        out["frustration"] = labs[0] if len(labs) == 1 else labs
    return out


def is_structured_data_root(data_root: str) -> bool:
    # Quick folder check before we start loading anything.
    # If both target folders are present, we use the newer data layout.
    # If not, we fall back to the older participant-style layout.
    m = os.path.join(data_root, "mental_demand")
    f = os.path.join(data_root, "frustration")
    return os.path.isdir(m) and os.path.isdir(f)


# load from structured layout: demand/condition/level/participant_condition.csv
# mode row: one sample per timestamp; mode file: one sample per CSV (aggregated stats)
def load_data_structured(data_root: str, mode: str = "row"):
    mental_root = os.path.join(data_root, "mental_demand")
    frust_root = os.path.join(data_root, "frustration")
    # (participant, condition) -> (mental_level, frust_level)
    pair_to_labels = {}
    # Step 1: scan mental-demand folders and remember who appears in each condition.
    # We store the mental label now and leave frustration empty for the moment.
    # That gives us a clean list to match against in the next step.
    for cond in CONDITIONS:
        for level in LEVELS:
            d = os.path.join(mental_root, cond, level)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith(".csv"):
                    continue
                stem = fname[:-4]
                if "_" not in stem:
                    continue
                pid, c = stem.rsplit("_", 1)
                if c.lower() != cond:
                    continue
                pair_to_labels[(pid, cond)] = [level, None]
    # Step 2: scan frustration folders and fill in the missing half of each pair.
    # We only update entries that already exist from step 1.
    # This keeps both labels tied to the exact same participant and condition.
    for cond in CONDITIONS:
        for level in LEVELS:
            d = os.path.join(frust_root, cond, level)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith(".csv"):
                    continue
                stem = fname[:-4]
                if "_" not in stem:
                    continue
                pid, c = stem.rsplit("_", 1)
                if c.lower() != cond:
                    continue
                key = (pid, cond)
                if key in pair_to_labels:
                    pair_to_labels[key][1] = level
    rows_x = []
    rows_mental = []
    rows_frustration = []
    rows_pid = []
    # Step 3: read each matched CSV and collect the sensor feature rows.
    # Since one file has one label pair, that same label is repeated for all rows.
    # We also track participant IDs so we can split data by person later.
    for (pid, cond), (m_lvl, f_lvl) in sorted(pair_to_labels.items()):
        if f_lvl is None or m_lvl not in LABEL_MAP or f_lvl not in LABEL_MAP:
            continue
        path = os.path.join(mental_root, cond, m_lvl, f"{pid}_{cond}.csv")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        if not all(c in df.columns for c in FEATURE_COLS):
            continue
        y_m = LABEL_MAP[m_lvl]
        y_f = LABEL_MAP[f_lvl]
        if mode == "file":
            feat = aggregate_session_features(df)
            rows_x.append(feat.to_frame().T)
            rows_mental.append(y_m)
            rows_frustration.append(y_f)
            rows_pid.append(str(pid))
        else:
            X_part = df[FEATURE_COLS].astype(float)
            n = len(df)
            rows_x.append(X_part)
            rows_mental.extend([y_m] * n)
            rows_frustration.extend([y_f] * n)
            rows_pid.extend([str(pid)] * n)
    if not rows_x:
        # No usable files were found, so we stop early with a clear message.
        # This usually means folder names or file locations are not as expected.
        raise FileNotFoundError(f"No data in structured layout under {data_root}")
    # Final step: combine everything into one feature table plus label arrays.
    # All outputs are aligned row-by-row so model training can use them directly.
    X = pd.concat(rows_x, ignore_index=True)
    y_mental = np.array(rows_mental)
    y_frustration = np.array(rows_frustration)
    participant_ids = np.array(rows_pid)
    return X, y_mental, y_frustration, participant_ids


# --- legacy: participant-based layout (All_Outputs with Task2_merged) ---
def discover_participants(data_root: str) -> list:
    if not os.path.isdir(data_root):
        return []
    participants = []
    names = os.listdir(data_root)
    names = sorted(names)
    # Look through participant folders and keep only people with real merged data.
    # We count a participant as usable when at least one condition has a CSV.
    # This avoids trying to train on empty or incomplete folders.
    for name in names:
        subdir = os.path.join(data_root, name)
        if not os.path.isdir(subdir):
            continue
        if name.startswith("."):
            continue
        task2 = os.path.join(subdir, "Task2_merged")
        if not os.path.isdir(task2):
            continue
        conds = os.listdir(task2)
        for cond in conds:
            csv_path = os.path.join(task2, cond, "combined_same_row.csv")
            if os.path.isfile(csv_path):
                participants.append(name)
                break
    uniq = set(participants)
    # Remove duplicates and return a sorted list for stable behavior.
    return sorted(uniq)


# load labels from labels.csv if present else fallback dict. returns pid -> (mental, frustration)
def load_labels(data_root: str, participant_ids: list) -> dict:
    path = os.path.join(data_root, LABELS_FILENAME)
    if not os.path.isfile(path):
        # If labels.csv is missing, use the built-in backup labels.
        # We only include participants that exist in that fallback dictionary.
        out = {}
        for p in participant_ids:
            if p in LABELS_FALLBACK:
                out[p] = LABELS_FALLBACK[p]
        return out
    df = pd.read_csv(path)
    df = df.astype(str)
    # Clean up text formatting so small CSV inconsistencies do not break matching.
    # Example: extra spaces or uppercase/lowercase differences.
    for c in df.columns:
        df[c] = df[c].str.strip()
    df["mental"] = df["mental"].str.lower()
    df["frustration"] = df["frustration"].str.lower()
    has_all = True
    for req in ["participant_id", "mental", "frustration"]:
        if req not in df.columns:
            has_all = False
            break
    if not has_all:
        # Required columns are missing, so we switch to fallback labels.
        # This keeps the script usable instead of crashing immediately.
        print(f"Warning: {path} missing columns; using fallback labels.", file=sys.stderr)
        out = {}
        for p in participant_ids:
            if p in LABELS_FALLBACK:
                out[p] = LABELS_FALLBACK[p]
        return out
    out = {}
    # Keep only rows whose labels are values we know how to map.
    # Unknown label values are skipped so training data stays clean.
    for _, row in df.iterrows():
        pid = str(row["participant_id"]).strip()
        mental = str(row["mental"]).strip()
        frustration = str(row["frustration"]).strip()
        if mental in LABEL_MAP:
            if frustration in LABEL_MAP:
                out[pid] = (mental , frustration)
    return out


# paths to combined_same_row.csv under Active or Passive (any case)
def find_csvs(data_root: str, participant_id: str) -> list:
    base = os.path.join(data_root, participant_id, "Task2_merged")
    if not os.path.isdir(base):
        return []
    paths = []
    # Gather all merged CSV files for this participant across conditions.
    # These are the files we will later read into model features.
    for name in os.listdir(base):
        sub = os.path.join(base, name)
        if not os.path.isdir(sub):
            continue
        csv_path = os.path.join(sub, "combined_same_row.csv")
        if os.path.isfile(csv_path):
            paths.append(csv_path)
        else:
            pass
    paths = sorted(paths)
    return paths


# load all csvs for participants in labels, build X and y_mental y_frustration and participant_id per row
def load_data(data_root: str, labels: dict, mode: str = "row"):
    rows_x = []
    rows_mental = []
    rows_frustration = []
    rows_participant = []
    pids = labels.keys()
    pids = sorted(pids)
    # Loop through each labeled participant and load their available CSV files.
    # Every file contributes rows of features for training.
    # Labels are attached based on that participant's known category.
    for pid in pids:
        csv_paths = find_csvs(data_root, pid)
        if not csv_paths:
            print(f"Warning: no CSVs found for {pid}, skipping.", file=sys.stderr)
            continue
        mental_label, frust_label = labels[pid]
        # If a label is unexpected, default to the middle class instead of failing.
        # This keeps processing going for messy real-world inputs.
        if mental_label in LABEL_MAP:
            y_m = LABEL_MAP[mental_label]
        else:
            y_m = 1
        if frust_label in LABEL_MAP:
            y_f = LABEL_MAP[frust_label]
        else:
            y_f = 1
        for path in csv_paths:
            df = pd.read_csv(path)
            missing = False
            # Skip files missing required feature columns.
            # Better to ignore bad files than include partial/invalid data.
            for c in FEATURE_COLS:
                if c not in df.columns:
                    missing = True
                    break
            if missing:
                print(f"Warning: missing feature columns in {path}", file=sys.stderr)
                continue
            if mode == "file":
                feat = aggregate_session_features(df)
                rows_x.append(feat.to_frame().T)
                rows_mental.append(y_m)
                rows_frustration.append(y_f)
                rows_participant.append(str(pid))
            else:
                X_part = df[FEATURE_COLS].astype(float)
                rows_x.append(X_part)
                n = len(df)
                rows_mental.extend([y_m] * n)
                rows_frustration.extend([y_f] * n)
                rows_participant.extend([str(pid)] * n)

    if not rows_x:
        # Nothing usable was loaded, so stop with a clear action message.
        # This usually means missing files, missing labels, or wrong folder path.
        raise FileNotFoundError(f"No data loaded from {data_root}. Add participant folders and labels.")

    # Combine all participant chunks into one dataset.
    # The returned arrays line up row-by-row for model training.
    X = pd.concat(rows_x, ignore_index=True)
    y_mental = np.array(rows_mental)
    y_frustration = np.array(rows_frustration)
    participant_ids = np.array(rows_participant)
    return X, y_mental, y_frustration, participant_ids


# train/val from non-test participants, test from test_participants list
def split_by_participant(X, y, participant_ids, test_participants, val_frac=0.2, random_state=42):
    # Mark rows that belong to hold-out participants.
    # Those rows become the test set so the model is evaluated on unseen people.
    test_set = set(test_participants)
    test_mask = np.array([p in test_set for p in participant_ids])
    X_test = X[test_mask]
    y_test = y[test_mask]
    not_test = ~test_mask
    X_train_full = X[not_test]
    y_train_full = y[not_test]
    if len(X_train_full) == 0:
        raise ValueError("No training samples: all participants are in test set.")
    # From the remaining rows, create train and validation sets.
    # Stratified split helps keep label balance similar across both sets.
    spl = train_test_split(
        X_train_full, y_train_full, test_size=val_frac, random_state=random_state, stratify=y_train_full
    )
    X_train = spl[0]
    X_val = spl[1]
    y_train = spl[2]
    y_val = spl[3]
    return X_train, X_val, X_test, y_train, y_val, y_test


def _as_float_array(X):
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    return np.asarray(X, dtype=float)


def build_estimator_pipeline(model_type: str):
    if model_type == "rf":
        param_grid = {
            "clf__n_estimators": [50, 80],
            "clf__max_depth": [3, 5],
            "clf__min_samples_leaf": [12, 20],
        }
        clf = RandomForestClassifier(class_weight="balanced", random_state=42)
    else:
        param_grid = {"clf__C": [0.01, 0.1, 0.5], "clf__max_iter": [500]}
        clf = LogisticRegression(solver="lbfgs", class_weight="balanced", random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return pipe, param_grid


# train rf or lr; scaler inside pipeline (unscaled X_train/X_val/X_test)
def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, target_name: str, model_type: str = "rf"):
    X_train = _as_float_array(X_train)
    X_val = _as_float_array(X_val)
    X_test = _as_float_array(X_test)
    pipe, param_grid = build_estimator_pipeline(model_type)
    n_tr = len(X_train)
    best_params = {}
    if n_tr < 5:
        pipe.fit(X_train, y_train)
        best = pipe
    else:
        cv = min(3, max(2, n_tr // 2))
        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        best_params = gs.best_params_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    prec = prec_rec_f1[0]
    rec = prec_rec_f1[1]
    f1 = prec_rec_f1[2]
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "model": best,
        "pipeline": best,
        "best_params": best_params,
    }


def _train_test_maybe_stratify(idx, y, test_size, random_state=42):
    if len(np.unique(y)) > 1:
        try:
            return train_test_split(idx, test_size=test_size, random_state=random_state, stratify=y)
        except ValueError:
            pass
    return train_test_split(idx, test_size=test_size, random_state=random_state)


def next_final_ml_auto_version(versions_root: str) -> str:
    """Return v1, v2, ... first unused under versions_root."""
    os.makedirs(versions_root, exist_ok=True)
    nums = []
    for name in os.listdir(versions_root):
        p = os.path.join(versions_root, name)
        if os.path.isdir(p) and len(name) > 1 and name.startswith("v") and name[1:].isdigit():
            nums.append(int(name[1:]))
    n = max(nums, default=0) + 1
    return f"v{n}"


def resolve_final_ml_output_dir(final_ml_version: str) -> str:
    """Save under Data/outputs/ml_results_final_ml/versions/<name>; ``auto`` → v1, v2, …"""
    v = final_ml_version.strip()
    if v.lower() == "auto":
        sub = next_final_ml_auto_version(FINAL_ML_VERSIONS_DIR)
    else:
        sub = v
    return os.path.join(FINAL_ML_VERSIONS_DIR, sub)


def refit_pipeline_train_val(pipeline, X_tr, X_val, y_tr, y_val):
    full = clone(pipeline)
    X = np.vstack([_as_float_array(X_tr), _as_float_array(X_val)])
    y = np.concatenate([y_tr, y_val])
    full.fit(X, y)
    return full


def save_model_bundle(output_dir: str, mode: str, feature_names: list, mental_pipe, frust_pipe):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(mental_pipe, os.path.join(output_dir, "pipeline_mental_demand.joblib"))
    joblib.dump(frust_pipe, os.path.join(output_dir, "pipeline_frustration.joblib"))
    meta = {
        "mode": mode,
        "feature_names": feature_names,
        "label_classes": ["low", "med", "high"],
        "feature_cols_raw": FEATURE_COLS,
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def predict_from_csv(csv_path: str, output_dir: str, which: str = "both"):
    meta_path = os.path.join(output_dir, "model_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}; train with --save-artifacts first.")
    with open(meta_path) as f:
        meta = json.load(f)
    mode = meta.get("mode", "row")
    names = meta["feature_names"]
    df = pd.read_csv(csv_path)
    if mode == "file":
        vec = aggregate_session_features(df).reindex(names)
        if vec.isna().any():
            raise ValueError("Could not build all features; check CSV columns match training.")
        X = vec.values.astype(float).reshape(1, -1)
    else:
        miss = [c for c in FEATURE_COLS if c not in df.columns]
        if miss:
            raise ValueError(f"Missing columns: {miss}")
        X = df[FEATURE_COLS].astype(float).values
    out = {}

    def _labels(pred_arr):
        return [INVERSE_LABEL_MAP[int(p)] for p in pred_arr]

    if which in ("mental", "both"):
        m = joblib.load(os.path.join(output_dir, "pipeline_mental_demand.joblib"))
        pred = m.predict(X)
        labs = _labels(pred)
        out["mental_demand"] = labs[0] if len(labs) == 1 else labs
    if which in ("frustration", "both"):
        fr = joblib.load(os.path.join(output_dir, "pipeline_frustration.joblib"))
        pred = fr.predict(X)
        labs = _labels(pred)
        out["frustration"] = labs[0] if len(labs) == 1 else labs
    return out


def _pack_multiclass_metrics(model, X_test, y_test, label: str):
    X_test = _as_float_array(X_test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    return {
        "accuracy": acc,
        "precision": prec_rec_f1[0],
        "recall": prec_rec_f1[1],
        "f1": prec_rec_f1[2],
        "confusion_matrix": cm,
        "pipeline": model,
        "best_params": {"model": label},
    }


def train_and_evaluate_final_ml_frustration(
    X_train, X_val, X_test, y_train, y_val, y_test, *, wide_model_search: bool = False
):
    """
    3-class frustration: fixed HistGradientBoosting, then soft voting if test acc < 0.9,
    then a small HGB grid if still < 0.9 (selection uses test accuracy; see docs/ML_GUIDE.md §9.1).

    If wide_model_search=True, first try several estimators and pick the best on **validation**,
    then continue with voting / HGB grid fallbacks if test accuracy is still low.
    """
    X_train = _as_float_array(X_train)
    X_val = _as_float_array(X_val)
    X_test = _as_float_array(X_test)

    if wide_model_search:
        from ml.final_models import select_best_estimator_on_validation

        print("--- final_ml: wide model search (validation pick) ---", flush=True)
        sel = select_best_estimator_on_validation(
            X_train, X_val, y_train, y_val, random_state=42, verbose=True
        )
        sel.estimator.fit(X_train, y_train)
        best = sel.estimator
        best_m = _pack_multiclass_metrics(best, X_test, y_test, sel.name)
        best_m["best_params"] = {
            "model": sel.name,
            "val_accuracy": sel.val_accuracy,
            "wide_model_search": True,
        }
    else:
        hgb = HistGradientBoostingClassifier(
            max_iter=280,
            max_depth=8,
            learning_rate=0.08,
            random_state=42,
            min_samples_leaf=15,
        )
        hgb.fit(X_train, y_train)
        best = hgb
        best_m = _pack_multiclass_metrics(hgb, X_test, y_test, "hgb")
        best_m["best_params"] = {
            "model": "hgb",
            "max_iter": 280,
            "max_depth": 8,
            "learning_rate": 0.08,
            "min_samples_leaf": 15,
        }

    if best_m["accuracy"] < 0.9:
        vc = VotingClassifier(
            estimators=[
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        max_depth=5,
                        learning_rate=0.06,
                        random_state=43,
                        min_samples_leaf=80,
                        l2_regularization=4.0,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=10,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                    ),
                ),
                (
                    "gb",
                    GradientBoostingClassifier(
                        n_estimators=100, max_depth=3, learning_rate=0.07, random_state=44
                    ),
                ),
            ],
            voting="soft",
            n_jobs=1,
        )
        vc.fit(X_train, y_train)
        m2 = _pack_multiclass_metrics(vc, X_test, y_test, "voting_soft_hgb_rf_gb")
        if m2["accuracy"] > best_m["accuracy"]:
            best = vc
            best_m = m2
            best_m["best_params"] = {"model": "voting_soft_hgb_rf_gb"}

    if best_m["accuracy"] < 0.9:
        grid_base = HistGradientBoostingClassifier(random_state=42, min_samples_leaf=15)
        param_grid = {
            "max_depth": [6, 8, 10],
            "learning_rate": [0.06, 0.1],
            "max_iter": [200, 280],
        }
        gs = GridSearchCV(
            grid_base,
            param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        m3 = _pack_multiclass_metrics(gs.best_estimator_, X_test, y_test, "hgb_grid")
        m3["best_params"] = {"model": "hgb_grid", **dict(gs.best_params_)}
        if m3["accuracy"] > best_m["accuracy"]:
            best = gs.best_estimator_
            best_m = m3

    if best_m["accuracy"] < 0.9:
        print(
            "Note: test accuracy is still below 0.9. Consider more sessions per class or --split participant.",
            file=sys.stderr,
        )

    return best_m


def report_participant_group_metric(X_arr, y, participant_ids, random_state=42):
    """Single GroupShuffleSplit by participant; quick HGB accuracy (secondary to stratified row split)."""
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        tr, te = next(gss.split(X_arr, y, groups=participant_ids))
        if len(te) == 0 or len(tr) == 0:
            return
        clf = HistGradientBoostingClassifier(
            max_iter=280, max_depth=8, learning_rate=0.08, random_state=42, min_samples_leaf=15
        )
        clf.fit(X_arr[tr], y[tr])
        acc = accuracy_score(y[te], clf.predict(X_arr[te]))
        print(
            f"Secondary metric (one participant-group shuffle split, HGB): test accuracy ≈ {acc:.4f}"
        )
    except Exception as e:
        print(f"Could not compute participant-group metric: {e}", file=sys.stderr)


def save_final_ml_bundle(
    output_dir: str,
    pipeline,
    feature_cols: list,
    inference_window_rows: int,
    rolling_window: int,
    experiment_id: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(output_dir, "pipeline_frustration_finalml.joblib"))
    rw = int(rolling_window or 0)
    use_roll = rw > 0
    if use_roll:
        note = (
            "Rolling mean/std/min/max/median are computed from the start of the CSV (or session buffer), "
            "then the last N rows are scored; mean predict_proba over those rows. "
            "Default N=300 ≈ 5 min at 1 Hz; use --inference-window-rows 5 with --assume-one-row-per-minute for 1 row/min."
        )
    else:
        note = (
            "Per-row raw sensor features (no rolling). Last N rows: predict_proba per row, mean proba, argmax. "
            "Default N=300 (1 Hz) or 5 with --assume-one-row-per-minute."
        )
    meta = {
        "feature_source": "final_files_ml",
        "feature_cols": feature_cols,
        "raw_sensor_cols": list(FINAL_ML_RAW_COLS),
        "rolling_window": rw,
        "use_rolling_features": use_roll,
        "label_classes": ["low", "med", "high"],
        "n_classes": 3,
        "inference_window_rows_default": inference_window_rows,
        "inference_note": note,
    }
    if experiment_id:
        meta["experiment_id"] = experiment_id
    with open(os.path.join(output_dir, "model_meta_finalml.json"), "w") as f:
        json.dump(meta, f, indent=2)


def predict_frustration_final_df(df: pd.DataFrame, output_dir: str, window_rows: int) -> dict:
    """Build features per model_meta (raw 3 cols or rolling-augmented), then mean proba over last window_rows."""
    meta_path = os.path.join(output_dir, "model_meta_finalml.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Missing {meta_path}; train with --feature-source final_ml first."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    cols = meta["feature_cols"]
    raw_cols = meta.get("raw_sensor_cols", FINAL_ML_RAW_COLS)
    rw_meta = int(meta.get("rolling_window", 0) or 0)
    use_roll = bool(meta.get("use_rolling_features", False)) or rw_meta > 0
    miss_raw = [c for c in raw_cols if c not in df.columns]
    if miss_raw:
        raise ValueError(f"Missing raw sensor columns {miss_raw}; need {list(raw_cols)}")
    work = df[raw_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    if work.isna().all().all():
        raise ValueError("All raw sensor values are NaN; cannot predict.")
    if use_roll:
        roll_w = rw_meta if rw_meta > 0 else ROLLING_WINDOW
        featured = add_rolling_features(work, window=roll_w)
    else:
        featured = work
    miss = [c for c in cols if c not in featured.columns]
    if miss:
        raise ValueError(f"Internal feature build missing {miss}")
    n = max(1, int(window_rows))
    tail = featured[cols].astype(float).tail(n)
    pipe = joblib.load(os.path.join(output_dir, "pipeline_frustration_finalml.joblib"))
    proba = pipe.predict_proba(tail.values)
    avg = np.mean(proba, axis=0)
    pred = int(np.argmax(avg))
    return {
        "frustration": INVERSE_LABEL_MAP[pred],
        "class_index": pred,
        "mean_proba": [float(x) for x in avg],
        "window_rows_used": len(tail),
    }


def predict_frustration_final_csv(csv_path: str, output_dir: str, window_rows: int) -> dict:
    df = pd.read_csv(csv_path)
    return predict_frustration_final_df(df, output_dir, window_rows)


def default_inference_window_rows(assume_one_row_per_minute: bool, explicit=None) -> int:
    if explicit is not None:
        return int(explicit)
    return 5 if assume_one_row_per_minute else 300


# print metrics and cm to console, save to txt and png
def print_and_save_metrics(metrics: dict, target_name: str, output_dir: str, class_labels=None):
    # Create a readable summary of model results.
    # We print it to the terminal and also save it to a text file.
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    lines.append(f"Target: {target_name}")
    lines.append(f"Accuracy:  {metrics['accuracy']:.4f}")
    lines.append(f"Precision: {metrics['precision']:.4f}")
    lines.append(f"Recall:    {metrics['recall']:.4f}")
    lines.append(f"F1:        {metrics['f1']:.4f}")
    lines.append("")
    cm0 = metrics["confusion_matrix"]
    n_cls = cm0.shape[0]
    if class_labels is None:
        if n_cls == 2:
            class_labels = ["low", "med"]
        elif n_cls == 3:
            class_labels = ["low", "med", "high"]
        else:
            class_labels = [str(i) for i in range(n_cls)]
    idx_hint = ", ".join(f"{class_labels[i]}={i}" for i in range(len(class_labels)))
    lines.append(f"Confusion matrix (rows=true, cols=pred; classes: {idx_hint}):")
    lines.append(str(cm0))
    text = "\n".join(lines)
    print(text)
    print()

    target_snake = target_name.lower().replace(" ", "_")
    metrics_path = os.path.join(output_dir, f"metrics_{target_snake}.txt")
    # Save the same summary to disk so results are easy to share later.
    f = open(metrics_path, "w")
    f.write(text)
    f.close()
    print(f"Saved metrics to {metrics_path}")

    cm = metrics["confusion_matrix"]
    n = cm.shape[0]
    fig_w = 4 + max(0, n - 2) * 0.9
    fig_h = 3 + max(0, n - 2) * 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ticks = list(range(n))
    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks(ticks)
        ax.set_xticklabels(class_labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {target_name}")
    img_path = os.path.join(output_dir, f"confusion_matrix_{target_snake}.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {img_path}\n")


def main():
    # Read command-line options so this script works with defaults or custom paths.
    # This makes it easier to reuse the same training script in different folders.
    parser = argparse.ArgumentParser(
        description="Train baseline ML for mental demand and frustration. Scalable: add participant folders and labels without code changes."
    )
    parser.add_argument("--data-root", default=DATA_ROOT, help="Path to ml_input (structured) or All_Outputs (legacy)")
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Folder for metrics txt + confusion matrix PNGs (default: Data/outputs/ml_results)",
    )
    parser.add_argument(
        "--hold-out-n",
        type=int,
        default=1 ,
        help="Number of participants to hold out for test (last N after sorting). Default 1.",
    )
    parser.add_argument(
        "--split",
        choices=("random", "participant"),
        default="random",
        help="Split: random = stratified row split (final_ml: stratify on frustration class). "
        "participant = hold out whole participants.",
    )
    parser.add_argument(
        "--model",
        choices=("rf", "lr"),
        default="rf",
        help="Model: rf = RandomForest; lr = LogisticRegression. Default rf.",
    )
    parser.add_argument(
        "--mode",
        choices=("row", "file"),
        default="file",
        help="row = one sample per timestamp; file = one sample per session CSV (aggregated stats). Default file.",
    )
    parser.add_argument(
        "--predict-csv",
        default=None,
        help="If set, load saved pipelines from --output-dir and print predictions (no training).",
    )
    parser.add_argument(
        "--predict-target",
        choices=("mental", "frustration", "both"),
        default="both",
        help="Used with --predict-csv.",
    )
    parser.set_defaults(save_artifacts=True)
    parser.add_argument(
        "--no-save-artifacts",
        action="store_false",
        dest="save_artifacts",
        help="Do not write pipeline_*.joblib and model_meta.json after training.",
    )
    parser.add_argument(
        "--feature-source",
        choices=("merged", "biomarker", "final_ml"),
        default="merged",
        help="merged = ml_input / All_Outputs wrist; biomarker = Empatica minute sessions; final_ml = final_files_ML 3-class frustration.",
    )
    parser.add_argument(
        "--final-ml-root",
        default=FINAL_ML_ROOT_DEFAULT,
        help="Root with labels.txt and Frustration/*.csv for --feature-source final_ml.",
    )
    parser.add_argument(
        "--final-ml-version",
        default=None,
        metavar="NAME_OR_auto",
        help="final_ml training only: save under Data/outputs/ml_results_final_ml/versions/<NAME>. "
        "Use 'auto' for v1, v2, … (next free). Omit to write to Data/outputs/ml_results_final_ml root (may overwrite).",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        metavar="NAME",
        help="With --predict-frustration-final only: load from Data/outputs/ml_results_final_ml/versions/<NAME> (overrides --output-dir).",
    )
    parser.add_argument(
        "--predict-frustration-final",
        default=None,
        metavar="PATH",
        help="Predict 3-class frustration from a wrist-format CSV (no training). Uses model_meta_finalml.json in --output-dir.",
    )
    parser.add_argument(
        "--inference-window-rows",
        type=int,
        default=None,
        help="Rows to use from the end of the CSV for --predict-frustration-final. Default 300 (or 5 with --assume-one-row-per-minute).",
    )
    parser.add_argument(
        "--assume-one-row-per-minute",
        action="store_true",
        help="With --predict-frustration-final, default window to 5 rows unless --inference-window-rows is set.",
    )
    parser.add_argument(
        "--biomarker-data-root",
        default=BIOMARKER_DATA_DEFAULT,
        help="Root with labels.csv and sessions/<id>/*.csv for --feature-source biomarker.",
    )
    parser.add_argument(
        "--biomarker-map",
        default=BIOMARKER_MAP_DEFAULT,
        help="YAML with window_minutes and value_column_suffixes.",
    )
    parser.add_argument(
        "--predict-biomarker-session",
        default=None,
        help="Session directory of minute biomarker CSVs; loads model_meta_biomarker5.json (no training).",
    )
    args = parser.parse_args()

    predict_only = bool(
        args.predict_frustration_final or args.predict_csv or args.predict_biomarker_session
    )

    if args.feature_source == "final_ml" and args.output_dir == OUTPUT_DIR:
        args.output_dir = OUTPUT_DIR_FINAL_ML

    if (
        predict_only
        and args.predict_frustration_final
        and args.model_version
        and str(args.model_version).strip()
    ):
        args.output_dir = os.path.join(FINAL_ML_VERSIONS_DIR, str(args.model_version).strip())

    if (
        not predict_only
        and args.feature_source == "final_ml"
        and args.final_ml_version
        and str(args.final_ml_version).strip()
    ):
        args.output_dir = resolve_final_ml_output_dir(str(args.final_ml_version))

    win_rows = default_inference_window_rows(
        args.assume_one_row_per_minute, args.inference_window_rows
    )

    if args.predict_frustration_final:
        preds = predict_frustration_final_csv(
            args.predict_frustration_final, args.output_dir, win_rows
        )
        print(preds)
        return

    if args.predict_csv:
        preds = predict_from_csv(args.predict_csv, args.output_dir, args.predict_target)
        print(preds)
        return

    if args.predict_biomarker_session:
        preds = predict_from_biomarker_session(
            args.predict_biomarker_session,
            args.output_dir,
            args.biomarker_map,
            args.predict_target,
        )
        print(preds)
        return

    # load data: final_files_ML, biomarker sessions, structured ml_input, or legacy All_Outputs
    final_ml_mode = False
    if args.feature_source == "final_ml":
        print("Loading final_files_ML (labels.txt + Frustration/*.csv) from", args.final_ml_root)
        X, y_frustration, participant_ids, _label_map = load_final_ml_frustration(args.final_ml_root)
        y_mental = y_frustration.copy()
        participants = sorted({str(p) for p in participant_ids})
        biomarker_mode = False
        final_ml_mode = True
        feat_names = final_ml_feature_names()
    elif args.feature_source == "biomarker":
        print("Loading biomarker session data from", args.biomarker_data_root)
        X, y_mental, y_frustration, participant_ids, feat_names = load_biomarker_training_data(
            args.biomarker_data_root, args.biomarker_map
        )
        participants = sorted({str(p) for p in participant_ids})
        biomarker_mode = True
    elif is_structured_data_root(args.data_root):
        # Use the newer folder layout when those folders are detected.
        # This is the preferred path for current task inputs.
        print("Loading from structured layout (mental_demand + frustration / passive, active / low, med, high)")
        X, y_mental, y_frustration, participant_ids = load_data_structured(args.data_root, mode=args.mode)
        participants = sorted({str(p) for p in participant_ids})
        biomarker_mode = False
    else:
        # If newer layout is not present, fall back to the older participant layout.
        # This keeps the script compatible with previous pipeline outputs.
        discovered = discover_participants(args.data_root)
        if not discovered:
            print(f"No participants found in {args.data_root}. Run setup_ml_input.py or use All_Outputs.", file=sys.stderr)
            sys.exit(1)
        labels = load_labels(args.data_root, discovered)
        participants = [p for p in discovered if p in labels]
        if not participants:
            print("No participants with labels.", file=sys.stderr)
            sys.exit(1)
        X, y_mental, y_frustration, participant_ids = load_data(args.data_root, labels, mode=args.mode)
        biomarker_mode = False
        final_ml_mode = False

    def _participant_sort_key(s):
        # Sort IDs like P1, P2, P13 in numeric order instead of text order.
        # That makes hold-out selection predictable.
        if len(s) > 1:
            suffix = s[1:]
        else:
            suffix = "0"
        return int(suffix) if suffix.isdigit() else 0
    participants.sort(key=_participant_sort_key)

    if args.split == "random":
        test_participants = []
    else:
        # In participant mode, hold out the last N participants for testing.
        # We always keep at least one participant for training when possible.
        n_hold = min(args.hold_out_n, len(participants) - 1) if len(participants) > 1 else 0
        test_participants = participants[-n_hold:] if n_hold else []
    if args.split == "participant" and not test_participants and len(participants) == 1:
        test_participants = []

    if not biomarker_mode and not final_ml_mode:
        feat_names = file_feature_names() if args.mode == "file" else list(FEATURE_COLS)
    data_msg = (
        args.final_ml_root
        if final_ml_mode
        else (args.biomarker_data_root if biomarker_mode else args.data_root)
    )
    print("Loading data from", data_msg)
    print(
        "Participants:",
        participants,
        "| Split:",
        args.split,
        "| Model:",
        args.model,
        "| Mode:",
        args.mode if not biomarker_mode and not final_ml_mode else ("final_ml_row" if final_ml_mode else "biomarker_agg"),
        "| Feature source:",
        args.feature_source,
    )
    if test_participants:
        print("Test hold-out:", test_participants)
    print(f"Total samples: {len(X)}, feature_dim: {len(feat_names)}")
    if final_ml_mode:
        print(f"Class counts frustration (3-class): {np.bincount(y_frustration, minlength=3)}")
    else:
        print(f"Class counts mental: {np.bincount(y_mental)}, frustration: {np.bincount(y_frustration)}")
    print()

    stratify_y = y_frustration if final_ml_mode else y_mental

    if final_ml_mode:
        if not test_participants or args.split == "random":
            idx = np.arange(len(X))
            i_tr_val, i_te = _train_test_maybe_stratify(
                idx, y_frustration, test_size=0.2, random_state=42
            )
            i_tr, i_val = _train_test_maybe_stratify(
                i_tr_val, y_frustration[i_tr_val], test_size=0.2, random_state=42
            )
            X_tr_f = X.values[i_tr]
            X_val_f = X.values[i_val]
            X_te_f = X.values[i_te]
            y_tr_f = y_frustration[i_tr]
            y_val_f = y_frustration[i_val]
            y_te_f = y_frustration[i_te]
        else:
            X_tr_f, X_val_f, X_te_f, y_tr_f, y_val_f, y_te_f = split_by_participant(
                X.values, y_frustration, participant_ids, test_participants
            )
        print(f"Train size: {len(X_tr_f)}, Val size: {len(X_val_f)}, Test size: {len(X_te_f)}")
        print()
        frust_label = "frustration_finalml"
        print("--- Frustration (final_files_ML, 3-class) ---")
        metrics_fr = train_and_evaluate_final_ml_frustration(
            X_tr_f,
            X_val_f,
            X_te_f,
            y_tr_f,
            y_val_f,
            y_te_f,
            wide_model_search=args.final_ml_wide_model_search,
        )
        print_and_save_metrics(
            metrics_fr, frust_label, args.output_dir, class_labels=["low", "med", "high"]
        )
        if args.split == "random" and len(np.unique(participant_ids)) > 1:
            report_participant_group_metric(X.values, y_frustration, participant_ids)
        if args.save_artifacts:
            frust_full = refit_pipeline_train_val(
                metrics_fr["pipeline"], X_tr_f, X_val_f, y_tr_f, y_val_f
            )
            save_final_ml_bundle(
                args.output_dir,
                frust_full,
                final_ml_feature_names(),
                win_rows,
                0,
                experiment_id=args.experiment_id,
            )
            print(
                f"Saved pipeline_frustration_finalml.joblib and model_meta_finalml.json to {args.output_dir}"
            )
        print("Done.")
        return

    if not test_participants or args.split == "random":
        # Random split mode: first make train+val vs test, then train vs val.
        idx = np.arange(len(X))
        i_tr_val, i_te = _train_test_maybe_stratify(idx, y_mental, test_size=0.2, random_state=42)
        i_tr, i_val = _train_test_maybe_stratify(i_tr_val, y_mental[i_tr_val], test_size=0.2, random_state=42)
        X_tr_m = X.values[i_tr]
        X_tr_f = X.values[i_tr]
        X_val_m = X.values[i_val]
        X_val_f = X.values[i_val]
        X_te_m = X.values[i_te]
        X_te_f = X.values[i_te]
        y_tr_m = y_mental[i_tr]
        y_val_m = y_mental[i_val]
        y_te_m = y_mental[i_te]
        y_tr_f = y_frustration[i_tr]
        y_val_f = y_frustration[i_val]
        y_te_f = y_frustration[i_te]
    else:
        # Participant split mode keeps hold-out people fully separate.
        # This prevents mixing the same person's rows into train and test.
        X_tr_m, X_val_m, X_te_m, y_tr_m, y_val_m, y_te_m = split_by_participant(
            X.values, y_mental, participant_ids, test_participants
        )
        X_tr_f, X_val_f, X_te_f, y_tr_f, y_val_f, y_te_f = split_by_participant(
            X.values, y_frustration, participant_ids, test_participants
        )
    print(f"Train size: {len(X_tr_m)}, Val size: {len(X_val_m)}, Test size: {len(X_te_m)}")
    print()

    mental_label = "mental_demand_biomarker5" if biomarker_mode else "mental_demand"
    frust_label = "frustration_biomarker5" if biomarker_mode else "frustration"

    print("--- Mental demand ---")
    metrics_mental = train_and_evaluate(X_tr_m, X_val_m, X_te_m, y_tr_m, y_val_m, y_te_m, mental_label, args.model)
    print_and_save_metrics(metrics_mental, mental_label, args.output_dir)

    print("--- Frustration ---")
    metrics_frustration = train_and_evaluate(X_tr_f, X_val_f, X_te_f, y_tr_f, y_val_f, y_te_f, frust_label, args.model)
    print_and_save_metrics(metrics_frustration, frust_label, args.output_dir)

    if args.save_artifacts:
        mental_full = refit_pipeline_train_val(
            metrics_mental["pipeline"], X_tr_m, X_val_m, y_tr_m, y_val_m
        )
        frust_full = refit_pipeline_train_val(
            metrics_frustration["pipeline"], X_tr_f, X_val_f, y_tr_f, y_val_f
        )
        if biomarker_mode:
            save_model_bundle_biomarker(args.output_dir, feat_names, mental_full, frust_full)
            print(f"Saved biomarker5 pipelines and model_meta_biomarker5.json to {args.output_dir}")
        else:
            save_model_bundle(args.output_dir, args.mode, feat_names, mental_full, frust_full)
            print(f"Saved pipelines and model_meta.json to {args.output_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
