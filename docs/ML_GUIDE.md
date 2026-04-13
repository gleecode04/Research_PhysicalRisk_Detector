# ML training and inference — detailed guide

**Audience:** Engineers new to this repository who need to train models, run batch inference, or wire **near–real-time** predictions into an application.

**Entry point:** [`train.py`](train.py) (project root: `VIP/`).

---

## Table of contents

1. [What this system does](#1-what-this-system-does)
2. [Repository map](#2-repository-map)
3. [How models are saved (retrieval format)](#3-how-models-are-saved-retrieval-format)
4. [Feature sources at a glance](#4-feature-sources-at-a-glance)
5. [Data formats](#5-data-formats)
6. [Installation and environment](#6-installation-and-environment)
7. [Training workflows](#7-training-workflows)
8. [`train.py` command-line reference](#8-trainpy-command-line-reference)
9. [What happens during training (internals)](#9-what-happens-during-training-internals)
10. [Evaluation: splits and how to read the numbers](#10-evaluation-splits-and-how-to-read-the-numbers)
11. [Artifacts written to disk](#11-artifacts-written-to-disk)
12. [Inference: batch, CLI, and real-time patterns](#12-inference-batch-cli-and-real-time-patterns)
13. [Programmatic examples](#13-programmatic-examples)
14. [Troubleshooting](#14-troubleshooting)
15. [FAQ](#15-faq)

---

## 1. What this system does

The VIP project trains **scikit-learn** classifiers to estimate **mental demand** and/or **frustration** from wearable sensor-style data. Depending on the **feature source**, the inputs may be:

- Raw wrist columns per row (`eda_eda`, `bvp_bvp`, `temperature_temp`), or  
- Per-session **aggregated** statistics (mean, std, min, max, median per channel), or  
- **Empatica minute biomarker** features produced under `emphatica/`.

The most recently added path, **`final_files_ML`** (under **`Data/final_files_ML/`** on disk), trains a **3-class frustration** model (`low`, `med`, `high`) from `labels.txt` plus CSVs under `Frustration/`.

---

## 2. Repository map

| Path | Role |
|------|------|
| [`train.py`](train.py) | Single script: load data, train, save `.joblib` + JSON, run prediction CLIs. |
| [`ml/final_files_ml_loader.py`](../ml/final_files_ml_loader.py) | Parses `labels.txt`, loads `Frustration/*.csv`, joins label by participant ID from **filename** (`P` + digits). |
| [`emphatica/ingest_minute_biomarkers.py`](emphatica/ingest_minute_biomarkers.py) | Builds aggregate feature rows from minute biomarker CSV directories (biomarker training path). |
| [`Data/final_files_ML/labels.txt`](../Data/final_files_ML/labels.txt) | **Source of truth** for frustration class per participant (`final_ml` mode). |
| [`Data/final_files_ML/Frustration/*.csv`](../Data/final_files_ML/Frustration) | Session/time-series CSVs; class comes from `labels.txt`, **not** from words in the filename. |
| `Data/ml_input/` | Typical root for `--feature-source merged` (structured mental_demand / frustration layout). |
| `Data/outputs/ml_results/` | Default output for merged + biomarker artifacts. |
| `Data/outputs/ml_results_final_ml/` | Default output for `final_ml` when you do not override `--output-dir`. |

---

## 3. How models are saved (retrieval format)

### Joblib (not a hand-rolled pickle file)

Trained estimators are persisted with **`joblib.dump(..., "*.joblib")`**. Joblib is the standard sklearn ecosystem choice: it uses **pickle** internally (with optimizations for large numpy arrays). You retrieve the model with:

```python
import joblib
model = joblib.load("Data/outputs/ml_results_final_ml/pipeline_frustration_finalml.joblib")
```

### Why not name it `.pkl`?

A `.joblib` file **is** retrievable the same way as pickle; the extension is convention. You could pickle manually with `pickle.dump`, but joblib is what `train.py` uses and what you should use to load.

### Version compatibility

Pickles are **not** stable across arbitrary Python or sklearn upgrades. For reproducible deployment:

- Record **Python version** and **`scikit-learn` version** used at train time.
- Use the same versions in production if you see `UnpicklingError` or attribute errors after upgrades.

### Metadata JSON

Each pipeline mode writes a companion **JSON** file listing feature column names, class names, and inference notes. Your inference code should read **`feature_cols`** from JSON so you do not hard-code column order.

---

## 4. Feature sources at a glance

| `--feature-source` | Labels | Typical inputs | Default `--output-dir` behavior |
|--------------------|--------|----------------|----------------------------------|
| `merged` | `labels.csv` or structured folders | Wrist CSVs under `Data/ml_input` or legacy `All_Outputs` | `Data/outputs/ml_results/` |
| `biomarker` | `labels.csv` under biomarker root | Minute CSVs per session | `Data/outputs/ml_results/` |
| `final_ml` | `labels.txt` (markdown table) | `Frustration/*.csv` | If `--output-dir` is still the default merged path, it is **automatically switched** to `Data/outputs/ml_results_final_ml/` to avoid overwriting merged artifacts. |

---

## 5. Data formats

### 5.1 `final_files_ML` — `labels.txt`

- Markdown **pipe table** with columns including **Participant** and **Level**.
- Parser strips `|`, skips header and separator lines.
- **Level** is mapped case-insensitively to classes: **Low → 0**, **Medium/Med → 1**, **High → 2**.
- **Only this file defines the class** for each participant. Filenames like `P6_Paasive Medium.csv` are **ignored** for labeling (typos in filenames do not change `y`).

Participants listed in `labels.txt` but without a usable CSV are skipped. CSVs without a matching participant row are skipped with a warning.

### 5.2 `final_files_ML` — `Frustration/*.csv`

Each file must include these **numeric** columns (names must match exactly):

- `eda_eda`
- `bvp_bvp`
- `temperature_temp`

`timestamp_local` is allowed and ignored for training. Rows are loaded at **native rate** (often ~1 Hz). Every row in a file receives the **same** frustration label for that participant.

**Training features (18 columns):** For each file, the loader keeps the three raw channels and adds **rolling** (causal) **mean, std, min, max, median** per channel over the last **300** rows of that file (`min_periods=1` at the start of the file). See `ml/final_files_ml_loader.py` (`add_rolling_features`, `ROLLING_WINDOW`). Inference builds the same columns by applying rolling from the **start of the CSV** (or full session buffer), then taking the last *N* rows for `predict_proba` averaging.

**Participant ID** is taken from the **start of the basename**: `^P(\d+)` (e.g. `P13_Active_Medium.csv` → `P13`). This avoids fragile reliance on “word boundaries” after digits when the next character is `_` (underscore is a “word” character in Python regex).

### 5.3 Merged / structured layout (`--feature-source merged`)

Two layouts are supported:

- **Structured:** `mental_demand/` and `frustration/` trees with condition and level folders (see `load_data_structured` in `train.py`).
- **Legacy:** participant folders with `Task2_merged/.../combined_same_row.csv`.

Feature columns are still the same three wrist names unless you are in biomarker mode.

### 5.4 Biomarker layout (`--feature-source biomarker`)

- Root contains `labels.csv` with `session_id`, `mental`, `frustration`.
- Under `sessions/<session_id>/`, minute CSVs are aggregated using [`emphatica/ingest_minute_biomarkers.py`](emphatica/ingest_minute_biomarkers.py) and the YAML map (`--biomarker-map`).

---

## 6. Installation and environment

From the directory that contains `train.py`:

```bash
pip install pandas numpy scikit-learn joblib matplotlib
```

For biomarker training/prediction:

```bash
pip install pyyaml
```

Use a **virtual environment** and pin versions for anything you ship to production.

---

## 7. Training workflows

All examples assume:

```bash
cd /path/to/VIP   # directory containing train.py
```

### 7.1 Three-class frustration (`final_ml`) — primary path for `final_files_ML`

```bash
python train.py --feature-source final_ml
```

Common variants:

```bash
# Custom data root
python train.py --feature-source final_ml --final-ml-root /data/my_final_ml   # e.g. Data/final_files_ML

# Custom artifact directory (explicit override)
python train.py --feature-source final_ml --output-dir ./artifacts/final_ml_run1

# Train metrics only, no model files
python train.py --feature-source final_ml --no-save-artifacts

# Participant holdout (last N participants by numeric sort, after split setup in script)
python train.py --feature-source final_ml --split participant --hold-out-n 1
```

### 7.2 Merged wrist data

```bash
python train.py --feature-source merged --data-root Data/ml_input --mode file
```

- `--mode row`: one training row per sensor timestamp.  
- `--mode file`: one row per session file (aggregated mean/std/min/max/median per channel).

### 7.3 Biomarker sessions

```bash
python train.py --feature-source biomarker \
  --biomarker-data-root Data/biomarker_input \
  --biomarker-map emphatica/biomarker_column_map.yaml
```

---

## 8. `train.py` command-line reference

| Argument | Used when | Description |
|----------|-----------|-------------|
| `--data-root` | `merged` | Structured `ml_input` or legacy `All_Outputs` root. |
| `--output-dir` | Always | Metrics, confusion matrices, and model artifacts. |
| `--split` | Training | `random` (default): stratified row splits. `participant`: hold out entire participants. |
| `--hold-out-n` | `split=participant` | Number of participants held out for test (sorted order). |
| `--model` | `merged` / biomarker (not `final_ml` frustration path) | `rf` or `lr`. |
| `--mode` | `merged` | `row` or `file` aggregation. |
| `--save-artifacts` | Default on | Saving pipelines is default; use `--no-save-artifacts` to disable. |
| `--feature-source` | Training | `merged`, `biomarker`, or `final_ml`. |
| `--final-ml-root` | `final_ml` | Folder with `labels.txt` and `Frustration/`. Default: `./Data/final_files_ML`. |
| `--final-ml-version` | `final_ml` training | Save under `Data/outputs/ml_results_final_ml/versions/<name>`; use `auto` for `v1`, `v2`, … |
| `--final-ml-wide-model-search` | `final_ml` training | Try several sklearn (and optional XGB/LGBM) models; pick best on **validation** before fallbacks. |
| `--experiment-id` | `final_ml` training | Optional string stored in `model_meta_finalml.json`. |
| `--model-version` | `--predict-frustration-final` | Load from `Data/outputs/ml_results_final_ml/versions/<name>` (overrides `--output-dir`). |
| `--biomarker-data-root` | `biomarker` | Biomarker dataset root. |
| `--biomarker-map` | `biomarker` | YAML for minute ingestion. |
| `--predict-frustration-final PATH` | Inference only | 3-class frustration on a wrist CSV; requires `model_meta_finalml.json` in `--output-dir`. |
| `--inference-window-rows` | With predict-frustration-final | Overrides default window length. |
| `--assume-one-row-per-minute` | With predict-frustration-final | Default window becomes 5 rows if `--inference-window-rows` unset. |
| `--predict-csv PATH` | Inference only | Merged pipelines + `model_meta.json`. |
| `--predict-target` | With predict flags | `mental`, `frustration`, or `both` (where applicable). |
| `--predict-biomarker-session DIR` | Inference only | Biomarker session directory. |

**Order of operations:** If you pass a prediction flag (`--predict-frustration-final`, `--predict-csv`, or `--predict-biomarker-session`), the script **exits after predicting** and does not train.

---

## 9. What happens during training (internals)

### 9.1 `final_ml` path

1. **`load_final_ml_frustration`** (`ml/final_files_ml_loader.py`) builds a row-level `X` with 3 columns and integer `y` in `{0,1,2}`, plus `participant_ids` per row.
2. **Split:**  
   - **Random:** nested `train_test_split` with stratification on `y_frustration` (falls back to unstratified if sklearn rejects stratify).  
   - **Participant:** rows for held-out participants go to test; remaining rows split train/val with stratified `train_test_split`.
3. **Model selection** (`train_and_evaluate_final_ml_frustration` in `train.py`):  
   - Fit **HistGradientBoostingClassifier** (fixed hyperparameters).  
   - If test accuracy **&lt; 0.9**, fit a **soft VotingClassifier** (HistGradientBoosting + RandomForest + GradientBoosting) and keep it if accuracy improves.  
   - If still **&lt; 0.9**, run a **small grid** over HistGradientBoosting (`max_depth`, `learning_rate`, `max_iter`).  
   - If still **&lt; 0.9**, print a stderr note suggesting more data or participant evaluation.
4. **Validation arrays** (`X_val`, `y_val`) are passed in for API symmetry; the current selection logic is driven by **test** performance (not early stopping on val).
5. **Refit:** The chosen estimator is **cloned** and refit on **train + val** before saving (`refit_pipeline_train_val`).
6. **Secondary metric (random split only):** One **`GroupShuffleSplit`** by `participant_ids` trains a quick HGB on the train slice and reports approximate accuracy on the held-out group slice. This is **not** the same as your primary test metric; it hints at **cross-person** difficulty.

### 9.2 `merged` / `biomarker` paths

- Grid search over RandomForest or LogisticRegression inside a `Pipeline` with `StandardScaler` (see `train_and_evaluate` and `build_estimator_pipeline` in `train.py`).
- Two targets: mental demand and frustration (except `final_ml`, which only trains frustration).

---

## 10. Evaluation splits and how to read the numbers

### Stratified random rows (default for `final_ml`)

- Many rows from the same participant appear in **both** train and test.  
- Accuracy can be **high** because the model may exploit person-specific sensor baselines.  
- This is useful for checking that features carry signal, but it **overstates** how well you will generalize to **new** wearers.

### Participant split (`--split participant`)

- All rows of held-out participants are in the test set.  
- Usually **much harder**; often closer to real deployment if new users are not in the training set.

### Secondary group metric

- Printed as: `Secondary metric (one participant-group shuffle split, HGB): test accuracy ≈ ...`  
- Single random split; treat as a **rough** indicator, not a formal benchmark.

---

## 11. Artifacts written to disk

### `final_ml` (default dir `Data/outputs/ml_results_final_ml/`)

| File | Purpose |
|------|---------|
| `pipeline_frustration_finalml.joblib` | Serialized trained model (HGB or VotingClassifier). |
| `model_meta_finalml.json` | `feature_cols`, `label_classes`, `n_classes`, default window hint, human-readable `inference_note`. |
| `metrics_frustration_finalml.txt` | Accuracy, macro P/R/F1, confusion matrix text. |
| `confusion_matrix_frustration_finalml.png` | Heatmap (seaborn if installed). |

### `merged` (default `Data/outputs/ml_results/`)

| File | Purpose |
|------|---------|
| `pipeline_mental_demand.joblib`, `pipeline_frustration.joblib` | Models. |
| `model_meta.json` | Includes `mode` (`row` or `file`) and feature name list. |

### `biomarker`

| File | Purpose |
|------|---------|
| `pipeline_mental_demand_biomarker5.joblib`, `pipeline_frustration_biomarker5.joblib` | Models. |
| `model_meta_biomarker5.json` | Biomarker feature list and class names. |

---

## 12. Inference: batch, CLI, and real-time patterns

### 12.1 Idea behind “real-time” here

The `final_ml` model was trained on **per-row** sensor vectors (3 features). At inference time the script does **not** re-aggregate the window into mean/std features; instead it:

1. Takes the last **N** rows from the CSV (or buffer).  
2. Calls **`predict_proba`** on **each** row in that window.  
3. **Averages** the probability vectors across rows.  
4. Takes **`argmax`** of that average → predicted class.

That matches the idea of a **stable label over a short recent interval** while staying in the same feature space as training.

### 12.2 Choosing N (window length)

| Sampling | Approximate meaning of N = 300 | Approximate meaning of N = 5 |
|----------|----------------------------------|------------------------------|
| ~1 row / second | ~5 minutes | ~5 seconds |
| ~1 row / minute | ~5 hours | ~5 minutes |

Defaults:

- **300** rows if you do not pass flags (assumes dense streaming like 1 Hz).  
- **5** rows if you pass **`--assume-one-row-per-minute`** and do not set `--inference-window-rows`.

Override explicitly:

```bash
python train.py --predict-frustration-final live_buffer.csv \
  --output-dir Data/outputs/ml_results_final_ml \
  --inference-window-rows 600
```

### 12.3 CLI batch prediction

```bash
python train.py \
  --predict-frustration-final path/to/file.csv \
  --output-dir Data/outputs/ml_results_final_ml
```

Output is a Python dict printed to stdout, e.g.:

- `frustration`: `"low"`, `"med"`, or `"high"`  
- `class_index`: `0`, `1`, or `2`  
- `mean_proba`: length-3 list of averaged probabilities  
- `window_rows_used`: can be &lt; N if the file has fewer than N rows (minimum 1 row is used)

### 12.4 Real-time integration loop (conceptual)

```
Load joblib + JSON once at startup.
Every tick (e.g. 1 s or 60 s):
  Append newest sample to ring buffer or CSV.
  tail = buffer.last_N_rows(feature_cols)
  P = model.predict_proba(tail)
  p_mean = mean(P, axis=0)
  label = argmax(p_mean)
  Emit label + optional p_mean to UI / API.
```

**Performance tips:**

- **Do not** `joblib.load` on every tick; load once.  
- Prefer a **ring buffer** of numpy arrays over rewriting a huge CSV every second.  
- If you only need updates every minute, set **N** and tick interval accordingly.

### 12.5 Other inference entry points

- **Merged:** `--predict-csv` + `model_meta.json` (behavior depends on `mode`).  
- **Biomarker:** `--predict-biomarker-session` + `model_meta_biomarker5.json`.

---

## 13. Programmatic examples

### 13.1 Load `final_ml` model and score a DataFrame

```python
import json
import joblib
import numpy as np
import pandas as pd

OUT = "Data/outputs/ml_results_final_ml"

with open(f"{OUT}/model_meta_finalml.json") as f:
    meta = json.load(f)

cols = meta["feature_cols"]
clf = joblib.load(f"{OUT}/pipeline_frustration_finalml.joblib")

def predict_window(df: pd.DataFrame, n_rows: int = 300) -> dict:
    tail = df[cols].astype(float).tail(max(1, n_rows))
    proba = clf.predict_proba(tail.values)
    avg = np.mean(proba, axis=0)
    pred = int(np.argmax(avg))
    names = meta["label_classes"]
    return {
        "frustration": names[pred],
        "class_index": pred,
        "mean_proba": avg.tolist(),
        "window_rows_used": len(tail),
    }
```

### 13.2 Optional: expose as HTTP microservice

A minimal pattern: **FastAPI** or **Flask** endpoint that accepts the last *N* rows as JSON rows or multipart CSV, runs `predict_window`, returns JSON. Ensure you validate column names against `meta["feature_cols"]` to avoid silent misalignment.

---

## 14. Troubleshooting

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `No rows loaded from final_files_ML/Frustration` (under `Data/final_files_ML/`) | No CSVs matched labels + required columns | Fix CSV columns; ensure `P<id>` at start of filename; check `labels.txt`. |
| Warnings: `missing ['eda_eda', ...]` | File lacks required feature columns | Fix export pipeline or drop broken sessions. |
| `Missing model_meta_finalml.json` | Wrong `--output-dir` | Point to the folder where training saved artifacts. |
| `ValueError: stratify` / split errors | Too few samples per class in a split | Use `--split random` with more data, or adjust holdout. |
| `UnpicklingError` / sklearn errors on load | Version mismatch | Match train-time Python/sklearn. |
| Great row-split accuracy, terrible secondary metric | Person-specific signal / leakage across rows | Report participant split metrics; collect more balanced participants. |

---

## 15. FAQ

**Is the saved model a pickle?**  
It is a **joblib** file, which uses pickling. Load with `joblib.load`.

**Can I use one CSV row for prediction?**  
Yes. The code uses `tail(1)` if that is all you have; probabilities are still averaged (over one row).

**Does filename sentiment (`_Low_`, `_Medium_`) affect training?**  
**No** for `final_ml`. Only `labels.txt` sets the class.

**Why two different default output folders?**  
So `final_ml` runs do not overwrite `pipeline_frustration.joblib` from merged training unless you explicitly set `--output-dir`.

**Where is the exact prediction logic?**  
See **`predict_frustration_final_df`** / **`predict_frustration_final_csv`** in [`train.py`](train.py).

---

## 16. Empatica minute CSVs → frustration (TA workflow)

Use minute CSVs from [`Data/examples/empatica_minute/`](../Data/examples/empatica_minute/) (`eda.csv`, `temperature.csv`, `pulse-rate.csv` with `timestamp_unix`, `eda_scl_usiemens`, etc.). Unity / poller **code** lives under [`examples/empatica_minute/`](../examples/empatica_minute/).

### Run in this order (one-time vs loop)

1. **Train the model** (if you do not already have artifacts):
   ```bash
   cd /path/to/VIP
   python train.py --feature-source final_ml
   ```
   Produces `Data/outputs/ml_results_final_ml/pipeline_frustration_finalml.joblib` and `model_meta_finalml.json`.

   **Versioned runs** (keeps each iteration separate): add `--final-ml-version auto` (creates `Data/outputs/ml_results_final_ml/versions/v1`, `v2`, …) or `--final-ml-version my_run` (writes under `Data/outputs/ml_results_final_ml/versions/my_run/`).

2. **One-shot prediction from three minute CSVs** (merge + score):
   ```bash
   cd /path/to/VIP
   python predict_from_emphatica_minute_csvs.py --dir Data/examples/empatica_minute --output-dir Data/outputs/ml_results_final_ml
   ```
   For a versioned model: `--model-version v1` (same as `--output-dir Data/outputs/ml_results_final_ml/versions/v1`).
   Or pass explicit paths: `--eda ... --temperature ... --pulse ...`.

   - Default scoring uses the **last 5 rows** (≈5 minutes at 1 row/min). For **1 Hz** merged wrist data instead, add **`--hz-wrist-window`** (uses 300 rows) or set **`--inference-window-rows N`**.
   - **`--save-merged PATH`** writes the merged `eda_eda` / `bvp_bvp` / `temperature_temp` table for debugging.
   - **Note:** TA `pulse_rate_bpm` is mapped to **`bvp_bvp`** as a stand-in (not true BVP). Empty numeric cells (e.g. `device_not_worn_correctly`) yield **NaN** after merge; prediction will **fail** until at least some rows have real numbers.

3. **Optional: poll a folder every 5 minutes** (same three filenames):
   ```bash
   export WRIST_MINUTE_DIR=/path/to/folder_with_csvs
   export VIP_MODEL_DIR=/path/to/VIP/Data/outputs/ml_results_final_ml
   python poll_wrist_frustration.py --dir "$WRIST_MINUTE_DIR" --output-dir "$VIP_MODEL_DIR" --interval 300
   ```
   Add **`--once`** for a single tick. Use **`--model-version v1`** or **`export VIP_MODEL_VERSION=v1`** to point at `Data/outputs/ml_results_final_ml/versions/v1`.

4. **Optional: Unity / HTTP** — run biomarker poller (TA script uses env for AWS keys) and prediction server **in separate terminals**:
   ```bash
   # Terminal A — after setting AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
   python examples/empatica_minute/avro_stream_poller.py
   # Terminal B
   export VIP_MODEL_DIR=/path/to/VIP/Data/outputs/ml_results_final_ml
   export VIP_MODEL_VERSION=v1   # optional: loads .../versions/v1
   python prediction_server.py
   ```
   Unity [`EmpaticaLatestClient.cs`](../examples/empatica_minute/EmpaticaLatestClient.cs) calls **`GET http://<host>:7000/latest`** and **`POST http://<host>:5000/predict`** with JSON `{ "eda", "bvp", "temp" }`. The server keeps an **in-memory buffer** of recent rows; set **`PREDICT_WINDOW_ROWS`** (default 5) and **`FLASK_PORT`** as needed. If the saved model uses **raw 3-column** features (default training), inference does not add rolling columns; older rolling-augmented checkpoints still work via `model_meta_finalml.json`.

### Supporting modules

| File | Role |
|------|------|
| [`wrist_minute_preprocess.py`](wrist_minute_preprocess.py) | Merge three TA-style CSVs on `timestamp_unix` → `eda_eda`, `bvp_bvp`, `temperature_temp`. |
| [`predict_from_emphatica_minute_csvs.py`](predict_from_emphatica_minute_csvs.py) | CLI: merge + `predict_frustration_final_df`. |
| [`poll_wrist_frustration.py`](poll_wrist_frustration.py) | Periodic merge + predict. |
| [`prediction_server.py`](prediction_server.py) | Flask **`/predict`** + **`/health`** for Unity. |

---

*This document describes the repository as implemented in `train.py` and `ml/final_files_ml_loader.py`. If behavior diverges after refactors, prefer the source and `--help` output.*
