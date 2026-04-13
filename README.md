# VIP — Wrist frustration ML & Empatica pipelines

Research codebase for **3-class frustration** models from wrist biosignals, **Empatica-style minute CSVs**, optional **Kafka** streaming, and Unity/HTTP demos.

## Repository layout

| Path | Purpose |
|------|---------|
| **`train.py`** | Main ML entry: merged/biomarker/`final_ml` training, metrics, artifacts, CSV prediction APIs. |
| **`train_final_ml_model.py`** | Thin wrapper: runs `train.py` with `--feature-source final_ml`. |
| **`ml/`** | `final_files_ml_loader.py` (training data), `final_models.py` (extra estimators / wide search), `training_utils.py` (seeds, CV helpers). |
| **`empatica_minute_filter_predict.py`** | Merge + filter + optional normalize + **one-shot** `final_ml` prediction for minute-wide CSV folders. |
| **`wrist_minute_preprocess.py`** | Minimal merge of `eda.csv` / `temperature.csv` / `pulse-rate.csv` → model column names. |
| **`predict_from_emphatica_minute_csvs.py`** | Merge + predict CLI (older sibling of `empatica_minute_filter_predict.py`). |
| **`prediction_server.py`**, **`poll_wrist_frustration.py`** | HTTP buffer inference; periodic folder poll. |
| **`emphatica/`** | S3 / biomarker ingest helpers (YAML-driven minute sessions). |
| **`streaming/kafka/`** | Example **Kafka** producer + windowed consumer calling the same predictor as batch CSV. |
| **`examples/empatica_minute/`** | Unity client + TA poller (**no CSVs**; sample CSVs live under `Data/examples/empatica_minute/`). |
| **`tools/`** | Legacy **Avro → spreadsheet** and **normalize/merge** research scripts (paths fixed to repo `Data/`). |
| **`docs/`** | `ML_GUIDE.md`, `DATAFLOW.md`, `DATA_REPOSITORY_LAYOUT.md`, `how_to_run.txt`. |
| **`scripts/`** | `git-add-code.sh` — stage only code folders (excludes `Data/`, `legacy/`). |
| **`Data/`** | **All local data** (CSV, PDF, Avro exports, `final_files_ML/`, `outputs/` joblibs). **Gitignored** — see `docs/DATA_REPOSITORY_LAYOUT.md`. |
| **`legacy/`** | Scratch copies (gitignored). |

## Quick start

```bash
cd /path/to/VIP
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train frustration model (needs Data/final_files_ML/ on disk — not in git)
python train_final_ml_model.py --final-ml-version auto

# Predict from bundled sample minute CSVs (under Data/)
python empatica_minute_filter_predict.py --dir Data/examples/empatica_minute --json-out
```

Optional extras (not required for core flows):

```bash
pip install kafka-python xgboost lightgbm
```

## ML practices in this repo

- **Stratified splits** for row-level `final_ml` evaluation; participant hold-out supported via `--split participant`.
- **Artifact versioning**: `--final-ml-version auto|name` writes under `Data/outputs/ml_results_final_ml/versions/`.
- **Wide model search** (`--final-ml-wide-model-search`): benchmarks ExtraTrees, RF, calibrated LinearSVC, small MLP, and optional **XGBoost / LightGBM** (if installed); selects the best on the **validation** split before the usual voting / HGB-grid fallbacks.
- **Traceability**: `--experiment-id` is copied into `model_meta_finalml.json`.
- **Reproducibility helpers**: `ml/training_utils.py` (`set_global_seed`, stratified K-fold index builder).

## Data flow (summary)

**Wristband → cloud → CSV download → model (5-minute style window)**

1. Device streams to vendor cloud; you export **per-minute** CSVs (or receive them via API).
2. Local scripts merge on `timestamp_unix`, optionally mask `missing_value_reason`, optionally resample.
3. `predict_frustration_final_df` loads `pipeline_frustration_finalml.joblib` + `model_meta_finalml.json`, builds features, averages **tail** `predict_proba`.

**Kafka variant:** see `docs/DATAFLOW.md` and `streaming/README.md`.

## Documentation

- **`docs/ML_GUIDE.md`** — CLI reference, split behavior, inference window semantics.
- **`docs/DATAFLOW.md`** — diagrams and stage-by-stage flow.
- **`docs/DATA_REPOSITORY_LAYOUT.md`** — what belongs under `Data/` locally.
- **`docs/how_to_run.txt`** — copy-paste command cheatsheet.

## Unity / TA client

Example C# client: `examples/empatica_minute/EmpaticaLatestClient.cs` (calls HTTP prediction server; adjust host/ports for your deployment).

