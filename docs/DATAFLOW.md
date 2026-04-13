# Data flow: wristband → cloud → CSV / Kafka → ML (5-minute analysis)

This document complements the root **`README.md`** with an end-to-end picture.

## 1. Physical path (Empatica / E4 style)

```text
Wristband (EDA, BVP/IBI proxy, temperature)
    → Vendor cloud (Empatica / research gateway)
        → Aggregated **per-minute** biomarkers (CSV export or API)
            → Your laptop / server
                → Merge + filter (`empatica_minute_filter_predict.py` or `wrist_minute_preprocess.py`)
                    → `predict_frustration_final_df` (`train.py`) using saved `final_ml` weights
```

Typical minute CSV columns: `timestamp_unix` (ms), `eda_scl_usiemens`, `temperature_celsius`, `pulse_rate_bpm`, optional `missing_value_reason`.

## 2. Training path (offline)

```text
Data/final_files_ML/labels.txt + Frustration/*.csv
    → ml/final_files_ml_loader.py
        → train.py --feature-source final_ml
            → Data/outputs/ml_results_final_ml/ (pipelines + model_meta_finalml.json)
```

Shortcut entry: `python train_final_ml_model.py` (same as adding `--feature-source final_ml`).

## 3. Kafka path (near real-time)

Assume a **gateway** or cloud function publishes one JSON object per minute (or per fused sample) to a topic.

```text
Wristband → Cloud → Kafka topic ``wrist.minute_biomarkers``
    → streaming/kafka/consumer_windowed_predict.py
        → in-memory buffer (last N rows)
            → every ANALYSIS_INTERVAL_SEC (default 300s)
                → frustration label + probabilities (stdout JSON)
```

Optional: land messages to **object storage** as CSV for audit, then reuse the same batch prediction scripts.

## 4. Unity / HTTP path

Unity posts single rows to **`prediction_server.py`** (`POST /predict`), which maintains a buffer and scores with the same tail-window logic as CSV inference.

## 5. Where each component lives

| Stage | Repo location |
|-------|----------------|
| Train `final_ml` | `train.py`, `train_final_ml_model.py`, `ml/final_files_ml_loader.py` (data under `Data/final_files_ML/`) |
| Minute merge (simple) | `wrist_minute_preprocess.py` |
| Minute merge + filter + predict CLI | `empatica_minute_filter_predict.py` |
| Kafka consumer | `streaming/kafka/consumer_windowed_predict.py` |
| Legacy Avro → Excel | `tools/avro_to_csv_example_script.py` |
| Legacy normalize / resample | `tools/normalize_and_merge.py` |

## 6. Five-minute analysis

- **Batch / folder:** run `empatica_minute_filter_predict.py` with default `--interval 300` (poll mode) or cron every 5 minutes.
- **Kafka:** set `ANALYSIS_INTERVAL_SEC=300` for `consumer_windowed_predict.py`.
- **HTTP:** `prediction_server.py` uses the last `PREDICT_WINDOW_ROWS` rows (often 5 for minute cadence).

The model scores a **tail window** of rows and averages `predict_proba` over that window (see `docs/ML_GUIDE.md` §12).
