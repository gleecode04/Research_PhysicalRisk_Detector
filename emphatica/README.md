# Empatica data retrieval (S3)

Empatica study data lives in **Amazon S3**. Keys and URLs come from the access credentials / URL the platform gives you (*Data Access and Retrieval* in Empatica docs, AWS SDK or CLI).

## Read-only policy

- Do **not** overwrite downloaded `aggregated_per_minute` CSVs in place when merging or feature-building.
- Keep canonical lab training data under `Data/ml_input` unchanged; write derived outputs to new paths.

## Setup

```bash
pip install boto3 pyyaml
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export S3_ACCESS_URL="s3://your-bucket/your/prefix/"
export LOCAL_PATH="$HOME/empatica_minute_csv"
```

## Minute digital biomarkers (not raw Avro)

Point at **`.../digital_biomarkers/aggregated_per_minute/`** or use:

```bash
python emphatica/fetch_s3.py --biomarkers-only --max-keys 200
python emphatica/fetch_s3.py --download --biomarkers-only --skip-raw-data --max-keys 500
```

`--biomarkers-only` appends `digital_biomarkers/aggregated_per_minute/` when missing.  
`--skip-raw-data` skips keys under `/raw_data/`.

## Merge + 5-minute window (local CSVs)

```bash
python emphatica/ingest_minute_biomarkers.py /path/to/session_folder --window 5 --mode agg --out /tmp/features.csv
```

Session folder = all `*.csv` for one participant window. Column allowlist: **`biomarker_column_map.yaml`**.

## Near–real-time poll (5 min)

```bash
export POLL_INTERVAL_SEC=300
export BIOMARKER_SESSION=/path/to/sessions/current_id
# optional: export RUN_FETCH=1
python emphatica/poll_inference.py
```

Trains/runs **`train.py --predict-biomarker-session`** each tick (requires biomarker models saved first).

## ML integration

| Command | Purpose |
|---------|---------|
| `python train.py --feature-source merged` | Default: `Data/ml_input` wrist merged features |
| `python train.py --feature-source biomarker` | Train on `Data/biomarker_input` (see `Data/biomarker_input/README.txt`) |
| `python train.py --predict-biomarker-session DIR` | Predict one session using `model_meta_biomarker5.json` |

Biomarker artifacts: `pipeline_*_biomarker5.joblib`, `model_meta_biomarker5.json` in `Data/outputs/ml_results/` (or `--output-dir`).

## Raw `.avro`

Still under `.../raw_data/<schema_version>/`. Use project Avro scripts if needed; this flow targets **per-minute biomarker CSVs** only.

## AWS CLI

`s3 sync` against your study prefix works the same as in Empatica’s export guide; `fetch_s3.py` is a small boto3 helper.
