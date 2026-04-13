#!/usr/bin/env bash
# Stage only code + docs + small examples (no Data/, no CSV, no PDF at repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DIRS=(ml streaming tools docs emphatica examples)
for d in "${DIRS[@]}"; do
  if [[ -d "$d" ]]; then
    git add -- "$d"
  fi
done

ROOT_FILES=(
  train.py
  train_final_ml_model.py
  empatica_minute_filter_predict.py
  predict_from_emphatica_minute_csvs.py
  poll_wrist_frustration.py
  prediction_server.py
  wrist_minute_preprocess.py
  setup_ml_input.py
  process_converted_data.py
  sriram_avro_to_csv.py
  read.py
  requirements.txt
  README.md
  .gitignore
)
for f in "${ROOT_FILES[@]}"; do
  if [[ -e "$f" ]]; then
    git add -- "$f"
  fi
done

if [[ -d scripts ]]; then
  git add -- scripts
fi

echo "Staged (code-only policy). Review with: git status"
git status -sb
