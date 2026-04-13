#!/usr/bin/env python3
"""
HTTP API for Unity EmpaticaLatestClient: POST /predict with JSON {eda, bvp, temp}.

Maintains an in-memory buffer of rows (eda_eda, bvp_bvp, temperature_temp). Appends each
POST as one new row, then runs final_ml prediction on the buffer tail (mean proba over
last N rows; rolling stats only if model_meta says use_rolling_features).

Env:
  VIP_MODEL_DIR        default Data/outputs/ml_results_final_ml under repo root
  VIP_MODEL_VERSION    if set, model loads from $VIP_MODEL_DIR/versions/<VERSION>
  PREDICT_WINDOW_ROWS  default 5 (minute-style tail)
  FLASK_PORT           default 5000
"""

from __future__ import annotations

import os
import sys
from collections import deque
from threading import Lock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import pandas as pd
from flask import Flask, jsonify, request

from train import predict_frustration_final_df

app = Flask(__name__)

VIP_ROOT = os.environ.get("VIP_ROOT", SCRIPT_DIR)
_BASE_MODEL = os.environ.get(
    "VIP_MODEL_DIR",
    os.path.join(VIP_ROOT, "Data", "outputs", "ml_results_final_ml"),
)
_VER = os.environ.get("VIP_MODEL_VERSION", "").strip()
MODEL_DIR = (
    os.path.join(_BASE_MODEL, "versions", _VER) if _VER else _BASE_MODEL
)
WINDOW_ROWS = int(os.environ.get("PREDICT_WINDOW_ROWS", "5"))
BUFFER_MAX = int(os.environ.get("PREDICT_BUFFER_MAX", "5000"))

_buffer: deque = deque(maxlen=BUFFER_MAX)
_lock = Lock()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "buffer_len": len(_buffer), "model_dir": MODEL_DIR})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    try:
        eda = float(data["eda"])
        bvp = float(data["bvp"])
        temp = float(data["temp"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "JSON body must include numeric eda, bvp, temp"}), 400

    row = {"eda_eda": eda, "bvp_bvp": bvp, "temperature_temp": temp}
    with _lock:
        _buffer.append(row)
        df = pd.DataFrame(list(_buffer))

    try:
        out = predict_frustration_final_df(df, MODEL_DIR, WINDOW_ROWS)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    idx = int(out["class_index"])
    lab = out["frustration"]
    # Shape matches EmpaticaLatestClient.PredictResponse (stress/cogload ints + labels)
    return jsonify(
        {
            "stress": idx,
            "stress_label": lab,
            "cogload": 0,
            "cogload_label": "n_a",
            "frustration": lab,
            "mean_proba": out["mean_proba"],
            "window_rows_used": out["window_rows_used"],
            "buffer_total_rows": len(df),
        }
    )


def main():
    port = int(os.environ.get("FLASK_PORT", "5000"))
    print(f"Model dir: {MODEL_DIR}")
    print(f"Window rows: {WINDOW_ROWS}")
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
