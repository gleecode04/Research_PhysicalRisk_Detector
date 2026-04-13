#!/usr/bin/env python3
"""
Kafka **consumer** that buffers minute-style biomarker JSON, then every *N* seconds
runs the same ``final_ml`` tail-window prediction as batch CSV inference.

Intended flow (see ``docs/DATAFLOW.md``):
  wristband → vendor cloud → export / gateway → **Kafka topic** → this consumer → ML output

Dependencies: ``kafka-python``, ``pandas``, repo ``train.py`` (for ``predict_frustration_final_df``).

Environment:
  KAFKA_BOOTSTRAP_SERVERS   default localhost:9092
  KAFKA_TOPIC_MINUTE        default wrist.minute_biomarkers
  KAFKA_GROUP_ID            default vip-frustration-consumer
  VIP_MODEL_DIR             default Data/outputs/ml_results_final_ml (see train.py)
  ANALYSIS_INTERVAL_SEC     default 300 (5 minutes)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from train import OUTPUT_DIR_FINAL_ML, default_inference_window_rows, predict_frustration_final_df


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a wrist-style frame from buffered minute events (post-merge column names)."""
    if not rows:
        return pd.DataFrame(columns=["eda_eda", "bvp_bvp", "temperature_temp"])
    df = pd.DataFrame(rows)
    # Accept either raw vendor names or already-renamed columns
    rename = {
        "eda_scl_usiemens": "eda_eda",
        "pulse_rate_bpm": "bvp_bvp",
        "temperature_celsius": "temperature_temp",
    }
    for a, b in rename.items():
        if a in df.columns and b not in df.columns:
            df[b] = df[a]
    need = ["eda_eda", "bvp_bvp", "temperature_temp"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Buffered rows missing columns {miss}; got {list(df.columns)}")
    return df[need].apply(pd.to_numeric, errors="coerce")


def main() -> None:
    try:
        from kafka import KafkaConsumer
    except ImportError as e:
        raise SystemExit("pip install kafka-python") from e

    servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.environ.get("KAFKA_TOPIC_MINUTE", "wrist.minute_biomarkers")
    group = os.environ.get("KAFKA_GROUP_ID", "vip-frustration-consumer")
    model_dir = os.environ.get("VIP_MODEL_DIR", OUTPUT_DIR_FINAL_ML)
    interval = int(os.environ.get("ANALYSIS_INTERVAL_SEC", "300"))
    assume_minute = os.environ.get("ASSUME_ONE_ROW_PER_MINUTE", "1") not in ("0", "false", "False")
    win = default_inference_window_rows(assume_minute, None)

    buf: deque[dict[str, Any]] = deque(maxlen=int(os.environ.get("KAFKA_BUFFER_MAX_ROWS", "10000")))
    lock = threading.Lock()
    stop = threading.Event()

    def analysis_tick() -> None:
        while not stop.wait(interval):
            with lock:
                snap = list(buf)
            if len(snap) < 1:
                print(json.dumps({"ts": time.time(), "note": "empty_buffer"}))
                continue
            try:
                df = _rows_to_dataframe(snap)
                out = predict_frustration_final_df(df, model_dir, win)
            except Exception as ex:
                print(json.dumps({"ts": time.time(), "error": str(ex)}))
                continue
            out["buffer_rows"] = len(snap)
            out["inference_window_rows"] = win
            print(json.dumps({"ts": time.time(), "prediction": out}))

    threading.Thread(target=analysis_tick, daemon=True).start()

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=servers.split(","),
        group_id=group,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="latest",
    )
    print(
        json.dumps(
            {
                "listening": topic,
                "bootstrap": servers,
                "model_dir": model_dir,
                "analysis_interval_sec": interval,
            }
        ),
        flush=True,
    )
    try:
        for msg in consumer:
            v = msg.value
            if not isinstance(v, dict):
                continue
            with lock:
                buf.append(v)
    except KeyboardInterrupt:
        stop.set()
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
