#!/usr/bin/env python3
"""
Near–real-time loop (default 5 min): optional S3 download, then biomarker-window predict.

Env:
  POLL_INTERVAL_SEC   default 300
  VIP_ROOT            repo root (default: parent of this script's directory)
  LOCAL_PATH          passed to fetch when RUN_FETCH=1
  BIOMARKER_SESSION   directory with minute CSVs for current predict tick
  RUN_FETCH           if "1", run fetch_s3.py --download --biomarkers-only --skip-raw-data first
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIP_ROOT = os.environ.get("VIP_ROOT", os.path.dirname(SCRIPT_DIR))


def main():
    interval = int(os.environ.get("POLL_INTERVAL_SEC", "300"))
    session = os.environ.get("BIOMARKER_SESSION", "").strip()
    if not session:
        print("Set BIOMARKER_SESSION to a folder of aggregated_per_minute CSVs.", file=sys.stderr)
        sys.exit(1)
    train_py = os.path.join(VIP_ROOT, "train.py")
    if not os.path.isfile(train_py):
        print(f"train.py not found at {train_py}", file=sys.stderr)
        sys.exit(1)

    while True:
        if os.environ.get("RUN_FETCH") == "1":
            fetch_py = os.path.join(SCRIPT_DIR, "fetch_s3.py")
            subprocess.run(
                [sys.executable, fetch_py, "--download", "--biomarkers-only", "--skip-raw-data", "--max-keys", "500"],
                cwd=VIP_ROOT,
                check=False,
            )
        r = subprocess.run(
            [sys.executable, train_py, "--predict-biomarker-session", session],
            cwd=VIP_ROOT,
        )
        if r.returncode != 0:
            print(f"predict exit {r.returncode}", file=sys.stderr)
        print(f"Sleeping {interval}s …")
        time.sleep(interval)


if __name__ == "__main__":
    main()
