#!/usr/bin/env python3
"""
Train the 3-class frustration model (final_files_ML layout).

This is the dedicated **model training** entry point. It delegates to train.py, which
requires the ``ml/`` package (``ml/final_files_ml_loader.py``) next to ``train.py``.

Typical:
  python train_final_ml_model.py
  python train_final_ml_model.py --final-ml-root ./Data/final_files_ML --final-ml-version auto

See also: train.py --help for all options (--split participant, --no-save-artifacts, etc.).
"""

from __future__ import annotations

import sys

import train as _train


def main() -> None:
    argv0 = sys.argv[0]
    user_args = sys.argv[1:]
    sys.argv = [argv0, "--feature-source", "final_ml", *user_args]
    _train.main()


if __name__ == "__main__":
    main()
