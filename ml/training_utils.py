"""
Reproducibility and evaluation helpers (optional use from train.py / notebooks).

Best-practice patterns: fixed seeds, explicit label order for metrics, stratified CV skeleton.
"""

from __future__ import annotations

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Make numpy / python RNG deterministic for a run (sklearn estimators still need random_state=)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int, shuffle: bool, random_state: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, val_idx) index pairs per fold for offline CV experiments."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    idx = np.arange(len(y))
    return list(skf.split(idx, y))
