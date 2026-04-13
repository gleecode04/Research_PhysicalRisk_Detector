"""
Additional scikit-learn estimators and validation-time model selection for 3-class final_ml.

Use ``--final-ml-wide-model-search`` with ``train.py`` to pick the best of several candidates
on the **validation** split before test evaluation (better practice than tuning on test).
Optional ``xgboost`` / ``lightgbm`` are tried if installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


def _as_float(X):
    return np.asarray(X, dtype=float)


def iter_final_ml_estimators(random_state: int = 42) -> Iterator[tuple[str, Any]]:
    """Named sklearn estimators suitable for low-dimensional tabular 3-class problems."""
    yield (
        "hgb_standard",
        HistGradientBoostingClassifier(
            max_iter=280,
            max_depth=8,
            learning_rate=0.08,
            random_state=random_state,
            min_samples_leaf=15,
        ),
    )
    yield (
        "extra_trees",
        ExtraTreesClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    yield (
        "rf_balanced",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    yield (
        "linear_svc_calibrated",
        CalibratedClassifierCV(
            estimator=LinearSVC(max_iter=8000, dual="auto", random_state=random_state),
            cv=3,
        ),
    )
    yield (
        "mlp_small",
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=12,
        ),
    )

    try:
        import xgboost as xgb

        yield (
            "xgboost",
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            ),
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb

        yield (
            "lightgbm",
            lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=-1,
                learning_rate=0.06,
                num_leaves=31,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        )
    except ImportError:
        pass


@dataclass
class SelectionResult:
    name: str
    estimator: Any
    val_accuracy: float


def select_best_estimator_on_validation(
    X_train,
    X_val,
    y_train,
    y_val,
    *,
    random_state: int = 42,
    verbose: bool = True,
) -> SelectionResult:
    """
    Fit each candidate on train, score on val; return unfitted clone of winner for refit on train+val.
    """
    X_train = _as_float(X_train)
    X_val = _as_float(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    best: SelectionResult | None = None
    for name, est in iter_final_ml_estimators(random_state):
        try:
            m = clone(est)
            m.fit(X_train, y_train)
            acc = float(accuracy_score(y_val, m.predict(X_val)))
            if verbose:
                print(f"  final_ml candidate {name}: val_accuracy={acc:.4f}", flush=True)
            if best is None or acc > best.val_accuracy:
                best = SelectionResult(name=name, estimator=m, val_accuracy=acc)
        except Exception as ex:
            if verbose:
                print(f"  final_ml candidate {name}: skipped ({ex})", flush=True)
            continue

    if best is None:
        raise RuntimeError("No estimator succeeded in wide model search; check data and optional deps.")

    return SelectionResult(
        name=best.name,
        estimator=clone(best.estimator),
        val_accuracy=best.val_accuracy,
    )
