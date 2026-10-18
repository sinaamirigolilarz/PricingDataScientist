"""Evaluation metrics and time-based rolling-origin backtesting.

Why not a random split: the problem is temporal, so a random split would let the
model train on future days and validate on past ones, leaking information and
inflating scores. We therefore validate strictly forward in time.

Backtesting design: a rolling-origin scheme. At each validation origin we train
on samples whose forward window ends on or before that origin (so training
labels are fully in the past), then score the held-out origin. Repeating over
several origins gives a distribution of errors rather than one fragile number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FORECAST_HORIZON_DAYS
from samples import ORIGIN_COL, TARGET_SUM_COL

EPS = 1e-6


def weighted_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WMAPE: total absolute error divided by total actual volume.

    Volume-weighted, so high-selling products dominate - it reflects aggregate
    business accuracy and avoids the blow-up plain MAPE suffers on near-zero
    actuals (common with intermittent demand).
    """
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + EPS))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in units: average miss regardless of product size."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error: penalizes large misses (spikes) more heavily."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error: positive means systematic over-forecasting."""
    return float(np.mean(y_pred - y_true))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the full metric set as a dictionary."""
    return {
        "WMAPE": weighted_mape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }


@dataclass
class BacktestResult:
    """Per-origin and aggregated metrics for one model."""

    per_origin: pd.DataFrame
    aggregate: dict[str, float]


def rolling_origin_backtest(
    model_factory,
    samples: pd.DataFrame,
    feature_cols: list[str],
    validation_origins: list[pd.Timestamp],
    horizon: int = FORECAST_HORIZON_DAYS,
) -> BacktestResult:
    """Backtest a model over several validation origins, training forward-safe.

    ``model_factory`` is a zero-argument callable returning a fresh model, so
    each fold trains an independent model and no state leaks between folds.
    """
    per_origin_rows = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for origin in validation_origins:
        # Training labels must be fully observed before this origin: a sample's
        # forward window ends horizon days after its own origin, so we require
        # training origins to be at least `horizon` days before the validation one.
        train_cutoff = origin - pd.Timedelta(days=horizon)
        train = samples[samples[ORIGIN_COL] <= train_cutoff]
        valid = samples[samples[ORIGIN_COL] == origin]
        if train.empty or valid.empty:
            continue

        model = model_factory()
        model.fit(train[feature_cols], train[TARGET_SUM_COL])
        predictions = model.predict(valid[feature_cols])

        truth = valid[TARGET_SUM_COL].to_numpy()
        metrics = compute_metrics(truth, predictions)
        metrics["origin"] = origin
        metrics["n_samples"] = len(valid)
        per_origin_rows.append(metrics)
        all_true.append(truth)
        all_pred.append(predictions)

    per_origin = pd.DataFrame(per_origin_rows)
    aggregate = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    return BacktestResult(per_origin=per_origin, aggregate=aggregate)
