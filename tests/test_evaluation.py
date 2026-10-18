"""Tests for evaluation metrics on known inputs."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import bias, compute_metrics, mae, rmse, weighted_mape


def test_perfect_prediction_scores_zero():
    y = np.array([10.0, 20.0, 30.0])
    metrics = compute_metrics(y, y.copy())
    assert metrics["WMAPE"] == 0.0
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["Bias"] == 0.0


def test_weighted_mape_known_value():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([90.0, 120.0])
    # Total absolute error = 10 + 20 = 30; total actual = 200 -> WMAPE = 0.15.
    assert weighted_mape(y_true, y_pred) == pytest.approx(0.15, abs=1e-6)


def test_bias_is_positive_when_overforecasting():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([12.0, 15.0])
    assert bias(y_true, y_pred) > 0


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 30.0])
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)
