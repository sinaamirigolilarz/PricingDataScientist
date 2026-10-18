"""Tests for the baseline and LightGBM forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import LightGBMForecaster, RecentAverageBaseline


def test_baseline_projects_recent_mean_over_horizon():
    features = pd.DataFrame({"sales_mean_28d": [2.0, 5.0]})
    model = RecentAverageBaseline(horizon=30)
    predictions = model.predict(features)
    assert list(predictions) == [60.0, 150.0]


def test_lightgbm_predictions_are_non_negative():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {"a": rng.normal(size=200), "b": rng.normal(size=200)}
    )
    target = pd.Series(np.abs(rng.normal(scale=10, size=200)))
    model = LightGBMForecaster(n_estimators=20).fit(features, target)
    predictions = model.predict(features)
    assert (predictions >= 0).all()


def test_lightgbm_exposes_feature_importance():
    rng = np.random.default_rng(1)
    features = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    target = pd.Series(rng.normal(size=100).clip(min=0))
    model = LightGBMForecaster(n_estimators=10).fit(features, target)
    assert set(model.feature_importance.index) == {"a", "b"}
