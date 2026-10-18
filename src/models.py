"""Forecasting models: a simple baseline and a LightGBM main model.

Baseline (recent-average): projects each product's recent average daily sales
onto the 30-day horizon. It is transparent, hard to beat for stable items, and
sets the bar the main model must clear.

Main model (LightGBM, Tweedie objective): gradient-boosted trees on the direct
30-day-sum target. LightGBM is chosen because the data is tabular, medium-sized,
and runs fast on CPU with low memory (no GPU needed) - a good fit for the stated
hardware. The Tweedie objective suits non-negative, zero-inflated, right-skewed
demand and balances low-volume and high-volume products better than plain MSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import FORECAST_HORIZON_DAYS, RANDOM_SEED
from samples import TARGET_SUM_COL

RECENT_MEAN_FEATURE = "sales_mean_28d"


class RecentAverageBaseline:
    """Predict the 30-day sum as the recent mean daily sales times the horizon."""

    def __init__(self, horizon: int = FORECAST_HORIZON_DAYS) -> None:
        self.horizon = horizon

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RecentAverageBaseline":
        """Stateless baseline; present for a scikit-learn-like interface."""
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        recent_daily_mean = features[RECENT_MEAN_FEATURE].fillna(0.0)
        return (recent_daily_mean * self.horizon).to_numpy()


class LightGBMForecaster:
    """LightGBM regressor wrapper for the direct 30-day-sum target."""

    def __init__(self, **overrides: object) -> None:
        params = dict(
            objective="tweedie",
            tweedie_variance_power=1.2,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        params.update(overrides)
        self.model = LGBMRegressor(**params)
        self.feature_names_: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LightGBMForecaster":
        self.feature_names_ = list(features.columns)
        self.model.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(features[self.feature_names_])
        # Demand cannot be negative; clip to keep predictions physically valid.
        return np.clip(predictions, a_min=0.0, a_max=None)

    @property
    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names_
        ).sort_values(ascending=False)


def split_features_target(samples: pd.DataFrame, feature_cols: list[str]):
    """Return (feature frame, target series) from a built sample table."""
    return samples[feature_cols], samples[TARGET_SUM_COL]
