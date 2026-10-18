"""Build direct-forecast training samples from the daily feature panel.

The business target is the SUM of sales over the next 30 days per product, so
we frame the problem as a DIRECT forecast: one row per (product, origin_date)
whose features are the panel snapshot known at the origin and whose label is the
realized 30-day forward sum.

Direct vs. recursive is discussed in the technical note; direct is chosen here
because it optimizes the exact reported quantity and avoids error accumulation
over a 30-step recursive roll-out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DATE_COL,
    FORECAST_HORIZON_DAYS,
    HISTORY_WINDOW_DAYS,
    PRODUCT_COL,
    TARGET_COL,
)

TRAINING_ORIGIN_STEP_DAYS = 7

TARGET_SUM_COL = "target_30d_sum"
ORIGIN_COL = "origin_date"
RAW_TARGET_COL = "sale_qty_raw"


def _forward_sum(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Per-product forward-looking sum of actual sales over the next ``horizon`` days.

    Uses the uncapped sales so the label reflects true future demand. The
    reversed-cumsum trick computes, for each day t, the sum over (t, t+horizon].
    """
    sales = frame[RAW_TARGET_COL] if RAW_TARGET_COL in frame else frame[TARGET_COL]

    def _per_product(series: pd.Series) -> pd.Series:
        future = series.shift(-1)  # start the window on the day AFTER the origin
        forward = future[::-1].rolling(horizon, min_periods=1).sum()[::-1]
        return forward

    return frame.assign(_s=sales).groupby(PRODUCT_COL, sort=False)["_s"].transform(
        _per_product
    )


def build_samples(
    frame: pd.DataFrame,
    origin_dates: list[pd.Timestamp],
    feature_cols: list[str],
    horizon: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    """Assemble (product, origin) training rows for the given origin dates.

    A sample is kept only when the full forward horizon is observable (the
    origin is at least ``horizon`` days before the panel end), guaranteeing the
    label is complete and never partially censored.
    """
    frame = frame.copy()
    frame[TARGET_SUM_COL] = _forward_sum(frame, horizon)

    max_date = frame[DATE_COL].max()
    usable_origins = [d for d in origin_dates if (max_date - d).days >= horizon]

    snapshots = []
    for origin in usable_origins:
        snapshot = frame[frame[DATE_COL] == origin].copy()
        snapshot[ORIGIN_COL] = origin
        snapshots.append(snapshot)

    samples = pd.concat(snapshots, ignore_index=True)
    keep = [PRODUCT_COL, ORIGIN_COL, TARGET_SUM_COL, *feature_cols]
    samples = samples[keep].replace([np.inf, -np.inf], np.nan)
    return samples.dropna(subset=[TARGET_SUM_COL]).reset_index(drop=True)


def make_origin_grid(
    frame: pd.DataFrame,
    step_days: int = TRAINING_ORIGIN_STEP_DAYS,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> list[pd.Timestamp]:
    """Create a dense grid of forecast origins for building training samples.

    Origins run weekly from one history window after the panel start up to one
    horizon before the panel end (so every sample has both enough look-back and
    a fully observed forward label). Overlapping forward windows are acceptable
    for TRAINING: they multiply the available samples, which boosted trees need.
    """
    first_origin = frame[DATE_COL].min() + pd.Timedelta(days=HISTORY_WINDOW_DAYS)
    last_origin = frame[DATE_COL].max() - pd.Timedelta(days=horizon)
    origins = pd.date_range(first_origin, last_origin, freq=f"{step_days}D")
    return list(origins)


def make_validation_origins(
    frame: pd.DataFrame,
    n_folds: int,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> list[pd.Timestamp]:
    """Pick the most recent origins, spaced by one horizon, for backtesting.

    Spacing validation origins by the horizon keeps their evaluation windows
    non-overlapping, so each fold reports an independent out-of-sample error.
    """
    last_origin = frame[DATE_COL].max() - pd.Timedelta(days=horizon)
    origins = [last_origin - pd.Timedelta(days=horizon * fold) for fold in range(n_folds)]
    return sorted(d for d in origins if d >= frame[DATE_COL].min())
