"""Preprocessing pipeline for the retail product x day panel.

The design answers the task's preprocessing questions explicitly:

1. Missing records vs. zero sales vs. zero stock
   - A MISSING RECORD is a (product, day) with no row at all. For an active
     product this usually means "no observation", not "zero demand", so we do
     NOT blindly fill it with 0. We reindex each product to a continuous daily
     calendar only within its own observed lifespan, and mark reconstructed
     days so the model can tell them apart.
   - ZERO SALES is a recorded row with sale_qty == 0: genuine demand signal
     (the product was tracked that day and sold nothing). Kept as 0.
   - ZERO STOCK (stock_qty == 0 / num_stores_with_stock == 0) means the product
     could not sell because it was unavailable. A zero sale under zero stock is
     a stock-out, not low demand, and is flagged so it can be treated
     differently from a true zero-demand day.

2. Outliers / sales spikes are detected per product with a robust (median /
   MAD) rule and WINSORIZED, not deleted, because many spikes are real
   promotion-driven demand we want the model to learn from.

All operations are leakage-safe: statistics used for flagging are descriptive
and recomputed inside each training fold by the caller when needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DATE_COL,
    DISCOUNT_COL,
    NUM_STORES_COL,
    PRICE_COL,
    PRODUCT_COL,
    STOCK_COL,
    TARGET_COL,
)

MAD_TO_STD = 1.4826  # Scale factor making MAD a consistent estimator of std.
SPIKE_THRESHOLD = 8.0  # Robust z-score above which a value is winsorized.
MIN_MAD = 1e-6  # Guards against division by zero for flat series.


def reindex_to_daily_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing calendar days within each product's observed lifespan.

    Each product is reindexed to a continuous daily range between its own first
    and last observed date (not the global range: a product that launches late
    should not get fabricated history before it existed). Reconstructed rows get
    ``is_imputed_day = 1`` and a sale_qty of 0, under the assumption that a
    tracked-but-unreported day most often reflects no sale. The flag lets the
    model discount these rows rather than trusting them as hard zeros.
    """
    filled_parts = []
    for product_id, group in frame.groupby(PRODUCT_COL, sort=False):
        group = group.set_index(DATE_COL)
        full_range = pd.date_range(group.index.min(), group.index.max(), freq="D")
        reindexed = group.reindex(full_range)
        reindexed[PRODUCT_COL] = product_id
        reindexed["is_imputed_day"] = reindexed[TARGET_COL].isna().astype("int8")
        filled_parts.append(reindexed)

    result = pd.concat(filled_parts)
    result.index.name = DATE_COL
    return result.reset_index()


def add_availability_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stock-out and zero-sale flags distinguishing demand from availability."""
    frame = frame.copy()
    had_no_stock = (frame[STOCK_COL].fillna(0) <= 0) & (
        frame[NUM_STORES_COL].fillna(0) <= 0
    )
    frame["is_stockout"] = had_no_stock.astype("int8")
    frame["is_zero_sale"] = (frame[TARGET_COL].fillna(0) <= 0).astype("int8")
    # A zero sale that happens while in stock is a true zero-demand signal;
    # a zero sale while out of stock is censored and less informative.
    frame["is_true_zero_demand"] = (
        (frame["is_zero_sale"] == 1) & (had_no_stock == False)  # noqa: E712
    ).astype("int8")
    return frame


def impute_static_and_price(frame: pd.DataFrame, static_cols: list[str]) -> pd.DataFrame:
    """Forward/backward fill per-product attributes that are structurally static.

    Category hierarchy and brand are product attributes and must be constant
    within a product, so gaps created by reindexing are filled from the product
    itself. Price is forward-filled (last known shelf price persists), and
    discount defaults to 0 when unobserved.
    """
    frame = frame.copy()
    group = frame.groupby(PRODUCT_COL, sort=False)
    for col in static_cols:
        frame[col] = group[col].ffill().bfill()
    frame[PRICE_COL] = group[PRICE_COL].ffill().bfill()
    frame[DISCOUNT_COL] = frame[DISCOUNT_COL].fillna(0.0)
    for col in (STOCK_COL, NUM_STORES_COL, TARGET_COL):
        frame[col] = frame[col].fillna(0.0)
    return frame


def winsorize_spikes(frame: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme sale_qty spikes per product using a robust MAD rule.

    We keep spikes (promotions are real demand) but cap only the most extreme
    values, which are more likely data-entry errors than genuine demand. The
    capped value and a flag are both retained.
    """
    frame = frame.copy()

    def _cap(series: pd.Series) -> pd.Series:
        median = series.median()
        mad = np.median(np.abs(series - median))
        scale = max(mad * MAD_TO_STD, MIN_MAD)
        upper = median + SPIKE_THRESHOLD * scale
        return series.clip(upper=upper)

    capped = frame.groupby(PRODUCT_COL, sort=False)[TARGET_COL].transform(_cap)
    frame["is_spike_capped"] = (capped < frame[TARGET_COL]).astype("int8")
    frame["sale_qty_raw"] = frame[TARGET_COL]
    frame[TARGET_COL] = capped
    return frame


def preprocess(frame: pd.DataFrame, static_cols: list[str]) -> pd.DataFrame:
    """Run the full preprocessing pipeline in a leakage-safe order."""
    frame = reindex_to_daily_calendar(frame)
    frame = add_availability_flags(frame)
    frame = impute_static_and_price(frame, static_cols)
    frame = winsorize_spikes(frame)
    return frame
