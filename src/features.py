"""Feature engineering: own-history, calendar, and competitive features.

Three feature families are produced on the preprocessed daily panel and then
snapshotted at each forecast origin to build training samples:

* Own-history features   - each product's recent demand dynamics (lags, rolling
                           statistics, intermittency). HISTORICAL (known only up
                           to the origin).
* Calendar features      - day-of-week, month, etc. FUTURE-KNOWN (computable for
                           any future date without data).
* Competitive features   - each product's position relative to its substitutes
                           (products sharing the same SUBSTITUTE_LEVEL group).

COMPETITIVE FEATURE CATALOG (the task asks for >= 5, each with definition,
business logic, and whether it is computable at prediction time):

  1. price_rank_in_group        Percentile rank of the product's price among its
                                substitutes that day. Business: cheaper than
                                rivals -> captures switching demand.
                                Prediction-time: FUTURE-KNOWN if the future price
                                plan is set; otherwise use last-known price.
  2. relative_discount          Product discount minus the group's mean discount.
                                Business: deeper promo than rivals pulls demand.
                                Prediction-time: FUTURE-KNOWN if the promo
                                calendar is planned; otherwise last-known.
  3. price_gap_to_cheapest      Product price minus the cheapest substitute's
                                price. Business: a large positive gap pushes
                                customers to alternatives. Prediction-time:
                                FUTURE-KNOWN with a price plan, else last-known.
  4. sales_share_in_group       Product's share of its group's total sales over a
                                trailing window. Business: incumbents hold share.
                                Prediction-time: HISTORICAL only (needs realized
                                peer sales) -> use the last trailing value as a
                                slow-moving carry-forward feature.
  5. group_sales_momentum       Group total-sales growth (recent vs. older
                                window). Business: the whole category can be
                                rising/falling (seasonality, trend).
                                Prediction-time: HISTORICAL (carry forward).
  6. relative_store_coverage    Product store coverage divided by the group mean.
                                Business: wider distribution than rivals -> more
                                demand captured. Prediction-time: HISTORICAL /
                                partially future-known if assortment is planned.

Features 1-3 and 6 are cross-sectional per day; 4-5 use trailing windows. All
are leakage-safe: they only use information dated at or before the origin.
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
    ROLLING_WINDOWS,
    SUBSTITUTE_LEVEL,
    TARGET_COL,
)
from data_io import JALALI_MONTH_COL, gregorian_to_jalali_parts

# Iran's weekend is Thursday/Friday; in pandas dayofweek Mon=0, so Thu=3, Fri=4.
IRAN_WEEKEND_DAYS = {3, 4}

SHARE_WINDOW_DAYS = 28
MOMENTUM_WINDOW_DAYS = 28
EPS = 1e-6


# --------------------------------------------------------------------------
# Calendar features (future-known)
# --------------------------------------------------------------------------
def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic date-derived features computable for any future day.

    Weekly seasonality uses the Gregorian weekday (a real calendar day) with an
    Iran-specific weekend flag; yearly seasonality uses the Jalali month, which
    is how this retailer's demand cycles (Nowruz, Persian seasons) actually move.
    """
    frame = frame.copy()
    dates = frame[DATE_COL]
    frame["dow"] = dates.dt.dayofweek.astype("int8")
    frame["is_weekend"] = dates.dt.dayofweek.isin(IRAN_WEEKEND_DAYS).astype("int8")

    jalali_parts = gregorian_to_jalali_parts(dates)
    frame["jalali_month"] = jalali_parts[JALALI_MONTH_COL].astype("int8")
    frame["jalali_day"] = jalali_parts["jday"].astype("int8")
    return frame


# --------------------------------------------------------------------------
# Own-history features (historical)
# --------------------------------------------------------------------------
def add_own_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add per-product lag, rolling, and intermittency features.

    All rolling windows are shifted by one day so the value for day t uses only
    data strictly before t, preventing target leakage.
    """
    frame = frame.copy()
    grouped = frame.groupby(PRODUCT_COL, sort=False)[TARGET_COL]
    shifted = grouped.shift(1)

    for window in ROLLING_WINDOWS:
        roll = shifted.rolling(window, min_periods=1)
        frame[f"sales_mean_{window}d"] = roll.mean()
        frame[f"sales_std_{window}d"] = roll.std().fillna(0.0)
        frame[f"sales_sum_{window}d"] = roll.sum()
        frame[f"zero_rate_{window}d"] = (
            shifted.eq(0).rolling(window, min_periods=1).mean()
        )

    frame["sales_lag_1"] = shifted
    frame["sales_lag_7"] = grouped.shift(7)

    # Intermittency: days since the last non-zero sale (large for sparse items).
    frame["days_since_last_sale"] = frame.groupby(PRODUCT_COL, sort=False)[
        TARGET_COL
    ].transform(_days_since_last_sale)
    return frame


def _days_since_last_sale(product_sales: pd.Series) -> pd.Series:
    """Rows since the previous positive sale, using only past days.

    For each day t we take the most recent sale position among days strictly
    before t (ffill then shift by one), so the current day's sale never informs
    its own feature. Days before the first recorded sale fall back to a running
    day index, which the model reads as "no sale seen yet".
    """
    positions = np.arange(len(product_sales))
    has_sale = (product_sales > 0).to_numpy()
    sale_position = np.where(has_sale, positions, np.nan)
    last_sale_position = (
        pd.Series(sale_position, index=product_sales.index).ffill().shift(1)
    )
    days_since = positions - last_sale_position
    return days_since.fillna(pd.Series(positions, index=product_sales.index))


# --------------------------------------------------------------------------
# Competitive features (relative to substitutes in the same group)
# --------------------------------------------------------------------------
def add_competitive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional competitive features within each substitute group."""
    frame = frame.copy()
    group_keys = [DATE_COL, SUBSTITUTE_LEVEL]
    day_group = frame.groupby(group_keys, sort=False)

    # 1. Price rank within the group that day (0 = cheapest, 1 = most expensive).
    frame["price_rank_in_group"] = day_group[PRICE_COL].rank(pct=True)

    # 2. Discount relative to the group's mean discount that day.
    frame["relative_discount"] = frame[DISCOUNT_COL] - day_group[DISCOUNT_COL].transform(
        "mean"
    )

    # 3. Price gap to the cheapest substitute that day.
    frame["price_gap_to_cheapest"] = frame[PRICE_COL] - day_group[PRICE_COL].transform(
        "min"
    )

    # 6. Store coverage relative to the group mean that day.
    group_mean_cov = day_group[NUM_STORES_COL].transform("mean")
    frame["relative_store_coverage"] = frame[NUM_STORES_COL] / (group_mean_cov + EPS)

    frame = _add_trailing_group_features(frame)
    return frame


def _add_trailing_group_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add trailing-window share and group momentum (features 4 and 5)."""
    frame = frame.sort_values([PRODUCT_COL, DATE_COL])

    # Trailing own sales and trailing group sales (shifted to avoid leakage).
    own_trailing = (
        frame.groupby(PRODUCT_COL, sort=False)[TARGET_COL]
        .shift(1)
        .rolling(SHARE_WINDOW_DAYS, min_periods=1)
        .sum()
    )
    frame["_own_trailing"] = own_trailing

    group_trailing = (
        frame.groupby([SUBSTITUTE_LEVEL, DATE_COL], sort=False)["_own_trailing"]
        .transform("sum")
    )
    # 4. Share of the group's recent sales held by this product.
    frame["sales_share_in_group"] = frame["_own_trailing"] / (group_trailing + EPS)

    # 5. Group momentum: recent group sales vs. the prior window.
    group_daily = (
        frame.groupby([SUBSTITUTE_LEVEL, DATE_COL], sort=False)[TARGET_COL]
        .transform("sum")
    )
    frame["_group_daily"] = group_daily
    recent = (
        frame.groupby(SUBSTITUTE_LEVEL, sort=False)["_group_daily"]
        .shift(1)
        .rolling(MOMENTUM_WINDOW_DAYS, min_periods=1)
        .mean()
    )
    older = (
        frame.groupby(SUBSTITUTE_LEVEL, sort=False)["_group_daily"]
        .shift(1 + MOMENTUM_WINDOW_DAYS)
        .rolling(MOMENTUM_WINDOW_DAYS, min_periods=1)
        .mean()
    )
    frame["group_sales_momentum"] = (recent - older) / (older.abs() + EPS)

    return frame.drop(columns=["_own_trailing", "_group_daily"])


COMPETITIVE_FEATURE_NAMES = [
    "price_rank_in_group",
    "relative_discount",
    "price_gap_to_cheapest",
    "relative_store_coverage",
    "sales_share_in_group",
    "group_sales_momentum",
]


def add_all_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply calendar, own-history, and competitive feature families."""
    frame = add_calendar_features(frame)
    frame = add_own_history_features(frame)
    frame = add_competitive_features(frame)
    return frame
