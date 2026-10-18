"""Tests for feature engineering, focused on leakage-safety and correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import (
    add_calendar_features,
    add_competitive_features,
    add_own_history_features,
)


def test_lag_1_equals_previous_day_sales(synthetic_panel):
    featured = add_own_history_features(synthetic_panel)
    product = featured[featured["product_id"] == "P0"].reset_index(drop=True)
    # Lag-1 on row t must equal the raw sale on row t-1 (strictly past).
    assert np.isnan(product["sales_lag_1"].iloc[0])
    assert product["sales_lag_1"].iloc[5] == product["sale_qty"].iloc[4]


def test_rolling_mean_uses_only_past(synthetic_panel):
    featured = add_own_history_features(synthetic_panel)
    product = featured[featured["product_id"] == "P0"].reset_index(drop=True)
    expected = product["sale_qty"].iloc[2:9].mean()  # days t-7..t-1 for t=9
    assert product["sales_mean_7d"].iloc[9] == expected


def test_days_since_last_sale_counts_gap():
    sales = pd.Series([5.0, 0.0, 0.0, 3.0, 0.0])
    frame = pd.DataFrame({"product_id": "A", "sale_qty": sales})
    featured = add_own_history_features(
        frame.assign(date=pd.date_range("2024-01-01", periods=5))
    )
    # Day 3 (index 3) last positive sale was index 0 -> 3 days since; shift keeps
    # the current day out of its own feature.
    assert featured["days_since_last_sale"].iloc[3] == 3


def test_price_rank_and_gap_within_group(synthetic_panel):
    featured = add_competitive_features(synthetic_panel)
    assert featured["price_rank_in_group"].between(0, 1).all()
    assert (featured["price_gap_to_cheapest"] >= -1e-9).all()


def test_sales_share_in_group_is_a_fraction(synthetic_panel):
    featured = add_competitive_features(synthetic_panel)
    share = featured["sales_share_in_group"].dropna()
    assert ((share >= -1e-9) & (share <= 1 + 1e-9)).all()


def test_calendar_features_are_present_and_bounded(synthetic_panel):
    featured = add_calendar_features(synthetic_panel)
    assert featured["jalali_month"].between(1, 12).all()
    assert featured["is_weekend"].isin({0, 1}).all()
