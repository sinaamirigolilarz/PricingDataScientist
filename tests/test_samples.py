"""Tests for direct-forecast sample construction (target and censoring)."""

from __future__ import annotations

import pandas as pd

from samples import _forward_sum, build_samples


def _toy_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "product_id": "A",
            "date": dates,
            "sale_qty": range(10),
            "sale_qty_raw": range(10),
            "feat": range(10),
        }
    )


def test_forward_sum_sums_next_horizon_days():
    frame = _toy_frame()
    forward = _forward_sum(frame, horizon=3)
    # Day 0: sum of days 1,2,3 = 1+2+3 = 6.
    assert forward.iloc[0] == 6
    # Day 6: sum of days 7,8,9 = 24.
    assert forward.iloc[6] == 24


def test_build_samples_drops_censored_origins():
    frame = _toy_frame()
    origins = [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-09")]
    # Horizon 3: the 2024-01-09 origin has fewer than 3 future days -> dropped.
    samples = build_samples(frame, origins, feature_cols=["feat"], horizon=3)
    assert samples["origin_date"].nunique() == 1
    assert samples["origin_date"].iloc[0] == pd.Timestamp("2024-01-03")


def test_build_samples_target_matches_forward_sum():
    frame = _toy_frame()
    samples = build_samples(
        frame, [pd.Timestamp("2024-01-01")], feature_cols=["feat"], horizon=3
    )
    assert samples["target_30d_sum"].iloc[0] == 6
