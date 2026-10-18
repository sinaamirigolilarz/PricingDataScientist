"""Shared pytest fixtures and import path setup.

Tests run against a small SYNTHETIC panel with known properties so assertions
are exact and fast, independent of the real dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A 4-product, 2-group daily panel with a controlled structure.

    - 180 consecutive days, Gregorian dates (already parsed).
    - Two Level4 groups of two products each.
    - product P0 has a deterministic ramp in sales; others are flat-ish.
    - One injected spike and one stock-out day for flag/winsorize tests.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    rows = []
    group_of = {"P0": "G0", "P1": "G0", "P2": "G1", "P3": "G1"}
    base_price = {"P0": 100.0, "P1": 120.0, "P2": 80.0, "P3": 200.0}
    for product, group in group_of.items():
        for day_index, date in enumerate(dates):
            sales = 10.0 + day_index * (0.1 if product == "P0" else 0.0)
            sales += rng.normal(0, 1)
            rows.append(
                {
                    "date": date,
                    "product_id": product,
                    "brand_id": "B" + product[-1],
                    "Level1": "L1",
                    "Level2": "L2",
                    "Level3": "L3",
                    "Level4": group,
                    "Level5": group + "x",
                    "sale_qty": max(sales, 0.0),
                    "stock_qty": 500.0,
                    "num_stores_with_stock": 10,
                    "discount_pct": 5.0,
                    "price": base_price[product],
                    "jyear": 1402,
                    "jmonth": 10,
                    "jday": 11,
                }
            )
    frame = pd.DataFrame(rows)

    spike_mask = (frame["product_id"] == "P0") & (frame["date"] == dates[90])
    frame.loc[spike_mask, "sale_qty"] = 10_000.0

    stockout_mask = (frame["product_id"] == "P1") & (frame["date"] == dates[50])
    frame.loc[stockout_mask, ["stock_qty", "num_stores_with_stock", "sale_qty"]] = 0.0

    return frame.sort_values(["product_id", "date"]).reset_index(drop=True)
