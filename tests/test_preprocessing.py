"""Tests for the preprocessing pipeline: imputation, flags, and winsorizing."""

from __future__ import annotations

import pandas as pd

from preprocessing import (
    add_availability_flags,
    preprocess,
    reindex_to_daily_calendar,
    winsorize_spikes,
)

STATIC_COLS = ["brand_id", "Level1", "Level2", "Level3", "Level4", "Level5"]


def test_reindex_adds_no_rows_for_continuous_panel(synthetic_panel):
    reindexed = reindex_to_daily_calendar(synthetic_panel)
    assert len(reindexed) == len(synthetic_panel)
    assert reindexed["is_imputed_day"].sum() == 0


def test_reindex_fills_internal_gap_and_flags_it(synthetic_panel):
    with_gap = synthetic_panel[
        ~(
            (synthetic_panel["product_id"] == "P0")
            & (synthetic_panel["date"] == pd.Timestamp("2024-02-01"))
        )
    ]
    reindexed = reindex_to_daily_calendar(with_gap)
    filled = reindexed[
        (reindexed["product_id"] == "P0")
        & (reindexed["date"] == pd.Timestamp("2024-02-01"))
    ]
    assert len(filled) == 1
    assert filled["is_imputed_day"].iloc[0] == 1


def test_stockout_day_is_flagged(synthetic_panel):
    flagged = add_availability_flags(synthetic_panel)
    stockout_row = flagged[
        (flagged["product_id"] == "P1") & (flagged["date"] == pd.Timestamp("2024-02-20"))
    ]
    assert stockout_row["is_stockout"].iloc[0] == 1


def test_zero_sale_under_stockout_is_not_true_zero_demand(synthetic_panel):
    flagged = add_availability_flags(synthetic_panel)
    stockout_row = flagged[
        (flagged["product_id"] == "P1") & (flagged["date"] == pd.Timestamp("2024-02-20"))
    ]
    # The product was unavailable, so its zero sale is censored, not real demand.
    assert stockout_row["is_true_zero_demand"].iloc[0] == 0


def test_spike_is_capped_not_dropped(synthetic_panel):
    capped = winsorize_spikes(synthetic_panel)
    spike_row = capped[
        (capped["product_id"] == "P0") & (capped["date"] == pd.Timestamp("2024-03-31"))
    ]
    assert spike_row["sale_qty_raw"].iloc[0] == 10_000.0
    assert spike_row["sale_qty"].iloc[0] < 10_000.0
    assert spike_row["is_spike_capped"].iloc[0] == 1
    # No rows are removed: winsorizing keeps the sample, only the value changes.
    assert len(capped) == len(synthetic_panel)


def test_preprocess_runs_and_preserves_products(synthetic_panel):
    processed = preprocess(synthetic_panel, static_cols=STATIC_COLS)
    assert set(processed["product_id"].unique()) == {"P0", "P1", "P2", "P3"}
    assert processed[STATIC_COLS].isna().sum().sum() == 0
