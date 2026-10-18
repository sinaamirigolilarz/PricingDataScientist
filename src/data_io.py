"""Dataset loading helpers.

A single entry point for reading the raw Parquet file keeps type, calendar, and
sort guarantees in one place, so every downstream module sees identical data.

The ``date`` column is stored as a Jalali (Persian/Shamsi) calendar string in
``YYYYMMDD`` form (e.g. ``14030631`` = 1403/06/31). We convert it to a Gregorian
timestamp for day arithmetic and ordering, and also keep the Jalali year/month
so seasonality features can use the Persian calendar (Nowruz, Ramadan-driven
demand, etc.), which aligns with how this retailer's year actually works.
"""

from __future__ import annotations

from pathlib import Path

import jdatetime
import pandas as pd

from config import (
    DATA_PATH,
    DATE_COL,
    DISCOUNT_COL,
    NUM_STORES_COL,
    PRICE_COL,
    PRODUCT_COL,
    STOCK_COL,
    TARGET_COL,
)

JALALI_YEAR_COL = "jyear"
JALALI_MONTH_COL = "jmonth"
JALALI_DAY_COL = "jday"

# Numeric columns are cast to plain numpy float64: the file uses pyarrow-backed
# dtypes, which gradient-boosting libraries cannot consume directly.
NUMERIC_RAW_COLS = [
    TARGET_COL,
    STOCK_COL,
    NUM_STORES_COL,
    DISCOUNT_COL,
    PRICE_COL,
]


def _jalali_string_to_parts(jalali_dates: pd.Series) -> pd.DataFrame:
    """Convert YYYYMMDD Jalali strings to Gregorian dates plus Jalali parts.

    Only the unique date strings are converted (hundreds, not ~1M rows) and the
    result is mapped back, which keeps the conversion fast.
    """
    unique_values = jalali_dates.dropna().unique()
    mapping = {}
    for value in unique_values:
        year, month, day = int(value[:4]), int(value[4:6]), int(value[6:8])
        gregorian = jdatetime.date(year, month, day).togregorian()
        mapping[value] = (pd.Timestamp(gregorian), year, month, day)

    converted = jalali_dates.map(mapping)
    parts = pd.DataFrame(
        converted.tolist(),
        columns=[DATE_COL, JALALI_YEAR_COL, JALALI_MONTH_COL, JALALI_DAY_COL],
        index=jalali_dates.index,
    )
    return parts


def gregorian_to_jalali_parts(timestamps: pd.Series) -> pd.DataFrame:
    """Return Jalali (year, month, day) columns for a series of Gregorian dates.

    Used after the panel is reindexed to a continuous daily calendar, so dates
    created during imputation also receive correct Persian-calendar parts.
    Conversion is cached over unique dates for speed.
    """
    unique_dates = pd.Index(timestamps.dropna().unique())
    mapping = {}
    for timestamp in unique_dates:
        jalali = jdatetime.date.fromgregorian(date=pd.Timestamp(timestamp).date())
        mapping[timestamp] = (jalali.year, jalali.month, jalali.day)

    converted = timestamps.map(mapping)
    return pd.DataFrame(
        converted.tolist(),
        columns=[JALALI_YEAR_COL, JALALI_MONTH_COL, JALALI_DAY_COL],
        index=timestamps.index,
    )


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw product x day table with Gregorian and Jalali calendar columns.

    Rows are sorted by (product_id, date) so that lag/rolling features computed
    later are well defined and never leak future information.
    """
    frame = pd.read_parquet(path)
    parts = _jalali_string_to_parts(frame[DATE_COL].astype("string"))
    frame[DATE_COL] = parts[DATE_COL]
    frame[JALALI_YEAR_COL] = parts[JALALI_YEAR_COL].astype("int16")
    frame[JALALI_MONTH_COL] = parts[JALALI_MONTH_COL].astype("int8")
    frame[JALALI_DAY_COL] = parts[JALALI_DAY_COL].astype("int8")

    frame[NUMERIC_RAW_COLS] = frame[NUMERIC_RAW_COLS].astype("float64")
    frame = frame.sort_values([PRODUCT_COL, DATE_COL]).reset_index(drop=True)
    return frame
