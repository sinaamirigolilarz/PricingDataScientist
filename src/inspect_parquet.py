"""Inspect and visualize the Parquet dataset.

Run this first to understand the data before any modeling. It prints the
schema and key structural facts to the console and writes a set of
exploratory figures (PNG) to the outputs directory.

Usage (from the project root, with the venv active):
    python src/inspect_parquet.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: save figures, never block.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import (
    DATA_PATH,
    DATE_COL,
    DISCOUNT_COL,
    HIERARCHY_COLS,
    NUM_STORES_COL,
    OUTPUTS_DIR,
    PRICE_COL,
    PRODUCT_COL,
    STOCK_COL,
    TARGET_COL,
)
from data_io import load_raw

FIGURE_DPI = 110
EXAMPLE_PRODUCT_COUNT = 3


def print_schema() -> None:
    """Print the Parquet file schema and on-disk metadata without loading rows."""
    parquet_file = pq.ParquetFile(DATA_PATH)
    metadata = parquet_file.metadata
    print("=" * 70)
    print("PARQUET FILE METADATA")
    print("=" * 70)
    print(f"Rows          : {metadata.num_rows:,}")
    print(f"Columns       : {metadata.num_columns}")
    print(f"Row groups    : {metadata.num_row_groups}")
    print(f"Created by    : {metadata.created_by}")
    print("\nArrow schema:")
    print(parquet_file.schema_arrow)


def print_structure(frame: pd.DataFrame) -> None:
    """Print dtypes, missing values, and the product x day panel structure."""
    print("\n" + "=" * 70)
    print("DATAFRAME STRUCTURE")
    print("=" * 70)
    print(f"Shape               : {frame.shape[0]:,} rows x {frame.shape[1]} cols")
    print(f"Memory (MB)         : {frame.memory_usage(deep=True).sum() / 1e6:.1f}")

    print("\nDtypes and null counts:")
    summary = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "n_null": frame.isna().sum(),
            "pct_null": (frame.isna().mean() * 100).round(2),
            "n_unique": frame.nunique(),
        }
    )
    print(summary.to_string())

    print("\nNumeric summary:")
    print(frame.describe(include="number").T.to_string())

    print("\n" + "-" * 70)
    print("PANEL STRUCTURE (product x day)")
    print("-" * 70)
    n_products = frame[PRODUCT_COL].nunique()
    date_min, date_max = frame[DATE_COL].min(), frame[DATE_COL].max()
    span_days = (date_max - date_min).days + 1
    print(f"Unique products     : {n_products:,}")
    print(f"Date range          : {date_min.date()} -> {date_max.date()} ({span_days} days)")

    rows_per_product = frame.groupby(PRODUCT_COL).size()
    print("\nRows (recorded days) per product:")
    print(rows_per_product.describe().round(1).to_string())

    # A dense panel would have n_products * span_days rows; the gap reveals
    # how many (product, day) combinations have no record at all.
    dense_rows = n_products * span_days
    coverage = len(frame) / dense_rows * 100
    print(f"\nPanel density       : {coverage:.1f}% "
          f"({len(frame):,} actual / {dense_rows:,} if fully dense)")
    print("  -> the remaining fraction are MISSING RECORDS (no row at all),")
    print("     which is conceptually different from a recorded zero sale.")

    for col in HIERARCHY_COLS:
        print(f"{col:<8} distinct values: {frame[col].nunique()}")


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(OUTPUTS_DIR.parent)}")


def plot_target_distribution(frame: pd.DataFrame) -> None:
    """Sale quantity distribution on linear and log(1+x) scales."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    frame[TARGET_COL].clip(upper=frame[TARGET_COL].quantile(0.99)).hist(
        bins=60, ax=axes[0]
    )
    axes[0].set_title("sale_qty (clipped at p99)")
    axes[0].set_xlabel("units sold per product-day")

    np.log1p(frame[TARGET_COL].clip(lower=0)).hist(bins=60, ax=axes[1])
    axes[1].set_title("log(1 + sale_qty)")
    axes[1].set_xlabel("log units")
    fig.suptitle("Target distribution (heavy right tail, many zeros)")
    _save(fig, "01_target_distribution.png")


def plot_daily_volume(frame: pd.DataFrame) -> None:
    """Total units sold per calendar day to reveal trend and seasonality."""
    daily = frame.groupby(DATE_COL)[TARGET_COL].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    daily.plot(ax=ax)
    ax.set_title("Total daily sales across all products")
    ax.set_ylabel("units")
    _save(fig, "02_daily_total_sales.png")


def plot_missingness_per_product(frame: pd.DataFrame) -> None:
    """Distribution of recorded-day counts: exposes short-history products."""
    rows_per_product = frame.groupby(PRODUCT_COL).size()
    fig, ax = plt.subplots(figsize=(8, 4))
    rows_per_product.hist(bins=50, ax=ax)
    ax.set_title("Recorded days per product (short history on the left)")
    ax.set_xlabel("number of recorded days")
    ax.set_ylabel("number of products")
    _save(fig, "03_history_length_per_product.png")


def plot_example_products(frame: pd.DataFrame) -> None:
    """Daily sales for the few best-recorded products as sanity-check series."""
    top_products = (
        frame.groupby(PRODUCT_COL).size().sort_values(ascending=False).head(EXAMPLE_PRODUCT_COUNT).index
    )
    fig, axes = plt.subplots(EXAMPLE_PRODUCT_COUNT, 1, figsize=(12, 7), sharex=True)
    for ax, product in zip(axes, top_products):
        series = frame[frame[PRODUCT_COL] == product].set_index(DATE_COL)[TARGET_COL]
        series.plot(ax=ax)
        ax.set_title(f"product_id={product}")
        ax.set_ylabel("units")
    fig.suptitle("Example daily sales series")
    _save(fig, "04_example_product_series.png")


def plot_price_discount(frame: pd.DataFrame) -> None:
    """Relationship between discount, price, stock exposure, and sales."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sample = frame.sample(min(50_000, len(frame)), random_state=0)
    axes[0].scatter(sample[DISCOUNT_COL], sample[TARGET_COL], s=3, alpha=0.2)
    axes[0].set(xlabel=DISCOUNT_COL, ylabel=TARGET_COL, title="Discount vs sales")
    axes[1].scatter(sample[PRICE_COL], sample[TARGET_COL], s=3, alpha=0.2)
    axes[1].set(xlabel=PRICE_COL, ylabel=TARGET_COL, title="Price vs sales")
    axes[2].scatter(sample[NUM_STORES_COL], sample[TARGET_COL], s=3, alpha=0.2)
    axes[2].set(xlabel=NUM_STORES_COL, ylabel=TARGET_COL, title="Store coverage vs sales")
    _save(fig, "05_price_discount_stock_vs_sales.png")


def main() -> None:
    print_schema()
    frame = load_raw()
    print_structure(frame)
    print("\n" + "=" * 70)
    print("WRITING FIGURES")
    print("=" * 70)
    plot_target_distribution(frame)
    plot_daily_volume(frame)
    plot_missingness_per_product(frame)
    plot_example_products(frame)
    plot_price_discount(frame)
    print("\nDone. Figures are in the outputs/ directory.")


if __name__ == "__main__":
    main()
