"""End-to-end experiment: baseline vs. LightGBM under identical evaluation.

Pipeline: load -> preprocess -> feature engineering -> build direct-forecast
samples -> rolling-origin backtest of both models -> report metrics and the
value added by the competitive features.

Usage (from the project root, with the venv active):
    python src/run_experiment.py --max-products 300

``--max-products`` runs on a product subset so the full experiment finishes in
minutes on a CPU-only laptop, as the task allows for limited compute.
"""

from __future__ import annotations

import argparse

import pandas as pd

from config import (
    BRAND_COL,
    HIERARCHY_COLS,
    OUTPUTS_DIR,
    PRODUCT_COL,
    RANDOM_SEED,
)
from data_io import (
    JALALI_DAY_COL,
    JALALI_MONTH_COL,
    JALALI_YEAR_COL,
    load_raw,
)
from evaluation import rolling_origin_backtest
from features import COMPETITIVE_FEATURE_NAMES, add_all_features
from models import LightGBMForecaster, RecentAverageBaseline
from preprocessing import preprocess
from samples import build_samples, make_origin_grid, make_validation_origins

VALIDATION_FOLDS = 3


def select_feature_columns(daily_features: pd.DataFrame) -> list[str]:
    """Pick numeric model inputs from the featured daily panel.

    Identifiers and the raw Jalali date parts are dropped (the latter are kept
    only for EDA and are superseded by the leakage-safe calendar features).
    Every remaining numeric column is dated at or before the origin, so all of
    them are leakage-safe as features.
    """
    exclude = {PRODUCT_COL, JALALI_YEAR_COL, JALALI_MONTH_COL, JALALI_DAY_COL}
    numeric = daily_features.select_dtypes(include="number").columns
    return [col for col in numeric if col not in exclude]


def subset_products(frame: pd.DataFrame, max_products: int | None) -> pd.DataFrame:
    """Keep a reproducible random subset of products to bound compute."""
    if max_products is None:
        return frame
    products = frame[PRODUCT_COL].drop_duplicates()
    chosen = products.sample(min(max_products, len(products)), random_state=RANDOM_SEED)
    return frame[frame[PRODUCT_COL].isin(chosen)]


def run(max_products: int | None) -> None:
    print("Loading data ...")
    raw = subset_products(load_raw(), max_products)
    print(f"  rows={len(raw):,}  products={raw[PRODUCT_COL].nunique():,}")

    print("Preprocessing ...")
    static_cols = [BRAND_COL, *HIERARCHY_COLS]
    daily = preprocess(raw, static_cols=static_cols)

    print("Building features ...")
    daily = add_all_features(daily)
    all_features = select_feature_columns(daily)

    print("Building direct-forecast samples ...")
    training_origins = make_origin_grid(daily)
    validation_origins = make_validation_origins(daily, n_folds=VALIDATION_FOLDS)
    all_origins = sorted(set(training_origins) | set(validation_origins))
    samples = build_samples(daily, all_origins, feature_cols=all_features)
    print(f"  samples={len(samples):,}  training origins={len(training_origins)}"
          f"  validation origins={[d.date() for d in validation_origins]}")

    with_competitive = all_features
    without_competitive = [c for c in all_features if c not in COMPETITIVE_FEATURE_NAMES]

    print("\nBacktesting baseline ...")
    baseline_result = rolling_origin_backtest(
        lambda: RecentAverageBaseline(), samples, with_competitive, validation_origins
    )
    print("Backtesting LightGBM WITHOUT competitive features ...")
    lgbm_plain = rolling_origin_backtest(
        LightGBMForecaster, samples, without_competitive, validation_origins
    )
    print("Backtesting LightGBM WITH competitive features ...")
    lgbm_full = rolling_origin_backtest(
        LightGBMForecaster, samples, with_competitive, validation_origins
    )

    report = pd.DataFrame(
        {
            "Baseline": baseline_result.aggregate,
            "LightGBM (no competitive)": lgbm_plain.aggregate,
            "LightGBM (full)": lgbm_full.aggregate,
        }
    ).T
    print("\n" + "=" * 60)
    print("AGGREGATE BACKTEST METRICS (lower is better except Bias->0)")
    print("=" * 60)
    print(report.round(3).to_string())

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report.round(4).to_csv(OUTPUTS_DIR / "backtest_metrics.csv")
    print(f"\nSaved metrics to {OUTPUTS_DIR / 'backtest_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-products",
        type=int,
        default=300,
        help="Number of products to sample (use a larger value or omit for all).",
    )
    parser.add_argument(
        "--all-products",
        action="store_true",
        help="Use every product (overrides --max-products).",
    )
    args = parser.parse_args()
    run(max_products=None if args.all_products else args.max_products)


if __name__ == "__main__":
    main()
