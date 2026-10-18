"""Show concrete model outputs on a held-out forecast origin.

This is the "see the model's output" entry point: it trains the baseline and the
LightGBM model on history, forecasts the 30-day sum for a held-out origin, and
prints an actual-vs-predicted table for a sample of products plus the top
feature importances. It complements run_experiment.py (which reports aggregate
backtest metrics).

Usage (from the project root, with the venv active):
    python src/sample_predictions.py --max-products 300 --show 12
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from config import BRAND_COL, HIERARCHY_COLS, OUTPUTS_DIR, PRODUCT_COL
from data_io import load_raw
from evaluation import compute_metrics
from features import add_all_features
from models import LightGBMForecaster, RecentAverageBaseline
from preprocessing import preprocess
from run_experiment import select_feature_columns, subset_products
from samples import (
    ORIGIN_COL,
    TARGET_SUM_COL,
    build_samples,
    make_origin_grid,
    make_validation_origins,
)

TOP_FEATURES_TO_SHOW = 15


def _prepare_samples(max_products: int):
    raw = subset_products(load_raw(), max_products)
    daily = preprocess(raw, static_cols=[BRAND_COL, *HIERARCHY_COLS])
    daily = add_all_features(daily)
    feature_cols = select_feature_columns(daily)

    holdout_origin = make_validation_origins(daily, n_folds=1)[0]
    origins = sorted(set(make_origin_grid(daily)) | {holdout_origin})
    samples = build_samples(daily, origins, feature_cols=feature_cols)
    return samples, feature_cols, holdout_origin


def _split_train_holdout(samples: pd.DataFrame, holdout_origin, horizon: int = 30):
    train_cutoff = holdout_origin - pd.Timedelta(days=horizon)
    train = samples[samples[ORIGIN_COL] <= train_cutoff]
    holdout = samples[samples[ORIGIN_COL] == holdout_origin]
    return train, holdout


def _print_sample_table(holdout: pd.DataFrame, predictions: dict, show: int) -> None:
    table = holdout[[PRODUCT_COL, TARGET_SUM_COL]].copy()
    table = table.rename(columns={TARGET_SUM_COL: "actual_30d"})
    for name, preds in predictions.items():
        table[name] = preds.round(0)
    table = table.sort_values("actual_30d", ascending=False).head(show)
    print("\nActual vs predicted 30-day sales (top sellers in the holdout):")
    print(table.to_string(index=False))


def _plot_actual_vs_predicted(holdout: pd.DataFrame, lgbm_pred) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    actual = holdout[TARGET_SUM_COL]
    ax.scatter(actual, lgbm_pred, s=8, alpha=0.3)
    upper = max(actual.max(), lgbm_pred.max())
    ax.plot([0, upper], [0, upper], "r--", linewidth=1)
    ax.set(
        xlabel="actual 30-day sales",
        ylabel="LightGBM predicted",
        title="Held-out origin: actual vs predicted",
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUTS_DIR / "06_actual_vs_predicted.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def run(max_products: int | None, show: int) -> None:
    samples, feature_cols, holdout_origin = _prepare_samples(max_products)
    train, holdout = _split_train_holdout(samples, holdout_origin)
    print(f"Holdout origin : {holdout_origin.date()}")
    print(f"Train samples  : {len(train):,}   Holdout products: {len(holdout):,}")

    baseline = RecentAverageBaseline().fit(train[feature_cols], train[TARGET_SUM_COL])
    lgbm = LightGBMForecaster().fit(train[feature_cols], train[TARGET_SUM_COL])

    baseline_pred = baseline.predict(holdout[feature_cols])
    lgbm_pred = lgbm.predict(holdout[feature_cols])
    truth = holdout[TARGET_SUM_COL].to_numpy()

    _print_sample_table(
        holdout, {"baseline": baseline_pred, "lightgbm": lgbm_pred}, show
    )

    print("\nHoldout metrics:")
    report = pd.DataFrame(
        {
            "Baseline": compute_metrics(truth, baseline_pred),
            "LightGBM": compute_metrics(truth, lgbm_pred),
        }
    ).T
    print(report.round(3).to_string())

    print(f"\nTop {TOP_FEATURES_TO_SHOW} LightGBM feature importances:")
    print(lgbm.feature_importance.head(TOP_FEATURES_TO_SHOW).to_string())

    _plot_actual_vs_predicted(holdout, lgbm_pred)
    print(f"\nSaved scatter to {OUTPUTS_DIR / '06_actual_vs_predicted.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-products", type=int, default=300)
    parser.add_argument("--all-products", action="store_true")
    parser.add_argument("--show", type=int, default=12, help="Rows in the sample table.")
    args = parser.parse_args()
    run(None if args.all_products else args.max_products, args.show)


if __name__ == "__main__":
    main()
