# Technical Note — 30-Day Demand Forecasting (English)

*Concise version (≤3 pages). Full detail: [`REPORT_EN.md`](REPORT_EN.md).*

## 1. Problem & data
Forecast each product's **total `sale_qty` over the next 30 days** from a
`product × day` retail panel (1,235 products, 234 brands, 902k rows). Dates are
**Jalali `YYYYMMDD`** strings over 732 days (2024-09-21 → 2026-09-22); we convert
to Gregorian for arithmetic and keep the **Jalali month** for seasonality. No null
cells; the panel is ~99.8% dense; `sale_qty` is continuous, right-skewed, 2.5%
zeros; `discount_pct` is 0–100; Level4 (88 groups of 2–49 products) defines
substitutes.

## 2. Key preprocessing decisions
- **Missing record ≠ zero sale ≠ zero stock.** Reindex each product to a daily
  calendar within its own lifespan and flag imputed days (`is_imputed_day`);
  keep recorded zeros as real demand; flag stock-outs (`is_stockout`) and define
  `is_true_zero_demand` = zero sale **while in stock**, so availability-censored
  zeros are not confused with no-demand.
- **Spikes are winsorized, not deleted** (robust median/MAD per product): most
  spikes are real promotions; only extreme likely-errors are capped, with the
  original kept in `sale_qty_raw`.
- **Short/sparse history:** intermittency features (`days_since_last_sale`,
  `zero_rate_*`), graceful `min_periods=1` rolling, and group/baseline fallback.
- **Scaling:** none needed for the tree model (scale-invariant); for linear/NN we
  would fit **robust, per-group** scalers **inside the training fold only**.
- **Leakage prevention:** all lag/rolling features use `shift(1)` (strictly past);
  time-based split only; any fitted statistic stays inside the fold; competitive
  features use only trailing peer data.

## 3. Proposed features
Three families: **calendar** (future-known: weekday, Iran weekend, Jalali month),
**own-history** (lags, rolling 7/14/28/56-day mean/std/sum, zero-rate,
intermittency), and **competitive** (vs. Level4 substitutes):

| Feature | Business logic | At prediction time |
|---|---|---|
| `price_rank_in_group` | cheaper than rivals → switching demand | future-known w/ price plan |
| `relative_discount` | deeper promo pulls demand | future-known w/ promo plan |
| `price_gap_to_cheapest` | big gap → customers switch away | future-known w/ price plan |
| `sales_share_in_group` | incumbents hold share | historical (carry forward) |
| `group_sales_momentum` | category rising/falling | historical (carry forward) |
| `relative_store_coverage` | wider distribution → more demand | historical / partly future |

Value is validated by **ablation** (with vs. without, same backtest) + feature
importance.

## 4. Model design
- **Baseline:** recent-average (mean daily sales × 30) — transparent, hard to beat
  for stable items.
- **Main model:** **LightGBM, Tweedie objective** — tabular, **CPU-fast,
  low-memory (no GPU)**, handles missing values and non-linear interactions;
  Tweedie fits non-negative, zero-inflated, right-skewed demand and does not let a
  few huge sellers dominate (protects low-volume items). Deep learning avoided
  (no GPU, no clear benefit at this scale).
- **Formulation:** **Direct** 30-day-sum forecast — optimizes the reported
  quantity and avoids recursive error accumulation (trade-off: no daily shape).
- **History window:** 90 days (tunable via the experiment script).
- **Static / historical / future-known** features are separated explicitly, which
  is what keeps the pipeline leakage-safe.

## 5. Evaluation
- **No random split** (temporal problem). **Rolling-origin backtest**: 3 recent
  origins spaced 30 days; retrain per fold on labels fully in the past; report an
  error distribution. Dense weekly training origins maximize samples.
- **Metrics:** WMAPE (volume-weighted accuracy, robust to near-zero actuals),
  MAE (per-product miss), RMSE (penalizes big/spike misses), Bias (systematic
  over/under-forecast → inventory risk).

## 6. Results (full data, 3-fold backtest)

| Model | WMAPE | MAE | RMSE | Bias |
|---|---|---|---|---|
| Baseline | 0.198 | 4,704 | 13,247 | −1,079 |
| LightGBM (no competitive) | 0.189 | 4,480 | 14,712 | −1,046 |
| **LightGBM (full)** | **0.183** | **4,342** | **13,480** | **−852** |

Held-out origin (2026-08-23): LightGBM WMAPE **0.160** vs baseline **0.197**.

**Conclusion.** LightGBM beats the baseline (~8% lower WMAPE/MAE, bias closer to
zero), and competitive features add real value (WMAPE 0.189→0.183, RMSE
14,712→13,480; several rank top in importance — `group_sales_momentum`,
`sales_share_in_group`, `relative_store_coverage`). The chosen model is justified
by better accuracy under an identical, leakage-safe, time-based evaluation, while
staying well within the CPU/16 GB budget.
