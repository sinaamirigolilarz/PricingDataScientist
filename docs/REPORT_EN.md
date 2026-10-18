# Pricing Data Scientist Assessment — Full Report (English)

**Task:** Forecast each product's **total sales quantity over the next 30 days**
from a retail `product × day` panel, and design a defensible
training / validation / evaluation pipeline. The emphasis is on **reasoning and
decision quality**, not code volume or heavy models.

This document is the end-to-end ("0 → 100") guide: environment setup, how to run
everything, the dataset we found, and detailed answers to every question in the
task, with the evidence behind each decision. A Persian version is in
[`REPORT_FA.md`](REPORT_FA.md). A concise ≤3-page version is in
[`TECHNICAL_NOTE_EN.md`](TECHNICAL_NOTE_EN.md).

---

## 1. Deliverables map

| Task requirement | Where it is |
|---|---|
| Runnable Jupyter Notebook (EDA + preprocessing + light experiment) | [`notebooks/pricing_forecast_assessment.ipynb`](../notebooks/pricing_forecast_assessment.ipynb) |
| Technical Note (≤3 pages) | [`TECHNICAL_NOTE_EN.md`](TECHNICAL_NOTE_EN.md) / [`TECHNICAL_NOTE_FA.md`](TECHNICAL_NOTE_FA.md) |
| Workflow diagram (Raw Data → Model Evaluation) | [`workflow_diagram.png`](workflow_diagram.png) |
| Parquet structure viewer + visualization | [`src/inspect_parquet.py`](../src/inspect_parquet.py) |
| Sample runs to see model outputs | [`src/sample_predictions.py`](../src/sample_predictions.py) |
| Tests | [`tests/`](../tests) (22 passing) |

---

## 2. Environment & hardware

Target machine: **Windows 10, Intel Core i7 (6th gen), 16 GB RAM, no GPU.**
Every choice respects this: the main model (LightGBM) trains in **seconds to a
few minutes on CPU** with a small memory footprint, and no deep learning is used.

- **Python 3.10** is used via a virtual environment. The machine also has Python
  3.14, but 3.10 was chosen because the full data-science stack
  (numpy/pandas/scikit-learn/lightgbm/pyarrow) ships mature, pre-built Windows
  wheels for it — no compilation, no surprises.

---

## 3. Installation (from zero)

Open **PowerShell** in the project root
(`C:\aliasaas\sina\OfoghTask_PricingDataScientist`).

### 3.1 Create and activate a virtual environment
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, allow it for the current user once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3.2 Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This installs numpy, pandas, pyarrow, scikit-learn, lightgbm, matplotlib,
seaborn, jdatetime (Persian-calendar parsing), and the notebook/test tooling.
Pinned versions are in [`requirements.txt`](../requirements.txt).

> **Tip:** if `pip install` is slow, install the core stack first
> (`pip install numpy pandas pyarrow scikit-learn lightgbm matplotlib seaborn jdatetime`)
> and the notebook tools afterwards — the `jupyter` meta-package pulls a large
> dependency tree and can be deferred.

---

## 4. Project structure
```
OfoghTask_PricingDataScientist/
├── candidate_dataset.parquet          # input data (product × day)
├── requirements.txt
├── README.md
├── src/
│   ├── config.py                      # paths, column names, modeling constants
│   ├── data_io.py                     # load + Jalali→Gregorian calendar handling
│   ├── inspect_parquet.py             # structure printout + EDA figures
│   ├── preprocessing.py               # missing/zero/stock logic, winsorizing
│   ├── features.py                    # calendar / own-history / competitive features
│   ├── samples.py                     # direct-forecast sample construction
│   ├── models.py                      # baseline + LightGBM
│   ├── evaluation.py                  # metrics + rolling-origin backtest
│   ├── run_experiment.py              # end-to-end backtest comparison
│   └── sample_predictions.py          # actual-vs-predicted outputs
├── tests/                             # pytest unit tests (synthetic data)
├── notebooks/pricing_forecast_assessment.ipynb
├── docs/                              # this report + technical notes + diagram
└── outputs/                           # generated figures + metrics CSV
```

---

## 5. How to run everything

All commands assume the venv is active and you are in the project root.

```powershell
# 1) Inspect the Parquet structure and write EDA figures to outputs/
python src/inspect_parquet.py

# 2) End-to-end backtest: baseline vs LightGBM (with/without competitive features)
python src/run_experiment.py --max-products 300     # fast subset
python src/run_experiment.py --all-products         # full 1,235 products

# 3) See concrete model outputs (actual vs predicted + feature importance)
python src/sample_predictions.py --max-products 300 --show 12

# 4) Run the tests
python -m pytest tests/ -q

# 5) Execute the notebook end to end
python -m nbconvert --to notebook --execute --inplace `
    notebooks/pricing_forecast_assessment.ipynb
```
The `--max-products` flag runs on a subset so everything finishes in minutes on a
CPU-only laptop (the task explicitly allows running on a subset under limited
compute). Use `--all-products` for the full dataset.

---

## 6. Dataset profile (what the data actually is)

Discovered with `src/inspect_parquet.py`:

- **902,438 rows**, 13 columns, **1,235 products**, **234 brands**.
- **Dates are Jalali (Persian) calendar strings** `YYYYMMDD` (e.g. `14030631`).
  Span: **1403/06/31 → 1405/06/31** (Gregorian 2024-09-21 → 2026-09-22), **732
  days**. We convert to Gregorian for day arithmetic and keep the Jalali month
  for seasonality. This was a key finding — a naive date parse fails because
  Jalali month 6 has 31 days.
- **No null cells**; "missingness" is therefore only about absent `(product, day)`
  rows. The panel is **~99.8% dense** (most products have a row almost every day;
  the shortest-history product has 492 days), so true missing records are rare
  but the pipeline still handles them correctly.
- **Target `sale_qty` is continuous/fractional** (weight-based items), heavy
  right tail (mean ≈ 766, max ≈ 129,289 per product-day), **2.5% exact zeros**.
- **`discount_pct`** is on a **0–100 scale** (max 82.2). **`price`** is in Rial
  (up to ~22.8M), with some zeros. Hierarchy sizes: Level1=5, …, **Level4=88**,
  Level5=289. Level4 groups hold **2–49 products** (median 10) — a good grain for
  defining substitutes.

EDA figures: `outputs/01_target_distribution.png` … `05_*.png`.

---

## 7. Answers to the task questions

### 7.1 Data understanding & preprocessing

**Missing records vs. zero sales vs. zero stock.** These are three different
things and we treat them differently:
- **Missing record** = no row for a `(product, day)`. For an active product this
  usually means "not observed", not "zero demand". We reindex each product to a
  continuous daily calendar **within its own first–last observed dates** (not the
  global range, so a late-launched product gets no fabricated pre-launch history),
  set `sale_qty = 0` for the reconstructed day, and flag it with `is_imputed_day`
  so the model can discount it rather than trust it as a hard zero.
- **Zero sale** = a recorded row with `sale_qty = 0`: a genuine demand signal
  (tracked that day, sold nothing). Kept as `0`.
- **Zero stock** (`stock_qty = 0` and `num_stores_with_stock = 0`) = the product
  could not sell. A zero sale under zero stock is a **stock-out / censored**
  observation, not low demand. We flag `is_stockout` and define
  `is_true_zero_demand` = zero sale **while in stock**, so the model can tell real
  zero-demand days from availability-driven zeros.

**Outliers / sales spikes.** Detected **per product** with a robust
median/MAD rule (`robust z > 8`). We **winsorize (cap), not delete**, because most
spikes are real promotion- or event-driven demand we want the model to learn; only
the most extreme values (likely data-entry errors) are capped. The original value
is kept in `sale_qty_raw` and the action is flagged with `is_spike_capped`. Not
all spikes should be removed — deleting promotional peaks would bias the model to
under-forecast exactly when demand matters most.

**Short history / sparse / many zero-sale days.** We (a) keep a minimum history
window and fall back gracefully via `min_periods=1` on rolling features; (b) add
intermittency features — `days_since_last_sale`, `zero_rate_{7,14,28,56}d` — so the
model recognizes sparse items; (c) for very short-history products the recent
**group-level** (Level4) signal and the baseline provide a sensible fallback. The
chosen 30-day *sum* target is also naturally more robust for sparse items than
daily forecasting.

**Scaling.** The main model is **tree-based (LightGBM), which is
scale-invariant**, so we do **not** scale inputs for it — avoiding an extra,
leakage-prone transform. If a linear or neural model were used, we would apply
**per-product or per-group robust scaling** (median/IQR, robust to the heavy tail)
and — crucially — **fit the scaler only on the training fold** and apply it to
validation/test, never on the full series.

**Two likely data-leakage sources (and prevention).**
1. **Target leakage from rolling/lag features.** Any rolling mean/lag that
   includes the current (or future) day leaks the label. **Prevention:** every
   rolling feature is computed on a `shift(1)` series (strictly past), and
   training labels use only forward windows that end before the validation origin.
2. **Look-ahead from global statistics / splits.** Scaling, winsorizing
   thresholds, or any statistic fit on the whole series — or a **random**
   train/test split — lets future information inform the past. **Prevention:**
   time-based split only, and any fitted statistic is computed inside the training
   fold. (A third, domain-specific one: competitive features built from peers'
   *realized future* sales — handled in §7.2 by using only trailing values.)

### 7.2 Competitive feature engineering (substitute products)

Substitutes are defined as products sharing the same **Level4** group (the data
dictionary calls Level4 a level usable for substitutes; with 2–49 products per
group it captures genuine competitors without isolating items). We engineer
**six** competitive features. For each: *definition · business logic · prediction-
time computability.*

1. **`price_rank_in_group`** — percentile rank of the product's price among its
   group that day. *Cheaper than rivals → captures switching demand.*
   **Future-known** if the future price plan is set; else carry the last price.
2. **`relative_discount`** — product discount minus the group's mean discount.
   *Deeper promotion than rivals pulls demand.* **Future-known** if the promo
   calendar is planned; else last-known.
3. **`price_gap_to_cheapest`** — product price minus the cheapest substitute's
   price. *A large positive gap pushes customers to alternatives.*
   **Future-known** with a price plan; else last-known.
4. **`sales_share_in_group`** — product's share of its group's total sales over a
   trailing 28-day window. *Incumbents hold share; a proxy for brand/position
   strength.* **Historical only** (needs realized peer sales) → at prediction time
   we carry the last trailing value forward (slow-moving).
5. **`group_sales_momentum`** — recent vs. older group-total-sales growth.
   *The whole category can be rising/falling (seasonality, trend).* **Historical**
   → carried forward.
6. **`relative_store_coverage`** — product store coverage ÷ group mean coverage.
   *Wider distribution than rivals → more demand captured.* **Historical /
   partially future-known** if the assortment plan is known.

**How they enter the model and how we validate their value.** They are plain
numeric columns in the same LightGBM feature matrix. We measure their value with
an **ablation under the identical backtest**: train LightGBM *with* vs *without*
the competitive block and compare metrics; we also read **feature importances**.
Result (full data, see §8): adding them lowers WMAPE **0.189 → 0.183** and RMSE
**14,712 → 13,480**, and several rank in the top of importance
(`group_sales_momentum` #2, `sales_share_in_group`, `relative_store_coverage`,
`price_rank_in_group`). So they add real, not cosmetic, value.

### 7.3 Model selection & training design

**Baseline — recent-average:** 30-day forecast = recent mean daily sales × 30.
Transparent and genuinely hard to beat for stable high-volume items.

**Main model — LightGBM (gradient-boosted trees), Tweedie objective.** Chosen
because the problem is **tabular and medium-sized**, it runs **fast on CPU with
low memory** (fits the hardware, no GPU), handles missing values and mixed feature
types natively, and captures non-linear interactions (price × discount × season)
that a linear model misses. Deep learning was deliberately avoided: without a GPU
it is slow, and for ~1.2k series over 2 years it offers no clear advantage.

**History window.** We summarize the recent **90 days** into features (lags,
rolling 7/14/28/56-day statistics). Rationale: long enough to capture ~a quarter
of seasonality, short enough to keep many training origins (including for
short-history products). The window length is a **tunable** decision — the
notebook/experiment make it easy to re-run with different rolling spans and
compare backtest error.

**Direct vs. recursive.** We use a **direct** forecast of the 30-day **sum** (one
model predicts the aggregate). *Pros:* optimizes exactly the reported quantity and
avoids the error accumulation of rolling a daily model 30 steps forward. *Cons:*
no daily shape, and a separate target per horizon if other horizons are needed.
*Recursive/daily-then-aggregate* gives daily granularity but compounds errors and
needs future covariates for each step — unnecessary here since only the 30-day sum
is requested.

**Static / historical / future-known features.** We separate them explicitly:
*Static* = brand, Level1–5 (product attributes); *Historical* = lags, rolling
stats, intermittency, trailing competitive features (known only up to the origin);
*Future-known* = calendar features (day-of-week, Iran weekend, Jalali month) and,
where a plan exists, future price/discount. This separation is what makes the
pipeline leakage-safe and lets the model use planned promotions at inference.

**Loss function.** **Tweedie deviance** (`variance_power ≈ 1.2`). Demand here is
**non-negative, zero-inflated, right-skewed** — Tweedie models exactly this family
and sits between Poisson and Gamma. Versus plain MSE it is far less dominated by a
few huge sellers, so **low-volume products are not ignored**; versus MAE it still
respects the multiplicative nature of demand. For **spikes**, the capping in
preprocessing plus Tweedie's variance assumption keeps a few extreme days from
distorting the fit, while WMAPE/RMSE in evaluation reveal how spikes are handled.

### 7.4 Validation & evaluation

**Why not random split.** The problem is temporal; a random split would train on
future days and test on past ones, leaking information and inflating scores. We
split **strictly forward in time**.

**Backtesting design — rolling origin.** We evaluate at **3 recent origins spaced
30 days apart** (non-overlapping forecast windows). At each origin we **retrain**
on samples whose forward label ends on or before that origin (so training labels
are fully in the past) and score the held-out origin. This yields an **error
distribution across folds**, not one fragile number, and mirrors how the model
would be refit in production. Training samples themselves use **dense weekly
origins** to maximize data.

**Metrics (≥2, complementary).**
- **WMAPE** (volume-weighted MAPE): aggregate business accuracy; robust to the
  near-zero actuals that break plain MAPE on intermittent items.
- **MAE**: average miss in units, treating all products equally.
- **RMSE**: penalizes large misses — sensitive to spike/high-volume errors.
- **Bias** (mean signed error): detects systematic over/under-forecasting, which
  matters directly for inventory (stock-outs vs. overstock).

### 7.5 Experiment & results

Baseline vs. LightGBM (with and without competitive features) under the **same**
rolling-origin backtest — see §8. LightGBM beats the baseline on WMAPE, MAE and
Bias, and the competitive features give a further, consistent improvement
(especially on RMSE, i.e. fewer large misses). The held-out-origin run and the
feature importances corroborate the story.

---

## 8. Results

**Full dataset — 1,235 products, 111,150 samples, 3-fold rolling-origin backtest**
(lower is better; Bias closer to 0 is better):

| Model | WMAPE | MAE | RMSE | Bias |
|---|---|---|---|---|
| Baseline (recent average) | 0.198 | 4,704 | 13,247 | −1,079 |
| LightGBM (no competitive) | 0.189 | 4,480 | 14,712 | −1,046 |
| **LightGBM (full)** | **0.183** | **4,342** | **13,480** | **−852** |

**Held-out origin (2026-08-23), all products:**

| Model | WMAPE | MAE | RMSE | Bias |
|---|---|---|---|---|
| Baseline | 0.197 | 4,889 | 13,763 | −1,875 |
| **LightGBM** | **0.160** | **3,966** | **12,643** | **−439** |

**Reading the evidence.**
- LightGBM beats the baseline: WMAPE **0.183 vs 0.198** (~8% relative), MAE ~8%
  lower, and Bias much closer to zero (−852 vs −1,079) — less systematic
  under-forecasting, which directly reduces stock-out risk.
- Competitive features add real value: WMAPE **0.189 → 0.183**, and a large RMSE
  drop **14,712 → 13,480** (they cut the big misses most).
- Top feature importances mix own-history (`sales_std_56d`, `sales_mean_56d`),
  calendar (`jalali_month`) and **competitive** (`group_sales_momentum` #2,
  `sales_share_in_group`, `relative_store_coverage`, `price_rank_in_group`).

Figures: `outputs/06_actual_vs_predicted.png` (held-out scatter), metrics in
`outputs/backtest_metrics.csv`.

---

## 9. Limitations & next steps
- Experiments reported on the full set; heavier search (hyper-parameter tuning,
  more folds) can run on a subset first, then scale — as the task allows.
- Promotion/price *plans* for the future horizon are not in the data, so
  price/discount competitive features are carried forward; with a real promo
  calendar they become genuinely future-known and should help more.
- Possible extensions: per-group or hierarchical models, quantile forecasts for
  safety stock, and modeling the stock-out censoring explicitly (e.g. only
  training the demand signal on in-stock days).
