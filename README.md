# Pricing Data Scientist Assessment — 30-Day Demand Forecasting

Forecast each product's **total sales over the next 30 days** from a retail
`product × day` Parquet panel. Baseline vs. **LightGBM** (CPU-only, no GPU),
with competitive "substitute product" features, under a leakage-safe
time-based backtest.

پیش‌بینی **مجموع فروش هر کالا در ۳۰ روز آینده** از یک پنل `کالا × روز`.
مقایسه‌ی Baseline و **LightGBM** (فقط CPU، بدون GPU) با ویژگی‌های رقابتیِ کالاهای
جایگزین، تحت یک بک‌تست زمانیِ بدون‌نشت.

## Quickstart
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python src/inspect_parquet.py                         # structure + EDA figures
python src/run_experiment.py --max-products 300       # baseline vs LightGBM
python src/sample_predictions.py --max-products 300    # see model outputs
python -m pytest tests/ -q                            # 22 tests
```
Use `--all-products` for the full 1,235-product dataset. Everything runs on a
CPU-only laptop within the time/compute budget.

## Documentation
- **Full report (0 → 100):** [`docs/REPORT_EN.md`](docs/REPORT_EN.md) ·
  [`docs/REPORT_FA.md`](docs/REPORT_FA.md) — setup, run, data profile, and
  detailed answers to every task question with evidence.
- **Technical Note (≤3 pages):** [`docs/TECHNICAL_NOTE_EN.md`](docs/TECHNICAL_NOTE_EN.md) ·
  [`docs/TECHNICAL_NOTE_FA.md`](docs/TECHNICAL_NOTE_FA.md)
- **Workflow diagram:** [`docs/workflow_diagram.png`](docs/workflow_diagram.png)
- **Notebook:** [`notebooks/pricing_forecast_assessment.ipynb`](notebooks/pricing_forecast_assessment.ipynb)

## Headline result (full data, 3-fold rolling-origin backtest)
| Model | WMAPE | MAE | RMSE | Bias |
|---|---|---|---|---|
| Baseline (recent average) | 0.198 | 4,704 | 13,247 | −1,079 |
| LightGBM (no competitive) | 0.189 | 4,480 | 14,712 | −1,046 |
| **LightGBM (full)** | **0.183** | **4,342** | **13,480** | **−852** |

LightGBM beats the baseline and the competitive features add consistent value.

## Notes
- **Code & comments:** English. **Docs & notes:** bilingual (EN/FA).
- The dataset's `date` column is the **Jalali (Persian) calendar** (`YYYYMMDD`);
  it is converted to Gregorian and the Jalali month is used for seasonality.
- Environment: Windows, Python 3.10, Intel i7 (6th gen), 16 GB RAM, **no GPU**.
