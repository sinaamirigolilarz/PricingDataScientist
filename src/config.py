"""Central configuration: paths, column names, and modeling constants.

Keeping these in one place avoids magic strings/numbers scattered across
modules and makes the forecasting assumptions explicit and reviewable.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "candidate_dataset.parquet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- Schema: column names --------------------------------------------------
DATE_COL = "date"
PRODUCT_COL = "product_id"
BRAND_COL = "brand_id"
TARGET_COL = "sale_qty"
STOCK_COL = "stock_qty"
NUM_STORES_COL = "num_stores_with_stock"
DISCOUNT_COL = "discount_pct"
PRICE_COL = "price"

# Category hierarchy from coarsest (Level1) to finest (Level5).
HIERARCHY_COLS = ["Level1", "Level2", "Level3", "Level4", "Level5"]

# Level used to define "substitute products" for competitive features.
# Level4 is called out in the data dictionary as a level usable for defining
# substitutes; it is granular enough to group genuine competitors without
# being so fine that each product sits alone.
SUBSTITUTE_LEVEL = "Level4"

# --- Forecasting setup -----------------------------------------------------
# Business objective: predict the SUM of sale_qty over the next 30 days
# for each product (a single direct forecast per product per origin).
FORECAST_HORIZON_DAYS = 30

# Length of recent history summarized into features at each forecast origin.
# 90 days balances capturing ~a quarter of seasonality against keeping enough
# training origins for short-history products (justified empirically in the notebook).
HISTORY_WINDOW_DAYS = 90

# Rolling-window lengths (days) used to build lag/rolling features.
ROLLING_WINDOWS = [7, 14, 28, 56]

RANDOM_SEED = 42
