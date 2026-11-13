"""Shared configuration for the LightGBM promo-information experiment (Hoàng).

Research question:
    Does incorporating *known future* promotional information improve
    daily retail demand forecasting accuracy?
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                # Hoang_LightGBM/
DATA_RAW = ROOT / "data" / "daily_panel.parquet"          # copy of ../dataset/daily_panel.parquet
DATA_FEAT = ROOT / "data" / "features.parquet"            # produced by 01_build_features.py

# Project layout
#   1_model/       model code + everything the model produces (models, predictions, tables, Excel)
#   2_report/      code that turns those results into the PDF report (+ its figures)
#   3_explainers/  beginner guides (metrics guide, model picture guide)
# HOANG_OUT / HOANG_DRYRUN are only used for a quick smoke test: all outputs then go under HOANG_OUT.
_BASE = Path(os.environ["HOANG_OUT"]) if os.environ.get("HOANG_OUT") else ROOT
DRYRUN = os.environ.get("HOANG_DRYRUN") == "1"
MODEL_DIR = _BASE / "1_model"
REPORT_DIR = _BASE / "2_report"
EXPLAIN_DIR = _BASE / "3_explainers"
MOD = MODEL_DIR / "models"            # LightGBM boosters, best params
PRED = MODEL_DIR / "predictions"      # predictions_{valid,test}.parquet, predictions_test.csv
TAB = MODEL_DIR / "tables"            # every result table (CSV)
FIG = REPORT_DIR / "figures"          # report figures
EXPLAIN_FIG = EXPLAIN_DIR / "figures" # figures of the beginner guides
for p in (MOD, PRED, TAB, FIG, EXPLAIN_FIG):
    p.mkdir(parents=True, exist_ok=True)

TARGET = "quantity"
KEYS = ["item_id", "store_id", "date"]
HORIZON = 7          # every lag/rolling feature is shifted >= 7 days -> 7-day-ahead forecast
SEED = 42
N_THREADS = 4 if DRYRUN else 12

# Official time split from the team pipeline (column `split` in daily_panel)
TRAIN_END = "2024-03-31"
VALID_START, VALID_END = "2024-04-01", "2024-06-30"
TEST_START, TEST_END = "2024-07-01", "2024-09-26"
# Hyper-parameter tuning (02_tune.py): manual grid search, each combination trained on TRAIN and
# scored on the hold-out VALID period with early stopping. The test period is never used for tuning.

CATEGORICAL = ["store_id", "group4", "dept_name", "class_name", "subclass_name", "item_type", "promo_type"]

# ---------------------------------------------------------------- feature sets
# A  : demand history + calendar + static product/store + regular shelf price.
#      NO promotion information at all.
CALENDAR = [
    "dow", "day_of_month", "week", "month", "year", "is_weekend", "is_friday",
    "ny_break", "is_holiday", "holiday_status", "minor_holiday",
    "days_to_holiday", "days_after_holiday", "pheic", "store4_period",
]
HISTORY = [
    "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_28", "roll_mean_91",
    "roll_zero_share_28", "dow_mean_4w", "ewm_7", "hist_mean", "series_age",
]
STATIC = [
    "store_id", "area", "group4", "is_food", "dept_name", "class_name",
    "subclass_name", "item_type", "weight_volume", "weight_netto",
    "regular_price", "log_regular_price",
]
# A+ : A + promotion information known only from the PAST (as of the forecast
#      origin t-7). Mirrors the MLP in the "Hierarchical" paper, which only saw
#      historical exogenous variables.
PAST_PROMO = [
    "promo_lag7", "depth_lag7", "day_in_promo_lag7", "days_since_promo_lag7",
    "promo_share_28_lag7", "hist_promo_lift",
]
# B  : A+ + promotion information for the forecast day itself, which the
#      retailer knows in advance because it schedules promotions.
FUT_FLAG = ["promo_status", "in_promo_any"]
FUT_DEPTH = ["depth", "price", "price_ratio", "promo_type", "depth_x_food"]
FUT_TIMING = [
    "day_in_promo", "days_to_promo_end", "promo_duration", "days_since_last_promo",
    "post_promo_1_7", "days_to_next_promo", "promo_days_next_7",
]

BASE_A = CALENDAR + HISTORY + STATIC
FEATURE_SETS = {
    "A_no_promo":        BASE_A,
    "A+_past_promo":     BASE_A + PAST_PROMO,
    "B1_+future_flag":   BASE_A + PAST_PROMO + FUT_FLAG,
    "B2_+future_depth":  BASE_A + PAST_PROMO + FUT_FLAG + FUT_DEPTH,
    "B_full_future":     BASE_A + PAST_PROMO + FUT_FLAG + FUT_DEPTH + FUT_TIMING,
}
MAIN_A, MAIN_B = "A_no_promo", "B_full_future"

# Columns that must NEVER be used as features (leakage / analysis labels)
FORBIDDEN = ["quantity", "revenue", "qty_norm", "phase", "status6", "density", "split"]
