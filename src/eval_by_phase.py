"""Tính MAE, RMSE, WAPE, bias theo giai đoạn khuyến mãi (phase) trên tập test, từ dự báo đã lưu.

Đầu vào : output/lstm/test_predictions.parquet (dự báo của mọi mô hình), data_model/daily_panel.parquet (cột phase)
Đầu ra  : output/lstm/metrics_test_by_phase.csv, output/lstm/metrics_test_by_phase_category.csv

phase: pre_1_3 (1–3 ngày trước KM), during (trong KM), post_1_3 / post_4_7 (1–3 / 4–7 ngày sau KM), other.
Chạy: python src/eval_by_phase.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "lstm"
PHASE_ORDER = ["pre_1_3", "during", "post_1_3", "post_4_7", "other"]

rows = pd.read_parquet(OUT / "test_predictions.parquet")
panel = pd.read_parquet(ROOT / "data_model" / "daily_panel.parquet", columns=["date", "item_id", "store_id", "phase"])
n0 = len(rows)
rows = rows.drop(columns=["phase"], errors="ignore").merge(panel, on=["date", "item_id", "store_id"], how="left")
assert len(rows) == n0 and rows.phase.notna().all(), "Không ghép được phase cho mọi dòng test"

MODELS = [m for m in ["lstm", "hgb", "hgb_default", "seasonal_naive", "ma7"] if m in rows.columns]


def m(g):
    y = g.quantity.values
    out = {"n": len(g), "sum_qty": y.sum()}
    for k in MODELS:
        e = g[k].values - y
        out[f"{k}_WAPE"] = np.abs(e).sum() / y.sum()
        out[f"{k}_MAE"] = np.abs(e).mean()
        out[f"{k}_RMSE"] = np.sqrt((e ** 2).mean())
        out[f"{k}_bias"] = e.sum() / y.sum()
    return pd.Series(out)


rows["phase"] = pd.Categorical(rows.phase, PHASE_ORDER, ordered=True)
by_phase = rows.groupby("phase", observed=True).apply(m, include_groups=False)
by_phase_cat = rows.groupby(["category", "phase"], observed=True).apply(m, include_groups=False)
by_phase.to_csv(OUT / "metrics_test_by_phase.csv")
by_phase_cat.to_csv(OUT / "metrics_test_by_phase_category.csv")

pd.set_option("display.width", 250)
for name, t in [("test_by_phase", by_phase), ("test_by_phase_category", by_phase_cat)]:
    for metric in ["WAPE", "MAE"]:
        cols = [f"{k}_{metric}" for k in MODELS]
        best = t[cols].idxmin(axis=1).str.replace(f"_{metric}", "")
        print(f"\n### {name} — {metric}\n{t[cols + ['n']].round(4).assign(tot_nhat=best).to_string()}")
