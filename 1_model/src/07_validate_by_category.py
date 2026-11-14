"""Step 7 - Validation (and test) results split by product category: food vs household.

Reads 1_model/predictions/predictions_{valid,test}.parquet produced by 03_train_eval.py
(no retraining needed) and writes:
  tables/metrics_by_category.csv       every model, valid + test, by category (food / household)
                                       and by product group (F1, F2 = food; H1, H2 = household)
  tables/significance_by_category.csv  A vs B and A+ vs B inside each category on VALID and TEST:
                                       Diebold-Mariano, series-cluster bootstrap 95% CI of dWAPE, Wilcoxon
"""
import numpy as np
import pandas as pd
from scipy import stats
from common import metrics, dm_test, cluster_bootstrap
from config import PRED, TAB

pv = pd.read_parquet(PRED / "predictions_valid.parquet")
pt = pd.read_parquet(PRED / "predictions_test.parquet")
MODELS = ["SeasonalNaive", "MovingAvg28", "A_no_promo", "A+_past_promo",
          "B1_+future_flag", "B2_+future_depth", "B_full_future",
          "A_no_promo@Bparams", "B_full_future@Aparams"]
CATS = ["food", "household"]

# ------------------------------------------------------------------ metrics by category / group
rows = []
for split, f in (("valid", pv), ("test", pt)):
    for level, col in (("all", None), ("category", "category"), ("group4", "group4")):
        parts = [("all", f)] if col is None else f.groupby(col, observed=True)
        for val, g in parts:
            for m in MODELS:
                rows.append({"split": split, "level": level, "value": str(val), "model": m,
                             **metrics(g["quantity"], g[f"pred_{m}"])})
mc = pd.DataFrame(rows)
mc.to_csv(TAB / "metrics_by_category.csv", index=False)
pd.set_option("display.width", 250)

# compact overall table: one row per category x model, valid and test side by side
ov = mc[mc.level.eq("category")].pivot_table(index=["value", "model"], columns="split",
                                             values=["WAPE", "MAE", "RMSE", "Bias_pct", "n", "actual_sum"], sort=False)
ov.columns = [f"{s}_{k}" for k, s in ov.columns]
ov = ov[[f"{s}_{k}" for s in ("valid", "test") for k in ("n", "actual_sum", "WAPE", "MAE", "RMSE", "Bias_pct")]]
ov = ov.reset_index().rename(columns={"value": "category"})
ov["model"] = pd.Categorical(ov["model"], MODELS)
ov = ov.sort_values(["category", "model"])
ov[[c for c in ov if c.endswith("_n")]] = ov[[c for c in ov if c.endswith("_n")]].astype(int)
ov.to_csv(TAB / "overall_by_category.csv", index=False, float_format="%.6f")
for split in ("valid", "test"):
    print(f"\n=== {split.upper()} WAPE by category / group ===")
    s = mc[mc.split.eq(split)]
    print(s.pivot_table(index="model", columns=["level", "value"], values="WAPE", sort=False)
          .loc[MODELS].round(4).to_string())
print("\n=== VALID Bias_pct by category ===")
s = mc[mc.split.eq("valid") & mc.level.isin(["all", "category"])]
print(s.pivot_table(index="model", columns="value", values="Bias_pct").loc[MODELS].round(4).to_string())

# ------------------------------------------------------------------ significance inside each category
PAIRS = [("A_no_promo", "B_full_future"), ("A+_past_promo", "B_full_future")]
rows = []
for split, f in (("valid", pv), ("test", pt)):
    for cat in CATS:
        fc = f[f["category"] == cat]
        subsets = {"all days": fc, "promo days": fc[fc["in_promo_any"] == 1],
                   "non-promo days": fc[fc["in_promo_any"] != 1]}
        for a, b in PAIRS:
            for sname, g in subsets.items():
                ea = (g["quantity"] - g[f"pred_{a}"]).abs(); eb = (g["quantity"] - g[f"pred_{b}"]).abs()
                daily = pd.DataFrame({"date": g["date"], "ea": ea, "eb": eb}).groupby("date")[["ea", "eb"]].sum()
                dm, dmp = dm_test(daily["ea"].values, daily["eb"].values)
                point, lo, hi, p_boot = cluster_bootstrap(g, a, b)
                ser = pd.DataFrame({"k": g["item_id"].astype(str) + "_" + g["store_id"].astype(str),
                                    "ea": ea, "eb": eb}).groupby("k")[["ea", "eb"]].mean()
                dif = ser["ea"] - ser["eb"]
                w = stats.wilcoxon(dif[dif != 0]) if (dif != 0).sum() > 10 else None
                wa = ea.sum() / g["quantity"].sum(); wb = eb.sum() / g["quantity"].sum()
                rows.append({"split": split, "category": cat, "model_a": a, "model_b": b, "subset": sname,
                             "n_rows": len(g), "n_series": len(ser), "n_days": len(daily),
                             "WAPE_a": wa, "WAPE_b": wb, "dWAPE_pp": point,
                             "rel_improvement_pct": (wa - wb) / wa * 100,
                             "boot_CI95_lo_pp": lo, "boot_CI95_hi_pp": hi, "boot_p_one_sided": p_boot,
                             "DM_stat": dm, "DM_p": dmp,
                             "series_improved_pct": float((dif > 0).mean() * 100),
                             "wilcoxon_p": (w.pvalue if w is not None else np.nan)})
        print(f"done {split} {cat}", flush=True)
sig = pd.DataFrame(rows)
sig.to_csv(TAB / "significance_by_category.csv", index=False)
print(sig[["split", "category", "model_a", "subset", "WAPE_a", "WAPE_b", "dWAPE_pp", "boot_CI95_lo_pp",
           "boot_CI95_hi_pp", "DM_p", "series_improved_pct", "wilcoxon_p"]].round(4).to_string())
