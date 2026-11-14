"""Step 4 - Detailed evaluation and statistical significance of the improvement.

Reads 1_model/predictions/predictions_test.parquet (and _valid) and writes:
  tables/metrics_overall.csv        every model, valid + test
  tables/metrics_by_segment.csv     test metrics by promo day / phase / group / category / store / day-in-promo
  tables/significance_tests.csv     Diebold-Mariano, series-cluster bootstrap CI of dWAPE, Wilcoxon
  tables/bias_calibration.csv       sensitivity: test metrics after a validation-fitted bias scale
"""
import numpy as np
import pandas as pd
from scipy import stats
from common import metrics, dm_test, cluster_bootstrap
from config import PRED, TAB

pt = pd.read_parquet(PRED / "predictions_test.parquet")
pv = pd.read_parquet(PRED / "predictions_valid.parquet")
MODELS = ["SeasonalNaive", "MovingAvg28", "A_no_promo", "A+_past_promo",
          "B1_+future_flag", "B2_+future_depth", "B_full_future",
          "A_no_promo@Bparams", "B_full_future@Aparams"]

# ------------------------------------------------------------------ overall
rows = []
for split, f in (("valid", pv), ("test", pt)):
    for m in MODELS:
        rows.append({"split": split, "model": m, **metrics(f["quantity"], f[f"pred_{m}"])})
ov = pd.DataFrame(rows)
ov.to_csv(TAB / "metrics_overall.csv", index=False)
print(ov.pivot(index="model", columns="split", values="WAPE").loc[MODELS].round(4))

# ------------------------------------------------------------------ segments (test)
f = pt.copy()
f["promo_day"] = np.where(f["in_promo_any"] == 1, "promo day", "non-promo day")
f["dip_bucket"] = pd.cut(f["day_in_promo"], [0, 3, 7, 14, 1000], labels=["1-3", "4-7", "8-14", "15+"]).astype(str)
f.loc[f["in_promo_any"] != 1, "dip_bucket"] = np.nan
segs = {"promo_day": "promo_day", "phase": "phase", "group4": "group4", "category": "category",
        "store_id": "store_id", "day_in_promo": "dip_bucket"}
rows = []
for seg, col in segs.items():
    for val, g in f.groupby(col, observed=True):
        for m in MODELS[:7]:
            rows.append({"segment": seg, "value": str(val), "model": m,
                         **metrics(g["quantity"], g[f"pred_{m}"])})
sg = pd.DataFrame(rows)
sg.to_csv(TAB / "metrics_by_segment.csv", index=False)


# ------------------------------------------------------------------ significance
PAIRS = [("SeasonalNaive", "A_no_promo"), ("A_no_promo", "A+_past_promo"),
         ("A+_past_promo", "B_full_future"), ("A_no_promo", "B_full_future"),
         ("A_no_promo@Bparams", "B_full_future"), ("A_no_promo", "B_full_future@Aparams")]
subsets = {"all days": f, "promo days": f[f["in_promo_any"] == 1], "non-promo days": f[f["in_promo_any"] != 1]}
for gname in ["F1", "F2", "H1", "H2"]:
    subsets[f"group {gname}"] = f[f["group4"] == gname]
for ph in ["pre_1_3", "during", "post_1_3", "post_4_7"]:
    subsets[f"phase {ph}"] = f[f["phase"] == ph]

rows = []
for a, b in PAIRS:
    for sname, g in subsets.items():
        ea = (g["quantity"] - g[f"pred_{a}"]).abs(); eb = (g["quantity"] - g[f"pred_{b}"]).abs()
        daily = pd.DataFrame({"date": g["date"], "ea": ea, "eb": eb}).groupby("date")[["ea", "eb"]].sum()
        dm, dmp = dm_test(daily["ea"].values, daily["eb"].values)
        point, lo, hi, p_boot = cluster_bootstrap(g, a, b)
        ser = pd.DataFrame({"k": g["item_id"].astype(str) + "_" + g["store_id"].astype(str), "ea": ea, "eb": eb}) \
            .groupby("k")[["ea", "eb"]].mean()
        dif = ser["ea"] - ser["eb"]
        w = stats.wilcoxon(dif[dif != 0]) if (dif != 0).sum() > 10 else None
        wa = (g["quantity"] - g[f"pred_{a}"]).abs().sum() / g["quantity"].sum()
        wb = (g["quantity"] - g[f"pred_{b}"]).abs().sum() / g["quantity"].sum()
        rows.append({"model_a": a, "model_b": b, "subset": sname, "n_rows": len(g),
                     "n_series": len(ser), "n_days": len(daily),
                     "WAPE_a": wa, "WAPE_b": wb, "dWAPE_pp": point,
                     "rel_improvement_pct": (wa - wb) / wa * 100,
                     "boot_CI95_lo_pp": lo, "boot_CI95_hi_pp": hi, "boot_p_one_sided": p_boot,
                     "DM_stat": dm, "DM_p": dmp,
                     "series_improved_pct": float((dif > 0).mean() * 100),
                     "wilcoxon_p": (w.pvalue if w is not None else np.nan)})
    print(f"done {a} vs {b}", flush=True)
sig = pd.DataFrame(rows)
sig.to_csv(TAB / "significance_tests.csv", index=False)
pd.set_option("display.width", 250)
print(sig[sig.model_a.isin(["A_no_promo", "A+_past_promo"]) & sig.model_b.eq("B_full_future")]
      [["model_a", "subset", "WAPE_a", "WAPE_b", "dWAPE_pp", "boot_CI95_lo_pp", "boot_CI95_hi_pp",
        "DM_stat", "DM_p", "series_improved_pct", "wilcoxon_p"]].round(4).to_string())

# ------------------------------------------------------------------ bias-calibration sensitivity
# Models tuned / early-stopped on WAPE (an absolute-error metric) learn a median-like forecast and
# under-forecast total volume. Check: scale each model's test forecast by the factor that removes its
# bias on the VALIDATION period (sum actual / sum forecast), then recompute test metrics.
rows = []
for m in MODELS[2:]:
    k = pv["quantity"].sum() / pv[f"pred_{m}"].sum()
    raw = metrics(pt["quantity"], pt[f"pred_{m}"])
    cal = metrics(pt["quantity"], pt[f"pred_{m}"] * k)
    rows.append({"model": m, "scale_from_valid": k, "test_WAPE_raw": raw["WAPE"], "test_Bias_raw": raw["Bias_pct"],
                 "test_WAPE_calibrated": cal["WAPE"], "test_Bias_calibrated": cal["Bias_pct"],
                 "test_RMSE_raw": raw["RMSE"], "test_RMSE_calibrated": cal["RMSE"]})
bc = pd.DataFrame(rows)
bc.to_csv(TAB / "bias_calibration.csv", index=False)
print(bc.round(4).to_string())
