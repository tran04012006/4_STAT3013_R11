"""Step 5 - Interpretation of the final model B (seed 42 booster trained on train+valid).

  tables/feature_importance_B.csv   gain + split importance per feature
  tables/shap_by_feature_B.csv      mean |SHAP| per feature on a test sample (log-link scale)
  tables/shap_by_group_B.csv        same, summed per feature group
  tables/promo_counterfactual.csv   model-implied promo uplift: predict test promo days with the
                                    promotion "switched off" vs on, per group
  tables/depth_response.csv         predicted quantity on test promo days when depth is set to a
                                    grid 5%..40% (within the observed range), per category
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load, between
from config import (FEATURE_SETS, MAIN_B, TEST_START, TEST_END, MOD, TAB, SEED,
                    CALENDAR, HISTORY, STATIC, PAST_PROMO, FUT_FLAG, FUT_DEPTH, FUT_TIMING)

feats = FEATURE_SETS[MAIN_B]
# read via Python (LightGBM C++ cannot open non-ASCII Windows paths)
bst = lgb.Booster(model_str=(MOD / f"lgbm_{MAIN_B}.txt").read_text(encoding="utf-8"))
d = load()
te = d[between(d, TEST_START, TEST_END)].copy()

group_of = {}
for gname, cols in [("Calendar", CALENDAR), ("Demand history", HISTORY), ("Static product/store", STATIC),
                    ("Past promo (as of t-7)", PAST_PROMO), ("Future promo: flag", FUT_FLAG),
                    ("Future promo: depth/price", FUT_DEPTH), ("Future promo: timing", FUT_TIMING)]:
    for c in cols:
        group_of[c] = gname

# ------------------------------------------------------------------ gain importance
imp = pd.DataFrame({"feature": bst.feature_name(),
                    "gain": bst.feature_importance("gain"),
                    "split": bst.feature_importance("split")})
imp["group"] = imp["feature"].map(group_of)
imp["gain_share_pct"] = imp["gain"] / imp["gain"].sum() * 100
imp = imp.sort_values("gain", ascending=False)
imp.to_csv(TAB / "feature_importance_B.csv", index=False)
print(imp.head(20).to_string())

# ------------------------------------------------------------------ SHAP (TreeSHAP built into LightGBM)
smp = te.sample(n=min(60000, len(te)), random_state=SEED)
contrib = bst.predict(smp[feats], pred_contrib=True)
sh = pd.DataFrame(contrib[:, :-1], columns=bst.feature_name())
shf = sh.abs().mean().rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
shf["group"] = shf["feature"].map(group_of)
shf = shf.sort_values("mean_abs_shap", ascending=False)
shf.to_csv(TAB / "shap_by_feature_B.csv", index=False)
# group-level: sum SHAP values of the group per row, then mean |.|
shg = pd.DataFrame({g: sh[[c for c in sh.columns if group_of.get(c) == g]].sum(axis=1)
                    for g in dict.fromkeys(group_of.values())})
shg_all = shg.abs().mean().rename("all_rows")
promo_mask = (smp["in_promo_any"] == 1).values
shg_pr = shg[promo_mask].abs().mean().rename("promo_days")
shg_np = shg[~promo_mask].abs().mean().rename("non_promo_days")
shg_tab = pd.concat([shg_all, shg_pr, shg_np], axis=1).reset_index().rename(columns={"index": "group"})
shg_tab.to_csv(TAB / "shap_by_group_B.csv", index=False)
print(shg_tab.round(4).to_string())


# ------------------------------------------------------------------ counterfactual: promotion off
def promo_off(df):
    x = df.copy()
    x["promo_status"] = 0.0; x["in_promo_any"] = 0.0; x["depth"] = 0.0; x["depth_x_food"] = 0.0
    x["price"] = x["regular_price"]; x["price_ratio"] = np.where(x["regular_price"].notna(), 1.0, np.nan)
    x["promo_type"] = x["promo_type"].where(np.zeros(len(x), bool))   # all missing, dtype kept
    for c in ["day_in_promo", "days_to_promo_end", "promo_duration"]:
        x[c] = np.nan
    x["days_to_next_promo"] = 60.0; x["promo_days_next_7"] = 0.0; x["post_promo_1_7"] = 0.0
    return x


pr = te[(te["in_promo_any"] == 1) & (te["depth"] > 0)].copy()
pr["p_on"] = bst.predict(pr[feats])
pr["p_off"] = bst.predict(promo_off(pr)[feats])
cf = pr.groupby("group4", observed=True).agg(n=("quantity", "size"), actual_mean=("quantity", "mean"),
                                             pred_on=("p_on", "mean"), pred_off=("p_off", "mean"),
                                             mean_depth=("depth", "mean"))
cf.loc["ALL"] = [len(pr), pr["quantity"].mean(), pr["p_on"].mean(), pr["p_off"].mean(), pr["depth"].mean()]
cf["implied_uplift"] = cf["pred_on"] / cf["pred_off"]
cf.reset_index().to_csv(TAB / "promo_counterfactual.csv", index=False)
print(cf.round(3).to_string())

# ------------------------------------------------------------------ depth response curve
rows = []
grid = np.round(np.arange(0.05, 0.401, 0.05), 2)
for dep in grid:
    x = pr.copy()
    x["depth"] = dep
    x["price_ratio"] = 1 - dep
    x["price"] = x["regular_price"] * (1 - dep)
    x["depth_x_food"] = dep * x["is_food"]
    p = bst.predict(x[feats])
    for (cat, g4), idx in x.groupby(["category", "group4"], observed=True).groups.items():
        rows.append({"category": cat, "group4": g4, "depth": dep, "mean_pred_qty": float(p[x.index.get_indexer(idx)].mean())})
    for cat, idx in x.groupby("category", observed=True).groups.items():
        rows.append({"category": cat, "group4": "ALL", "depth": dep, "mean_pred_qty": float(p[x.index.get_indexer(idx)].mean())})
dr = pd.DataFrame(rows)
base = dr[dr.depth == 0.05].set_index(["category", "group4"])["mean_pred_qty"]
dr["index_vs_5pct"] = dr.apply(lambda r: r.mean_pred_qty / base[(r.category, r.group4)], axis=1)
dr.to_csv(TAB / "depth_response.csv", index=False)
print(dr[dr.group4 == "ALL"].round(3).to_string())
