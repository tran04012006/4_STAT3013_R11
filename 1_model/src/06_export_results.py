"""Step 6 - Export all model results into one Excel workbook + a CSV of test predictions.

  1_model/LightGBM_results_Hoang.xlsx           (one sheet per result table, with a README sheet)
  1_model/predictions/predictions_test.csv      (every test row: keys, actual quantity, each model's forecast)
"""
import json
import pandas as pd
from config import MODEL_DIR, PRED, TAB, MOD, MAIN_A, MAIN_B, FEATURE_SETS

pt = pd.read_parquet(PRED / "predictions_test.parquet")
pcols = [c for c in pt.columns if c.startswith("pred_") and "@" not in c]
pt[["item_id", "store_id", "date", "group4", "category", "phase", "in_promo_any", "depth", "quantity"] + pcols] \
    .to_csv(PRED / "predictions_test.csv", index=False, float_format="%.4f")

readme = pd.DataFrame({"sheet": [
    "run_summary", "metrics_overall", "metrics_by_segment", "significance_tests", "bias_calibration", "seed_metrics",
    "feature_importance_B", "shap_by_group_B", "promo_counterfactual", "depth_response",
    "best_params", "feature_sets", "pilot_objective", "tuning_A", "tuning_B"],
    "content": [
    "One row per model: validation + test WAPE/MAE/RMSE/Bias, number of trees",
    "WAPE, MAE, RMSE, Bias for every model on valid and test",
    "Test metrics by promo day / promo phase / product group / category / store / day-in-promo",
    "A vs B (and other pairs): Diebold-Mariano test, series-cluster bootstrap 95% CI of dWAPE, Wilcoxon",
    "Sensitivity: test metrics after scaling forecasts by a validation-fitted bias factor",
    "Test metrics of each of the 3 random seeds (robustness)",
    "Model B gain/split importance per feature",
    "Model B mean |SHAP| per feature group, promo vs non-promo days",
    "Model-implied promo uplift on test promo days (promo on vs switched off)",
    "What-if: predicted quantity when discount depth is set to 5-40%",
    "Tuned LightGBM hyper-parameters for A and B",
    "Which features belong to which experimental feature set",
    "Pilot: choice of loss function",
    "Grid search for model A: one row per combination, hold-out validation WAPE and early-stopped trees",
    "Grid search for model B: one row per combination, hold-out validation WAPE and early-stopped trees"]})

sheets = {"README": readme}
for name in ["run_summary", "metrics_overall", "metrics_by_segment", "significance_tests", "bias_calibration",
             "seed_metrics",
             "feature_importance_B", "shap_by_group_B", "promo_counterfactual", "depth_response",
             "pilot_objective"]:
    p = TAB / f"{name}.csv"
    if p.exists():
        sheets[name] = pd.read_csv(p)
bp = []
for n in (MAIN_A, MAIN_B):
    j = json.load(open(MOD / f"best_params_{n}.json"))
    for k, v in j["params"].items():
        bp.append({"feature_set": n, "param": k, "value": v})
    bp.append({"feature_set": n, "param": "best_valid_wape (hold-out)", "value": j["best_valid_wape"]})
    bp.append({"feature_set": n, "param": "best_iter (early stopping)", "value": j["best_iter"]})
sheets["best_params"] = pd.DataFrame(bp)
fs = [{"feature_set": k, "feature": f} for k, v in FEATURE_SETS.items() for f in v]
sheets["feature_sets"] = pd.DataFrame(fs).pivot_table(index="feature", columns="feature_set", aggfunc=len,
                                                      fill_value=0).reset_index()
sheets["tuning_A"] = pd.read_csv(TAB / f"tuning_{MAIN_A}.csv")
sheets["tuning_B"] = pd.read_csv(TAB / f"tuning_{MAIN_B}.csv")

with pd.ExcelWriter(MODEL_DIR / "LightGBM_results_Hoang.xlsx", engine="openpyxl") as xw:
    for n, df in sheets.items():
        df.to_excel(xw, sheet_name=n[:31], index=False)
        ws = xw.sheets[n[:31]]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
print("saved", MODEL_DIR / "LightGBM_results_Hoang.xlsx", "and", PRED / "predictions_test.csv")
