"""Step 3 - Final training and one-shot test evaluation.

For every model configuration:
  1. fit on TRAIN (<= 2024-03-31) with early stopping on VALID (Apr-Jun 2024)
     -> best number of trees + validation predictions;
  2. refit on TRAIN+VALID with round(best_iter * 1.1) trees, 3 random seeds,
     predict TEST (Jul - 26 Sep 2024) once; the final forecast is the seed average.

Model configurations
  * the 5 feature sets of config.FEATURE_SETS
    - A_no_promo uses its own tuned params, every other set uses B's tuned params
  * robustness: A with B's params and B with A's params (so the A-vs-B gap
    cannot be an artefact of tuning luck)
  * baselines: seasonal naive (same weekday last week = lag_7) and 28-day moving average
Outputs: 1_model/predictions/predictions_{valid,test}.parquet, 1_model/tables/run_summary.csv,
         1_model/models/*.txt (LightGBM boosters), 1_model/tables/seed_metrics.csv
"""
import json, time
import numpy as np
import pandas as pd
from common import load, between, fit, predict, metrics
from config import (FEATURE_SETS, MAIN_A, MAIN_B, TRAIN_END, VALID_START, VALID_END,
                    TEST_START, TEST_END, MOD, TAB, PRED, DRYRUN)

SEEDS = [42] if DRYRUN else [42, 7, 2024]
MAX_ROUNDS = 60 if DRYRUN else 5000
t0 = time.time()
d = load()
if DRYRUN:  # smoke test on 10% of the series
    keys = d[["item_id", "store_id"]].drop_duplicates().sample(frac=0.1, random_state=1)
    d = d.merge(keys, on=["item_id", "store_id"])
tr = d[between(d, None, TRAIN_END)]
va = d[between(d, VALID_START, VALID_END)]
trva = d[between(d, None, VALID_END)]
te = d[between(d, TEST_START, TEST_END)]
print("rows train/valid/test:", len(tr), len(va), len(te))

hpA = json.load(open(MOD / f"best_params_{MAIN_A}.json"))["params"]
hpB = json.load(open(MOD / f"best_params_{MAIN_B}.json"))["params"]

configs = [(name, name, hpA if name == MAIN_A else hpB, "A" if name == MAIN_A else "B")
           for name in FEATURE_SETS]
configs += [("A_no_promo@Bparams", MAIN_A, hpB, "B"),
            ("B_full_future@Aparams", MAIN_B, hpA, "A")]

keep = ["item_id", "store_id", "date", "group4", "category", "dept_name", "phase",
        "in_promo_any", "promo_status", "day_in_promo", "depth", "quantity"]
pv = va[keep].copy()
pt = te[keep].copy()

# ---------------------------------------------------------------- baselines
for frame, src in ((pv, va), (pt, te)):
    frame["pred_SeasonalNaive"] = src["lag_7"].fillna(src["roll_mean_28"]).fillna(0).values
    frame["pred_MovingAvg28"] = src["roll_mean_28"].fillna(0).values

rows, seed_rows = [], []
for label, fs, hp, hp_src in configs:
    feats = FEATURE_SETS[fs]
    t1 = time.time()
    b = fit(tr, feats, hp, va, num_boost_round=MAX_ROUNDS, early_stop=150)
    best_it = b.best_iteration
    pv[f"pred_{label}"] = predict(b, va, feats, num_iteration=best_it)
    n_final = int(round(best_it * 1.1))
    preds = []
    for s in SEEDS:
        bf = fit(trva, feats, hp, num_boost_round=n_final, seed=s)
        p = predict(bf, te, feats)
        preds.append(p)
        m = metrics(te["quantity"].values, p)
        seed_rows.append({"model": label, "seed": s, **m})
        if s == SEEDS[0] and "@" not in label:
            # LightGBM C++ cannot open non-ASCII Windows paths -> write the model text from Python
            (MOD / f"lgbm_{label}.txt").write_text(bf.model_to_string(), encoding="utf-8")
    pt[f"pred_{label}"] = np.mean(preds, axis=0)
    mv = metrics(va["quantity"].values, pv[f"pred_{label}"].values)
    mt = metrics(te["quantity"].values, pt[f"pred_{label}"].values)
    rows.append({"model": label, "feature_set": fs, "params_from": hp_src, "n_features": len(feats),
                 "best_iter_valid": best_it, "n_trees_final": n_final,
                 "valid_WAPE": mv["WAPE"], "test_WAPE": mt["WAPE"], "test_MAE": mt["MAE"],
                 "test_RMSE": mt["RMSE"], "test_Bias_pct": mt["Bias_pct"],
                 "minutes": round((time.time() - t1) / 60, 2)})
    print(f"{label:24s} iters={best_it:5d} valid WAPE={mv['WAPE']:.4f} test WAPE={mt['WAPE']:.4f} "
          f"({time.time()-t1:.0f}s)", flush=True)

for name in ("SeasonalNaive", "MovingAvg28"):
    mv = metrics(va["quantity"].values, pv[f"pred_{name}"].values)
    mt = metrics(te["quantity"].values, pt[f"pred_{name}"].values)
    rows.append({"model": name, "feature_set": "baseline", "params_from": "-", "n_features": 1,
                 "valid_WAPE": mv["WAPE"], "test_WAPE": mt["WAPE"], "test_MAE": mt["MAE"],
                 "test_RMSE": mt["RMSE"], "test_Bias_pct": mt["Bias_pct"]})

summary = pd.DataFrame(rows)
summary.to_csv(TAB / "run_summary.csv", index=False)
sm = pd.DataFrame(seed_rows)
sm.to_csv(TAB / "seed_metrics.csv", index=False)
pv.to_parquet(PRED / "predictions_valid.parquet", index=False)
pt.to_parquet(PRED / "predictions_test.parquet", index=False)
print(summary.to_string())
print(sm.groupby("model")[["WAPE", "MAE", "RMSE"]].agg(["mean", "std"]).to_string())
print(f"total {(time.time()-t0)/60:.1f} min")
