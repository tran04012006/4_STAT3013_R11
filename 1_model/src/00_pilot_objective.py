"""Step 0 (pilot) - choose the loss function before tuning.

Fits models A and B with default hyper-parameters and three candidate
objectives (Tweedie, Poisson, L2 on log1p), training on <= 2024-03-31 and
scoring WAPE on the validation period. Tweedie won for both feature sets, so
02_tune.py fixes objective = tweedie and tunes its variance power.
Output: 1_model/tables/pilot_objective.csv
"""
import time
import pandas as pd
from common import load, between, fit, predict, wape
from config import FEATURE_SETS, MAIN_A, MAIN_B, TRAIN_END, VALID_START, VALID_END, TAB

d = load()
tr = d[between(d, None, TRAIN_END)]
va = d[between(d, VALID_START, VALID_END)]
rows = []
for name in (MAIN_A, MAIN_B):
    for obj in ("tweedie", "poisson", "l2log"):
        t = time.time()
        b = fit(tr, FEATURE_SETS[name], {"objective": obj, "learning_rate": 0.05}, va, num_boost_round=3000)
        p = predict(b, va, FEATURE_SETS[name], num_iteration=b.best_iteration)
        rows.append({"feature_set": name, "objective": obj, "best_iter": b.best_iteration,
                     "valid_WAPE": wape(va["quantity"].values, p), "seconds": round(time.time() - t, 1)})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(TAB / "pilot_objective.csv", index=False)
