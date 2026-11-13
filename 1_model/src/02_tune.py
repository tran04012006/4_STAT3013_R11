"""Step 2 - Hyper-parameter tuning: Manual Grid Search + Hold-out Validation + Early Stopping.

Usage:  python 02_tune.py <feature_set_name> [max_combinations]
        e.g. python 02_tune.py A_no_promo          (full grid, 36 combinations)
             python 02_tune.py A_no_promo 2        (quick check: first 2 combinations only)

Method
  * Manual grid: the hand-picked values in GRID below; every combination is tried (3 x 3 x 2 x 2 = 36).
    The other settings are fixed at the values in FIXED.
  * Hold-out validation: each combination is trained on TRAIN (<= 2024-03-31) and scored on the
    hold-out VALID period (1 Apr - 30 Jun 2024) by WAPE. The test period (Jul-Sep 2024) is never used.
  * Early stopping: up to MAX_ROUNDS trees; training stops when the validation WAPE has not improved
    for EARLY_STOP trees, and the score is taken at the best tree count.
The combination with the lowest validation WAPE is kept.

Every finished combination is appended to the results CSV at once, so an interrupted run
(e.g. power cut) resumes where it stopped when the same command is run again.
Results -> 1_model/tables/tuning_<set>.csv and 1_model/models/best_params_<set>.json
"""
import sys, json, time, itertools
import pandas as pd
from common import load, between, fit, predict, wape
from config import FEATURE_SETS, TRAIN_END, VALID_START, VALID_END, TAB, MOD

GRID = {                                   # hand-picked values, every combination is tried
    "num_leaves": [63, 255, 511],          # tree size: small / medium / large
    "min_data_in_leaf": [50, 200, 800],    # minimum rows per leaf: less / more smoothing
    "feature_fraction": [0.5, 0.8],        # share of inputs each tree may use
    "tweedie_variance_power": [1.3, 1.6],  # shape of the Tweedie loss (closer to 1 = Poisson-like)
}
FIXED = {"objective": "tweedie", "learning_rate": 0.05, "bagging_fraction": 0.8,
         "lambda_l1": 0.01, "lambda_l2": 1.0, "cat_smooth": 10.0}
MAX_ROUNDS = 4000
EARLY_STOP = 100

name = sys.argv[1]
max_combos = int(sys.argv[2]) if len(sys.argv) > 2 else None
feats = FEATURE_SETS[name]
combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
if max_combos:
    combos = combos[:max_combos]

d = load()
tr = d[between(d, None, TRAIN_END)]
va = d[between(d, VALID_START, VALID_END)]
y_va = va["quantity"].values

out_csv = TAB / f"tuning_{name}.csv"
done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
done_ids = set(done["number"]) if len(done) else set()
print(f"[{name}] grid of {len(combos)} combinations, {len(done_ids & set(range(len(combos))))} already done; "
      f"hold-out {VALID_START}..{VALID_END}", flush=True)

t0 = time.time()
for i, c in enumerate(combos):
    if i in done_ids:
        continue
    t1 = time.time()
    hp = {**FIXED, **c}
    b = fit(tr, feats, hp, va, num_boost_round=MAX_ROUNDS, early_stop=EARLY_STOP)
    score = wape(y_va, predict(b, va, feats, num_iteration=b.best_iteration))
    row = {"number": i, "value": score, "best_iter": b.best_iteration, "state": "COMPLETE",
           "seconds": round(time.time() - t1, 1), **{f"params_{k}": v for k, v in c.items()}}
    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
    print(f"[{name}] combo {i:2d}/{len(combos)} {c} -> valid WAPE={score:.5f} trees={b.best_iteration} "
          f"({row['seconds']:.0f}s, elapsed {time.time()-t0:.0f}s)", flush=True)

res = pd.read_csv(out_csv)
res = res[res["number"] < len(combos)].sort_values("number")
res.to_csv(out_csv, index=False)
best = res.loc[res["value"].idxmin()]
params = {**FIXED, **{k: (int if isinstance(GRID[k][0], int) else float)(best[f"params_{k}"]) for k in GRID}}
out = {"feature_set": name,
       "method": "manual grid search + hold-out validation + early stopping",
       "holdout": [VALID_START, VALID_END], "train_end": TRAIN_END,
       "best_valid_wape": float(best["value"]), "best_iter": int(best["best_iter"]),
       "params": params, "grid": GRID, "fixed": FIXED,
       "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOP,
       "n_combinations": int(len(res)), "minutes": round(res["seconds"].sum() / 60, 1)}
json.dump(out, open(MOD / f"best_params_{name}.json", "w"), indent=2)
print(json.dumps(out, indent=2))
