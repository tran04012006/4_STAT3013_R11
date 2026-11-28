"""Pairwise comparison of the promotion-aware forecasting models on the test set.

Models: LSTM (this branch), NLinear-X, LightGBM set B and XGBoost (other branches of the group repository).
Forecasts of the other models are read directly from their branches with `git show`, so no checkout is needed:

    origin/NLinear-X_forecasting      outputs/final_result.csv
    origin/Hoang_LightGBM_GridSearch  1_model/predictions/predictions_test.parquet
    origin/XGBoost                    xgboost/xgboost_train.py + outputs/checkpoints/tweedie1.2_rs02.json

XGBoost does not store test predictions, so they are regenerated from the selected checkpoint with exactly the
feature engineering of xgboost_train.py (same approach as xgboost/evaluate_breakdown.py; the script checks that the
validation MSE of the checkpoint, 76.106, is reproduced).

Tests (report, Section 5.4 and Table 14):
  * Diebold-Mariano on the daily loss differential d_t = sum_i |e_A| - sum_i |e_B| over the 88 test days,
    Newey-West variance with H-1 = 6 lags and the Harvey-Leybourne-Newbold correction, t_{T-1}, two-sided p.
    Negative DM* / negative dWAPE favour model A.
  * Series-cluster bootstrap 95% CI for dWAPE = WAPE_A - WAPE_B (2,000 resamples overall, 1,000 per subset).
  * Share of series for which model A has the lower MAE.
  * Seven-day-horizon comparison: rows that the weekly-origin models (LSTM, NLinear-X) forecast 7 days ahead
    (12 Sundays), i.e. with the same information lag as the tree models; DM on the weekly differential, 1 lag, t_11.

Usage:  git fetch origin
        python src/compare_models_dm.py [--panel data_model/daily_panel.parquet] [--out output/compare]
Outputs: pairwise_dm.csv, pairwise_h7.csv, wape_by_segment.csv, wape_by_horizon.csv, xgb_vs_naive.csv
"""
import argparse
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["item_id", "store_id", "date"]
SEED = 42
MODELS = {"LSTM": "pred_lstm", "NLinear-X": "pred_nlinear_x", "LightGBM-B": "pred_lgb_b", "XGBoost": "pred_xgb"}
PAIRS = [("LSTM", "NLinear-X"), ("LSTM", "LightGBM-B"), ("LSTM", "XGBoost"),
         ("NLinear-X", "LightGBM-B"), ("NLinear-X", "XGBoost"), ("LightGBM-B", "XGBoost")]
PHASES = ["pre_1_3", "during", "post_1_3", "post_4_7", "other"]


def git_bytes(ref, path):
    """Content of `path` on branch `ref`, read without checking the branch out."""
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, check=True, capture_output=True).stdout


def norm_keys(d):
    d = d.copy()
    d["item_id"] = d["item_id"].astype(str)
    d["store_id"] = d["store_id"].astype(str).astype(int)
    d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
    return d


def xgb_test_predictions(panel_path):
    """Regenerate XGBoost test predictions from the selected checkpoint (features as in xgboost_train.py)."""
    import xgboost as xgb
    src = git_bytes("origin/XGBoost", "xgboost/xgboost_train.py").decode("utf-8")
    prep = src.split("# 4. Tuning")[0]                       # data loading + feature engineering only
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prep = prep.replace("__file__", repr(str(tmp / "xgboost_train.py")))
        prep = prep.replace("HERE.parent / 'data' / 'daily_panel.parquet'", repr(str(Path(panel_path).resolve())))
        prep = prep.replace("print(", "(lambda *a, **k: None)(")  # silence the script's own logging
        ns = {}
        exec(compile(prep, "xgboost_train.py[prep]", "exec"), ns)
        df, feats = ns["df"], ns["FEATURES"]
        ckpt = tmp / "tweedie1.2_rs02.json"
        ckpt.write_bytes(git_bytes("origin/XGBoost", "xgboost/outputs/checkpoints/tweedie1.2_rs02.json"))
        bst = xgb.Booster()
        bst.load_model(str(ckpt))

    def predict(part):
        return np.clip(bst.predict(xgb.DMatrix(part[feats], enable_categorical=True)), 0, None)

    va = df[df.split == "valid"]
    mse = float(np.mean((va.quantity.values - predict(va)) ** 2))
    assert abs(mse - 76.106) < 0.01, f"XGBoost checkpoint does not reproduce the validation MSE ({mse:.3f})"
    te = df[df.split == "test"][["item_id", "store_id", "date", "quantity"]].copy()
    te["pred_xgb"] = predict(df[df.split == "test"])
    print(f"XGBoost: validation MSE {mse:.3f} reproduced; {len(te):,} test predictions regenerated")
    return norm_keys(te.rename(columns={"quantity": "q_xgb"}))


def load_all(panel_path):
    lstm = pd.read_parquet(ROOT / "output/lstm/test_predictions.parquet")
    lstm = norm_keys(lstm[KEYS + ["h", "lstm", "quantity", "category", "group4", "phase"]]
                     .rename(columns={"lstm": "pred_lstm"}))
    nlx = pd.read_csv(io.BytesIO(git_bytes("origin/NLinear-X_forecasting", "outputs/final_result.csv")),
                      dtype={"item_id": str})
    nlx = norm_keys(nlx[KEYS + ["horizon_step", "actual_quantity", "pred_nlinear_x", "pred_seasonal_naive"]])
    lgb = pd.read_parquet(io.BytesIO(git_bytes("origin/Hoang_LightGBM_GridSearch",
                                               "1_model/predictions/predictions_test.parquet")))
    lgb = norm_keys(lgb[KEYS + ["quantity", "pred_B_full_future"]]
                    .rename(columns={"quantity": "q_lgb", "pred_B_full_future": "pred_lgb_b"}))
    m = lstm.merge(nlx, on=KEYS).merge(lgb, on=KEYS).merge(xgb_test_predictions(panel_path), on=KEYS)
    assert len(m) == 177_713, len(m)
    for c in ["actual_quantity", "q_lgb", "q_xgb"]:
        assert np.allclose(m.quantity, m[c]), f"actual quantities differ ({c})"
    assert (m.h == m.horizon_step).all(), "LSTM and NLinear-X forecast origins differ"
    return m.sort_values(KEYS).reset_index(drop=True)


def dm_test(ea, eb, dates, lags):
    """DM* with Bartlett weights (bandwidth lags+1) and HLN correction; two-sided p from t_{T-1}."""
    d = pd.Series(ea - eb).groupby(dates).sum().sort_index().to_numpy()
    T, h = len(d), lags + 1
    dc = d - d.mean()
    var = dc @ dc / T
    for k in range(1, h):
        var += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / T
    dm = d.mean() / np.sqrt(var / T)
    if lags == 6:                                            # HLN correction for horizon H = 7
        dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return dm, 2 * stats.t.sf(abs(dm), T - 1), T


class Bootstrap:
    """Series-cluster bootstrap of dWAPE = WAPE_A - WAPE_B, in percentage points."""

    def __init__(self, series, seed=SEED):
        self.ids, self.inv = np.unique(series, return_inverse=True)
        self.rng = np.random.default_rng(seed)

    def __call__(self, y, ea, eb, mask, n_boot):
        n = len(self.ids)
        A = np.bincount(self.inv, ea * mask, n)
        B = np.bincount(self.inv, eb * mask, n)
        Y = np.bincount(self.inv, y * mask, n)
        draws = []
        for _ in range(n_boot):
            s = self.rng.integers(0, n, n)
            draws.append((A[s].sum() - B[s].sum()) / Y[s].sum())
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return 100 * (A.sum() - B.sum()) / Y.sum(), 100 * lo, 100 * hi


def wape(y, p):
    return 100 * np.abs(y - p).sum() / y.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=str(ROOT / "data_model/daily_panel.parquet"))
    ap.add_argument("--out", default=str(ROOT / "output/compare"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    m = load_all(args.panel)
    y, dates = m.quantity.to_numpy(np.float64), m.date.to_numpy()
    series = (m.item_id + "_" + m.store_id.astype(str)).to_numpy()
    err = {k: np.abs(y - m[c].to_numpy(np.float64)) for k, c in MODELS.items()}
    boot = Bootstrap(series)

    subsets = {"All": np.ones(len(m), bool),
               "Food": (m.category == "food").to_numpy(), "Household": (m.category == "household").to_numpy()}
    subsets.update({ph: (m.phase == ph).to_numpy() for ph in PHASES})

    rows = []
    for a, b in PAIRS:
        for name, mask in subsets.items():
            dm, p, T = dm_test(err[a][mask], err[b][mask], dates[mask], lags=6)
            pt, lo, hi = boot(y, err[a], err[b], mask.astype(float), 2000 if name == "All" else 1000)
            sa = pd.Series(err[a][mask]).groupby(series[mask]).mean()
            sb = pd.Series(err[b][mask]).groupby(series[mask]).mean()
            rows.append({"model_A": a, "model_B": b, "subset": name, "n_rows": int(mask.sum()), "T_days": T,
                         "WAPE_A": 100 * err[a][mask].sum() / y[mask].sum(),
                         "WAPE_B": 100 * err[b][mask].sum() / y[mask].sum(),
                         "dWAPE_pp": pt, "boot_CI95_lo": lo, "boot_CI95_hi": hi, "DM_HLN": dm, "p_value": p,
                         "share_series_A_better_pct": 100 * (sa < sb).mean()})
    pd.DataFrame(rows).to_csv(out / "pairwise_dm.csv", index=False)

    h7 = (m.h == 7).to_numpy()
    boot = Bootstrap(series)                                 # fresh RNG: each table is reproducible on its own
    rows = []
    for a, b in PAIRS:
        dm, p, T = dm_test(err[a][h7], err[b][h7], dates[h7], lags=1)
        pt, lo, hi = boot(y, err[a], err[b], h7.astype(float), 2000)
        rows.append({"model_A": a, "model_B": b, "n_rows": int(h7.sum()), "T_days": T,
                     "WAPE_A": 100 * err[a][h7].sum() / y[h7].sum(), "WAPE_B": 100 * err[b][h7].sum() / y[h7].sum(),
                     "dWAPE_pp": pt, "boot_CI95_lo": lo, "boot_CI95_hi": hi, "t_stat": dm, "p_value": p})
    pd.DataFrame(rows).to_csv(out / "pairwise_h7.csv", index=False)

    seg = []
    for col in ["category", "group4", "phase"]:
        for v, idx in m.groupby(col).indices.items():
            r = {"by": col, "group": v, "n": len(idx)}
            for k, c in MODELS.items():
                p_ = m[c].to_numpy(np.float64)[idx]
                r[f"WAPE_{k}"] = wape(y[idx], p_)
                r[f"Bias_{k}"] = 100 * (p_ - y[idx]).sum() / y[idx].sum()
            seg.append(r)
    pd.DataFrame(seg).to_csv(out / "wape_by_segment.csv", index=False)

    hz = [{"h": h, "n": int((m.h == h).sum()),
           **{f"WAPE_{k}": wape(y[(m.h == h).to_numpy()], m[c].to_numpy(np.float64)[(m.h == h).to_numpy()])
              for k, c in MODELS.items()}} for h in range(1, 8)]
    pd.DataFrame(hz).to_csv(out / "wape_by_horizon.csv", index=False)

    en = np.abs(y - m.pred_seasonal_naive.to_numpy(np.float64))
    dm, p, T = dm_test(err["XGBoost"], en, dates, lags=6)
    boot = Bootstrap(series)
    pt, lo, hi = boot(y, err["XGBoost"], en, np.ones(len(m)), 2000)
    sx = pd.Series(err["XGBoost"]).groupby(series).mean()
    sn = pd.Series(en).groupby(series).mean()
    pd.DataFrame([{"model_A": "XGBoost", "model_B": "SeasonalNaive", "WAPE_A": wape(y, m.pred_xgb.to_numpy()),
                   "WAPE_B": wape(y, m.pred_seasonal_naive.to_numpy()), "dWAPE_pp": pt, "boot_CI95_lo": lo,
                   "boot_CI95_hi": hi, "DM_HLN": dm, "p_value": p,
                   "share_series_A_better_pct": 100 * (sx < sn).mean()}]).to_csv(out / "xgb_vs_naive.csv", index=False)

    pw = pd.read_csv(out / "pairwise_dm.csv")
    pd.set_option("display.width", 200)
    print(pw[pw.subset == "All"].round(3).to_string(index=False))
    print(pd.read_csv(out / "pairwise_h7.csv").round(3).to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
