"""Step 1 - Feature engineering.

Input : data/daily_panel.parquet  (a byte-identical copy of the team file
        ../dataset/daily_panel.parquet, which is never modified)
Output: data/features.parquet

Every history feature is shifted by >= 7 days (forecast origin = t-7), so the
experiment is a 7-day-ahead forecast. "Future promo" features describe the
forecast day itself and come from the promotion schedule, which the retailer
plans in advance.
"""
import time
import numpy as np
import pandas as pd
from config import DATA_RAW, DATA_FEAT, HORIZON, FORBIDDEN, TAB

t0 = time.time()
raw = pd.read_parquet(DATA_RAW)
print("raw", raw.shape)

# ------------------------------------------------------------------ full daily grid per series
# The panel has 252 one-day gaps (abnormal store days removed by the team
# pipeline). Lags must be computed by calendar date, so we expand each
# item-store series to a complete daily grid, compute features, then keep only
# the original rows.
raw = raw.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
raw["_orig"] = True
span = raw.groupby(["item_id", "store_id"])["date"].agg(["min", "max"]).reset_index()
grid = pd.concat(
    [pd.DataFrame({"item_id": r.item_id, "store_id": r.store_id,
                   "date": pd.date_range(r.min, r.max, freq="D")})
     for r in span.itertuples(index=False)],
    ignore_index=True,
)
d = grid.merge(raw, on=["item_id", "store_id", "date"], how="left")
d["_orig"] = d["_orig"].fillna(False).astype(bool)
d = d.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
print("grid", d.shape, "gap rows", (~d._orig).sum())

key = [d["item_id"], d["store_id"]]
g = d.groupby(["item_id", "store_id"], sort=False)
q = d["quantity"].astype("float64")          # NaN on gap days

def gshift(s, k):
    return s.groupby(key, sort=False).shift(k)

def groll(s, w, fn="mean", minp=1):
    r = s.groupby(key, sort=False).rolling(w, min_periods=minp)
    out = getattr(r, fn)()
    return out.reset_index(level=[0, 1], drop=True).sort_index()

# ------------------------------------------------------------------ demand history (past only)
s7 = gshift(q, HORIZON)
for k in (7, 14, 21, 28):
    d[f"my_lag_{k}"] = gshift(q, k)
d["roll_mean_14"] = groll(s7, 14, "mean", 1)
d["roll_mean_91"] = groll(s7, 91, "mean", 7)
d["roll_zero_share_28"] = groll((s7 == 0).astype(float).where(s7.notna()), 28, "mean", 7)
d["my_roll_mean_7"] = groll(s7, 7, "mean", 1)
d["my_roll_mean_28"] = groll(s7, 28, "mean", 1)
d["my_roll_std_28"] = groll(s7, 28, "std", 2)
d["dow_mean_4w"] = d[[f"my_lag_{k}" for k in (7, 14, 21, 28)]].mean(axis=1)
d["ewm_7"] = s7.groupby(key, sort=False).transform(lambda x: x.ewm(halflife=7, ignore_na=True).mean())
cs = s7.fillna(0).groupby(key, sort=False).cumsum()
cn = s7.notna().astype(int).groupby(key, sort=False).cumsum()
d["hist_mean"] = (cs / cn.replace(0, np.nan))
d["series_age"] = (d["date"] - g["date"].transform("min")).dt.days

# Sanity check against the team's own lag features (computed the same way)
chk = d[d._orig]
for mine, theirs in [("my_lag_7", "lag_7"), ("my_lag_14", "lag_14"), ("my_lag_28", "lag_28"),
                     ("my_roll_mean_7", "roll_mean_7"), ("my_roll_mean_28", "roll_mean_28")]:
    m = np.isclose(chk[mine], chk[theirs], equal_nan=True, rtol=1e-4, atol=1e-4).mean()
    print(f"  check {mine:16s} vs {theirs:14s}: {m:.4%} identical")
# Use our date-consistent versions everywhere
for k in (7, 14, 21, 28):
    d[f"lag_{k}"] = d.pop(f"my_lag_{k}")
d["roll_mean_7"] = d.pop("my_roll_mean_7")
d["roll_mean_28"] = d.pop("my_roll_mean_28")
d["roll_std_28"] = d.pop("my_roll_std_28")

# ------------------------------------------------------------------ static / calendar extras
d["is_friday"] = (d["dow"] == 4).astype("float")          # dow: Monday = 0
d["is_food"] = (d["category"] == "food").astype("float")
d["log_regular_price"] = np.log(d["regular_price"])

# ------------------------------------------------------------------ promotion: known future (schedule)
d["in_promo_any"] = d["day_in_promo"].notna().astype(float).where(d._orig)
d["depth_x_food"] = d["depth"] * d["is_food"]
d["post_promo_1_7"] = ((d["in_promo_any"] == 0) & d["days_since_last_promo"].between(1, 7)).astype(float)

# days until the next scheduled promo day (0 if today is a promo day), capped at 60
ip = d["in_promo_any"].fillna(0).values
pos = np.arange(len(d))
nxt = np.where(ip == 1, pos, np.nan)
nxt = pd.Series(nxt).groupby([d["item_id"].values, d["store_id"].values], sort=False).bfill().values
d["days_to_next_promo"] = np.minimum(nxt - pos, 60)
d.loc[np.isnan(nxt), "days_to_next_promo"] = 60
# number of promo days in the next 7 days (t+1 .. t+7)
ipS = pd.Series(ip)
rev = ipS[::-1].groupby([d["item_id"].values[::-1], d["store_id"].values[::-1]], sort=False) \
    .rolling(7, min_periods=1).sum().reset_index(level=[0, 1], drop=True).sort_index()
d["promo_days_next_7"] = gshift(rev.reindex(d.index), -1).fillna(0)

# ------------------------------------------------------------------ promotion: known only from the past (as of t-7)
d["promo_lag7"] = gshift(d["in_promo_any"], HORIZON)
d["depth_lag7"] = gshift(d["depth"], HORIZON)
d["day_in_promo_lag7"] = gshift(d["day_in_promo"], HORIZON)
d["days_since_promo_lag7"] = gshift(d["days_since_last_promo"], HORIZON)
d["promo_share_28_lag7"] = groll(gshift(d["in_promo_any"], HORIZON), 28, "mean", 7)
# historical promo lift of this item-store, using only sales up to t-7
pm = (d["in_promo_any"] == 1)
qp, qn = q.where(pm), q.where(~pm & d["in_promo_any"].notna())
sp = gshift(qp.fillna(0), HORIZON).groupby(key, sort=False).cumsum()
np_ = gshift(qp.notna().astype(int), HORIZON).groupby(key, sort=False).cumsum()
sn = gshift(qn.fillna(0), HORIZON).groupby(key, sort=False).cumsum()
nn = gshift(qn.notna().astype(int), HORIZON).groupby(key, sort=False).cumsum()
lift = ((sp / np_.replace(0, np.nan)) + 0.1) / ((sn / nn.replace(0, np.nan)) + 0.1)
d["hist_promo_lift"] = lift.where(np_ >= 3)

# ------------------------------------------------------------------ keep original rows only
d = d[d._orig].drop(columns="_orig").reset_index(drop=True)
assert len(d) == len(raw), "row count changed!"
for c in ["store_id", "group4", "dept_name", "class_name", "subclass_name", "item_type"]:
    d[c] = d[c].astype("category")
d["promo_type"] = d["promo_type"].astype("Int64").astype("category")

# float32 to save memory
for c in d.select_dtypes("float64").columns:
    d[c] = d[c].astype("float32")

d.to_parquet(DATA_FEAT, index=False)
print("saved", DATA_FEAT, d.shape, f"{time.time()-t0:.1f}s")

# feature missingness table for the report
from config import FEATURE_SETS
allf = FEATURE_SETS["B_full_future"]
assert not set(allf) & set(FORBIDDEN), "forbidden column in feature set!"
miss = pd.DataFrame({"feature": allf,
                     "dtype": [str(d[c].dtype) for c in allf],
                     "missing_pct": [round(100 * d[c].isna().mean(), 2) for c in allf]})
miss.to_csv(TAB / "feature_missingness.csv", index=False)
print(miss.to_string())
