"""Shared helpers: data loading, model fitting, metrics."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from config import DATA_FEAT, TARGET, CATEGORICAL, N_THREADS, SEED, DRYRUN

_DF = None


def load():
    global _DF
    if _DF is None:
        _DF = pd.read_parquet(DATA_FEAT)
    return _DF


def between(df, start=None, end=None):
    m = np.ones(len(df), bool)
    if start is not None:
        m &= (df["date"] >= pd.Timestamp(start)).values
    if end is not None:
        m &= (df["date"] <= pd.Timestamp(end)).values
    return m


# ------------------------------------------------------------------ metrics
def wape(y, p):
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = p - y
    return {
        "n": len(y),
        "WAPE": wape(y, p),
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "Bias_pct": float(e.sum() / y.sum()),
        "actual_sum": float(y.sum()),
    }


def make_lgb_wape(transform):
    """LightGBM custom eval: WAPE on the original quantity scale."""
    def _f(preds, data):
        y = data.get_label()
        if transform == "log1p":
            y, preds = np.expm1(y), np.expm1(preds)
        return "wape", wape(y, np.clip(preds, 0, None)), False
    return _f


# ------------------------------------------------------------------ significance
def dm_test(loss_a, loss_b, h=7):
    """Diebold-Mariano with Newey-West variance (lag h-1) and the Harvey et al. (1997)
    small-sample correction. Positive statistic -> model b has lower loss."""
    dlt = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(dlt); mu = dlt.mean()
    gam = [np.sum((dlt[k:] - mu) * (dlt[:T - k] - mu)) / T for k in range(h)]
    var = (gam[0] + 2 * sum((1 - k / h) * gam[k] for k in range(1, h))) / T
    stat = mu / np.sqrt(var)
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return stat, p


def cluster_bootstrap(g, a, b, n_boot=200 if DRYRUN else 2000, seed=SEED):
    """Resample item-store series with replacement; CI of WAPE_a - WAPE_b (pp)."""
    g = g.assign(ea=(g["quantity"] - g[f"pred_{a}"]).abs(), eb=(g["quantity"] - g[f"pred_{b}"]).abs())
    s = g.groupby(["item_id", "store_id"])[["ea", "eb", "quantity"]].sum().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    ea, eb, y = s[:, 0][idx].sum(1), s[:, 1][idx].sum(1), s[:, 2][idx].sum(1)
    diff = (ea - eb) / y * 100
    point = (s[:, 0].sum() - s[:, 1].sum()) / s[:, 2].sum() * 100
    return point, np.percentile(diff, 2.5), np.percentile(diff, 97.5), float((diff <= 0).mean())


# ------------------------------------------------------------------ model
def build_params(hp):
    """Translate a hyper-parameter dict into LightGBM params + target transform."""
    hp = dict(hp)
    obj = hp.pop("objective")
    p = dict(
        boosting_type="gbdt", learning_rate=hp.pop("learning_rate", 0.05),
        num_leaves=hp.pop("num_leaves", 127), min_data_in_leaf=hp.pop("min_data_in_leaf", 100),
        feature_fraction=hp.pop("feature_fraction", 0.8), bagging_fraction=hp.pop("bagging_fraction", 0.8),
        bagging_freq=1, lambda_l1=hp.pop("lambda_l1", 0.0), lambda_l2=hp.pop("lambda_l2", 1.0),
        max_bin=255, cat_smooth=hp.pop("cat_smooth", 10.0), min_data_per_group=50,
        num_threads=N_THREADS, seed=SEED, verbose=-1, metric="None",
    )
    transform = "none"
    if obj == "tweedie":
        p["objective"] = "tweedie"
        p["tweedie_variance_power"] = hp.pop("tweedie_variance_power", 1.3)
    elif obj == "poisson":
        p["objective"] = "poisson"
    elif obj == "l2log":
        p["objective"] = "regression"
        transform = "log1p"
    else:
        raise ValueError(obj)
    hp.pop("tweedie_variance_power", None)
    p.update(hp)
    return p, transform


def make_dataset(df, feats, transform, reference=None):
    y = df[TARGET].values.astype(float)
    if transform == "log1p":
        y = np.log1p(y)
    cats = [c for c in feats if c in CATEGORICAL]
    return lgb.Dataset(df[feats], label=y, categorical_feature=cats, reference=reference,
                       free_raw_data=False)


def fit(train_df, feats, hp, valid_df=None, num_boost_round=3000, early_stop=100, seed=None):
    params, transform = build_params(hp)
    if seed is not None:
        params["seed"] = seed
    dtr = make_dataset(train_df, feats, transform)
    kw = {}
    if valid_df is not None:
        dva = make_dataset(valid_df, feats, transform, reference=dtr)
        kw = dict(valid_sets=[dva], valid_names=["valid"], feval=make_lgb_wape(transform),
                  callbacks=[lgb.early_stopping(early_stop, verbose=False)])
    booster = lgb.train({k: v for k, v in params.items()}, dtr, num_boost_round=num_boost_round, **kw)
    booster._target_transform = transform
    return booster


def predict(booster, df, feats, transform=None, num_iteration=None):
    tr = transform or getattr(booster, "_target_transform", "none")
    p = booster.predict(df[feats], num_iteration=num_iteration)
    if tr == "log1p":
        p = np.expm1(p)
    return np.clip(p, 0, None)
