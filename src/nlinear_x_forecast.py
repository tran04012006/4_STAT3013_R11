"""NLinear-X demand forecasting (item_id x store_id, daily, H = 7, rolling origin every 7 days).

Pipeline (see Task.md):
  1. Build dense grids Q[S, D] (target) and Qf (forward-filled input) from daily_panel.parquet.
  2. Build the 35 exogenous features z of each target day.
  3. Tune NLinear-X on valid (L x loss grid, early stopping on valid WAPE).
  4. Retrain the selected config on train+valid for the best number of epochs.
  5. Rolling-origin forecast on test (NLinear-X + seasonal naive benchmark), run once.
  6. Metrics, bottom-up, Diebold-Mariano, promo-episode evaluation, coefficient export.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs"
MET_DIR = OUT_DIR / "metrics"
MOD_DIR = OUT_DIR / "models"

H = 7
STRIDE = 7
SEED = 42
LR = 0.002
BATCH = 4096
MAX_EPOCHS = 12
PATIENCE = 3
MIN_DELTA = 1e-5
L_GRID = [14, 28, 56]
LOSS_GRID = ["mae_raw", "mse_scaled"]
PRIOR_DAYS = 28

TRAIN_END = pd.Timestamp("2024-03-31")
VALID_START, VALID_END = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30")
TEST_START, TEST_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-26")
EXPECTED_TEST_ROWS = 177_713

GROUPS = ["F1", "F2", "H1", "H2"]
PANEL_COLS = ["date", "item_id", "store_id", "group4", "category", "dept_name", "quantity",
              "promo_status", "depth", "day_in_promo", "days_to_promo_end", "is_holiday",
              "holiday_status", "minor_holiday", "ny_break", "phase", "split"]
SERIES_FEATS = ["promo_on", "promo_nodisc", "depth", "depth_sq", "promo_first3", "promo_last2",
                "pre_promo_1_3", "post_promo_1_3", "post_promo_4_7"]
CAL_FEATS = ([f"dow_{i}" for i in range(7)]
             + ["is_holiday", "pre_holiday", "post_holiday", "ny_break", "minor_holiday"]
             + [f"month_{m}" for m in range(1, 13)])
WIN_FEATS = ["lookback_promo_share", "last_day_promo"]
EXO_FEATS = SERIES_FEATS + CAL_FEATS + WIN_FEATS
assert len(EXO_FEATS) == 35

torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

# ----------------------------------------------------------------------------- logging
_LOG_LINES: list[str] = []
_T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def flush_log() -> None:
    (OUT_DIR / "run_log.txt").write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- data
class Data:
    """Dense grids indexed [series, day]. Day axis is padded with H extra days so that
    the last (truncated) block can be indexed without bounds checks."""

    def __init__(self) -> None:
        panel = pd.read_parquet(DATA_DIR / "daily_panel.parquet", columns=PANEL_COLS)
        panel["date"] = pd.to_datetime(panel["date"])
        self.d0 = panel["date"].min()
        self.n_days = (panel["date"].max() - self.d0).days + 1
        self.Dp = self.n_days + H
        self.dates = pd.date_range(self.d0, periods=self.Dp, freq="D")

        keys = panel[["item_id", "store_id"]].drop_duplicates().sort_values(["item_id", "store_id"])
        keys = keys.reset_index(drop=True)
        keys["sid"] = np.arange(len(keys))
        panel = panel.merge(keys, on=["item_id", "store_id"], how="left")
        panel["di"] = (panel["date"] - self.d0).dt.days.astype(np.int64)
        self.S = len(keys)

        static = (panel.groupby("sid")[["item_id", "store_id", "group4", "category", "dept_name"]]
                  .first().sort_index())
        self.static = static
        self.g = static["group4"].map({k: i for i, k in enumerate(GROUPS)}).to_numpy(np.int64)
        self.store = static["store_id"].to_numpy(np.int64)

        S, Dp = self.S, self.Dp
        si, di = panel["sid"].to_numpy(), panel["di"].to_numpy()

        def grid(values, fill=np.nan, dtype=np.float32):
            a = np.full((S, Dp), fill, dtype=dtype)
            a[si, di] = values
            return a

        # --- target / input grids
        Q = grid(panel["quantity"].to_numpy(np.float32))
        active = ~np.isnan(Q)
        day_idx = np.arange(Dp)
        self.first_di = np.where(active.any(1), active.argmax(1), -1)
        self.last_di = np.where(active.any(1), Dp - 1 - active[:, ::-1].argmax(1), -1)
        Qf = pd.DataFrame(Q.T).ffill().to_numpy(np.float32).T
        outside = (day_idx[None, :] < self.first_di[:, None]) | (day_idx[None, :] > self.last_di[:, None])
        Qf[outside] = np.nan
        self.Q, self.Qf = Q, Qf

        # --- per-series exogenous features (9)
        ps = panel["promo_status"].to_numpy()
        promo_on = grid((ps == 1).astype(np.float32), 0.0)
        nodisc = grid(np.isnan(ps).astype(np.float32), 0.0)
        depth = grid(np.nan_to_num(panel["depth"].to_numpy(np.float32)), 0.0) * promo_on
        dip = grid(np.nan_to_num(panel["day_in_promo"].to_numpy(np.float32), nan=1e9), 1e9)
        dte = grid(np.nan_to_num(panel["days_to_promo_end"].to_numpy(np.float32), nan=1e9), 1e9)
        first3 = promo_on * (dip <= 3)
        last2 = promo_on * (dte <= 2)

        on = promo_on > 0
        big = 10 ** 6
        # forward scan: index of most recent promo_on day <= d
        last_on = np.where(on, day_idx[None, :], -big)
        last_on = np.maximum.accumulate(last_on, axis=1)
        since = day_idx[None, :] - last_on
        # backward scan: index of next promo_on day >= d
        next_on = np.where(on, day_idx[None, :], 2 * big)
        next_on = np.minimum.accumulate(next_on[:, ::-1], axis=1)[:, ::-1]
        until = next_on - day_idx[None, :]
        pre13 = ((until >= 1) & (until <= 3)).astype(np.float32)
        post13 = ((since >= 1) & (since <= 3)).astype(np.float32)
        post47 = ((since >= 4) & (since <= 7)).astype(np.float32)

        P = np.stack([promo_on, nodisc, depth, depth ** 2, first3, last2, pre13, post13, post47], axis=-1)
        self.P = torch.from_numpy(P.astype(np.float32))
        self.promo_on = promo_on
        cs = np.zeros((S, Dp + 1), dtype=np.float64)
        cs[:, 1:] = np.cumsum(promo_on, axis=1)
        self.promo_cs = torch.from_numpy(cs.astype(np.float32))
        self.promo_on_t = torch.from_numpy(promo_on)

        # --- calendar features (24), same for every series
        hs = panel["holiday_status"]
        daily = pd.DataFrame({
            "di": panel["di"],
            "is_holiday": panel["is_holiday"].astype(np.float32),
            "pre_holiday": (hs == -1).astype(np.float32),
            "post_holiday": (hs == 1).astype(np.float32),
            "ny_break": panel["ny_break"].astype(np.float32),
            "minor_holiday": panel["minor_holiday"].astype(np.float32),
        }).groupby("di").max().reindex(day_idx, fill_value=0.0)
        dow = np.eye(7, dtype=np.float32)[self.dates.dayofweek.to_numpy()]
        month = np.eye(12, dtype=np.float32)[self.dates.month.to_numpy() - 1]
        C = np.concatenate([dow, daily[["is_holiday", "pre_holiday", "post_holiday", "ny_break",
                                        "minor_holiday"]].to_numpy(np.float32), month], axis=1)
        self.C = torch.from_numpy(C)

        self.Qf_t = torch.from_numpy(np.nan_to_num(Qf))      # only read inside valid windows
        self.Q_t = torch.from_numpy(Q)
        self.g_t = torch.from_numpy(self.g)

        # panel rows kept for joining evaluation columns
        self.panel = panel[["sid", "di", "phase", "promo_status", "depth", "split"]]
        log(f"Data: {len(panel):,} rows, S={S}, D={self.n_days} "
            f"({self.d0.date()} -> {self.dates[self.n_days - 1].date()})")

    def di(self, ts: pd.Timestamp) -> int:
        return int((ts - self.d0).days)

    def exo(self, s: torch.Tensor, t: torch.Tensor, L: int) -> torch.Tensor:
        """z of the 7 target days t..t+6 for origins t: (B, 7, 35)."""
        hh = torch.arange(H)
        days = t[:, None] + hh[None, :]
        zs = self.P[s[:, None], days]                        # (B, 7, 9)
        zc = self.C[days]                                    # (B, 7, 24)
        share = (self.promo_cs[s, t] - self.promo_cs[s, t - L]) / L
        lastp = self.promo_on_t[s, t - 1]
        zw = torch.stack([share, lastp], dim=1)[:, None, :].expand(-1, H, -1)
        return torch.cat([zs, zc, zw], dim=2)


# ----------------------------------------------------------------------------- model
class NLinearX(nn.Module):
    def __init__(self, L: int, n_exo: int = 35, n_groups: int = 4) -> None:
        super().__init__()
        self.L = L
        self.linear = nn.Linear(L, H)                        # default PyTorch init
        self.V = nn.Parameter(torch.zeros(n_groups, n_exo))
        self.c = nn.Parameter(torch.zeros(n_groups, H))

    def forward(self, x: torch.Tensor, z: torch.Tensor, g: torch.Tensor):
        last = x[:, -1:]
        s = x.mean(dim=1, keepdim=True) + 1.0
        base = last + self.linear(x - last)
        exo = torch.einsum("bhk,bk->bh", z, self.V[g]) + self.c[g]
        return base + s * exo, s


def masked_loss(pred, y, mask, s, kind):
    m = mask.float()
    n = m.sum().clamp_min(1.0)
    if kind == "mae_raw":
        return ((pred - y).abs() * m).sum() / n
    if kind == "mse_scaled":
        return ((((pred - y) / s) ** 2) * m).sum() / n
    raise ValueError(kind)


# ----------------------------------------------------------------------------- training
def train_samples(data: Data, L: int, end_di: int):
    """All (s, t) with t - L >= first_di[s] and t + 6 <= min(last_di[s], end_di)."""
    lo = data.first_di + L
    hi = np.minimum(data.last_di, end_di) - (H - 1)
    n = np.clip(hi - lo + 1, 0, None)
    s = np.repeat(np.arange(data.S), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    t = np.repeat(lo, n) + offs
    return torch.from_numpy(s.astype(np.int64)), torch.from_numpy(t.astype(np.int64))


def train_model(data: Data, L: int, loss_kind: str, end_di: int, epochs: int,
                early_stop: bool, tag: str):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = NLinearX(L)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    S_idx, T_idx = train_samples(data, L, end_di)
    n = len(S_idx)
    back = torch.arange(-L, 0)
    fwd = torch.arange(H)
    log(f"[{tag}] L={L} loss={loss_kind} samples={n:,} end={data.dates[end_di].date()}")

    history = []
    best = (np.inf, 0, None)
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            s, t = S_idx[b], T_idx[b]
            x = data.Qf_t[s[:, None], t[:, None] + back[None, :]]
            yraw = data.Q_t[s[:, None], t[:, None] + fwd[None, :]]
            mask = ~torch.isnan(yraw)
            y = torch.nan_to_num(yraw)
            z = data.exo(s, t, L)
            pred, sc = model(x, z, data.g_t[s])
            loss = masked_loss(pred, y, mask, sc, loss_kind)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        msg = f"[{tag}] epoch {ep:2d} train_loss={tot / cnt:.5f}"
        if early_stop:
            res = rolling_forecast(data, model, L, VALID_START, VALID_END, with_naive=False)
            w = np.abs(res["actual"] - res["nl"]).sum() / res["actual"].sum()
            history.append(w)
            msg += f" valid_WAPE={w:.5f}"
            if w < best[0] - MIN_DELTA:
                best = (w, ep, {k: v.clone() for k, v in model.state_dict().items()})
                bad = 0
            else:
                bad += 1
        log(msg)
        if early_stop and bad >= PATIENCE:
            log(f"[{tag}] early stop after epoch {ep}")
            break
    if early_stop:
        model.load_state_dict(best[2])
        return model, best[0], best[1], history
    return model, None, epochs, history


# ----------------------------------------------------------------------------- rolling forecast
@torch.no_grad()
def rolling_forecast(data: Data, model: NLinearX, L: int, start: pd.Timestamp, end: pd.Timestamp,
                     with_naive: bool = True) -> dict:
    model.eval()
    Lmax = max(L, PRIOR_DAYS)
    d_start, d_end = data.di(start), data.di(end)
    S = data.S
    all_s = torch.arange(S)
    gs_key = data.g * 10 + data.store
    out = {k: [] for k in ["sid", "di", "origin", "h", "history_days", "actual", "nl", "sn"]}
    for t in range(d_start, d_end + 1, STRIDE):
        blk_len = min(H, d_end - t + 1)
        tgt = data.Q[:, t:t + blk_len]
        act = ~np.isnan(tgt)
        sel = act.any(1)
        win = data.Qf[:, t - Lmax:t].copy()                 # data up to t-1 only
        valid = ~np.isnan(win)
        hist = valid.sum(1)
        # cold-start prior: mean over (group4, store) of each series' last-28-day mean
        w28 = win[:, -PRIOR_DAYS:]
        n28 = (~np.isnan(w28)).sum(1)
        m28 = np.where(n28 > 0, np.nansum(w28, 1) / np.maximum(n28, 1), np.nan)
        prior_df = pd.DataFrame({"k": gs_key, "m": m28}).groupby("k")["m"].mean()
        prior = pd.Series(gs_key).map(prior_df).to_numpy(np.float64)
        prior = np.nan_to_num(prior, nan=0.0)
        mean_part = np.where(hist > 0, np.nansum(win, 1) / np.maximum(hist, 1), 0.0)
        fill = np.where(hist == 0, prior, mean_part).astype(np.float32)
        win = np.where(valid, win, fill[:, None]).astype(np.float32)

        x = torch.from_numpy(win[:, -L:])
        tt = torch.full((S,), t, dtype=torch.int64)
        z = data.exo(all_s, tt, L)
        pred, _ = model(x, z, data.g_t)
        pred = pred.clamp_min(0.0).numpy()[:, :blk_len]
        sn = win[:, Lmax - 7:Lmax - 7 + blk_len]              # day t-8+h for h = 1..blk_len

        ss, hh = np.nonzero(act & sel[:, None])
        out["sid"].append(ss)
        out["di"].append(t + hh)
        out["origin"].append(np.full(len(ss), t))
        out["h"].append(hh + 1)
        out["history_days"].append(hist[ss])
        out["actual"].append(tgt[ss, hh])
        out["nl"].append(pred[ss, hh])
        out["sn"].append(sn[ss, hh])
    return {k: np.concatenate(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- metrics
MODELS = [("seasonal_naive", "pred_seasonal_naive"), ("nlinear_x", "pred_nlinear_x")]


def metric_row(y, p) -> dict:
    e = p - y
    sy = y.sum()
    return {"n": len(y), "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "WAPE": np.abs(e).sum() / sy if sy > 0 else np.nan,
            "Bias": (p.sum() - sy) / sy if sy > 0 else np.nan}


def grouped_metrics(df: pd.DataFrame, by: str | None, actual="actual_quantity") -> pd.DataFrame:
    rows = []
    parts = [("ALL", df)] if by is None else list(df.groupby(by, sort=True, dropna=False))
    for key, part in parts:
        for name, col in MODELS:
            r = {by or "subset": key, "model": name}
            r.update(metric_row(part[actual].to_numpy(np.float64), part[col].to_numpy(np.float64)))
            rows.append(r)
    return pd.DataFrame(rows)


def diebold_mariano(df: pd.DataFrame, h: int = H) -> dict:
    ea = (df["actual_quantity"] - df["pred_nlinear_x"]).abs()
    eb = (df["actual_quantity"] - df["pred_seasonal_naive"]).abs()
    d = (ea - eb).groupby(df["date"]).sum().sort_index().to_numpy(np.float64)
    T = len(d)
    dbar = d.mean()
    dc = d - dbar
    var = (dc @ dc) / T
    for k in range(1, h):
        var += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / T
    dm = dbar / np.sqrt(var / T)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(hln), df=T - 1)
    return {"T": T, "mean_d": dbar, "hac_var": var, "DM": dm, "HLN": hln, "p_value": p}


def episode_eval(res: pd.DataFrame) -> pd.DataFrame:
    ep = pd.read_csv(DATA_DIR / "episodes.csv", usecols=["source", "item_id", "store_id",
                                                         "category", "start", "end"])
    ep = ep[ep["source"] == "promo"].copy()
    ep["start"], ep["end"] = pd.to_datetime(ep["start"]), pd.to_datetime(ep["end"])
    ep = ep[(ep["start"] >= TEST_START) & (ep["end"] <= TEST_END)].reset_index(drop=True)
    ep["ep_id"] = np.arange(len(ep))
    m = ep.merge(res[["item_id", "store_id", "date", "actual_quantity", "pred_seasonal_naive",
                      "pred_nlinear_x"]], on=["item_id", "store_id"], how="inner")
    m = m[(m["date"] >= m["start"]) & (m["date"] <= m["end"])]
    per = m.groupby(["ep_id", "category"])[["actual_quantity", "pred_seasonal_naive",
                                            "pred_nlinear_x"]].mean().reset_index()
    rows = []
    for subset, part in [("ALL", per)] + list(per.groupby("category")):
        a = part["actual_quantity"].to_numpy()
        for name, col in MODELS:
            p = part[col].to_numpy()
            rows.append({"subset": subset, "model": name, "n_episodes": len(part),
                         "sum_mean_actual": a.sum(), "sum_mean_pred": p.sum(),
                         "capture_ratio": p.sum() / a.sum(),
                         "episode_WAPE": np.abs(p - a).sum() / a.sum()})
    return pd.DataFrame(rows), len(ep)


# ----------------------------------------------------------------------------- main
def main() -> None:
    for d in (OUT_DIR, MET_DIR, MOD_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log(f"Python {sys.version.split()[0]} | torch {torch.__version__} | threads {torch.get_num_threads()}")
    data = Data()
    di_train_end, di_valid_end = data.di(TRAIN_END), data.di(VALID_END)

    # ---------------- tuning
    tuning = []
    for L in L_GRID:
        for loss_kind in LOSS_GRID:
            _, w, ep, hist = train_model(data, L, loss_kind, di_train_end, MAX_EPOCHS,
                                         early_stop=True, tag="tune")
            tuning.append({"L": L, "loss": loss_kind, "best_epoch": ep, "valid_WAPE": w,
                           "epochs_run": len(hist),
                           "valid_WAPE_by_epoch": " ".join(f"{v:.5f}" for v in hist)})
            log(f"  -> L={L} {loss_kind}: best valid WAPE {w:.5f} at epoch {ep}")
    tun = pd.DataFrame(tuning)
    best = tun.loc[tun["valid_WAPE"].idxmin()]
    tun["selected"] = tun.index == best.name
    tun.to_csv(MET_DIR / "tuning_validation.csv", index=False)
    L, loss_kind, n_ep = int(best["L"]), str(best["loss"]), int(best["best_epoch"])
    log(f"Selected: L={L}, loss={loss_kind}, epochs={n_ep}, valid WAPE={best['valid_WAPE']:.5f}")

    # ---------------- retrain on train+valid
    model, _, _, _ = train_model(data, L, loss_kind, di_valid_end, n_ep, early_stop=False,
                                 tag="final")
    torch.save({"state_dict": model.state_dict(), "L": L, "loss": loss_kind, "epochs": n_ep,
                "exo_features": EXO_FEATS, "groups": GROUPS}, MOD_DIR / "nlinear_x.pt")

    # ---------------- test (single run)
    r = rolling_forecast(data, model, L, TEST_START, TEST_END, with_naive=True)
    st = data.static
    res = pd.DataFrame({
        "sid": r["sid"], "di": r["di"],
        "item_id": st["item_id"].to_numpy()[r["sid"]],
        "store_id": st["store_id"].to_numpy()[r["sid"]],
        "date": data.dates[r["di"]],
        "group4": st["group4"].to_numpy()[r["sid"]],
        "category": st["category"].to_numpy()[r["sid"]],
        "dept_name": st["dept_name"].to_numpy()[r["sid"]],
        "forecast_origin": data.dates[r["origin"]],
        "horizon_step": r["h"], "history_days": r["history_days"],
        "actual_quantity": r["actual"].astype(np.float64),
        "pred_seasonal_naive": r["sn"].astype(np.float64),
        "pred_nlinear_x": r["nl"].astype(np.float64),
    })
    test_rows = data.panel[data.panel["split"] == "test"]
    res = res.merge(test_rows[["sid", "di", "phase", "promo_status", "depth"]], on=["sid", "di"],
                    how="left", validate="one_to_one")
    n_orig = res["forecast_origin"].nunique()
    dup = res.duplicated(["sid", "di"]).sum()
    log(f"Test: {len(res):,} rows (expected {EXPECTED_TEST_ROWS:,}), origins={n_orig}, "
        f"duplicates={dup}, rows missing from panel test split={res['phase'].isna().sum()}, "
        f"panel test rows not forecast={len(test_rows) - len(res)}")
    assert len(res) == EXPECTED_TEST_ROWS and dup == 0
    res = res.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    cols = ["item_id", "store_id", "date", "group4", "category", "dept_name", "phase",
            "promo_status", "depth", "forecast_origin", "horizon_step", "history_days",
            "actual_quantity", "pred_seasonal_naive", "pred_nlinear_x"]
    out = res[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["forecast_origin"] = out["forecast_origin"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUT_DIR / "final_result.csv", index=False)

    # ---------------- metrics
    res["promo_label"] = np.select([res["promo_status"] == 1, res["promo_status"] == 0],
                                   ["promo", "no_promo"], "promo_no_discount")
    overall = grouped_metrics(res, None)
    overall.to_csv(MET_DIR / "overall.csv", index=False)
    grouped_metrics(res, "category").to_csv(MET_DIR / "by_category.csv", index=False)
    grouped_metrics(res, "group4").to_csv(MET_DIR / "by_group4.csv", index=False)
    grouped_metrics(res, "phase").to_csv(MET_DIR / "by_phase.csv", index=False)
    grouped_metrics(res, "horizon_step").to_csv(MET_DIR / "by_horizon.csv", index=False)
    grouped_metrics(res, "promo_label").rename(columns={"promo_label": "promo_status"}) \
        .to_csv(MET_DIR / "by_promo_status.csv", index=False)
    grouped_metrics(res, "store_id").to_csv(MET_DIR / "by_store.csv", index=False)

    agg = res.groupby(["group4", "store_id", "date"])[["actual_quantity", "pred_seasonal_naive",
                                                       "pred_nlinear_x"]].sum().reset_index()
    bu = pd.concat([grouped_metrics(agg, "group4").rename(columns={"group4": "subset"}),
                    grouped_metrics(agg, None)], ignore_index=True)
    bu.insert(0, "level", "group4 x store x date")
    bu.to_csv(MET_DIR / "bottom_up.csv", index=False)

    dm_rows = []
    for subset, part in [("ALL", res)] + list(res.groupby("category")):
        r_ = {"subset": subset, "model_A": "nlinear_x", "model_B": "seasonal_naive", "H": H}
        r_.update(diebold_mariano(part))
        dm_rows.append(r_)
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(MET_DIR / "diebold_mariano.csv", index=False)

    eps, n_ep_total = episode_eval(res)
    eps.to_csv(MET_DIR / "episodes_summary.csv", index=False)

    W = model.linear.weight.detach().numpy()
    b = model.linear.bias.detach().numpy()
    wdf = pd.DataFrame(W, columns=[f"lag_{L - j}" for j in range(L)])   # column j -> x_{j+1}
    wdf.insert(0, "horizon_step", np.arange(1, H + 1))
    wdf["bias"] = b
    wdf.to_csv(MET_DIR / "nlinear_x_weights.csv", index=False)
    V = model.V.detach().numpy()
    vdf = pd.DataFrame(V.T, columns=GROUPS)
    vdf.insert(0, "feature", EXO_FEATS)
    vdf.to_csv(MET_DIR / "nlinear_x_exo_coefficients.csv", index=False)
    cdf = pd.DataFrame(model.c.detach().numpy(), index=GROUPS,
                       columns=[f"h{i}" for i in range(1, H + 1)])
    cdf.index.name = "group4"
    cdf.to_csv(MET_DIR / "nlinear_x_group_intercepts.csv")

    # ---------------- summary log
    log("Overall test metrics:\n" + overall.to_string(index=False))
    log("By category WAPE:\n" + grouped_metrics(res, "category")
        .pivot(index="category", columns="model", values="WAPE").to_string())
    log("By group4 WAPE:\n" + grouped_metrics(res, "group4")
        .pivot(index="group4", columns="model", values="WAPE").to_string())
    log("By phase WAPE:\n" + grouped_metrics(res, "phase")
        .pivot(index="phase", columns="model", values="WAPE").to_string())
    log("By horizon WAPE:\n" + grouped_metrics(res, "horizon_step")
        .pivot(index="horizon_step", columns="model", values="WAPE").to_string())
    log("Bottom-up:\n" + bu.to_string(index=False))
    log("Diebold-Mariano:\n" + dm.to_string(index=False))
    log(f"Episodes in test: {n_ep_total}\n" + eps.to_string(index=False))

    cfg = {
        "selected": {"L": L, "loss": loss_kind, "epochs": n_ep, "valid_WAPE": float(best["valid_WAPE"])},
        "H": H, "stride": STRIDE, "seed": SEED, "lr": LR, "batch_size": BATCH,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
        "L_grid": L_GRID, "loss_grid": LOSS_GRID, "prior_days": PRIOR_DAYS,
        "train_end": str(TRAIN_END.date()), "valid": [str(VALID_START.date()), str(VALID_END.date())],
        "test": [str(TEST_START.date()), str(TEST_END.date())],
        "exo_features": EXO_FEATS, "groups": GROUPS,
        "n_series": data.S, "n_days": data.n_days, "n_test_rows": len(res),
        "negative_clip": "applied to forecasts only (not inside the training loss)",
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "pyarrow": __import__("pyarrow").__version__, "scipy": scipy.__version__,
                     "torch": torch.__version__,
                     "matplotlib": __import__("matplotlib").__version__},
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    log(f"Done in {time.time() - _T0:.1f}s")
    flush_log()


if __name__ == "__main__":
    main()
