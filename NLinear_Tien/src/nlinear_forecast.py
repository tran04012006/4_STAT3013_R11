"""
NLinear demand forecasting - task cua Tien (STATISTICS ANALYSIS project).

Bai toan : du bao luong ban theo ngay (quantity) o cap san pham x cua hang,
           tam du bao H = 7 ngay, rolling-origin moi tuan.
Mo hinh  : 1) Seasonal naive (baseline: cung thu tuan truoc)
           2) NLinear       (Zeng et al., 2023 - "Are Transformers Effective for
                             Time Series Forecasting?"), global, channel-independent
           3) NLinear-X     NLinear + bien ngoai sinh TUONG LAI da biet truoc
                             (lich KM da len ke hoach + lich/ngay le)
           4) NLinear-X-cal ablation: chi bien lich, khong co bien KM
Chia du lieu: train <= 2024-03-31, valid 2024-04-01..06-30, test 2024-07-01..09-26
              (theo cot `split` cua pipeline nhom). Tune tren valid, retrain tren
              train+valid, chay test DUY NHAT 1 lan.

Chay:  python src/nlinear_forecast.py --data-dir ../dataset --out-dir outputs
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message="Mean of empty slice")   # chuoi chua co lich su -> xu ly bang prior

# ----------------------------------------------------------------------------
# Hang so
# ----------------------------------------------------------------------------
H = 7                                   # tam du bao (ngay)
TRAIN_END = pd.Timestamp("2024-03-31")
VALID_START, VALID_END = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30")
TEST_START, TEST_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-26")
GROUPS = ["F1", "F2", "H1", "H2"]
LOOKBACKS = [14, 28, 56]
LOSSES = ["mae_raw", "mse_scaled"]
MAX_EPOCHS, PATIENCE, BATCH, LR = 12, 3, 4096, 2e-3
SEED = 42

SERIES_FEATS = ["promo_on", "promo_nodisc", "depth", "depth_sq", "promo_first3",
                "promo_last2", "pre_promo_1_3", "post_promo_1_3", "post_promo_4_7"]
CAL_FEATS = ([f"dow_{i}" for i in range(7)] +
             ["is_holiday", "pre_holiday", "post_holiday", "ny_break", "minor_holiday"] +
             [f"month_{m}" for m in range(1, 13)])
WINDOW_FEATS = ["lookback_promo_share", "last_day_promo"]
ALL_FEATS = SERIES_FEATS + CAL_FEATS + WINDOW_FEATS
PROMO_FEAT_IDX = [ALL_FEATS.index(f) for f in SERIES_FEATS + WINDOW_FEATS]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------
# 1. Doc du lieu va dung tensor dang [series, day]
# ----------------------------------------------------------------------------
PANEL_COLS = ["date", "item_id", "store_id", "group4", "category", "dept_name",
              "quantity", "promo_status", "depth", "day_in_promo", "days_to_promo_end",
              "is_holiday", "holiday_status", "minor_holiday", "ny_break", "dow",
              "month", "phase", "split"]


def build_tensors(data_dir: Path) -> dict:
    p = pd.read_parquet(data_dir / "daily_panel.parquet", columns=PANEL_COLS)
    p = p.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    dates = pd.date_range(p["date"].min(), p["date"].max(), freq="D")
    D = len(dates)
    di = ((p["date"] - dates[0]).dt.days).to_numpy()

    keys = p[["item_id", "store_id"]].drop_duplicates().reset_index(drop=True)
    keys["sid"] = np.arange(len(keys))
    p = p.merge(keys, on=["item_id", "store_id"], how="left")
    S = len(keys)
    sid = p["sid"].to_numpy()

    static = (p.groupby("sid")[["item_id", "store_id", "group4", "category", "dept_name"]]
              .first().reindex(range(S)))
    gidx = static["group4"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy()

    def dense(values, fill=np.nan, dtype=np.float32):
        a = np.full((S, D), fill, dtype=dtype)
        a[sid, di] = values
        return a

    Q = dense(p["quantity"].to_numpy(np.float32))            # NaN = khong co du lieu
    active = dense(np.ones(len(p), np.float32), 0.0) > 0      # dong co trong panel
    first_di = np.where(active.any(1), active.argmax(1), -1)
    last_di = D - 1 - active[:, ::-1].argmax(1)

    # Q dung lam INPUT: ffill cac ngay trong khoang hoat dong bi khuyet (ngay CH bat thuong)
    Qf = pd.DataFrame(Q.T).ffill().to_numpy(np.float32).T
    for s in range(S):                                        # ngoai khoang hoat dong -> NaN
        Qf[s, :first_di[s]] = np.nan
        Qf[s, last_di[s] + 1:] = np.nan

    # --- dac trung KM theo chuoi (biet truoc vi nha ban le tu len lich KM) ---
    ps = p["promo_status"]
    promo_on = dense((ps == 1).to_numpy(np.float32), 0.0)
    promo_nodisc = dense(ps.isna().to_numpy(np.float32), 0.0)
    depth = dense(np.where(ps == 1, p["depth"].fillna(0), 0).astype(np.float32), 0.0)
    first3 = dense(((ps == 1) & (p["day_in_promo"] <= 3)).to_numpy(np.float32), 0.0)
    last2 = dense(((ps == 1) & (p["days_to_promo_end"] <= 2)).to_numpy(np.float32), 0.0)

    # so ngay ke tu lan KM gan nhat / den lan KM ke tiep (tinh tu lich KM)
    BIG = 10_000
    since = np.full((S, D), BIG, np.int32)
    until = np.full((S, D), BIG, np.int32)
    last_seen = np.full(S, -BIG, np.int64)
    for d in range(D):
        on = promo_on[:, d] > 0
        since[:, d] = np.where(on, 0, d - last_seen)
        last_seen = np.where(on, d, last_seen)
    next_seen = np.full(S, 2 * BIG, np.int64)
    for d in range(D - 1, -1, -1):
        on = promo_on[:, d] > 0
        until[:, d] = np.where(on, 0, next_seen - d)
        next_seen = np.where(on, d, next_seen)
    pre13 = ((until >= 1) & (until <= 3)).astype(np.float32)
    post13 = ((since >= 1) & (since <= 3)).astype(np.float32)
    post47 = ((since >= 4) & (since <= 7)).astype(np.float32)

    ZS = np.stack([promo_on, promo_nodisc, depth, depth ** 2, first3, last2,
                   pre13, post13, post47], axis=-1).astype(np.float32)   # [S, D, 9]

    # --- dac trung lich (giong nhau moi chuoi) ---
    cal = (p.groupby(di)[["is_holiday", "holiday_status", "minor_holiday", "ny_break"]]
           .max().reindex(range(D)).fillna(0))
    C = np.zeros((D, len(CAL_FEATS)), np.float32)
    C[np.arange(D), dates.dayofweek] = 1.0                    # dow_0 = Thu Hai
    C[:, 7] = cal["is_holiday"].to_numpy()
    C[:, 8] = (cal["holiday_status"] == -1).to_numpy()
    C[:, 9] = (cal["holiday_status"] == 1).to_numpy()
    C[:, 10] = cal["ny_break"].to_numpy()
    C[:, 11] = cal["minor_holiday"].to_numpy()
    C[np.arange(D), 12 + dates.month - 1] = 1.0

    return dict(panel=p, dates=dates, D=D, S=S, static=static, gidx=gidx,
                Q=Q, Qf=Qf, active=active, first_di=first_di, last_di=last_di,
                ZS=ZS, C=C, store=static["store_id"].to_numpy())


# ----------------------------------------------------------------------------
# 2. Mo hinh
# ----------------------------------------------------------------------------
class NLinear(nn.Module):
    """NLinear (Zeng et al., 2023): tru gia tri cuoi cua cua so, qua 1 lop Linear
    L -> H, cong lai gia tri cuoi. Trong so dung chung cho moi chuoi (global).

    Mo rong NLinear-X: cong them  s * (z_{t+h} . V_g + b_{g,h}),
    trong do z la bien ngoai sinh cua NGAY CAN DU BAO (da biet truoc),
    V_g la he so rieng cho tung nhom hang g, s = muc trung binh cua cua so + 1
    (nen hieu ung KM ty le voi quy mo ban cua san pham)."""

    def __init__(self, L: int, n_exo: int = 0, n_groups: int = 4):
        super().__init__()
        self.L, self.n_exo = L, n_exo
        self.lin = nn.Linear(L, H)
        if n_exo:
            self.V = nn.Parameter(torch.zeros(n_groups, n_exo))
            self.b = nn.Parameter(torch.zeros(n_groups, H))

    def forward(self, x, z=None, g=None, s=None):
        last = x[:, -1:]
        y = self.lin(x - last) + last
        if self.n_exo:
            eff = (z * self.V[g][:, None, :]).sum(-1) + self.b[g]
            y = y + s[:, None] * eff
        return y


# ----------------------------------------------------------------------------
# 3. Tao batch
# ----------------------------------------------------------------------------
class Data:
    def __init__(self, T: dict):
        self.T = T
        self.Qf = torch.from_numpy(np.nan_to_num(T["Qf"], nan=0.0))
        self.Q0 = torch.from_numpy(np.nan_to_num(T["Q"], nan=0.0))
        self.M = torch.from_numpy(~np.isnan(T["Q"]))
        self.ZS = torch.from_numpy(T["ZS"])
        self.C = torch.from_numpy(T["C"])
        self.g = torch.from_numpy(T["gidx"].astype(np.int64))

    def exo(self, s_idx, tgt_idx, x_promo, feat_mask):
        """z [B, H, F] cho cac ngay muc tieu."""
        zs = self.ZS[s_idx[:, None], tgt_idx]                          # [B,H,9]
        zc = self.C[tgt_idx]                                           # [B,H,24]
        w = torch.stack([x_promo.mean(1), x_promo[:, -1]], 1)          # [B,2]
        zw = w[:, None, :].expand(-1, H, -1)
        z = torch.cat([zs, zc, zw], -1)
        return z * feat_mask

    def train_samples(self, L: int, last_target_di: int):
        """Moi (series, origin t) co cua so [t-L, t-1] nam tron trong khoang hoat dong
        va ngay muc tieu t..t+H-1 <= min(last_di, last_target_di)."""
        T = self.T
        s_list, t_list = [], []
        for s in range(T["S"]):
            lo = T["first_di"][s] + L
            hi = min(T["last_di"][s], last_target_di) - H + 1
            if hi >= lo:
                t = np.arange(lo, hi + 1)
                s_list.append(np.full(len(t), s))
                t_list.append(t)
        return (torch.from_numpy(np.concatenate(s_list)),
                torch.from_numpy(np.concatenate(t_list)))


def make_feat_mask(exo_set: str) -> torch.Tensor:
    m = torch.ones(len(ALL_FEATS))
    if exo_set == "calendar":
        m[PROMO_FEAT_IDX] = 0.0
    return m


def batch_tensors(data: Data, s_idx, t_idx, L, n_exo, feat_mask):
    ar_in = torch.arange(-L, 0)
    ar_out = torch.arange(0, H)
    in_idx = t_idx[:, None] + ar_in
    tgt_idx = t_idx[:, None] + ar_out
    x = data.Qf[s_idx[:, None], in_idx]
    y = data.Q0[s_idx[:, None], tgt_idx]
    m = data.M[s_idx[:, None], tgt_idx]
    g = data.g[s_idx]
    scale = x.mean(1) + 1.0
    z = None
    if n_exo:
        x_promo = data.ZS[s_idx[:, None], in_idx, 0]
        z = data.exo(s_idx, tgt_idx, x_promo, feat_mask)
    return x, y, m, g, scale, z


# ----------------------------------------------------------------------------
# 4. Du bao rolling-origin (dung cho valid va test)
# ----------------------------------------------------------------------------
def forecast_blocks(data: Data, models: dict, L_of: dict, start: pd.Timestamp,
                    end: pd.Timestamp, masks: dict) -> pd.DataFrame:
    """Origin moi 7 ngay tu `start`. Moi ngay trong [start, end] duoc du bao dung 1 lan.
    Cua so dau vao dung du lieu thuc te den ngay origin-1 (rolling origin).
    Chuoi chua du L ngay lich su: dem trai bang trung binh phan lich su co san;
    chuoi chua co ngay nao: dung muc trung binh 28 ngay cua cac chuoi cung nhom x cua hang."""
    T = data.T
    dates = T["dates"]
    d_start = (start - dates[0]).days
    d_end = (end - dates[0]).days
    Lmax = max(max(L_of.values()), 28)
    Qf_np = T["Qf"]
    out = []
    for t in range(d_start, d_end + 1, H):
        h_last = min(t + H - 1, d_end)
        act = T["active"][:, t:h_last + 1]
        sids = np.where(act.any(1))[0]
        if len(sids) == 0:
            continue
        win = Qf_np[sids, t - Lmax:t].copy()                         # [n, Lmax]
        hist_days = (~np.isnan(win)).sum(1)
        # prior cho chuoi chua co lich su: mean 28 ngay cua cac chuoi cung group x store
        m28 = np.nanmean(Qf_np[:, t - 28:t], axis=1) if t >= 28 else np.full(T["S"], np.nan)
        prior_tbl = (pd.DataFrame({"g": T["gidx"], "st": T["store"], "m": m28})
                     .groupby(["g", "st"])["m"].mean())
        for k, s in enumerate(sids):
            row = win[k]
            if np.isnan(row).all():
                v = prior_tbl.get((T["gidx"][s], T["store"][s]), np.nan)
                row[:] = 0.0 if np.isnan(v) else v
            elif np.isnan(row).any():
                mu = np.nanmean(row)
                row[np.isnan(row)] = mu
        win_t = torch.from_numpy(win)
        s_idx = torch.from_numpy(sids)
        tgt_idx = torch.clamp(torch.arange(t, t + H)[None, :].expand(len(sids), -1), max=T["D"] - 1)

        preds = {"seasonal_naive": win[:, Lmax - 7: Lmax - 7 + H]}
        for name, model in models.items():
            L = L_of[name]
            x = win_t[:, -L:]
            z = None
            if model.n_exo:
                x_promo = data.ZS[s_idx[:, None], torch.arange(t - L, t)[None, :].expand(len(sids), -1), 0]
                z = data.exo(s_idx, tgt_idx, x_promo, masks[name])
            with torch.no_grad():
                yh = model(x, z, data.g[s_idx], x.mean(1) + 1.0).numpy()
            preds[name] = yh

        for h in range(h_last - t + 1):
            d = t + h
            keep = T["active"][sids, d]
            if not keep.any():
                continue
            rec = {"sid": sids[keep], "di": d, "origin_di": t, "horizon_step": h + 1,
                   "history_days": hist_days[keep]}
            for name, arr in preds.items():
                rec["pred_" + name] = np.clip(arr[keep, h], 0, None)
            out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


def wape(y, yh):
    return float(np.abs(y - yh).sum() / max(np.abs(y).sum(), 1e-9))


def metrics(df: pd.DataFrame, col: str) -> dict:
    y, yh = df["actual_quantity"].to_numpy(), df[col].to_numpy()
    return {"n": len(df), "MAE": float(np.abs(y - yh).mean()),
            "RMSE": float(np.sqrt(((y - yh) ** 2).mean())), "WAPE": wape(y, yh),
            "Bias_pct": float((yh.sum() - y.sum()) / max(y.sum(), 1e-9))}


# ----------------------------------------------------------------------------
# 5. Huan luyen
# ----------------------------------------------------------------------------
def train_model(data: Data, L: int, loss: str, exo_set: str, last_target_di: int,
                epochs: int | None = None, val_fn=None, log=print):
    """Neu epochs=None: early stopping theo WAPE valid (val_fn). Tra ve (model, best_epoch, curve)."""
    set_seed()
    n_exo = 0 if exo_set == "none" else len(ALL_FEATS)
    fm = make_feat_mask(exo_set)
    model = NLinear(L, n_exo)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    s_all, t_all = data.train_samples(L, last_target_di)
    n = len(s_all)
    n_ep = epochs if epochs is not None else MAX_EPOCHS
    best, best_state, best_ep, bad, curve = np.inf, None, 0, 0, []
    for ep in range(1, n_ep + 1):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            x, y, m, g, sc, z = batch_tensors(data, s_all[b], t_all[b], L, n_exo, fm)
            yh = model(x, z, g, sc)
            if loss == "mae_raw":
                e = (yh - y).abs()
            else:
                e = ((yh - y) / sc[:, None]) ** 2
            l = (e * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            l.backward()
            opt.step()
            tot += float(l) * len(b)
        model.eval()
        rec = {"epoch": ep, "train_loss": tot / n}
        if val_fn is not None:
            rec["val_WAPE"] = val_fn(model)
            if rec["val_WAPE"] < best - 1e-5:
                best, best_ep, bad = rec["val_WAPE"], ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        curve.append(rec)
        log(f"      ep{ep:02d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch"))
        if val_fn is not None and bad >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_ep = n_ep
    model.n_samples = n
    return model, best_ep, curve


# ----------------------------------------------------------------------------
# 6. Diebold-Mariano (sai so tuyet doi, tong theo ngay, HAC lag H-1)
# ----------------------------------------------------------------------------
def diebold_mariano(df: pd.DataFrame, a: str, b: str) -> dict:
    from scipy import stats
    e = df.assign(ea=(df["actual_quantity"] - df[a]).abs(),
                  eb=(df["actual_quantity"] - df[b]).abs()).groupby("date")[["ea", "eb"]].sum()
    d = (e["ea"] - e["eb"]).to_numpy()
    T_ = len(d)
    dc = d - d.mean()
    lr_var = (dc @ dc) / T_
    for k in range(1, H):
        lr_var += 2 * (1 - k / H) * (dc[k:] @ dc[:-k]) / T_
    dm = d.mean() / np.sqrt(lr_var / T_)
    hln = dm * np.sqrt((T_ + 1 - 2 * H + H * (H - 1) / T_) / T_)   # Harvey-Leybourne-Newbold
    p = 2 * stats.t.sf(abs(hln), df=T_ - 1)
    return {"model_A": a, "model_B": b, "n_days": T_, "mean_daily_AE_diff(A-B)": float(d.mean()),
            "DM_stat": float(dm), "HLN_stat": float(hln), "p_value": float(p)}


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--data-dir", default=str(here.parent / "dataset"))
    ap.add_argument("--out-dir", default=str(here / "outputs"))
    ap.add_argument("--threads", type=int, default=0, help="so luong CPU thread cho torch (0 = mac dinh)")
    ap.add_argument("--quick", action="store_true", help="chay nhanh: chi L=28, loss=mae_raw (de kiem tra cai dat)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    data_dir, out = Path(args.data_dir), Path(args.out_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logf = open(out / "run_log.txt", "w", encoding="utf-8")

    def log(msg):
        # gan tien to thoi gian cho TUNG dong de bang nhieu dong (DataFrame) khong bi lech cot
        prefix = f"[{time.time() - t0:7.1f}s] "
        text = "\n".join(prefix + ln for ln in str(msg).splitlines())
        print(text, flush=True)
        logf.write(text + "\n")
        logf.flush()

    log(f"Doc du lieu tu {data_dir}")
    T = build_tensors(data_dir)
    data = Data(T)
    log(f"  {T['S']} chuoi item x store, {T['D']} ngay ({T['dates'][0].date()} -> {T['dates'][-1].date()})")
    dates = T["dates"]
    di_of = lambda ts: (ts - dates[0]).days
    panel = T["panel"]

    def attach_actual(fc):
        key = panel[["sid", "date", "quantity", "phase", "promo_status", "depth"]].copy()
        key["di"] = (key["date"] - dates[0]).dt.days
        fc = fc.merge(key, on=["sid", "di"], how="left").rename(columns={"quantity": "actual_quantity"})
        st = T["static"].reset_index()
        return fc.merge(st, on="sid", how="left")

    # ---------------- Tuning tren valid ----------------
    lookbacks, losses = ([28], ["mae_raw"]) if args.quick else (LOOKBACKS, LOSSES)
    tuning, best_cfg = [], {}
    for exo_set, name in [("none", "nlinear"), ("full", "nlinear_x")]:
        best_cfg[name] = None
        for L in lookbacks:
            for loss in losses:
                log(f"[tune] {name} L={L} loss={loss}")
                fm = make_feat_mask(exo_set)

                def val_fn(model, L=L, fm=fm, name=name):
                    fc = forecast_blocks(data, {name: model}, {name: L}, VALID_START, VALID_END, {name: fm})
                    fc = attach_actual(fc)
                    return wape(fc["actual_quantity"].to_numpy(), fc["pred_" + name].to_numpy())

                model, ep, curve = train_model(data, L, loss, exo_set, di_of(TRAIN_END), None, val_fn, log)
                best_w = min(c["val_WAPE"] for c in curve)
                tuning.append({"model": name, "lookback_L": L, "loss": loss, "best_epoch": ep,
                               "val_WAPE": best_w, "n_train_windows": model.n_samples})
                if best_cfg[name] is None or best_w < best_cfg[name]["val_WAPE"]:
                    best_cfg[name] = {"L": L, "loss": loss, "epochs": ep, "val_WAPE": best_w}
    best_cfg["nlinear_x_cal"] = dict(best_cfg["nlinear_x"])   # ablation dung cung cau hinh NLinear-X
    pd.DataFrame(tuning).to_csv(out / "metrics" / "tuning_validation.csv", index=False)
    log(f"Cau hinh tot nhat: {json.dumps(best_cfg)}")

    # ---------------- Retrain tren train+valid, du bao test ----------------
    final_models, L_of, masks = {}, {}, {}
    for name, exo_set in [("nlinear", "none"), ("nlinear_x", "full"), ("nlinear_x_cal", "calendar")]:
        cfg = best_cfg[name]
        log(f"[final] {name} L={cfg['L']} loss={cfg['loss']} epochs={cfg['epochs']} (train+valid)")
        m, _, _ = train_model(data, cfg["L"], cfg["loss"], exo_set, di_of(VALID_END), cfg["epochs"], None, log)
        final_models[name], L_of[name], masks[name] = m, cfg["L"], make_feat_mask(exo_set)
        torch.save({"state_dict": m.state_dict(), "L": cfg["L"], "H": H, "exo_set": exo_set,
                    "features": ALL_FEATS, "groups": GROUPS}, out / "models" / f"{name}.pt")

    log("Du bao test (rolling origin moi 7 ngay)")
    fc = forecast_blocks(data, final_models, L_of, TEST_START, TEST_END, masks)
    fc = attach_actual(fc)
    fc["forecast_origin"] = dates[fc["origin_di"].to_numpy()].date
    fc["date"] = fc["date"].dt.date
    cols = ["item_id", "store_id", "date", "group4", "category", "dept_name", "phase",
            "promo_status", "depth", "forecast_origin", "horizon_step", "history_days",
            "actual_quantity", "pred_seasonal_naive", "pred_nlinear", "pred_nlinear_x", "pred_nlinear_x_cal"]
    fc = fc[cols].sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    fc["promo_status"] = fc["promo_status"].map({1.0: "promo", 0.0: "no_promo"}).fillna("promo_no_discount")
    for c in [c for c in cols if c.startswith("pred_")] + ["depth"]:
        fc[c] = fc[c].astype(float).round(4)
    fc.to_csv(out / "final_result.csv", index=False, encoding="utf-8-sig")
    n_panel_test = int((panel["split"] == "test").sum())
    log(f"final_result.csv: {len(fc)} dong (panel test co {n_panel_test} dong)")

    # ---------------- Metrics ----------------
    model_cols = {"Seasonal naive": "pred_seasonal_naive", "NLinear": "pred_nlinear",
                  "NLinear-X (KM + lich tuong lai)": "pred_nlinear_x",
                  "NLinear-X-cal (chi lich)": "pred_nlinear_x_cal"}

    def table(by):
        rows = []
        groups = [((), fc)] if by is None else fc.groupby(by, observed=True)
        for k, sub in groups:
            for mname, col in model_cols.items():
                r = {"model": mname, **metrics(sub, col)}
                if by is not None:
                    r = {by if isinstance(by, str) else "group": k, **r}
                rows.append(r)
        return pd.DataFrame(rows)

    M = out / "metrics"
    table(None).to_csv(M / "overall.csv", index=False)
    table("category").to_csv(M / "by_category.csv", index=False)
    table("group4").to_csv(M / "by_group4.csv", index=False)
    table("phase").to_csv(M / "by_phase.csv", index=False)
    table("horizon_step").to_csv(M / "by_horizon.csv", index=False)
    table("promo_status").to_csv(M / "by_promo_status.csv", index=False)
    table("store_id").to_csv(M / "by_store.csv", index=False)

    # bottom-up: cong du bao len nhom x cua hang x ngay
    agg = fc.groupby(["group4", "store_id", "date"])[["actual_quantity"] + list(model_cols.values())].sum().reset_index()
    rows = []
    for g, sub in agg.groupby("group4"):
        for mname, col in model_cols.items():
            rows.append({"group4": g, "model": mname, **metrics(sub, col)})
    for mname, col in model_cols.items():
        rows.append({"group4": "ALL", "model": mname, **metrics(agg, col)})
    pd.DataFrame(rows).to_csv(M / "bottom_up_group_store_day.csv", index=False)

    # Diebold-Mariano
    dm = [diebold_mariano(fc, "pred_nlinear_x", "pred_nlinear"),
          diebold_mariano(fc, "pred_nlinear", "pred_seasonal_naive"),
          diebold_mariano(fc, "pred_nlinear_x", "pred_nlinear_x_cal"),
          diebold_mariano(fc, "pred_nlinear_x_cal", "pred_nlinear")]
    for cat in ["food", "household"]:
        r = diebold_mariano(fc[fc["category"] == cat], "pred_nlinear_x", "pred_nlinear")
        r["subset"] = cat
        dm.append(r)
    pd.DataFrame(dm).to_csv(M / "diebold_mariano.csv", index=False)

    # trong so NLinear va he so ngoai sinh
    W = final_models["nlinear"].lin.weight.detach().numpy()
    pd.DataFrame(W, index=[f"h{h+1}" for h in range(H)],
                 columns=[f"t-{L_of['nlinear'] - j}" for j in range(L_of["nlinear"])]).to_csv(M / "nlinear_weights.csv")
    Vx = final_models["nlinear_x"].V.detach().numpy()
    pd.DataFrame(Vx.T, index=ALL_FEATS, columns=GROUPS).to_csv(M / "nlinear_x_exo_coefficients.csv")
    Wx = final_models["nlinear_x"].lin.weight.detach().numpy()
    pd.DataFrame(Wx, index=[f"h{h+1}" for h in range(H)],
                 columns=[f"t-{L_of['nlinear_x'] - j}" for j in range(L_of["nlinear_x"])]).to_csv(M / "nlinear_x_weights.csv")

    # ---------------- Danh gia cap dot KM (episodes.csv) ----------------
    ep = pd.read_csv(data_dir / "episodes.csv", parse_dates=["start", "end"])
    ep = ep[(ep["source"] == "promo") & (ep["start"] >= TEST_START) & (ep["end"] <= TEST_END)].copy()
    fcd = fc.assign(date=pd.to_datetime(fc["date"]))
    recs = []
    for r in ep.itertuples():
        sub = fcd[(fcd["item_id"] == r.item_id) & (fcd["store_id"] == r.store_id) &
                  (fcd["date"] >= r.start) & (fcd["date"] <= r.end)]
        if len(sub) == 0:
            continue
        rec = {"item_id": r.item_id, "store_id": r.store_id, "category": r.category, "group4": r.group4,
               "start": r.start.date(), "end": r.end.date(), "depth": r.depth, "pre_qty": r.pre_qty,
               "actual_during_qty": sub["actual_quantity"].mean()}
        for col in model_cols.values():
            rec[col.replace("pred_", "pred_during_")] = sub[col].mean()
        recs.append(rec)
    epi = pd.DataFrame(recs)
    epi.to_csv(M / "episodes_test_during.csv", index=False)
    erows = []
    for cat, sub in [("ALL", epi)] + list(epi.groupby("category")):
        for mname, col in model_cols.items():
            pc = col.replace("pred_", "pred_during_")
            erows.append({"category": cat, "model": mname, "n_episodes": len(sub),
                          "sum_actual_during": sub["actual_during_qty"].sum(),
                          "sum_pred_during": sub[pc].sum(),
                          "capture_ratio": sub[pc].sum() / sub["actual_during_qty"].sum(),
                          "episode_WAPE": wape(sub["actual_during_qty"].to_numpy(), sub[pc].to_numpy())})
    pd.DataFrame(erows).to_csv(M / "episodes_summary.csv", index=False)

    run = {"best_config": best_cfg, "H": H, "seed": SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
           "batch": BATCH, "lr": LR, "features": ALL_FEATS, "n_series": T["S"],
           "n_test_predictions": len(fc), "n_panel_test_rows": n_panel_test,
           "runtime_sec": round(time.time() - t0, 1), "torch": torch.__version__,
           "numpy": np.__version__, "pandas": pd.__version__}
    json.dump(run, open(out / "run_config.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log(pd.read_csv(M / "overall.csv").to_string())
    log("XONG")


if __name__ == "__main__":
    main()
