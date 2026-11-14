"""LSTM encoder-decoder dự báo lượng bán theo ngày ở cấp mã hàng × cửa hàng.

Đầu vào : data_model/daily_panel.parquet (bảng C, đã chia train / valid / test theo thời gian)
Đầu ra  : output/lstm/  (log huấn luyện, metric, dự báo test, độ quan trọng nhóm feature, mô phỏng depth)

Thiết kế
- 1 mô hình toàn cục cho cả 4.063 chuỗi mã hàng × cửa hàng.
- Encoder đọc L ngày quá khứ (lượng bán log1p + cờ có dữ liệu + feature biết trước), decoder nhận feature
  biết trước của H = 7 ngày tới (lịch, lễ, KM đã lên kế hoạch, giá) và xuất 7 dự báo.
- Feature tĩnh (store, nhóm hàng, ngành hàng, class, format) qua embedding.
- Đánh giá theo khối tuần không chồng lấn: mỗi ngày trong valid / test được dự báo đúng 1 lần, tầm 1–7 ngày.
- Tuning (thống nhất cho cả LSTM và HGB): Manual Grid Search + Hold-out Validation + Early Stopping.
  * Hold-out: train ≤ 31/03/2024, valid 01/04–30/06/2024 (chọn siêu tham số + dừng sớm), test 07–09/2024
    chỉ chạy 1 lần cho cấu hình tốt nhất.
  * LSTM: lưới đầy đủ lookback × hidden × learning rate; mỗi cấu hình dừng sớm theo WAPE valid
    (patience 4 epoch, khôi phục trọng số của epoch tốt nhất).
  * HGB : lưới đầy đủ learning rate × max_leaf_nodes × min_samples_leaf; mỗi cấu hình thêm cây theo từng bước
    25 cây (warm start) và dừng sớm theo WAPE valid (patience 4 bước).

Chạy: python src/train_lstm.py            (đầy đủ)
      python src/train_lstm.py --smoke    (chạy thử nhanh)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor

import os
ROOT = Path(__file__).resolve().parents[1]
# Chạy local: đọc/ghi trong dự án. Chạy Kaggle: đặt PANEL_PATH / OUT_DIR qua biến môi trường.
PANEL_PATH = Path(os.environ.get("PANEL_PATH", ROOT / "data_model" / "daily_panel.parquet"))
OUT = Path(os.environ.get("OUT_DIR", ROOT / "output" / "lstm"))
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = "--smoke" in sys.argv

SEED, H = 42, 7
SEED_PERM = SEED
TRAIN_END, VALID_END = pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")
# Manual Grid Search cho LSTM (lưới đầy đủ). Hàm mất mát cố định = MAE trên lượng gốc: WAPE = Σ|e| / Σy nên MAE
# tối ưu đúng đại lượng được đánh giá; ở v1 (lưới loss × lookback) log-MSE kém hơn ~3 điểm WAPE ở cả L=28 và L=56.
LOSS = "raw_mae"
GRID_LOOKBACK, GRID_HID, GRID_LR = [28, 56], [64, 128], [1e-3, 2e-3]
DROPOUT = 0.1
VARIANTS = [(L, hd, lr) for L in GRID_LOOKBACK for hd in GRID_HID for lr in GRID_LR]
# Early stopping theo WAPE trên hold-out valid
MAX_EPOCHS, PATIENCE, MIN_DELTA, WIN_PER_EPOCH, BATCH = 40, 4, 1e-4, 150_000, 1024
# HGB: Manual Grid Search + early stopping theo WAPE valid
HGB_GRID_LR, HGB_GRID_LEAVES, HGB_GRID_MINLEAF = [0.05, 0.1], [31, 63, 127], [20, 100]
HGB_STEP, HGB_PATIENCE, HGB_MAX_IT = 25, 4, 3000
if SMOKE:
    VARIANTS, MAX_EPOCHS, WIN_PER_EPOCH = [(28, 32, 2e-3), (28, 32, 1e-3)], 2, 20_000
    HGB_GRID_LR, HGB_GRID_LEAVES, HGB_GRID_MINLEAF, HGB_MAX_IT = [0.1], [31], [20, 100], 75
PADL, PADR = max(v[0] for v in VARIANTS), H

torch.set_num_threads(min(16, os.cpu_count() or 1))
LOG_NAME = "run_smoke.log" if SMOKE else ("run_importance.log" if "--importance-only" in sys.argv else "run.log")
LOG = open(OUT / LOG_NAME, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


t0 = time.time()
# ------------------------------------------------------------------ 1. Đọc panel
df = pd.read_parquet(PANEL_PATH)
dates = pd.date_range(df.date.min(), df.date.max())
ND = len(dates)
df["d"] = (df.date - dates[0]).dt.days.astype(int)
pairs = df[["item_id", "store_id"]].drop_duplicates().sort_values(["item_id", "store_id"]).reset_index(drop=True)
pairs["pair"] = np.arange(len(pairs))
df = df.merge(pairs, on=["item_id", "store_id"])
P = len(pairs)
I_TRAIN_END, I_VALID_END = (TRAIN_END - dates[0]).days, (VALID_END - dates[0]).days
say(f"Thiết bị: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV.type == "cuda" else "")
    + f", torch {torch.__version__}, {os.cpu_count()} CPU")
say(f"Panel {df.shape}, {P} cặp, {ND} ngày; train ≤ ngày {I_TRAIN_END}, valid ≤ ngày {I_VALID_END}")

# ------------------------------------------------------------------ 2. Feature
# 2a. Theo ngày (giống nhau cho mọi chuỗi): lịch + lễ
cal = pd.DataFrame({"date": dates})
day_info = df.drop_duplicates("d").set_index("d").sort_index()
for c in ["is_holiday", "holiday_status", "minor_holiday", "ny_break", "days_to_holiday", "days_after_holiday",
          "pheic", "store4_period"]:
    cal[c] = day_info[c].reindex(range(ND)).values
cal[["is_holiday", "minor_holiday", "ny_break", "pheic", "store4_period"]] = \
    cal[["is_holiday", "minor_holiday", "ny_break", "pheic", "store4_period"]].ffill().bfill()
CAL_COLS = []
dow = cal.date.dt.dayofweek
for k in range(7):
    cal[f"dow_{k}"] = (dow == k).astype(float); CAL_COLS.append(f"dow_{k}")
cal["is_weekend"] = (dow >= 5).astype(float)
cal["is_friday"] = (dow == 4).astype(float)
cal["dom_sin"], cal["dom_cos"] = np.sin(2 * np.pi * cal.date.dt.day / 31), np.cos(2 * np.pi * cal.date.dt.day / 31)
cal["mon_sin"], cal["mon_cos"] = np.sin(2 * np.pi * cal.date.dt.month / 12), np.cos(2 * np.pi * cal.date.dt.month / 12)
cal["hol_pre"] = (cal.holiday_status == -1).astype(float)
cal["hol_post"] = (cal.holiday_status == 1).astype(float)
cal["days_to_hol"] = cal.days_to_holiday.clip(0, 30).fillna(30) / 30
cal["days_after_hol"] = cal.days_after_holiday.clip(0, 30).fillna(30) / 30
CAL_GROUPS = {
    "dow_weekend_friday": CAL_COLS + ["is_weekend", "is_friday"],
    "month_dom": ["dom_sin", "dom_cos", "mon_sin", "mon_cos"],
    "holiday": ["is_holiday", "hol_pre", "hol_post", "minor_holiday", "ny_break", "days_to_hol", "days_after_hol"],
    "regime_store4_pheic": ["pheic", "store4_period"],
}
CAL_COLS = [c for g in CAL_GROUPS.values() for c in g]
CAL = cal[CAL_COLS].values.astype(np.float32)

# 2b. Theo chuỗi × ngày: KM đã lên kế hoạch + giá (biết trước)
f = pd.DataFrame(index=df.index)
f["promo_disc"] = (df.promo_status == 1).astype(float)
f["promo_nodisc"] = df.promo_status.isna().astype(float)          # KM không giảm giá
in_promo = (f.promo_disc + f.promo_nodisc) > 0
f["depth"] = df.depth.fillna(0).clip(0, 0.9)
f["depth_x_food"] = f.depth * (df.category == "food")              # tương tác depth × nhóm hàng
f["price_ratio"] = df.price_ratio.fillna(1).clip(0.2, 1.5)
f["promo_progress"] = (df.day_in_promo / df.promo_duration).fillna(0)
f["days_to_promo_end"] = (df.days_to_promo_end.clip(0, 30) / 30).fillna(0)
f["promo_duration"] = (df.promo_duration.clip(0, 60) / 60).fillna(0)
for code in [5, 6, 10, 8]:
    f[f"ptype_{code}"] = (df.promo_type == code).astype(float)
f["ptype_other"] = (df.promo_type.notna() & ~df.promo_type.isin([5, 6, 10, 8])).astype(float)
dslp = df.days_since_last_promo
f["days_since_promo"] = (dslp.clip(0, 60) / 60).fillna(1)        # ≈ số ngày kể từ khi KM kết thúc
f["never_promo"] = dslp.isna().astype(float)
f["post_promo_1_7"] = (~in_promo & (dslp <= 7)).astype(float)    # trạng thái hậu KM
f["post_promo_8_14"] = (~in_promo & (dslp > 7) & (dslp <= 14)).astype(float)
lrp = np.log(df.regular_price)
mu, sd = lrp[df.split == "train"].mean(), lrp[df.split == "train"].std()   # chỉ học từ train
f["log_reg_price"] = ((lrp - mu) / sd).fillna(0)
f["reg_price_missing"] = lrp.isna().astype(float)
PR_GROUPS = {
    "promo_schedule_depth": ["promo_disc", "promo_nodisc", "depth", "depth_x_food", "price_ratio", "promo_progress",
                             "days_to_promo_end", "promo_duration", "ptype_5", "ptype_6", "ptype_10", "ptype_8",
                             "ptype_other"],
    "post_promo": ["days_since_promo", "never_promo", "post_promo_1_7", "post_promo_8_14"],
    "price_level": ["log_reg_price", "reg_price_missing"],
}
PR_COLS = [c for g in PR_GROUPS.values() for c in g]
NPR, NCAL = len(PR_COLS), len(CAL_COLS)

# Mảng đệm: [cặp, PADL + ngày + PADR]
W = PADL + ND + PADR
PR = np.zeros((P, W, NPR), np.float32)
Q = np.full((P, W), np.nan, np.float32)               # lượng bán gốc (NaN = không có dữ liệu)
pi, di = df.pair.values, df.d.values + PADL
PR[pi, di] = f[PR_COLS].values.astype(np.float32)
Q[pi, di] = df.quantity.values
OBS = (~np.isnan(Q)).astype(np.float32)
Y = np.where(OBS > 0, np.log1p(np.nan_to_num(Q)), 0).astype(np.float32)
CALP = np.zeros((W, NCAL), np.float32)
CALP[PADL:PADL + ND] = CAL
del f

# 2c. Tĩnh
st_df = df.drop_duplicates("pair").set_index("pair").sort_index()
STATIC_COLS = ["store_id", "group4", "dept_name", "class_name", "format"]
STATIC = np.stack([pd.factorize(st_df[c])[0] for c in STATIC_COLS], 1).astype(np.int64)
CARDS = [int(STATIC[:, i].max()) + 1 for i in range(len(STATIC_COLS))]
EMB_DIMS = [3, 3, 4, 8, 2]
say(f"Feature: {NPR} KM/giá + {NCAL} lịch/lễ + 2 lịch sử bán + {len(STATIC_COLS)} tĩnh; dựng xong {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ 3. Cửa sổ
def train_windows():
    """(cặp, origin) sao cho 7 ngày dự báo nằm trong train và có ≥ 1 ngày có dữ liệu."""
    o = OBS[:, PADL:PADL + ND]
    cs = np.concatenate([np.zeros((P, 1)), np.cumsum(o, 1)], 1)
    t = np.arange(0, I_TRAIN_END - H + 1)
    cnt = cs[:, t + H + 1] - cs[:, t + 1]                  # số ngày quan sát trong t+1..t+H
    p, k = np.nonzero(cnt > 0)
    return p, t[k]


def block_windows(start_idx, end_idx):
    """Khối tuần không chồng lấn: origin = start-1, start+6, ...; chỉ tính ngày ≤ end_idx."""
    ps, ts = [], []
    for t in range(start_idx - 1, end_idx, H):
        lo, hi = t + 1, min(t + H, end_idx)
        has = OBS[:, PADL + lo:PADL + hi + 1].sum(1) > 0
        ps.append(np.nonzero(has)[0]); ts.append(np.full(has.sum(), t))
    return np.concatenate(ps), np.concatenate(ts)


def make_batch(p, t, L, end_idx=None, perm=None):
    e = t[:, None] + np.arange(-L + 1, 1) + PADL
    d = t[:, None] + np.arange(1, H + 1) + PADL
    pp = p[:, None]
    pr_e, pr_d, cal_e, cal_d = PR[pp, e], PR[pp, d], CALP[e], CALP[d]
    y_e, o_e, st = Y[pp, e], OBS[pp, e], STATIC[p]
    if perm is not None:                     # hoán vị 1 nhóm feature giữa các mẫu (permutation importance)
        kind, cols, ix = perm
        if kind == "pr":
            pr_e, pr_d = pr_e.copy(), pr_d.copy(); pr_e[..., cols] = pr_e[ix][..., cols]; pr_d[..., cols] = pr_d[ix][..., cols]
        elif kind == "cal":
            # Feature lịch giống nhau cho mọi chuỗi trong cùng 1 khối (cùng origin, cùng thứ), nên hoán vị giữa
            # các mẫu không đổi gì. Thay vào đó lấy lịch của 1 ngày ngẫu nhiên khác (độ lệch không chia hết cho 7).
            off = np.random.default_rng(int(ix[0])).integers(1, ND, len(p))
            off = off + (off % 7 == 0)
            se = (e - PADL + off[:, None]) % ND + PADL
            sd = (d - PADL + off[:, None]) % ND + PADL
            cal_e, cal_d = cal_e.copy(), cal_d.copy()
            cal_e[..., cols] = CALP[se][..., cols]; cal_d[..., cols] = CALP[sd][..., cols]
        elif kind == "hist":
            y_e, o_e = y_e[ix], o_e[ix]
        elif kind == "static":
            st = st[ix]
    enc = np.concatenate([y_e[..., None], o_e[..., None], pr_e, cal_e], -1)
    dec = np.concatenate([pr_d, cal_d], -1)
    y_d, o_d = Y[pp, d], OBS[pp, d].copy()
    if end_idx is not None:
        o_d[(d - PADL) > end_idx] = 0
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV, non_blocking=True)
    return T(enc), T(dec), T(st), T(y_d), T(o_d)


# ------------------------------------------------------------------ 4. Mô hình
class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_enc, n_dec, hid=64, dropout=DROPOUT):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, k) for c, k in zip(CARDS, EMB_DIMS)])
        ne = sum(EMB_DIMS)
        self.enc = nn.LSTM(n_enc + ne, hid, batch_first=True)
        self.dec = nn.LSTM(n_dec + ne, hid, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hid + n_dec + ne, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, enc, dec, st):
        emb = torch.cat([m(st[:, i]) for i, m in enumerate(self.embs)], -1)
        level = (enc[..., 0] * enc[..., 1]).sum(1) / enc[..., 1].sum(1).clamp(min=1)   # mức nền log1p của encoder
        e_in = torch.cat([enc, emb[:, None].expand(-1, enc.size(1), -1)], -1)
        _, hc = self.enc(e_in)
        d_in = torch.cat([dec, emb[:, None].expand(-1, dec.size(1), -1)], -1)
        out, _ = self.dec(d_in, hc)
        z = self.head(torch.cat([self.drop(out), d_in], -1)).squeeze(-1)
        return level[:, None] + z                                                     # dự báo log1p(qty)


def loss_fn(pred, y, o, kind):
    if kind == "log_mse":
        return (((pred - y) ** 2) * o).sum() / o.sum().clamp(min=1)
    q_hat = torch.expm1(pred.clamp(-5, 9))
    return ((q_hat - torch.expm1(y)).abs() * o).sum() / o.sum().clamp(min=1)


@torch.no_grad()
def predict(model, p, t, L, end_idx, perm_spec=None):
    model.eval()
    rng = np.random.default_rng(SEED_PERM)
    preds, ys, obs = [], [], []
    for i in range(0, len(p), 4096):
        pb, tb = p[i:i + 4096], t[i:i + 4096]
        perm = None
        if perm_spec is not None:
            perm = (perm_spec[0], perm_spec[1], rng.permutation(len(pb)))
        enc, dec, st, y, o = make_batch(pb, tb, L, end_idx, perm)
        preds.append(model(enc, dec, st).cpu().numpy()); ys.append(y.cpu().numpy()); obs.append(o.cpu().numpy())
    pred, y, o = np.concatenate(preds), np.concatenate(ys), np.concatenate(obs)
    return np.clip(np.expm1(pred), 0, None), np.expm1(y), o


def wape(q_hat, q, o):
    m = o > 0
    return np.abs(q_hat[m] - q[m]).sum() / q[m].sum()


def train_variant(L, hid, lr, tw, vw, kind=LOSS):
    """Huấn luyện 1 cấu hình của lưới; dừng sớm theo WAPE trên hold-out valid, trả về trọng số epoch tốt nhất."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Seq2SeqLSTM(2 + NPR + NCAL, NPR + NCAL, hid=hid).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    name = f"L{L}_h{hid}_lr{lr:g}"
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        te = time.time()
        model.train()
        idx = rng.choice(len(tw[0]), min(WIN_PER_EPOCH, len(tw[0])), replace=False)
        tot = 0.0
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            enc, dec, st, y, o = make_batch(tw[0][b], tw[1][b], L)
            loss = loss_fn(model(enc, dec, st), y, o, kind)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        q_hat, q, o = predict(model, vw[0], vw[1], L, I_VALID_END)
        vw_ = wape(q_hat, q, o)
        sched.step(vw_)
        hist.append({"variant": name, "epoch": ep, "train_loss": tot / len(idx), "valid_wape": vw_,
                     "lr_now": opt.param_groups[0]["lr"], "sec": time.time() - te})
        say(f"  [{name}] epoch {ep:2d}  train_loss={tot/len(idx):.4f}  valid WAPE={vw_:.4f}  ({time.time()-te:.0f}s)")
        if vw_ < best - MIN_DELTA:
            best, bad = vw_, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, hist


def feature_importance(model, L, n_rep=3):
    """Tăng WAPE trên valid khi phá liên kết của 1 nhóm feature (trung bình n_rep lần).
    Nhóm theo chuỗi / tĩnh / lịch sử: hoán vị giữa các mẫu. Nhóm lịch: lấy lịch của 1 ngày ngẫu nhiên khác."""
    global SEED_PERM
    base_q, q_true, o_v = predict(model, vw[0], vw[1], L, I_VALID_END)
    base_w = wape(base_q, q_true, o_v)
    groups = [("history_sales", "hist", None)]
    groups += [(g, "pr", [PR_COLS.index(c) for c in cols]) for g, cols in PR_GROUPS.items()]
    groups += [(g, "cal", [CAL_COLS.index(c) for c in cols]) for g, cols in CAL_GROUPS.items()]
    groups += [("static_store_product", "static", None)]
    imp = []
    for name, kind_, cols in groups:
        ws = []
        for r in range(n_rep):
            SEED_PERM = SEED + r
            q_hat, _, _ = predict(model, vw[0], vw[1], L, I_VALID_END, (kind_, cols))
            ws.append(wape(q_hat, q_true, o_v))
        imp.append({"feature_group": name, "valid_wape_permuted": np.mean(ws), "delta_wape": np.mean(ws) - base_w,
                    "delta_sd": np.std(ws)})
    imp = pd.DataFrame(imp).sort_values("delta_wape", ascending=False)
    imp.to_csv(OUT / "feature_group_importance.csv", index=False)
    say(f"\n### Feature-group importance (valid, WAPE gốc {base_w:.4f}, {n_rep} lần)\n{imp.round(4).to_string(index=False)}")


# ------------------------------------------------------------------ 5. Huấn luyện + chọn biến thể trên valid
tw = train_windows()
vw = block_windows(I_TRAIN_END + 1, I_VALID_END)
tst = block_windows(I_VALID_END + 1, ND - 1)
say(f"Cửa sổ train: {len(tw[0]):,} | khối valid: {len(vw[0]):,} | khối test: {len(tst[0]):,}")

if "--importance-only" in sys.argv:          # dùng lại mô hình đã lưu, chỉ tính lại mục 8
    ck = torch.load(OUT / "lstm_best.pt", map_location=DEV)
    best_model = Seq2SeqLSTM(2 + NPR + NCAL, NPR + NCAL, hid=ck.get("hid", 64)).to(DEV)
    best_model.load_state_dict(ck["state"])
    feature_importance(best_model, ck["L"])
    sys.exit(0)

say(f"Manual Grid Search LSTM: {len(VARIANTS)} cấu hình (lookback {GRID_LOOKBACK} × hidden {GRID_HID} × lr {GRID_LR}), "
    f"loss={LOSS}, dropout={DROPOUT}; early stopping patience={PATIENCE}, tối đa {MAX_EPOCHS} epoch")
results, all_hist = [], []
best_model, best_cfg, best_w = None, None, np.inf
for L, hid, lr in VARIANTS:
    tv = time.time()
    m, w, h = train_variant(L, hid, lr, tw, vw)
    all_hist += h
    best_ep = int(np.argmin([x["valid_wape"] for x in h]) + 1)
    results.append({"variant": f"L{L}_h{hid}_lr{lr:g}", "loss": LOSS, "lookback": L, "hidden": hid, "lr": lr,
                    "dropout": DROPOUT, "n_params": sum(p.numel() for p in m.parameters()), "best_valid_wape": w,
                    "best_epoch": best_ep, "epochs_run": len(h), "early_stopped": len(h) < MAX_EPOCHS,
                    "sec": time.time() - tv})
    say(f"==> L={L} hidden={hid} lr={lr:g}: valid WAPE tốt nhất = {w:.4f} ở epoch {best_ep}/{len(h)}"
        f"{'' if len(h) < MAX_EPOCHS else ' (CHẠM MAX_EPOCHS, chưa dừng sớm)'}  ({time.time()-tv:.0f}s)")
    if w < best_w:
        best_model, best_cfg, best_w = m, (L, hid, lr), w
    pd.DataFrame(all_hist).to_csv(OUT / "training_log.csv", index=False)       # ghi dần để theo dõi
    pd.DataFrame(results).sort_values("best_valid_wape").to_csv(OUT / "variant_selection.csv", index=False)
L, best_hid, best_lr = best_cfg
kind = LOSS
torch.save({"state": {k: v.cpu() for k, v in best_model.state_dict().items()}, "kind": kind, "L": L, "hid": best_hid, "lr": best_lr,
            "dropout": DROPOUT, "pr_cols": PR_COLS, "cal_cols": CAL_COLS, "static_cols": STATIC_COLS,
            "cards": CARDS}, OUT / "lstm_best.pt")
say(f"Cấu hình được chọn: L={L} hidden={best_hid} lr={best_lr:g} (valid WAPE {best_w:.4f})  — {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ 6. Baseline
def to_rows(p, t, q_hat, end_idx):
    h = np.tile(np.arange(1, H + 1), (len(p), 1))
    d = t[:, None] + h
    o = np.zeros_like(h, dtype=bool)
    ok = d <= end_idx
    o[ok] = OBS[np.repeat(p[:, None], H, 1)[ok], d[ok] + PADL] > 0
    rows = pd.DataFrame({"pair": np.repeat(p, H)[o.ravel()], "d": d.ravel()[o.ravel()],
                         "origin": np.repeat(t, H)[o.ravel()], "h": h.ravel()[o.ravel()],
                         "lstm": q_hat.ravel()[o.ravel()]})
    return rows


def naive_baselines(rows):
    p, d, t = rows.pair.values, rows.d.values + PADL, rows.origin.values + PADL
    # Trung bình 7 ngày gần nhất tính đến origin (bỏ ngày không có dữ liệu)
    last7 = Q[p[:, None], t[:, None] + np.arange(-6, 1)]
    ma7 = np.nan_to_num(np.nanmean(np.where(np.isnan(last7), np.nan, last7), 1), nan=0.0)
    # Seasonal naive: cùng thứ tuần trước (d - 7 ≤ origin); thiếu thì dùng MA7
    sn = Q[p, d - 7]
    sn = np.where(np.isnan(sn), ma7, sn)
    rows["seasonal_naive"], rows["ma7"] = sn, ma7
    return rows


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

HGB_FEATS = ["lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_28", "dow", "day_of_month",
             "month", "is_weekend", "is_holiday", "holiday_status", "minor_holiday", "ny_break", "days_to_holiday",
             "days_after_holiday", "pheic", "store4_period", "depth", "promo_status", "promo_type", "day_in_promo",
             "days_to_promo_end", "promo_duration", "days_since_last_promo", "regular_price", "price_ratio"]
CAT_FEATS = ["store_id", "group4", "dept_name", "format"]


def hgb_baseline():
    """HGB Poisson học trên train, tune bằng Manual Grid Search + hold-out valid + early stopping.
    Mỗi cấu hình: thêm HGB_STEP cây mỗi bước (warm start), đo WAPE valid; dừng khi HGB_PATIENCE bước liên tiếp không
    giảm được ≥ MIN_DELTA. Không dùng early stopping có sẵn của sklearn vì nó tách valid ngẫu nhiên từ train
    (rò rỉ theo thời gian) và theo dõi loss Poisson chứ không phải WAPE.
    Giữ thêm cấu hình mặc định không tune (hgb_default, 400 cây) để thấy tác dụng của việc tune."""
    X = df[HGB_FEATS].copy()
    for c in CAT_FEATS:
        X[c] = pd.factorize(df[c])[0]
    cat_idx = [X.columns.get_loc(c) for c in CAT_FEATS]
    X = X.values.astype(np.float64)
    tr, va = (df.split == "train").values, (df.split == "valid").values
    y = df.quantity.values.astype(np.float64)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    mk = lambda lr, nl, msl, it, ws=False: HistGradientBoostingRegressor(
        loss="poisson", max_iter=it, learning_rate=lr, max_leaf_nodes=nl, min_samples_leaf=msl,
        early_stopping=False, warm_start=ws, categorical_features=cat_idx, random_state=SEED)

    default = mk(0.05, 63, 50, 60 if SMOKE else 400).fit(Xtr, ytr)

    grid = [(lr, nl, msl) for lr in HGB_GRID_LR for nl in HGB_GRID_LEAVES for msl in HGB_GRID_MINLEAF]
    say(f"Manual Grid Search HGB: {len(grid)} cấu hình (lr {HGB_GRID_LR} × leaves {HGB_GRID_LEAVES} × "
        f"min_leaf {HGB_GRID_MINLEAF}); early stopping mỗi {HGB_STEP} cây, patience {HGB_PATIENCE}, tối đa {HGB_MAX_IT}")
    curve, summ, best = [], [], (np.inf, None)
    for lr, nl, msl in grid:
        te_ = time.time()
        m = mk(lr, nl, msl, 0, ws=True)
        cfg_best, cfg_it, bad, it = np.inf, 0, 0, 0
        while it < HGB_MAX_IT:
            it += HGB_STEP
            m.set_params(max_iter=it)
            m.fit(Xtr, ytr)
            w = np.abs(m.predict(Xva) - yva).sum() / yva.sum()
            curve.append({"learning_rate": lr, "max_leaf_nodes": nl, "min_samples_leaf": msl, "n_trees": it,
                          "valid_wape": w})
            if w < cfg_best - MIN_DELTA:
                cfg_best, cfg_it, bad = w, it, 0
            else:
                bad += 1
                if bad >= HGB_PATIENCE:
                    break
        stopped = bad >= HGB_PATIENCE
        summ.append({"learning_rate": lr, "max_leaf_nodes": nl, "min_samples_leaf": msl, "best_n_trees": cfg_it,
                     "trees_trained": it, "best_valid_wape": cfg_best, "early_stopped": stopped,
                     "sec": time.time() - te_})
        say(f"  [HGB lr={lr} leaves={nl} min_leaf={msl}] valid WAPE tốt nhất = {cfg_best:.4f} ở {cfg_it} cây "
            f"(đã train {it}{'' if stopped else ', CHẠM HGB_MAX_IT'})  ({time.time()-te_:.0f}s)")
        if cfg_best < best[0]:
            best = (cfg_best, (lr, nl, msl, cfg_it))
        pd.DataFrame(curve).to_csv(OUT / "hgb_tuning.csv", index=False)
        pd.DataFrame(summ).sort_values("best_valid_wape").to_csv(OUT / "hgb_grid_summary.csv", index=False)
    lr, nl, msl, it = best[1]
    say(f"==> HGB được chọn: lr={lr}, leaves={nl}, min_leaf={msl}, {it} cây (valid WAPE {best[0]:.4f})")
    tuned = mk(lr, nl, msl, it).fit(Xtr, ytr)          # học lại chỉ trên train, dừng ở số cây tốt nhất
    w_chk = np.abs(tuned.predict(Xva) - yva).sum() / yva.sum()
    say(f"    kiểm tra: mô hình học lại cho valid WAPE {w_chk:.4f}")
    cfg = {"learning_rate": lr, "max_leaf_nodes": nl, "min_samples_leaf": msl, "n_trees": it,
           "valid_wape": float(best[0]), "grid_size": len(grid), "step": HGB_STEP, "patience": HGB_PATIENCE}
    return pd.DataFrame({"pair": df.pair.values, "d": df.d.values, "hgb": tuned.predict(X),
                         "hgb_default": default.predict(X)}), cfg


te = time.time()
hgb_pred, hgb_cfg = hgb_baseline()
say(f"HistGradientBoosting (Poisson) tune + train xong {time.time()-te:.0f}s")

# ------------------------------------------------------------------ 7. Đánh giá
meta = df[["pair", "d", "date", "quantity", "promo_status", "category", "group4", "store_id", "item_id", "phase"]]


def evaluate(p, t, end_idx, model, L):
    q_hat, _, _ = predict(model, p, t, L, end_idx)
    rows = naive_baselines(to_rows(p, t, q_hat, end_idx))
    rows = rows.merge(hgb_pred, on=["pair", "d"], how="left").merge(meta, on=["pair", "d"], how="left")
    rows["promo_day"] = np.select([rows.promo_status == 1, rows.promo_status == 0], ["promo (giảm giá)", "không KM"],
                                  "KM không giảm giá")
    return rows


MODELS = ["lstm", "hgb", "hgb_default", "seasonal_naive", "ma7"]


def metric_table(rows, by=None):
    def m(g):
        y = g.quantity.values
        out = {"n": len(g), "sum_qty": y.sum()}
        for k in MODELS:
            e = g[k].values - y
            out[f"{k}_WAPE"] = np.abs(e).sum() / y.sum()
            out[f"{k}_MAE"] = np.abs(e).mean()
            out[f"{k}_RMSE"] = np.sqrt((e ** 2).mean())
            out[f"{k}_bias"] = e.sum() / y.sum()
        return pd.Series(out)
    if by is None:
        return m(rows).to_frame("all").T
    return rows.groupby(by).apply(m, include_groups=False)


trw = block_windows(1, I_TRAIN_END)                 # chỉ để chẩn đoán overfitting (mô hình đã thấy dữ liệu này)
train_rows = evaluate(trw[0], trw[1], I_TRAIN_END, best_model, L)
valid_rows = evaluate(vw[0], vw[1], I_VALID_END, best_model, L)
test_rows = evaluate(tst[0], tst[1], ND - 1, best_model, L)
test_rows.to_parquet(OUT / "test_predictions.parquet", index=False)

tables = {
    "train_overall": metric_table(train_rows),
    "valid_overall": metric_table(valid_rows),
    "test_overall": metric_table(test_rows),
    "test_by_promo": metric_table(test_rows, "promo_day"),
    "test_by_category": metric_table(test_rows, "category"),
    "test_by_phase": metric_table(test_rows, "phase"),             # pre_1_3 / during / post_1_3 / post_4_7 / other
    "test_by_phase_category": metric_table(test_rows, ["category", "phase"]),
    "test_by_group4": metric_table(test_rows, "group4"),
    "test_by_horizon": metric_table(test_rows, "h"),
    "test_by_store": metric_table(test_rows, "store_id"),
}
with open(OUT / "metrics.txt", "w", encoding="utf-8") as fh:
    for k, v in tables.items():
        v.to_csv(OUT / f"metrics_{k}.csv")
        fh.write(f"\n### {k}\n{v.round(4).to_string()}\n")
for k in ["train_overall", "valid_overall", "test_overall", "test_by_promo", "test_by_category", "test_by_phase",
          "test_by_group4", "test_by_horizon"]:
    cols = [c for c in tables[k].columns if c.endswith("WAPE")] + ["n"]
    say(f"\n### {k}\n{tables[k][cols].round(4).to_string()}")

# ------------------------------------------------------------------ 8. Độ quan trọng nhóm feature (trên valid)
feature_importance(best_model, L)

# ------------------------------------------------------------------ 9. Mô phỏng depth trên các ngày KM giảm giá của test
promo_mask_cols = [PR_COLS.index("depth"), PR_COLS.index("depth_x_food"), PR_COLS.index("price_ratio")]
is_food_pair = (st_df.category.values == "food")
p_t, t_t = tst
d_idx = t_t[:, None] + np.arange(1, H + 1) + PADL
promo_d = (PR[p_t[:, None], d_idx, PR_COLS.index("promo_disc")] > 0) & (OBS[p_t[:, None], d_idx] > 0) \
          & ((d_idx - PADL) <= ND - 1)
keep = promo_d.any(1)
p_s, t_s, mask_s, d_s = p_t[keep], t_t[keep], promo_d[keep], d_idx[keep]
reg_price = np.full((P, W), np.nan, np.float32)
reg_price[pi, di] = df.regular_price.values
rp = reg_price[p_s[:, None], d_s]
mask_s = mask_s & ~np.isnan(rp)
saved = PR[p_s[:, None], d_s][..., promo_mask_cols].copy()
sim = []
for dep in [None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    if dep is not None:
        vals = PR[p_s[:, None], d_s]
        vals[..., promo_mask_cols[0]] = np.where(mask_s, dep, vals[..., promo_mask_cols[0]])
        vals[..., promo_mask_cols[1]] = np.where(mask_s, dep * is_food_pair[p_s][:, None], vals[..., promo_mask_cols[1]])
        vals[..., promo_mask_cols[2]] = np.where(mask_s, 1 - dep, vals[..., promo_mask_cols[2]])
        PR[p_s[:, None], d_s] = vals
    q_hat, _, _ = predict(best_model, p_s, t_s, L, ND - 1)
    actual_depth = saved[..., 0]
    price = rp * (1 - (actual_depth if dep is None else dep))
    for cat_name, sel in [("food", is_food_pair[p_s]), ("household", ~is_food_pair[p_s])]:
        m = mask_s & sel[:, None]
        sim.append({"depth": "thực tế" if dep is None else dep, "category": cat_name, "n_days": int(m.sum()),
                    "pred_qty": float(q_hat[m].sum()), "pred_revenue": float((q_hat * price)[m].sum()),
                    "actual_qty": float(Q[p_s[:, None], d_s][m].sum())})
    tmp = PR[p_s[:, None], d_s]; tmp[..., promo_mask_cols] = saved; PR[p_s[:, None], d_s] = tmp   # khôi phục
sim = pd.DataFrame(sim)
for c in ["pred_qty", "pred_revenue"]:
    ref = sim[sim.depth == 0.20].set_index("category")[c]
    sim[c + "_idx_vs_20pct"] = sim.apply(lambda r: 100 * r[c] / ref[r.category], axis=1)
sim.to_csv(OUT / "depth_simulation.csv", index=False)
say(f"\n### Mô phỏng depth (ngày KM giảm giá trong test)\n{sim.round(2).to_string(index=False)}")

summary = {"tuning": "Manual Grid Search + Hold-out Validation + Early Stopping",
           "lstm_grid": {"lookback": GRID_LOOKBACK, "hidden": GRID_HID, "lr": GRID_LR, "loss": LOSS,
                         "dropout": DROPOUT, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA},
           "chosen_variant": f"L{L}_h{best_hid}_lr{best_lr:g}", "chosen_lookback": L, "chosen_hidden": best_hid,
           "chosen_lr": best_lr, "valid_wape": float(best_w), "runtime_sec": time.time() - t0,
           "n_train_windows": int(len(tw[0])), "n_test_rows": int(len(test_rows)), "hgb_tuned": hgb_cfg,
           "train_overall": tables["train_overall"].iloc[0].to_dict(),
           "valid_overall": tables["valid_overall"].iloc[0].to_dict(),
           "test_overall": tables["test_overall"].iloc[0].to_dict()}
json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
say(f"\nXong trong {time.time()-t0:.0f}s. Kết quả ở {OUT}")
