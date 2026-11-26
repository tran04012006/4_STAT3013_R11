"""Kiểm định giả thuyết cho kết quả dự báo LSTM trên tập test (07–09/2024).

Đầu vào : output/lstm/test_predictions.parquet (do src/train_lstm.py tạo)
Đầu ra  : output/lstm/hypothesis_tests.csv + in kết quả ra màn hình

Các kiểm định
- H1/H3  Diebold–Mariano (hiệu chỉnh Harvey–Leybourne–Newbold), một phía: LSTM có tổng sai số tuyệt đối
         nhỏ hơn baseline không? Chuỗi chênh lệch d_t tính theo ngày (gộp mọi chuỗi), nên trung bình d_t
         tỷ lệ thuận với chênh lệch WAPE. Phương sai HAC (Bartlett, h − 1 = 6 lag). Kèm KTC 95% của ΔWAPE
         bằng moving block bootstrap theo ngày (khối 7 ngày, 5.000 lần).
- H2     Wilcoxon signed-rank, một phía, trên MAE của từng cặp mã hàng × cửa hàng; cỡ hiệu ứng rank-biserial.
- H4     Không chệch: t-test với sai số chuẩn HAC trên tổng sai số (dự báo − thực tế) theo ngày.
- H5     Ljung–Box trên sai số tương đối theo ngày (lag 7, 14).
- Phụ    Skewness, kurtosis, Jarque–Bera của phần dư cấp dòng.

Chạy: python src/lstm_hypothesis_tests.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "lstm"
H, SEED, B = 7, 42, 5000

t = pd.read_parquet(OUT / "test_predictions.parquet")
rows = []


def hac_var(x, lags):
    """Phương sai dài hạn Newey–West (trọng số Bartlett) của trung bình chuỗi x."""
    x = x - x.mean()
    n = len(x)
    v = (x @ x) / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return v


def dm_test(sub, a, b, label):
    """H0: E[d_t] = 0; H1: E[d_t] < 0 (mô hình a có tổng |sai số| nhỏ hơn b)."""
    day = sub.assign(ea=(sub[a] - sub.quantity).abs(), eb=(sub[b] - sub.quantity).abs()) \
             .groupby("date")[["ea", "eb", "quantity"]].sum()
    d = (day.ea - day.eb).values
    n = len(d)
    dm = d.mean() / np.sqrt(hac_var(d, H - 1) / n)
    hln = dm * np.sqrt((n + 1 - 2 * H + H * (H - 1) / n) / n)
    p = stats.t.cdf(hln, n - 1)
    d_wape = (day.ea.sum() - day.eb.sum()) / day.quantity.sum()
    rng = np.random.default_rng(SEED)
    ea, eb, q = day.ea.values, day.eb.values, day.quantity.values
    nb, boot = int(np.ceil(n / H)), np.empty(B)
    for r in range(B):
        idx = (rng.integers(0, n - H + 1, nb)[:, None] + np.arange(H)).ravel()[:n]
        boot[r] = (ea[idx].sum() - eb[idx].sum()) / q[idx].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append({"test": "Diebold-Mariano (HLN)", "subset": label, "comparison": f"{a} vs {b}", "n": n,
                 "estimate": d_wape, "ci_low": lo, "ci_high": hi, "statistic": hln, "p_value": p})


def wilcoxon_pairs(a, b):
    """H0: trung vị chênh lệch MAE theo cặp = 0; H1: MAE của a nhỏ hơn."""
    g = t.assign(ea=(t[a] - t.quantity).abs(), eb=(t[b] - t.quantity).abs()).groupby("pair")[["ea", "eb"]].mean()
    diff = g.ea - g.eb
    w = stats.wilcoxon(g.ea, g.eb, alternative="less")
    nz = diff[diff != 0]
    rk = stats.rankdata(nz.abs())
    rb = (rk[nz < 0].sum() - rk[nz > 0].sum()) / rk.sum()          # rank-biserial, > 0 là a tốt hơn
    rows.append({"test": "Wilcoxon signed-rank", "subset": "per pair MAE", "comparison": f"{a} vs {b}",
                 "n": len(g), "estimate": diff.median(), "statistic": w.statistic, "p_value": w.pvalue,
                 "share_a_better": (diff < 0).mean(), "rank_biserial": rb})


def bias_test(sub, label):
    """H0: E[Σ_i (q̂ − q) trong 1 ngày] = 0 (dự báo không chệch)."""
    day = sub.assign(e=sub.lstm - sub.quantity).groupby("date")[["e", "quantity"]].sum()
    e = day.e.values
    n = len(e)
    tt = e.mean() / np.sqrt(hac_var(e, H - 1) / n)
    rows.append({"test": "Bias t-test (HAC)", "subset": label, "comparison": "lstm", "n": n,
                 "estimate": e.sum() / day.quantity.sum(), "mean_daily_error": e.mean(), "statistic": tt,
                 "p_value": 2 * stats.t.sf(abs(tt), n - 1)})
    return day


# H1: toàn bộ test
for base in ["ma7", "seasonal_naive"]:
    dm_test(t, "lstm", base, "all")
# H3: theo nhóm hàng và giai đoạn KM
for c in ["food", "household"]:
    dm_test(t[t.category == c], "lstm", "ma7", c)
for ph in ["pre_1_3", "during", "post_1_3", "post_4_7"]:
    dm_test(t[t.phase == ph], "lstm", "ma7", ph)
# H2
for base in ["ma7", "seasonal_naive"]:
    wilcoxon_pairs("lstm", base)
# H4
day_all = bias_test(t, "all")
for c in ["food", "household"]:
    bias_test(t[t.category == c], c)
# H5
rel = (day_all.e / day_all.quantity).values
lb = acorr_ljungbox(rel - rel.mean(), lags=[7, 14], return_df=True)
for lag, r in lb.iterrows():
    rows.append({"test": "Ljung-Box", "subset": f"daily relative error, lag {lag}", "comparison": "lstm",
                 "n": len(rel), "statistic": r.lb_stat, "p_value": r.lb_pvalue})
# Phân phối phần dư cấp dòng
e = (t.lstm - t.quantity).values
jb = stats.jarque_bera(e)
rows.append({"test": "Jarque-Bera", "subset": "row residuals", "comparison": "lstm", "n": len(e),
             "statistic": jb.statistic, "p_value": jb.pvalue, "skew": stats.skew(e),
             "excess_kurtosis": stats.kurtosis(e)})

res = pd.DataFrame(rows)
# Hiệu chỉnh Benjamini–Hochberg cho họ kiểm định DM (H1 + H3)
m = res.test.str.startswith("Diebold")
p = res.loc[m, "p_value"].values
order = np.argsort(p)
adj = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
res.loc[res.index[m][order], "p_bh"] = np.minimum(adj, 1)
res.to_csv(OUT / "hypothesis_tests.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(res.round(4).to_string(index=False))
