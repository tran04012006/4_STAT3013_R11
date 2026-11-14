"""Lọc dữ liệu theo output/quy_tac_loc_du_lieu.md và tạo bộ dữ liệu để train model + chạy thống kê.

Chạy: python src/prepare_data.py   (khoảng vài phút; dataset/ gốc không bị sửa)
Ra (thư mục data_model/):
  daily_panel.parquet  1 dòng = mã hàng × cửa hàng × ngày  -> ML/DL, elasticity cấp ngày
  episodes.csv         1 dòng = 1 đợt KM thật hoặc đợt giả  -> ANOVA, t-test, chi-square, logistic, elasticity
  group_day.csv        1 dòng = nhóm × cửa hàng × ngày × trạng thái KM -> tái lập 6 nhãn của paper
  output/bao_cao_loc_du_lieu.md: số dòng sau từng bước, bảng kiểu Bảng 2 của paper, từ điển cột
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT, REPORT = ROOT / "dataset", ROOT / "data_model", ROOT / "output" / "bao_cao_loc_du_lieu.md"

D0, D1 = pd.Timestamp("2022-08-28"), pd.Timestamp("2024-09-26")
DATES = pd.date_range(D0, D1)
ND = len(DATES)
DAY = np.arange(ND)

GROUPS = {  # mục 6 của file quy tắc
    "F1": ("Rau quả tươi", "food", ["ФРУКТЫ", "ОВОЩИ"]),
    "F2": ("Thịt tươi và thịt chế biến ướp lạnh", "food",
           ["СВЕЖЕЕ МЯСО", "ПТИЦА", "КОЛБАСЫ ВАРЕНЫЕ", "СОСИСКИ.САРДЕЛЬКИ.ШПИКАЧКИ"]),
    "H1": ("Chăm sóc cá nhân và giấy vệ sinh", "household", ["ПЕРСОНАЛЬНЫЙ УХОД", "БУМАЖНО-ВАТНАЯ ПРОДУКЦИЯ"]),
    "H2": ("Giặt giũ và tẩy rửa", "household",
           ["СРЕДСТВА ДЛЯ СТИРКИ", "СОПУТСТВУЮЩИЕ ТОВАРЫ ДЛЯ СТИРКИ", "ЧИСТЯЩИЕ,МОЮЩИЕ СРЕДСТВА"]),
}
BAD_STORE_DAYS = [(3, "2022-10-16"), (3, "2022-10-22"), (4, "2023-12-13")]  # B4: đóng cửa / hoạt động không đủ giờ
MIN_DENSITY = 0.2        # C3: bán ≥ 20% số ngày giữa lần bán đầu và cuối
PRE, POST = 3, 7         # D1: cửa sổ trước 1–3 ngày, sau 1–3 và 4–7 ngày
C = 1 / 3                # hằng số làm trơn tỷ lệ ≈ 1 đơn vị trong cửa sổ 3 ngày
PLACEBO_LEN, PLACEBO_PER_PAIR = 14, 10
TRAIN_END, VALID_END = "2024-03-31", "2024-06-30"
PHEIC_END = "2023-05-05"
MAJOR = ["2022-03-08", "2023-01-01", "2023-02-23", "2023-03-08", "2024-01-01", "2024-02-23", "2024-03-08", "2025-01-01"]
MINOR = ["2022-11-04", "2023-05-01", "2023-05-09", "2023-06-12", "2023-11-04", "2024-05-01", "2024-05-09", "2024-06-12"]

log = []


def step(df, mask, code, table, rule):
    log.append((code, table, rule, -int(mask.sum()), int((~mask).sum())))
    return df[~mask]


def didx(s):
    return (pd.Timestamp(s) - D0).days


def holiday_calendar():
    """Mục 3: −1 trước lễ, 1 sau lễ, 0 ngày thường; chính ngày lễ = NaN + is_holiday."""
    status = np.zeros(ND)
    is_hol = np.zeros(ND, bool)

    def mark(day, v):
        i = didx(day)
        if 0 <= i < ND:
            status[i] = v
    for y in (2022, 2023):                       # Tết Dương lịch: 27–31/12 trước, 02–08/01 sau
        for k in range(27, 32):
            mark(f"{y}-12-{k}", -1)
        for k in range(2, 9):
            mark(f"{y + 1}-01-0{k}", 1)
    for y in (2023, 2024):                       # 23/02, 08/03: 1–3 ngày trước / sau
        for md in ("02-23", "03-08"):
            h = pd.Timestamp(f"{y}-{md}")
            for k in (1, 2, 3):
                mark(h - pd.Timedelta(days=k), -1)
                mark(h + pd.Timedelta(days=k), 1)
    for h in MAJOR:
        if 0 <= didx(h) < ND:
            is_hol[didx(h)] = True
    status[is_hol] = np.nan
    minor = np.isin(DATES, pd.to_datetime(MINOR))
    hd = np.array([didx(h) for h in MAJOR])
    to_next = np.array([hd[hd >= d].min() - d for d in DAY])
    after_last = np.array([d - hd[hd <= d].max() for d in DAY])
    ny_break = (DATES.month == 1) & (DATES.day <= 8)
    return status, is_hol, minor, to_next, after_last, ny_break


def cum(a):
    """Mảng cộng dồn theo trục ngày, thêm cột 0 ở đầu: tổng [a, b) = c[:, b] - c[:, a]."""
    return np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, 1, dtype=np.float64)], 1)


def wsum(c, p, a, b):
    return c[p, b] - c[p, a]


def window_outcomes(df, CQ, CR, CPOS, hol_c, holiday_any_c):
    """Lượng bán / doanh thu trung bình mỗi ngày ở các cửa sổ quanh một đợt [a, b)."""
    p, a, b = df.pair.values, df.a.values, df.b.values
    dur = b - a
    for name, c in [("qty", CQ), ("rev", CR)]:
        df[f"pre_{name}"] = wsum(c, p, a - PRE, a) / PRE
        df[f"during_{name}"] = wsum(c, p, a, b) / dur
        df[f"post13_{name}"] = wsum(c, p, b, b + 3) / 3
        df[f"post47_{name}"] = wsum(c, p, b + 3, b + POST) / (POST - 3)
    df["pre_days"] = wsum(CPOS, p, a - PRE, a)
    df["lift"] = df.during_qty / df.pre_qty.where(df.pre_qty > 0)
    df["log_lift"] = np.log((df.during_qty + C) / (df.pre_qty + C))
    df["log_post13"] = np.log((df.post13_qty + C) / (df.pre_qty + C))
    df["log_post47"] = np.log((df.post47_qty + C) / (df.pre_qty + C))
    df["dip_1_3"] = (df.log_post13 < np.log(0.9)).astype(int)
    df["dip_4_7"] = (df.log_post47 < np.log(0.9)).astype(int)
    # trạng thái lễ của đợt (ưu tiên: chứa ngày lễ > chạm trước lễ > chạm sau lễ)
    h_in = lambda key: (hol_c[key][b] - hol_c[key][a]) > 0
    df["hol_status"] = np.select([h_in("hol"), h_in("pre"), h_in("post")],
                                 ["holiday", "pre_holiday", "post_holiday"], "normal")
    df["window_has_holiday"] = ((holiday_any_c[a] - holiday_any_c[a - PRE]) + (holiday_any_c[b + POST] - holiday_any_c[b]) > 0).astype(int)
    df["start"], df["end"] = DATES[a], DATES[b - 1]
    df["duration"] = dur
    df["start_month"], df["start_dow"], df["year"] = df.start.dt.month, df.start.dt.dayofweek, df.start.dt.year
    df["pheic"] = (df.start <= PHEIC_END).astype(int)
    df["store4_period"] = (df.start >= "2023-12-14").astype(int)
    return df


def main():
    OUT.mkdir(exist_ok=True)
    hstatus, is_hol, minor, to_next, after_last, ny_break = holiday_calendar()

    # ---------- Catalog, stores: phạm vi 4 nhóm (B1–B2)
    cat = pd.read_csv(SRC / "catalog.csv", dtype={"item_id": str})
    cat = cat.loc[:, ~cat.columns.str.startswith("Unnamed")].drop(columns="fatness")
    d2g = {d: g for g, (_, _, ds) in GROUPS.items() for d in ds}
    log.append(("—", "catalog", "Dòng gốc", 0, len(cat)))
    cat = step(cat, ~cat.dept_name.isin(d2g), "B2", "catalog", "Ngoài 11 ngành hàng của 4 nhóm")
    cat["group4"] = cat.dept_name.map(d2g)
    cat["group_name"] = cat.group4.map(lambda g: GROUPS[g][0])
    cat["category"] = cat.group4.map(lambda g: GROUPS[g][1])
    for col in ["item_type", "weight_volume", "weight_netto"]:
        cat[f"{col}_missing"] = cat[col].isna().astype(int)
    stores = pd.read_csv(SRC / "stores.csv", usecols=["store_id", "division", "format", "city", "area"])

    # ---------- Sales (A2, A3, B1–B2, B4)
    s = pd.read_csv(SRC / "sales.csv", usecols=["date", "item_id", "quantity", "price_base", "sum_total", "store_id"],
                    parse_dates=["date"], dtype={"item_id": str})
    log.append(("—", "sales", "Dòng gốc", 0, len(s)))
    s = step(s, (s.quantity <= 0) | (s.sum_total <= 0) | (s.price_base <= 0), "A2", "sales",
             "quantity / sum_total / price_base ≤ 0 (trả hàng, hủy, dòng rỗng)")
    s = step(s, (s.sum_total - s.quantity * s.price_base).abs() > 1, "A3", "sales", "|sum_total − quantity × price_base| > 1")
    s = step(s, ~s.item_id.isin(cat.item_id), "B1–B2", "sales", "Mã hàng ngoài catalog hoặc ngoài 4 nhóm")
    bad = pd.DataFrame(BAD_STORE_DAYS, columns=["store_id", "date"]).assign(date=lambda d: pd.to_datetime(d.date), bad=1)
    s = step(s, s.merge(bad, on=["store_id", "date"], how="left").bad.notna().values, "B4", "sales",
             "Ngày cửa hàng hoạt động bất thường")
    s = s.groupby(["item_id", "store_id", "date"], as_index=False).agg(quantity=("quantity", "sum"), revenue=("sum_total", "sum"))

    # ---------- C3: chỉ giữ cặp mã hàng – cửa hàng bán đều
    pr = s.groupby(["item_id", "store_id"]).agg(first=("date", "min"), last=("date", "max"), days=("date", "nunique")).reset_index()
    pr["density"] = pr.days / ((pr["last"] - pr["first"]).dt.days + 1)
    keep = pr.density >= MIN_DENSITY
    kept_rows = s.merge(pr.loc[keep, ["item_id", "store_id"]], on=["item_id", "store_id"]).shape[0]
    n = lambda v: f"{int(v):,}".replace(",", ".")
    log.append(("C3", "sales", f"Cặp mã hàng – cửa hàng có mật độ bán < {MIN_DENSITY} "
                f"({n((~keep).sum())} cặp bị loại, còn {n(keep.sum())} cặp)", kept_rows - len(s), kept_rows))
    pr = pr[keep].reset_index(drop=True)
    pr["pair"] = np.arange(len(pr))
    P = len(pr)
    s = s.merge(pr[["item_id", "store_id", "pair"]], on=["item_id", "store_id"])

    # Mảng dày: cặp × ngày. Trong khoảng bán đầu–cuối, ngày không có dòng = 0 (C3); ngoài khoảng / ngày lỗi = NaN
    f, l = (pr["first"] - D0).dt.days.values, (pr["last"] - D0).dt.days.values
    valid = (DAY[None, :] >= f[:, None]) & (DAY[None, :] <= l[:, None])
    for st, d in BAD_STORE_DAYS:
        valid[(pr.store_id.values == st), didx(d)] = False
    Q = np.full((P, ND), np.nan, np.float32)
    R = np.full((P, ND), np.nan, np.float32)
    Q[valid], R[valid] = 0, 0
    sd = (s.date - D0).dt.days.values
    Q[s.pair.values, sd], R[s.pair.values, sd] = s.quantity.values, s.revenue.values

    # ---------- Discounts (A4, A5, B)
    disc = pd.read_csv(SRC / "discounts_history.csv",
                       usecols=["date", "item_id", "sale_price_before_promo", "sale_price_time_promo",
                                "promo_type_code", "doc_id", "store_id"], parse_dates=["date"], dtype={"item_id": str})
    log.append(("—", "discounts_history", "Dòng gốc", 0, len(disc)))
    disc = step(disc, disc.date > D1, "A4", "discounts_history", "Ngày sau 26/09/2024")
    disc = step(disc, (disc.sale_price_before_promo <= 0) | (disc.sale_price_time_promo > disc.sale_price_before_promo),
                "A5", "discounts_history", "Giá trước KM ≤ 0 hoặc giá KM > giá trước KM")
    disc = step(disc, ~disc.item_id.isin(cat.item_id), "B1–B2", "discounts_history", "Mã hàng ngoài 4 nhóm")
    n_before = len(disc)
    disc = disc.merge(pr[["item_id", "store_id", "pair"]], on=["item_id", "store_id"])
    log.append(("C3", "discounts_history", "Thuộc cặp mã hàng – cửa hàng không đạt C3", len(disc) - n_before, len(disc)))
    disc["depth"] = 1 - disc.sale_price_time_promo / disc.sale_price_before_promo
    disc["d"] = (disc.date - D0).dt.days

    day_promo = disc.sort_values("depth", ascending=False).drop_duplicates(["pair", "d"])  # nhiều chứng từ cùng ngày: lấy mức giảm sâu nhất
    ANY = np.zeros((P, ND), np.int8)
    DEPTH, BEFORE, PPRICE, PTYPE = (np.full((P, ND), np.nan, np.float32) for _ in range(4))
    ip, idd = day_promo.pair.values, day_promo.d.values
    ANY[ip, idd] = 1
    DEPTH[ip, idd], BEFORE[ip, idd] = day_promo.depth.values, day_promo.sale_price_before_promo.values
    PPRICE[ip, idd], PTYPE[ip, idd] = day_promo.sale_price_time_promo.values, day_promo.promo_type_code.values
    PROMO = np.where(ANY == 0, 0.0, np.where(DEPTH > 0, 1.0, np.nan))  # C1: KM không giảm giá -> NaN

    # Đợt KM = gộp các doc chồng lấn / liền kề của cùng cặp
    ep = disc.groupby(["pair", "doc_id"]).d.agg(["min", "max"]).reset_index().sort_values(["pair", "min"])
    ep["e"] = (ep["min"] > ep.groupby("pair")["max"].shift() + 1).groupby(ep.pair).cumsum()
    disc = disc.merge(ep[["pair", "doc_id", "e"]], on=["pair", "doc_id"])
    mode = lambda x: x.mode().iloc[0] if x.notna().any() else np.nan
    E = disc.groupby(["pair", "e"]).agg(
        a=("d", "min"), last_d=("d", "max"), n_docs=("doc_id", "nunique"), depth=("depth", "median"),
        price_before=("sale_price_before_promo", "median"), price_promo=("sale_price_time_promo", "median"),
        promo_type=("promo_type_code", mode)).reset_index()
    E["b"] = E.last_d + 1

    # Đặc trưng KM theo ngày (biết trước được vì KM do cửa hàng lên lịch)
    DAY_IN, TO_END, DUR = (np.full((P, ND), np.nan, np.float32) for _ in range(3))
    for p, a, b in zip(E.pair.values, E.a.values, E.b.values):
        n = b - a
        DAY_IN[p, a:b], TO_END[p, a:b], DUR[p, a:b] = np.arange(1, n + 1), np.arange(n - 1, -1, -1), n
    PHASE = np.zeros((P, ND), np.int8)  # 0 other, 1 pre_1_3, 2 post_4_7, 3 post_1_3, 4 during (số lớn ghi đè số nhỏ)
    Ereal = E[E.depth > 0]
    for code, lo, hi in [(1, lambda a, b: a - PRE, lambda a, b: a), (2, lambda a, b: b + 3, lambda a, b: b + POST),
                         (3, lambda a, b: b, lambda a, b: b + 3), (4, lambda a, b: a, lambda a, b: b)]:
        for p, a, b in zip(Ereal.pair.values, Ereal.a.values, Ereal.b.values):
            x, y = max(lo(a, b), 0), min(hi(a, b), ND)
            PHASE[p, x:y] = np.maximum(PHASE[p, x:y], code)
    lp = np.maximum.accumulate(np.where(ANY == 1, DAY[None, :], -10**6), axis=1)
    lp_prev = np.concatenate([np.full((P, 1), -10**6), lp[:, :-1]], 1)
    SINCE = np.where(lp_prev < 0, np.nan, DAY[None, :] - lp_prev)

    # Giá thường: giá trước KM (ngày có KM) hoặc giá gần nhất trong price_history (A6, A7)
    ph = pd.read_csv(SRC / "price_history.csv", usecols=["date", "item_id", "price", "code", "store_id"],
                     parse_dates=["date"], dtype={"item_id": str})
    log.append(("—", "price_history", "Dòng gốc", 0, len(ph)))
    ph = step(ph, ph.duplicated(), "A6", "price_history", "Trùng hoàn toàn")
    ph = step(ph, ph.price <= 0, "A7", "price_history", "price ≤ 0 (gồm code 9999999999)")
    ph = ph.merge(pr[["item_id", "store_id", "pair"]], on=["item_id", "store_id"])
    ph = ph[ph.date <= D1].groupby(["pair", "date"]).price.median().reset_index()
    PH = np.full((P, ND), np.nan, np.float32)
    PH[ph.pair.values, (ph.date - D0).dt.days.values] = ph.price.values
    PH = pd.DataFrame(PH.T).ffill().values.T
    REG = np.where(np.isnan(BEFORE), PH, BEFORE)
    PRICE = np.where(PROMO == 1, PPRICE, REG)

    # Lịch sử bán: chỉ dùng quá khứ, dịch ≥ 7 ngày (đủ cho dự báo tối đa 7 ngày tới)
    Qd = pd.DataFrame(Q.T)
    lags = {f"lag_{k}": Qd.shift(k).values.T for k in (7, 14, 28)}
    sh = Qd.shift(7)
    lags["roll_mean_7"] = sh.rolling(7, min_periods=4).mean().values.T
    lags["roll_mean_28"] = sh.rolling(28, min_periods=14).mean().values.T
    lags["roll_std_28"] = sh.rolling(28, min_periods=14).std().values.T
    mn, mx = np.nanmin(Q, 1, keepdims=True), np.nanmax(Q, 1, keepdims=True)
    NORM = np.where(mx > mn, (Q - mn) / np.where(mx > mn, mx - mn, 1), np.nan)  # min-max theo từng cặp, như paper

    # ---------- Bảng 1: daily panel
    pi, di = np.nonzero(valid)
    static = pr.merge(cat, on="item_id").merge(stores, on="store_id").sort_values("pair").reset_index(drop=True)
    panel = pd.DataFrame({"date": DATES[di]})
    for col in ["item_id", "store_id", "group4", "group_name", "category", "dept_name", "class_name", "subclass_name"]:
        panel[col] = static[col].values[pi]
    panel["quantity"], panel["revenue"] = Q[pi, di], R[pi, di]
    panel["promo_status"] = PROMO[pi, di]
    panel["holiday_status"], panel["is_holiday"], panel["minor_holiday"] = hstatus[di], is_hol[di].astype(int), minor[di].astype(int)
    ok = panel.promo_status.notna() & panel.holiday_status.notna()
    panel["status6"] = np.where(ok, panel.holiday_status.fillna(0).astype(int).astype(str) + panel.promo_status.fillna(0).astype(int).astype(str), None)
    panel["phase"] = np.array(["other", "pre_1_3", "post_4_7", "post_1_3", "during"])[PHASE[pi, di]]
    panel["qty_norm"] = NORM[pi, di]
    dt = panel.date.dt
    panel["dow"], panel["day_of_month"], panel["week"] = dt.dayofweek, dt.day, dt.isocalendar().week.astype(int).values
    panel["month"], panel["year"], panel["is_weekend"] = dt.month, dt.year, (dt.dayofweek >= 5).astype(int)
    panel["ny_break"], panel["days_to_holiday"], panel["days_after_holiday"] = ny_break[di].astype(int), to_next[di], after_last[di]
    panel["pheic"] = (panel.date <= PHEIC_END).astype(int)
    panel["store4_period"] = (panel.date >= "2023-12-14").astype(int)
    panel["depth"] = np.where(ANY[pi, di] == 1, DEPTH[pi, di], 0)
    panel["promo_type"] = PTYPE[pi, di]
    panel["day_in_promo"], panel["days_to_promo_end"], panel["promo_duration"] = DAY_IN[pi, di], TO_END[pi, di], DUR[pi, di]
    panel["days_since_last_promo"] = SINCE[pi, di]
    panel["regular_price"], panel["price"] = REG[pi, di], PRICE[pi, di]
    panel["price_ratio"] = panel.price / panel.regular_price
    for k, v in lags.items():
        panel[k] = v[pi, di]
    for col in ["item_type", "weight_volume", "weight_netto", "item_type_missing", "weight_volume_missing",
                "weight_netto_missing", "division", "format", "city", "area", "density"]:
        panel[col] = static[col].values[pi]
    panel["split"] = np.select([panel.date <= TRAIN_END, panel.date <= VALID_END], ["train", "valid"], "test")
    panel.to_parquet(OUT / "daily_panel.parquet", index=False)

    # ---------- Bảng 2: episodes (D1–D4) + đợt giả
    Qz, Rz = np.nan_to_num(Q), np.nan_to_num(R)
    CQ, CR, CPOS, CV, CA = cum(Qz), cum(Rz), cum((Qz > 0).astype(np.int8)), cum(valid.astype(np.int8)), cum(ANY)
    hol_c = {"hol": np.concatenate([[0], np.cumsum(is_hol)]), "pre": np.concatenate([[0], np.cumsum(hstatus == -1)]),
             "post": np.concatenate([[0], np.cumsum(hstatus == 1)])}
    holiday_any_c = np.concatenate([[0], np.cumsum(is_hol | (np.nan_to_num(hstatus) != 0))])

    n_ep = len(E)
    log.append(("—", "episodes", "Đợt KM (gộp chứng từ) của các cặp đạt C3", 0, n_ep))
    E = step(E, ~(E.depth > 0), "C1", "episodes", "KM không giảm giá (mức giảm trung vị = 0)")
    E = step(E, (E.a - PRE < 0) | (E.b + POST > ND), "D3", "episodes", "Cửa sổ vượt khoảng 28/08/2022 – 26/09/2024")
    p, a, b = E.pair.values, E.a.values, E.b.values
    full = wsum(CV, p, a - PRE, b + POST) == (b + POST - a + PRE)
    E = step(E, ~full, "D3/B4", "episodes", "Cửa sổ chạm ngày ngoài kỳ bán của cặp hoặc ngày cửa hàng bất thường")
    p, a, b = E.pair.values, E.a.values, E.b.values
    clean = (wsum(CA, p, a - PRE, a) == 0) & (wsum(CA, p, b, b + POST) == 0)
    E = step(E, ~clean, "D2", "episodes", "Cửa sổ không sạch (có KM khác trong 3 ngày trước / 7 ngày sau)")
    E = window_outcomes(E.copy(), CQ, CR, CPOS, hol_c, holiday_any_c)
    E["log_price_ratio"] = np.log(E.price_promo / E.price_before)
    E["incr_rev"] = ((E.during_rev - E.pre_rev) * E.duration + (E.post13_rev - E.pre_rev) * 3
                     + (E.post47_rev - E.pre_rev) * (POST - 3))
    E["arc_elasticity"] = E.log_lift / E.log_price_ratio
    E["source"] = "promo"

    rng = np.random.default_rng(42)
    pl = pd.DataFrame({"pair": np.repeat(np.arange(P), PLACEBO_PER_PAIR)})
    pl["a"] = rng.integers(PRE, ND - PLACEBO_LEN - POST, len(pl))
    pl["b"] = pl.a + PLACEBO_LEN
    pp, pa, pb = pl.pair.values, pl.a.values, pl.b.values
    ok = (wsum(CV, pp, pa - PRE, pb + POST) == (pb + POST - pa + PRE)) & (wsum(CA, pp, pa - PRE, pb + POST) == 0)
    pl = window_outcomes(pl[ok].copy(), CQ, CR, CPOS, hol_c, holiday_any_c)
    pl["source"] = "placebo"
    log.append(("—", "episodes", f"Đợt giả (không KM, dài {PLACEBO_LEN} ngày, {PLACEBO_PER_PAIR} lần thử / cặp) hợp lệ", 0, len(pl)))

    eps = pd.concat([E, pl], ignore_index=True).merge(static, on="pair")
    eps = eps.drop(columns=["pair", "e", "a", "b", "last_d", "first", "last", "days"])
    front = ["source", "item_id", "store_id", "group4", "group_name", "category", "dept_name", "start", "end", "duration"]
    eps = eps[front + [c for c in eps.columns if c not in front]]
    eps.to_csv(OUT / "episodes.csv", index=False)

    # ---------- Bảng 3: group_day (đơn vị của paper)
    gd = (panel[panel.status6.notna()]
          .groupby(["group4", "category", "store_id", "date", "status6", "holiday_status", "promo_status"])
          .agg(qty_norm_mean=("qty_norm", "mean"), quantity_sum=("quantity", "sum"), n_items=("item_id", "nunique"))
          .reset_index())
    gd.to_csv(OUT / "group_day.csv", index=False)

    write_report(panel, eps, gd)


def md_table(df):
    cols = list(df.columns)
    fmt = lambda v: f"{v:,}".replace(",", ".") if isinstance(v, (int, np.integer)) else str(v)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def write_report(panel, eps, gd):
    lg = pd.DataFrame(log, columns=["Bước", "Bảng", "Quy tắc", "Số dòng thay đổi", "Còn lại"])
    by_group = panel.groupby("group4").agg(dong=("quantity", "size"), ma_hang=("item_id", "nunique"))
    by_group["cap_ma_cua_hang"] = panel.drop_duplicates(["group4", "item_id", "store_id"]).groupby("group4").size()
    by_group = by_group.reset_index().rename(columns={"group4": "Nhóm", "dong": "Dòng panel",
                                                      "cap_ma_cua_hang": "Cặp mã hàng – cửa hàng", "ma_hang": "Mã hàng"})
    split = panel.pivot_table(index="group4", columns="split", values="quantity", aggfunc="size")[["train", "valid", "test"]].reset_index()
    ep_tab = eps.pivot_table(index="group4", columns="source", values="duration", aggfunc="size").reset_index()
    hol_tab = eps[eps.source == "promo"].pivot_table(index="group4", columns="hol_status", values="duration", aggfunc="size").fillna(0).astype(int).reset_index()
    t2 = gd.pivot_table(index="group4", columns="status6", values="date", aggfunc="size").fillna(0).astype(int)
    t2 = t2[[c for c in ["00", "01", "-10", "-11", "10", "11"] if c in t2.columns]]
    small = int((t2 < 30).sum().sum())
    t2 = t2.reset_index()
    text = f"""# Báo cáo lọc dữ liệu

*Tạo tự động bởi `src/prepare_data.py` theo `output/quy_tac_loc_du_lieu.md`. Dữ liệu gốc trong `dataset/` không bị sửa.*

## 1. Đầu ra

| File | 1 dòng là | Dùng cho | Đọc bằng |
|---|---|---|---|
| `data_model/daily_panel.parquet` | mã hàng × cửa hàng × ngày | Train ML/DL; elasticity cấp ngày | `pd.read_parquet(...)` |
| `data_model/episodes.csv` | 1 đợt KM thật (`source = promo`) hoặc đợt giả không KM (`source = placebo`) | Two-way ANOVA, t-test, chi-square, logistic, elasticity cấp đợt | `pd.read_csv(..., parse_dates=["start", "end"])` |
| `data_model/group_day.csv` | nhóm × cửa hàng × ngày × trạng thái | Tái lập phân tích 6 nhãn của paper (ANOVA + so sánh cặp có hiệu chỉnh BH) | `pd.read_csv(...)` |

## 2. Số dòng sau từng bước

{md_table(lg)}

## 3. Kết quả theo nhóm

**Daily panel**

{md_table(by_group)}

Chia dữ liệu theo thời gian: train đến {TRAIN_END}, valid đến {VALID_END}, phần còn lại là test.

{md_table(split)}

**Episodes: số đợt KM thật / đợt giả**

{md_table(ep_tab)}

**Đợt KM thật theo trạng thái lễ**

{md_table(hol_tab)}

**Bảng kiểu Bảng 2 của paper:** số nhóm × cửa hàng × ngày theo 6 nhãn, đã bỏ chính ngày lễ và ngày KM không giảm giá.

{md_table(t2)}

Số ô < 30 (quy tắc E3): **{small}**.

## 4. Từ điển cột

### daily_panel.parquet

| Nhóm cột | Cột | Ghi chú |
|---|---|---|
| Khóa | `date`, `item_id`, `store_id`, `group4`, `group_name`, `category`, `dept_name`, `class_name`, `subclass_name` | `group4` ∈ F1, F2, H1, H2; `category` ∈ food, household |
| Biến mục tiêu | `quantity` | Ngày không có dòng bán = 0 (chỉ với cặp đạt C3) |
| Không dùng làm feature | `revenue`, `qty_norm`, `phase`, `status6` | `revenue` sẽ rò rỉ thông tin của biến mục tiêu; `qty_norm` dùng min/max của toàn kỳ, chỉ dùng cho thống kê; `phase`, `status6` là nhãn dùng cho phân tích |
| Nhãn thống kê | `promo_status` (1 / 0 / NaN = KM không giảm giá), `holiday_status` (−1 / 0 / 1; NaN = chính ngày lễ), `is_holiday`, `minor_holiday`, `status6`, `phase` (pre_1_3, during, post_1_3, post_4_7, other) | Mục 1 và 3 của file quy tắc |
| Lịch (biết trước) | `dow`, `day_of_month`, `week`, `month`, `year`, `is_weekend`, `ny_break`, `days_to_holiday`, `days_after_holiday`, `pheic`, `store4_period` | |
| KM đã lên kế hoạch (biết trước) | `depth`, `promo_type`, `day_in_promo`, `days_to_promo_end`, `promo_duration`, `days_since_last_promo` | `depth` = 0 khi không có KM |
| Giá (biết trước) | `regular_price`, `price`, `price_ratio` | Giá thường lấy từ giá trước KM hoặc price_history (giá gần nhất). Không dùng `price_base` của sales |
| Lịch sử bán | `lag_7`, `lag_14`, `lag_28`, `roll_mean_7`, `roll_mean_28`, `roll_std_28` | Đã dịch ≥ 7 ngày, chỉ dùng quá khứ |
| Tĩnh | `item_type`, `weight_volume`, `weight_netto` + các cột `*_missing`, `division`, `format`, `city`, `area`, `density` | |
| Chia dữ liệu | `split` | train / valid / test |

### episodes.csv

| Nhóm cột | Cột | Ghi chú |
|---|---|---|
| Nhận diện | `source`, `item_id`, `store_id`, `group4`, `group_name`, `category`, `dept_name`, `start`, `end`, `duration` | Đợt giả có `duration` = {PLACEBO_LEN} |
| KM (chỉ đợt thật) | `n_docs`, `depth`, `price_before`, `price_promo`, `promo_type`, `log_price_ratio` | |
| Kết quả | `pre_qty`, `during_qty`, `post13_qty`, `post47_qty` (lượng bán trung bình mỗi ngày); các cột `*_rev` tương ứng; `pre_days` | Cửa sổ: 1–3 ngày trước, trong đợt, 1–3 ngày sau, 4–7 ngày sau |
| Biến phụ thuộc | `lift`, `log_lift`, `log_post13`, `log_post47`, `dip_1_3`, `dip_4_7`, `incr_rev`, `arc_elasticity` | `log_x = ln((x + 1/3) / (pre + 1/3))`; `dip` = tỷ lệ đó < 0,9 |
| Thời điểm | `hol_status` (holiday / pre_holiday / post_holiday / normal), `window_has_holiday`, `start_month`, `start_dow`, `year`, `pheic`, `store4_period` | |
| Tĩnh | Như daily panel | |

### group_day.csv

`group4`, `category`, `store_id`, `date`, `status6`, `holiday_status`, `promo_status`, `qty_norm_mean` (trung bình lượng bán đã chuẩn hóa min-max theo từng sản phẩm, như paper), `quantity_sum`, `n_items`.

## 5. Lưu ý khi dùng

0. **Chia train/test:** chỉ áp dụng cho ML/DL. Dùng cột `split` của daily panel (chia theo thời gian, không chia ngẫu nhiên, để không lấy tương lai dự đoán quá khứ). `episodes.csv` và `group_day.csv` dùng cho kiểm định thống kê nên **dùng toàn bộ**, không chia: kiểm định là để suy luận trên toàn mẫu, không phải để dự báo.
1. **Trạng thái lễ của đợt KM:** đợt KM thường dài 14 ngày nên hầu như luôn chứa luôn ngày lễ; vì vậy rất ít đợt có nhãn `pre_holiday` / `post_holiday`. Phân tích 6 nhãn theo kiểu paper nên dùng `group_day.csv`.
2. **Thống kê:** luôn so đợt KM thật với đợt giả (`source`), vì lượng bán sau/trước giảm ngay cả khi không có KM. Dùng Welch / HC3, hiệu chỉnh BH, và báo cáo cỡ hiệu ứng (quy tắc E4).
3. **ML:** feature = các cột "biết trước" + lịch sử bán + tĩnh. Không dùng `revenue`, `qty_norm`, `phase`, `status6`. Mọi phép điền thiếu / mã hóa chỉ học từ `split == "train"`.
4. **Mẫu không cân bằng:** không cắt bớt nhóm lớn; nếu cần, kiểm tra lại bằng trọng số hoặc lấy mẫu phân tầng (quy tắc E1, E5).
"""
    REPORT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
