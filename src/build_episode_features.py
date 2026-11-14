"""Bảng feature cấp đợt khuyến mãi: 1 dòng = 1 đợt KM của 1 mã hàng tại 1 cửa hàng.

Chạy: python src/build_episode_features.py
Ra:   output/features/episode_features.csv  (đợt KM thật)
      output/features/placebo_episodes.csv  (đợt giả không KM, làm nhóm đối chứng)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "dataset", ROOT / "output" / "features"
W = 14                  # độ dài cửa sổ trước / sau KM (ngày)
MIN_PRE_DAYS = 7        # số ngày có bán tối thiểu trong cửa sổ trước để coi là hàng bán đều
HOL_WIN = 7             # số ngày trước / sau lễ
D0, D1 = pd.Timestamp("2022-08-28"), pd.Timestamp("2024-09-26")  # khoảng có dữ liệu sales

# Lịch lễ Nga (giả thuyết từ chữ Cyrillic + đỉnh doanh số 30/12, 07/03; cần xác nhận với nguồn dataset)
HOLIDAYS = pd.to_datetime([
    "2022-11-04", "2023-01-01", "2023-02-23", "2023-03-08", "2023-05-01", "2023-05-09", "2023-06-12",
    "2023-11-04", "2024-01-01", "2024-02-23", "2024-03-08", "2024-05-01", "2024-05-09", "2024-06-12"])

# Hàng tiêu dùng (phi thực phẩm, để lâu, tích trữ được)
HOUSEHOLD = ["ПЕРСОНАЛЬНЫЙ УХОД", "БУМАЖНО-ВАТНАЯ ПРОДУКЦИЯ", "ТОВАРЫ ДЛЯ ДОМА", "ЧИСТЯЩИЕ,МОЮЩИЕ СРЕДСТВА",
             "СРЕДСТВА ДЛЯ СТИРКИ", "ТОВАРЫ ДЛЯ УБОРКИ", "ОСВЕЖИТЕЛИ,ИНСЕКТИЦИДЫ",
             "СОПУТСТВУЮЩИЕ ТОВАРЫ ДЛЯ СТИРКИ", "ТОВАРЫ ДЛЯ ЛИЧНОГО ПОЛЬЗОВАНИЯ"]
# Phi thực phẩm khác / nhóm kỹ thuật: không đưa vào so sánh Thực phẩm - Hàng tiêu dùng
OTHER = ["ТАБАЧНЫЕ ИЗДЕЛИЯ", "АКСЕССУАРЫ ДЛЯ КУРЕНИЯ", "ЦВЕТЫ", "ДЕТСКИЙ МИР", "ТОВАРЫ ДЛЯ ПРАЗДНИКА",
         "ТОВАРЫ ДЛЯ ШКОЛЫ И ОФИСА", "АВТОТОВАРЫ", "ЗООТОВАРЫ", "КОРМА ДЛЯ КОШЕК", "КОРМА ДЛЯ СОБАК",
         "ЛАКОМСТВА ДЛЯ ЖИВОТНЫХ", "КОРМА ДЛЯ ДРУГИХ ЖИВОТНЫХ", "НАПОЛНИТЕЛИ", "ИНФОРМАЦИОННЫЕ ТОВАРЫ",
         "НАРОДНЫЕ ПРОМЫСЛЫ", "СЕЗОННЫЙ АССОРТИМЕНТ", "ВСПОМОГАТЕЛЬНАЯ ГРУППА", "ГЛОБАЛЬНЫЙ КАТАЛОГ",
         "АКЦИЯ ЛОЯЛЬНОСТИ", "АКЦИЯ ЛОЯЛЬНОСТИ (НЕ ИСПОЛЬЗОВАТЬ)", "ПОДАРОЧНЫЕ КОРЗИНЫ", "ПРИКАССА",
         "РЫБА СВЕЖАЯ НЕ ИСПОЛЬЗОВАТЬ", "СДОБНЫЕ ИЗДЕЛИЯ(НЕ ИСПОЛЬЗОВАТЬ)"]
# Để đối chiếu với Darbanian et al. (2025): rau quả tươi vs đông lạnh
FRESH_FV = ["ФРУКТЫ", "ОВОЩИ", "ЗЕЛЕНЬ СВЕЖАЯ"]
FROZEN_FV = ["ОВОЩИ ЗАМОРОЖЕННЫЕ", "ЯГОДЫ ЗАМОРОЖЕННЫЕ", "ГРИБЫ ЗАМОРОЖЕННЫЕ"]


def window_sum(C, p, a, b):
    """Tổng theo ngày [a, b) của từng hàng p, dùng mảng cộng dồn C."""
    return C[p, b] - C[p, a]


def outcome_features(df, Q, R, CQ, CR):
    p, a, b = df.pair.values, df.d_start.values, df.d_end.values + 1
    dur = b - a
    df["pre_qty"] = window_sum(CQ, p, a - W, a) / W
    df["during_qty"] = window_sum(CQ, p, a, b) / dur
    df["post_qty"] = window_sum(CQ, p, b, b + W) / W
    df["pre_rev"] = window_sum(CR, p, a - W, a) / W
    df["during_rev"] = window_sum(CR, p, a, b) / dur
    df["post_rev"] = window_sum(CR, p, b, b + W) / W
    idx = np.arange(W)
    df["pre_days"] = (Q[p[:, None], a[:, None] - W + idx] > 0).sum(1)
    df["post_days"] = (Q[p[:, None], b[:, None] + idx] > 0).sum(1)
    c = 1 / W  # hằng số làm trơn: tương đương 1 đơn vị bán trong cả cửa sổ
    df["lift"] = df.during_qty / df.pre_qty.where(df.pre_qty > 0)
    df["post_ratio"] = df.post_qty / df.pre_qty.where(df.pre_qty > 0)
    df["log_lift"] = np.log((df.during_qty + c) / (df.pre_qty + c))
    df["log_post_ratio"] = np.log((df.post_qty + c) / (df.pre_qty + c))
    df["dip"] = (df.post_ratio < 0.9).astype(int)
    return df


def product_store_features(df, cat, stores):
    df = df.merge(cat, on="item_id", how="left").merge(stores, on="store_id", how="left")
    df["catalog_missing"] = df.dept_name.isna().astype(int)
    df["group"] = np.select([df.dept_name.isin(HOUSEHOLD), df.dept_name.isin(OTHER) | df.dept_name.isna()],
                            ["household", "other"], "food")
    df["is_food"] = (df.group == "food").astype(int)
    df["paper_group"] = np.select([df.dept_name.isin(FRESH_FV), df.dept_name.isin(FROZEN_FV)],
                                  ["fresh_fv", "frozen_fv"], "")
    for col in ["item_type", "weight_volume", "weight_netto"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def holiday_features(df):
    s, e = df.start.values[:, None], df.end.values[:, None]
    h, k = HOLIDAYS.values[None, :], np.timedelta64(HOL_WIN, "D")
    covers = ((h >= s) & (h <= e)).any(1)
    pre = ((s < h) & (e >= h - k)).any(1)        # KM chạm vào 7 ngày trước lễ
    post = ((e > h) & (s <= h + k)).any(1)       # KM chạm vào 7 ngày sau lễ
    df["hol_status"] = np.select([covers, pre, post], ["holiday", "pre_holiday", "post_holiday"], "normal")
    nxt = np.where(h >= s, (h - s) / np.timedelta64(1, "D"), np.inf).min(1)
    df["days_to_next_holiday"] = nxt
    df["start_month"], df["start_dow"] = df.start.dt.month, df.start.dt.dayofweek
    df["year"] = df.start.dt.year
    df["store4_period"] = (df.start >= "2023-12-13").astype(int)  # sau khi CH4 xuất hiện
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ndays = (D1 - D0).days + 1

    sales = pd.read_csv(DATA / "sales.csv", usecols=["date", "item_id", "quantity", "sum_total", "store_id"],
                        parse_dates=["date"], dtype={"item_id": str})
    s = sales.groupby(["item_id", "store_id", "date"], as_index=False)[["quantity", "sum_total"]].sum()
    s["pair"] = s.groupby(["item_id", "store_id"]).ngroup()
    s["d"] = (s.date - D0).dt.days
    pairs = s.drop_duplicates("pair")[["item_id", "store_id", "pair"]]
    Q = np.zeros((len(pairs), ndays), np.float32)
    R = np.zeros((len(pairs), ndays), np.float32)
    Q[s.pair, s.d], R[s.pair, s.d] = s.quantity, s.sum_total
    # quantity âm (trả hàng) được tính ròng theo ngày; ngày không có dòng = 0 (chỉ dùng cho hàng bán đều, xem MIN_PRE_DAYS)
    CQ = np.concatenate([np.zeros((len(pairs), 1)), Q.cumsum(1, dtype=np.float64)], 1)
    CR = np.concatenate([np.zeros((len(pairs), 1)), R.cumsum(1, dtype=np.float64)], 1)
    del sales, s

    disc = pd.read_csv(DATA / "discounts_history.csv",
                       usecols=["date", "item_id", "sale_price_before_promo", "sale_price_time_promo",
                                "promo_type_code", "doc_id", "store_id"],
                       parse_dates=["date"], dtype={"item_id": str})
    disc = disc[disc.date <= D1]  # bỏ các dòng sau kỳ bán hàng (đến 2045)
    disc["depth"] = np.where(disc.sale_price_before_promo > 0,
                             1 - disc.sale_price_time_promo / disc.sale_price_before_promo, np.nan)

    # Gộp các doc chồng lấn / liền kề của cùng mã hàng - cửa hàng thành một đợt
    ep = (disc.groupby(["item_id", "store_id", "doc_id"]).date.agg(["min", "max"]).reset_index()
          .sort_values(["item_id", "store_id", "min"]))
    prev_end = ep.groupby(["item_id", "store_id"])["max"].shift() + pd.Timedelta(days=1)
    ep["episode"] = (ep["min"] > prev_end).groupby([ep.item_id, ep.store_id]).cumsum()
    disc = disc.merge(ep[["item_id", "store_id", "doc_id", "episode"]], on=["item_id", "store_id", "doc_id"])
    mode = lambda x: x.mode().iloc[0] if x.notna().any() else np.nan
    ep = disc.groupby(["item_id", "store_id", "episode"]).agg(
        start=("date", "min"), end=("date", "max"), promo_days=("date", "nunique"), n_docs=("doc_id", "nunique"),
        price_before=("sale_price_before_promo", "median"), price_promo=("sale_price_time_promo", "median"),
        depth=("depth", "median"), promo_type=("promo_type_code", mode)).reset_index()
    ep["duration"] = (ep.end - ep.start).dt.days + 1
    ep["promo_type"] = ep.promo_type.fillna(-1).astype(int).astype(str).replace("-1", "missing")
    ep["zero_depth"] = (ep.depth == 0).astype(int)
    ep["log_price_before"] = np.log(ep.price_before.where(ep.price_before > 0))
    ep["log_price_ratio"] = np.log((ep.price_promo / ep.price_before).where((ep.price_before > 0) & (ep.price_promo > 0)))

    # Lịch KM theo mã hàng - cửa hàng, để kiểm tra cửa sổ sạch
    dp = disc[["item_id", "store_id", "date"]].drop_duplicates().merge(pairs, on=["item_id", "store_id"])
    P = np.zeros((len(pairs), ndays), np.int8)
    P[dp.pair, (dp.date - D0).dt.days] = 1
    CP = np.concatenate([np.zeros((len(pairs), 1)), P.cumsum(1, dtype=np.int32)], 1)

    ep = ep.merge(pairs, on=["item_id", "store_id"])  # mã hàng không có dòng sales nào thì bỏ
    ep["d_start"], ep["d_end"] = (ep.start - D0).dt.days, (ep.end - D0).dt.days
    ep = ep[(ep.d_start - W >= 0) & (ep.d_end + W < ndays)].copy()
    p, a, b = ep.pair.values, ep.d_start.values, ep.d_end.values + 1
    ep["clean"] = ((window_sum(CP, p, a - W, a) == 0) & (window_sum(CP, p, b, b + W) == 0)).astype(int)
    ep["source"] = "promo"

    # Đợt giả: cùng mã hàng - cửa hàng, dài 14 ngày, không có ngày KM nào trong [start - W, end + W]
    rng = np.random.default_rng(42)
    k = 4
    pl = pd.DataFrame({"pair": np.repeat(pairs.pair.values, k)})
    pl["d_start"] = rng.integers(W, ndays - W - 14, len(pl))
    pl["d_end"] = pl.d_start + 13
    pp, pa, pb = pl.pair.values, pl.d_start.values, pl.d_end.values + 1
    pl = pl[window_sum(CP, pp, pa - W, pb + W) == 0].merge(pairs, on="pair")
    pl["start"] = D0 + pd.to_timedelta(pl.d_start, "D")
    pl["end"] = D0 + pd.to_timedelta(pl.d_end, "D")
    pl["duration"], pl["clean"], pl["source"] = 14, 1, "placebo"

    cat = pd.read_csv(DATA / "catalog.csv", usecols=lambda c: not c.startswith("Unnamed") and c != "fatness",
                      dtype={"item_id": str})
    stores = pd.read_csv(DATA / "stores.csv", usecols=["store_id", "division", "format", "city", "area"])

    for df, name in [(ep, "episode_features.csv"), (pl, "placebo_episodes.csv")]:
        df = outcome_features(df, Q, R, CQ, CR)
        df = holiday_features(df)
        df = product_store_features(df, cat, stores)
        df["eligible"] = ((df.clean == 1) & (df.pre_days >= MIN_PRE_DAYS) & df.group.isin(["food", "household"])).astype(int)
        if df.source.iat[0] == "promo":
            # doanh thu tăng thêm = phần tăng trong KM - phần sụt sau KM, so với nền trước KM
            df["incr_rev"] = (df.during_rev - df.pre_rev) * df.duration + (df.post_rev - df.pre_rev) * W
            df["arc_elasticity"] = df.log_lift / df.log_price_ratio.where(df.log_price_ratio < 0)
        df.drop(columns=["pair", "d_start", "d_end"]).to_csv(OUT / name, index=False)
        print(name, len(df), "dòng;", int(df.eligible.sum()), "đợt đủ điều kiện phân tích")


if __name__ == "__main__":
    main()
