"""Lọc dữ liệu thô -> dataset_clean/ để nhóm triển khai. Không sửa dataset/ gốc.

Chạy: python src/filter_data.py
Ra:   dataset_clean/*.csv  +  output/bao_cao_loc_du_lieu.md (số dòng bị loại theo từng quy tắc)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from build_episode_features import HOUSEHOLD, OTHER, D1

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "dataset", ROOT / "dataset_clean"
log = []


def read(name, **kw):
    df = pd.read_csv(SRC / name, dtype={"item_id": str}, **kw)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]  # bỏ cột chỉ mục xuất CSV


def apply_rules(table, df, rules):
    """rules: [(tên quy tắc, hàm trả về mask dòng cần loại)], áp dụng lần lượt."""
    log.append((table, "Dòng gốc", len(df)))
    for name, drop in rules:
        mask = drop(df)
        log.append((table, name, -int(mask.sum())))
        df = df[~mask]
    log.append((table, "**Còn lại**", len(df)))
    df.to_csv(DST / table, index=False)
    return df


DST.mkdir(exist_ok=True)

# Catalog: phạm vi Thực phẩm + Hàng tiêu dùng
cat = read("catalog.csv")
cat["group"] = np.select([cat.dept_name.isin(HOUSEHOLD), cat.dept_name.isin(OTHER)], ["household", "other"], "food")
cat = apply_rules("catalog.csv", cat.drop(columns="fatness"), [
    ("Ngoài phạm vi (thuốc lá, hoa, thú cưng, nhóm kỹ thuật, 'НЕ ИСПОЛЬЗОВАТЬ'...)", lambda d: d.group == "other"),
])
scope = set(cat.item_id)
out_of_scope = lambda d: ~d.item_id.isin(scope)

apply_rules("stores.csv", read("stores.csv"), [])

for name in ["sales.csv", "online.csv"]:
    apply_rules(name, read(name), [
        ("Mã hàng ngoài catalog hoặc ngoài phạm vi", out_of_scope),
        ("quantity = 0 và sum_total = 0 (dòng rỗng)", lambda d: (d.quantity == 0) & (d.sum_total == 0)),
        # quantity âm (trả hàng) được giữ lại, cộng ròng khi tổng hợp theo ngày
    ])

apply_rules("discounts_history.csv", read("discounts_history.csv"), [
    ("Ngày sau 26/09/2024 (ngoài kỳ bán hàng, đến 2045)", lambda d: d.date > str(D1.date())),
    ("Mã hàng ngoài catalog hoặc ngoài phạm vi", out_of_scope),
    ("Giá trước KM <= 0 (không tính được mức giảm)", lambda d: d.sale_price_before_promo <= 0),
    ("Giá KM > giá trước KM", lambda d: d.sale_price_time_promo > d.sale_price_before_promo),
])

apply_rules("price_history.csv", read("price_history.csv"), [
    ("Trùng hoàn toàn", lambda d: d.duplicated()),
    ("Mã hàng ngoài catalog hoặc ngoài phạm vi", out_of_scope),
    ("price <= 0 (gồm code 9999999999)", lambda d: d.price <= 0),
])

apply_rules("markdowns.csv", read("markdowns.csv"), [
    ("Trùng hoàn toàn", lambda d: d.duplicated()),
    ("Mã hàng ngoài catalog hoặc ngoài phạm vi", out_of_scope),
])

apply_rules("actual_matrix.csv", read("actual_matrix.csv"), [
    ("Mã hàng ngoài catalog hoặc ngoài phạm vi", out_of_scope),
])

rows = ["| Bảng | Quy tắc | Số dòng |", "|---|---|---|"]
rows += [f"| {t} | {r} | {n:,} |".replace(",", ".") for t, r, n in log]
(ROOT / "output" / "bao_cao_loc_du_lieu.md").write_text(
    "# Báo cáo lọc dữ liệu\n\nNguồn: `dataset/` (giữ nguyên). Kết quả: `dataset_clean/`. "
    "Script: `src/filter_data.py`.\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
print("\n".join(rows))
