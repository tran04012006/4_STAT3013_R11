# Báo cáo lọc dữ liệu

*Tạo tự động bởi `src/prepare_data.py` theo `output/quy_tac_loc_du_lieu.md`. Dữ liệu gốc trong `dataset/` không bị sửa.*

## 1. Đầu ra

| File | 1 dòng là | Dùng cho | Đọc bằng |
|---|---|---|---|
| `data_model/daily_panel.parquet` | mã hàng × cửa hàng × ngày | Train ML/DL; elasticity cấp ngày | `pd.read_parquet(...)` |
| `data_model/episodes.csv` | 1 đợt KM thật (`source = promo`) hoặc đợt giả không KM (`source = placebo`) | Two-way ANOVA, t-test, chi-square, logistic, elasticity cấp đợt | `pd.read_csv(..., parse_dates=["start", "end"])` |
| `data_model/group_day.csv` | nhóm × cửa hàng × ngày × trạng thái | Tái lập phân tích 6 nhãn của paper (ANOVA + so sánh cặp có hiệu chỉnh BH) | `pd.read_csv(...)` |

## 2. Số dòng sau từng bước

| Bước | Bảng | Quy tắc | Số dòng thay đổi | Còn lại |
|---|---|---|---|---|
| — | catalog | Dòng gốc | 0 | 219.810 |
| B2 | catalog | Ngoài 11 ngành hàng của 4 nhóm | -198.267 | 21.543 |
| — | sales | Dòng gốc | 0 | 7.432.685 |
| A2 | sales | quantity / sum_total / price_base ≤ 0 (trả hàng, hủy, dòng rỗng) | -9.204 | 7.423.481 |
| A3 | sales | |sum_total − quantity × price_base| > 1 | -3.075 | 7.420.406 |
| B1–B2 | sales | Mã hàng ngoài catalog hoặc ngoài 4 nhóm | -6.675.074 | 745.332 |
| B4 | sales | Ngày cửa hàng hoạt động bất thường | -17 | 745.315 |
| C3 | sales | Cặp mã hàng – cửa hàng có mật độ bán < 0.2 (3.994 cặp bị loại, còn 4.063 cặp) | -122.115 | 623.200 |
| — | discounts_history | Dòng gốc | 0 | 3.746.744 |
| A4 | discounts_history | Ngày sau 26/09/2024 | -314.913 | 3.431.831 |
| A5 | discounts_history | Giá trước KM ≤ 0 hoặc giá KM > giá trước KM | -42.306 | 3.389.525 |
| B1–B2 | discounts_history | Mã hàng ngoài 4 nhóm | -2.960.623 | 428.902 |
| C3 | discounts_history | Thuộc cặp mã hàng – cửa hàng không đạt C3 | -234.403 | 194.499 |
| — | price_history | Dòng gốc | 0 | 698.626 |
| A6 | price_history | Trùng hoàn toàn | -18.641 | 679.985 |
| A7 | price_history | price ≤ 0 (gồm code 9999999999) | -60.421 | 619.564 |
| — | episodes | Đợt KM (gộp chứng từ) của các cặp đạt C3 | 0 | 16.163 |
| C1 | episodes | KM không giảm giá (mức giảm trung vị = 0) | -2.685 | 13.478 |
| D3 | episodes | Cửa sổ vượt khoảng 28/08/2022 – 26/09/2024 | -922 | 12.556 |
| D3/B4 | episodes | Cửa sổ chạm ngày ngoài kỳ bán của cặp hoặc ngày cửa hàng bất thường | -1.302 | 11.254 |
| D2 | episodes | Cửa sổ không sạch (có KM khác trong 3 ngày trước / 7 ngày sau) | -1.210 | 10.044 |
| — | episodes | Đợt giả (không KM, dài 14 ngày, 10 lần thử / cặp) hợp lệ | 0 | 9.700 |

## 3. Kết quả theo nhóm

**Daily panel**

| Nhóm | Dòng panel | Mã hàng | Cặp mã hàng – cửa hàng |
|---|---|---|---|
| F1 | 381.968 | 613 | 1.437 |
| F2 | 270.230 | 505 | 965 |
| H1 | 431.835 | 826 | 1.342 |
| H2 | 113.294 | 222 | 319 |

Chia dữ liệu theo thời gian: train đến 2024-03-31, valid đến 2024-06-30, phần còn lại là test.

| group4 | train | valid | test |
|---|---|---|---|
| F1 | 271.352 | 62.297 | 48.319 |
| F2 | 184.372 | 46.992 | 38.866 |
| H1 | 284.026 | 76.023 | 71.786 |
| H2 | 75.201 | 19.351 | 18.742 |

**Episodes: số đợt KM thật / đợt giả**

| group4 | placebo | promo |
|---|---|---|
| F1 | 3.458 | 3.039 |
| F2 | 1.963 | 2.769 |
| H1 | 3.626 | 2.686 |
| H2 | 653 | 1.550 |

**Đợt KM thật theo trạng thái lễ**

| group4 | holiday | normal | post_holiday | pre_holiday |
|---|---|---|---|---|
| F1 | 198 | 2.657 | 38 | 146 |
| F2 | 319 | 2.438 | 4 | 8 |
| H1 | 318 | 2.364 | 1 | 3 |
| H2 | 178 | 1.363 | 0 | 9 |

**Bảng kiểu Bảng 2 của paper:** số nhóm × cửa hàng × ngày theo 6 nhãn, đã bỏ chính ngày lễ và ngày KM không giảm giá.

| group4 | 00 | 01 | -10 | -11 | 10 | 11 |
|---|---|---|---|---|---|---|
| F1 | 2.380 | 2.365 | 77 | 61 | 91 | 51 |
| F2 | 2.380 | 2.379 | 77 | 77 | 91 | 91 |
| H1 | 2.380 | 2.185 | 77 | 74 | 91 | 88 |
| H2 | 2.359 | 1.534 | 77 | 50 | 91 | 55 |

Số ô < 30 (quy tắc E3): **0**.

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
| Nhận diện | `source`, `item_id`, `store_id`, `group4`, `group_name`, `category`, `dept_name`, `start`, `end`, `duration` | Đợt giả có `duration` = 14 |
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
