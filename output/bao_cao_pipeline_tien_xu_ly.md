# Báo cáo pipeline tiền xử lý dữ liệu

**Đề tài:** Phân tích doanh số trước, trong và sau khuyến mãi; dự báo nhu cầu mua hàng; đề xuất kinh doanh
**Dữ liệu:** Retail Sales Forecasting (Kaggle `svizor/retail-sales-forecasting-data`), 8 tệp CSV
**Bài báo tham chiếu:** Darbanian et al. (2025), "Timing Matters: How pre- and post-holiday promotions affect fresh and frozen product sales in grocery retail", *Journal of Retailing and Consumer Services* 85, 104317
**Script:** `src/prepare_data.py` · **Quy tắc lọc:** `output/quy_tac_loc_du_lieu.md` · **Ngày lập:** 26/10/2026

---

## 1. Tổng quan

Pipeline biến 8 tệp CSV thô thành **3 bộ dữ liệu sẵn sàng phân tích**:

| Đầu ra | 1 dòng là | Kích thước | Dùng cho |
|---|---|---|---|
| `data_model/daily_panel.parquet` | mã hàng × cửa hàng × ngày | 1.197.327 dòng × 56 cột (26,6 MB) | Train mô hình ML/DL |
| `data_model/episodes.csv` | 1 đợt khuyến mãi (KM) thật, hoặc 1 đợt giả không KM | 19.744 dòng × 53 cột (9,9 MB) | Two-way ANOVA, t-test, chi-square, logistic, elasticity |
| `data_model/group_day.csv` | nhóm hàng × cửa hàng × ngày × trạng thái | 19.181 dòng × 10 cột (1,1 MB) | Tái lập phân tích 6 nhãn của bài báo |

Phạm vi được giữ lại: **4 nhóm hàng** (2 nhóm Thực phẩm, 2 nhóm Đồ tiêu dùng), **4 cửa hàng**, giai đoạn **28/08/2022 – 26/09/2024**.

**Sơ đồ pipeline**

```
 8 tệp CSV thô (dataset/, không bị sửa)
        │
        ▼
 [A] LÀM SẠCH ─────── bỏ dòng âm / 0, dòng sai lệch tiền, KM sau 2024,
        │              giá KM bất hợp lệ, giá trùng / giá 0
        ▼
 [B] PHẠM VI ──────── 11 ngành hàng thuộc 4 nhóm; bỏ ngày cửa hàng bất thường;
        │              chỉ dùng kênh cửa hàng (sales)
        ▼
 [C] GÁN NHÃN ─────── giữ cặp mã hàng – cửa hàng bán đều (≥ 20% số ngày) và điền ngày bán 0;
        │              trạng thái KM (0/1), trạng thái lễ (−1/0/1), 6 nhãn, giai đoạn
        ▼
 [D] ĐỢT KM ───────── gộp chứng từ thành đợt; giữ đợt có giảm giá, cửa sổ đủ dữ liệu, cửa sổ sạch;
        │              tạo thêm đợt giả không KM làm nhóm đối chứng
        ▼
 [F] FEATURE ──────── lịch, KM đã lên kế hoạch, giá, lịch sử bán (lag), thông tin tĩnh
        │
        ▼
 [S] CHIA DỮ LIỆU ─── train ≤ 31/03/2024 · valid 04–06/2024 · test 07–09/2024 (chỉ cho ML/DL)
        │
        ▼
 daily_panel.parquet · episodes.csv · group_day.csv · output/bao_cao_loc_du_lieu.md
```

---

## 2. Dữ liệu đầu vào

| Tệp | Số dòng gốc | Nội dung | Có dùng không | Lý do |
|---|---|---|---|---|
| `sales.csv` | 7.432.685 | Bán hàng theo ngày × mã hàng × cửa hàng | **Có**, là bảng nền | Nguồn của biến mục tiêu `quantity` |
| `discounts_history.csv` | 3.746.744 | Lịch KM: giá trước / trong KM, loại KM, chứng từ | **Có** | Xác định ngày KM, đợt KM, mức giảm |
| `price_history.csv` | 698.626 | Lịch sử thay đổi giá | **Có** | Giá thường cho những ngày không KM |
| `catalog.csv` | 219.810 | Danh mục sản phẩm, 3 cấp phân loại | **Có** | Xác định nhóm hàng, thuộc tính sản phẩm |
| `stores.csv` | 4 | Thông tin cửa hàng | **Có** | Đặc trưng cửa hàng |
| `online.csv` | 1.123.412 | Bán online (chỉ CH1, CH4) | Không | Quan hệ với sales chưa rõ (906.603 bộ khóa trùng nhau); để phân tích riêng |
| `markdowns.csv` | 8.979 | Giảm giá xả hàng | Không | Chỉ có 313 mã hàng; quá nhỏ |
| `actual_matrix.csv` | 35.202 | Danh sách mã hàng theo ngày | Không | Quá thưa, ý nghĩa chưa rõ; không suy ra được hàng có trên kệ hay không |

---

## 3. Tham số chính

| Tham số | Giá trị | Căn cứ |
|---|---|---|
| Khoảng thời gian | 28/08/2022 – 26/09/2024 (761 ngày) | Khoảng có dữ liệu bán hàng |
| Nhóm hàng | F1, F2, H1, H2 (11 ngành hàng) | Mục 6 của file quy tắc |
| Mật độ bán tối thiểu | 0,2 | Đồ tiêu dùng có mật độ trung vị chỉ 0,12; ngưỡng 0,5 sẽ loại gần hết |
| Cửa sổ trước KM | 1–3 ngày | Đề xuất của nhóm; 95,3% số đợt có cửa sổ sạch |
| Cửa sổ sau KM | 1–3 ngày và 4–7 ngày | Đề xuất của nhóm |
| Hằng số làm trơn tỷ lệ | c = 1/3 | Tương đương 1 đơn vị bán trong cửa sổ 3 ngày; tránh chia cho 0 |
| Ngưỡng "sụt sau KM" (`dip`) | tỷ lệ sau / trước < 0,9 | Giảm hơn 10% |
| Đợt giả | dài 14 ngày, 10 lần thử ngẫu nhiên mỗi cặp, seed = 42 | 14 ngày là độ dài trung vị của đợt KM |
| Chia dữ liệu | train ≤ 31/03/2024; valid 01/04 – 30/06/2024; test 01/07 – 26/09/2024 | Chia theo thời gian |

---

## 4. Chi tiết từng bước

### 4.1 Giai đoạn A: Làm sạch

| Bước | Bảng | Quy tắc | Số dòng bị loại | Lý do |
|---|---|---|---|---|
| A1 | Tất cả | Bỏ cột chỉ mục không tên (`Unnamed: 0`) | — | Cột do xuất CSV sinh ra, không phải dữ liệu |
| A2 | sales | Bỏ dòng có `quantity`, `sum_total` hoặc `price_base` ≤ 0 | 9.204 (0,12%) | Trả hàng, hủy, dòng rỗng. Bài báo cũng bỏ giá trị âm. Không có trường "đơn hủy" riêng nên dùng dấu hiệu này |
| A3 | sales | Bỏ dòng có \|sum_total − quantity × price_base\| > 1 | 3.075 | Các trường tiền mâu thuẫn nhau, coi là lỗi dữ liệu |
| A4 | discounts_history | Bỏ ngày sau 26/09/2024 | 314.913 | Dữ liệu KM kéo dài đến năm 2045, ngoài kỳ bán hàng |
| A5 | discounts_history | Bỏ dòng có giá trước KM ≤ 0, hoặc giá KM > giá trước KM | 42.306 | Không tính được mức giảm, hoặc giá mâu thuẫn |
| A6 | price_history | Bỏ dòng trùng hoàn toàn | 18.641 | Bản ghi lặp |
| A7 | price_history | Bỏ dòng có `price` ≤ 0 (gồm code 9999999999) | 60.421 | Không phải giá bán |

**Những gì không lọc ở giai đoạn này:**

- **Ngoại lệ thống kê** (lọc theo IQR, z-score): nếu lọc sẽ xóa luôn đỉnh doanh số ngày KM và ngày lễ, mà đó chính là đối tượng nghiên cứu.
- **Giai đoạn COVID-19:** nếu bỏ sẽ mất 25% dữ liệu, và dataset không có giai đoạn trước dịch để so sánh. Bài báo cũng giữ giai đoạn này. Thay vào đó, pipeline thêm cờ `pheic`.

### 4.2 Giai đoạn B: Phạm vi

| Bước | Quy tắc | Kết quả |
|---|---|---|
| B1–B2 | Chỉ giữ mã hàng có trong catalog và thuộc 11 ngành hàng của 4 nhóm (bảng dưới) | catalog: 219.810 → 21.543 mã hàng; sales: còn 745.332 dòng |
| B3 | Các ngành hàng được chọn đều có KM thường xuyên: ≥ 100 đợt KM và ≥ 20% số mã hàng từng được KM | Đây là tiêu chí dùng để chọn nhóm |
| B4 | Bỏ các ngày cửa hàng hoạt động bất thường: CH3 ngày 16/10/2022 (đóng cửa) và 22/10/2022 (chỉ có 10% số dòng thường ngày); CH4 ngày 13/12/2023 (ngày đầu xuất hiện, chỉ 0,6% số dòng thường ngày) | sales: bỏ 17 dòng. Đợt KM nào có cửa sổ chạm các ngày này cũng bị loại (bước D3) |
| B5 | Chỉ dùng `sales` (kênh cửa hàng) | `online` không cộng vào |
| B6 | Không lọc giai đoạn COVID; thêm cờ `pheic` (ngày ≤ 05/05/2023) | Dùng để kiểm tra độ vững |

**Bốn nhóm hàng phân tích:**

| Phía | Nhóm | Ngành hàng (`dept_name`) | Lý do chọn |
|---|---|---|---|
| Thực phẩm (dễ hỏng, mua thường xuyên) | **F1 Rau quả tươi** | ФРУКТЫ (trái cây), ОВОЩИ (rau củ) | Trùng nhóm "Fresh Fruits & Vegetables" của bài báo |
| | **F2 Thịt tươi và thịt chế biến ướp lạnh** | СВЕЖЕЕ МЯСО (thịt tươi), ПТИЦА (gia cầm), КОЛБАСЫ ВАРЕНЫЕ (xúc xích luộc), СОСИСКИ.САРДЕЛЬКИ.ШПИКАЧКИ (xúc xích) | Tương ứng nhóm "Fresh Meat & Fish"; quy mô mẫu cân bằng với các nhóm khác |
| Đồ tiêu dùng (để lâu, tích trữ được) | **H1 Chăm sóc cá nhân và giấy vệ sinh** | ПЕРСОНАЛЬНЫЙ УХОД, БУМАЖНО-ВАТНАЯ ПРОДУКЦИЯ | Nhiều đợt KM nhất trong phía đồ tiêu dùng; giấy vệ sinh là hàng điển hình được tích trữ |
| | **H2 Giặt giũ và tẩy rửa** | СРЕДСТВА ДЛЯ СТИРКИ, СОПУТСТВУЮЩИЕ ТОВАРЫ ДЛЯ СТИРКИ, ЧИСТЯЩИЕ,МОЮЩИЕ СРЕДСТВА | Bột giặt là sản phẩm tích trữ kinh điển trong nghiên cứu về KM |

Dataset không tái lập được thiết kế "tươi vs đông lạnh" của bài báo: thịt đông lạnh chỉ có 45 dòng bán, rau quả đông lạnh khoảng 300 đợt KM dùng được. Vì vậy đề tài so sánh **Thực phẩm vs Đồ tiêu dùng**, dựa trên cùng cơ chế "dễ hỏng vs tích trữ được".

### 4.3 Giai đoạn C: Chọn cặp bán đều và gán nhãn

**C3. Chọn cặp mã hàng – cửa hàng bán đều và điền ngày bán 0**

`sales.csv` không ghi dòng cho ngày không bán. Với các cửa sổ ngắn 1–3 ngày, nếu không điền 0 thì lượng bán trung bình sẽ bị phóng đại; nhưng nếu điền 0 cho hàng bán rất thưa thì sẽ tạo ra hàng loạt số 0 giả.

- Mật độ bán = số ngày có bán / số ngày từ lần bán đầu đến lần bán cuối của mỗi cặp mã hàng – cửa hàng.
- Giữ cặp có mật độ ≥ 0,2: **4.063 cặp** được giữ, 3.994 cặp bị loại. Số dòng sales bị loại theo đó: 122.115, còn 623.200 dòng.
- Với cặp được giữ: mọi ngày trong khoảng từ lần bán đầu đến lần bán cuối mà không có dòng bán thì gán `quantity = 0`. Ngoài khoảng này, và các ngày ở bước B4, là "không có dữ liệu".

**C1. Trạng thái KM theo ngày (`promo_status`)**

| Giá trị | Ý nghĩa | Ghi chú |
|---|---|---|
| 1 | Ngày nằm trong KM có giảm giá thật (`depth > 0`) | Nếu một ngày có nhiều chứng từ KM thì lấy mức giảm sâu nhất |
| 0 | Không có bản ghi KM nào | |
| NaN | Có KM nhưng không giảm giá (giá trước = giá KM) | Không xếp vào 0 hay 1, để không làm bẩn nhóm "không KM". Chiếm 2,5% số dòng panel |

**C2. Trạng thái lễ (`holiday_status`)**

Ngày lễ theo lịch Nga. Đây là **giả thuyết** dựa trên chữ Cyrillic trong dữ liệu và các đỉnh doanh số. Chỉ 3 ngày lễ có hiệu ứng rõ trong dữ liệu được dùng làm ngày lễ chính:

| Ngày lễ | −1 (trước lễ) | Chính ngày lễ (`is_holiday = 1`, `holiday_status` = NaN) | 1 (sau lễ) |
|---|---|---|---|
| Tết Dương lịch | 27/12 – 31/12 | 01/01 | 02/01 – 08/01 |
| 23/02, 08/03 | 1–3 ngày trước | chính ngày | 1–3 ngày sau |

Các ngày lễ 01/05, 09/05, 12/06, 04/11 được tính là ngày thường (0) và có cờ `minor_holiday`. Tổng cộng mỗi cửa hàng có 22 ngày trước lễ và 26 ngày sau lễ trong 2 năm.

**C4. Sáu nhãn kết hợp (`status6`)**, theo đúng cách của bài báo:

| Nhãn | Ý nghĩa |
|---|---|
| `00` | Ngày thường, không KM |
| `01` | Ngày thường, có KM |
| `-10` | Trước lễ, không KM |
| `-11` | Trước lễ, có KM |
| `10` | Sau lễ, không KM |
| `11` | Sau lễ, có KM |

Ngày lễ chính và ngày KM không giảm giá có `status6` để trống.

**C5. Giai đoạn quanh đợt KM (`phase`)**: `pre_1_3`, `during`, `post_1_3`, `post_4_7`, `other`. Nếu một ngày rơi vào nhiều giai đoạn cùng lúc thì ưu tiên theo thứ tự during > post_1_3 > post_4_7 > pre_1_3.

### 4.4 Giai đoạn D: Đợt khuyến mãi và đợt giả

**Tạo đợt KM:** các chứng từ (`doc_id`) chồng lấn hoặc liền kề nhau của cùng một cặp mã hàng – cửa hàng được gộp thành một đợt. Với mỗi đợt, lấy trung vị của mức giảm, giá trước KM và giá KM; loại KM là giá trị xuất hiện nhiều nhất.

| Bước | Quy tắc | Đợt bị loại | Còn lại |
|---|---|---|---|
| — | Tất cả đợt KM của 4.063 cặp đạt C3 | — | 16.163 |
| C1 | KM không giảm giá (mức giảm trung vị = 0) | 2.685 | 13.478 |
| D3 | Cửa sổ (3 ngày trước, 7 ngày sau) vượt ra ngoài 28/08/2022 – 26/09/2024 | 922 | 12.556 |
| D3 / B4 | Cửa sổ chạm vào ngày không có dữ liệu của cặp, hoặc ngày cửa hàng bất thường | 1.302 | 11.254 |
| D2 | Cửa sổ không sạch: có KM khác của cùng cặp trong 3 ngày trước hoặc 7 ngày sau | 1.210 | **10.044** |

**Đợt giả (nhóm đối chứng):** mỗi cặp thử 10 lần chọn ngẫu nhiên một khoảng 14 ngày. Một đợt giả hợp lệ khi toàn bộ khoảng [3 ngày trước, 7 ngày sau] đều có dữ liệu và **không có ngày KM nào**. Kết quả có **9.700 đợt giả**.

Lý do cần đợt giả: ngay cả khi không có KM, lượng bán sau một khoảng thời gian vẫn có thể thấp hơn trước (do hồi quy về trung bình, vòng đời sản phẩm…). Nếu chỉ so "sau KM" với "trước KM" thì sẽ phóng đại mức sụt do KM. So đợt KM thật với đợt giả mới tách được hiệu ứng của KM.

**Các biến kết quả của mỗi đợt:**

| Biến | Công thức |
|---|---|
| `pre_qty`, `during_qty`, `post13_qty`, `post47_qty` | Lượng bán trung bình mỗi ngày ở 1–3 ngày trước / trong đợt / 1–3 ngày sau / 4–7 ngày sau. Các cột `*_rev` tương tự nhưng tính theo doanh thu |
| `lift` | during_qty / pre_qty (để trống nếu pre_qty = 0) |
| `log_lift`, `log_post13`, `log_post47` | ln((x + 1/3) / (pre_qty + 1/3)) |
| `dip_1_3`, `dip_4_7` | 1 nếu tỷ lệ tương ứng < 0,9 |
| `incr_rev` | (during_rev − pre_rev) × duration + (post13_rev − pre_rev) × 3 + (post47_rev − pre_rev) × 4 |
| `log_price_ratio`, `arc_elasticity` | ln(giá KM / giá trước KM); log_lift / log_price_ratio |
| `hol_status` | `holiday` (đợt chứa ngày lễ chính) > `pre_holiday` > `post_holiday` > `normal` |

### 4.5 Tạo feature cho ML/DL (`daily_panel`)

| Nhóm | Feature | Biết trước tại thời điểm dự báo? |
|---|---|---|
| Lịch | `dow`, `day_of_month`, `week`, `month`, `year`, `is_weekend`, `ny_break`, `is_holiday`, `holiday_status`, `minor_holiday`, `days_to_holiday`, `days_after_holiday`, `pheic`, `store4_period` | Có |
| KM đã lên kế hoạch | `promo_status`, `depth`, `promo_type`, `day_in_promo`, `days_to_promo_end`, `promo_duration`, `days_since_last_promo` | Có (cửa hàng tự lên lịch KM) |
| Giá | `regular_price` (giá trước KM, hoặc giá gần nhất trong price_history), `price`, `price_ratio` | Có |
| Lịch sử bán | `lag_7`, `lag_14`, `lag_28`, `roll_mean_7`, `roll_mean_28`, `roll_std_28` (trung bình / độ lệch chuẩn trượt, tính trên chuỗi đã dịch 7 ngày) | Chỉ dùng quá khứ, dự báo được tối đa 7 ngày tới |
| Tĩnh | `group4`, `category`, `dept_name`, `class_name`, `subclass_name`, `item_type`, `weight_volume`, `weight_netto` + các cột `*_missing`, `store_id`, `division`, `format`, `city`, `area`, `density` | Có |

**Cột không được dùng làm feature:**

- `revenue`: gần bằng lượng × giá, sẽ rò rỉ thông tin của biến mục tiêu.
- `qty_norm`: dùng min/max của toàn bộ thời kỳ, nên chứa thông tin tương lai.
- `phase`, `status6`: nhãn dùng cho phân tích.
- `price_base` của sales: vào ngày KM nó đã bằng giá KM và có thể được tính ngược từ doanh thu. Pipeline không đưa cột này vào panel.

### 4.6 Chia train / valid / test

| Tập | Thời gian | F1 | F2 | H1 | H2 |
|---|---|---|---|---|---|
| train | 28/08/2022 – 31/03/2024 | 271.352 | 184.372 | 284.026 | 75.201 |
| valid | 01/04 – 30/06/2024 | 62.297 | 46.992 | 76.023 | 19.351 |
| test | 01/07 – 26/09/2024 | 48.319 | 38.866 | 71.786 | 18.742 |

- Chia **theo thời gian**, không chia ngẫu nhiên, để không dùng tương lai dự đoán quá khứ.
- Chỉ áp dụng cho ML/DL. `episodes.csv` và `group_day.csv` dùng cho kiểm định thống kê nên dùng toàn bộ.

---

## 5. Kết quả đầu ra

### 5.1 Daily panel theo nhóm

| Nhóm | Dòng | Mã hàng | Cặp mã hàng – cửa hàng |
|---|---|---|---|
| F1 Rau quả tươi | 381.968 | 613 | 1.437 |
| F2 Thịt tươi và thịt chế biến ướp lạnh | 270.230 | 505 | 965 |
| H1 Chăm sóc cá nhân và giấy vệ sinh | 431.835 | 826 | 1.342 |
| H2 Giặt giũ và tẩy rửa | 113.294 | 222 | 319 |
| **Tổng** | **1.197.327** | **2.166** | **4.063** |

### 5.2 Episodes theo nhóm

| Nhóm | Đợt KM thật | Đợt giả | Trung vị `lift` (KM) | Trung vị `lift` (giả) | Tỷ lệ `dip_1_3` (KM) | Tỷ lệ `dip_1_3` (giả) |
|---|---|---|---|---|---|---|
| F1 | 3.039 | 3.458 | 1,51 | 0,96 | 31,0% | 29,0% |
| F2 | 2.769 | 1.963 | 2,62 | 0,95 | 34,5% | 39,4% |
| H1 | 2.686 | 3.626 | 1,82 | 0,86 | 30,2% | 36,0% |
| H2 | 1.550 | 653 | 1,62 | 0,69 | 25,9% | 34,6% |

- Nhóm lớn nhất (F1) có số đợt KM gấp khoảng 2 lần nhóm nhỏ nhất (H2), đạt tiêu chí cân bằng đã đặt.
- Đợt giả có `lift` gần hoặc dưới 1, đúng như kỳ vọng khi không có KM. Điều này cho thấy nhóm đối chứng hoạt động đúng.

**Đợt KM thật theo trạng thái lễ:**

| Nhóm | normal | holiday | pre_holiday | post_holiday |
|---|---|---|---|---|
| F1 | 2.657 | 198 | 146 | 38 |
| F2 | 2.438 | 319 | 8 | 4 |
| H1 | 2.364 | 318 | 3 | 1 |
| H2 | 1.363 | 178 | 9 | 0 |

Một đợt KM thường dài 14 ngày nên hầu như luôn chứa luôn ngày lễ. Vì vậy có rất ít đợt mang nhãn `pre_holiday` / `post_holiday`. Phân tích theo thời điểm lễ nên dùng `group_day.csv`.

### 5.3 Group-day theo 6 nhãn (tương ứng Bảng 2 của bài báo)

Đơn vị đếm: nhóm × cửa hàng × ngày.

| Nhóm | 00 | 01 | −10 | −11 | 10 | 11 |
|---|---|---|---|---|---|---|
| F1 | 2.380 | 2.365 | 77 | 61 | 91 | 51 |
| F2 | 2.380 | 2.379 | 77 | 77 | 91 | 91 |
| H1 | 2.380 | 2.185 | 77 | 74 | 91 | 88 |
| H2 | 2.359 | 1.534 | 77 | 50 | 91 | 55 |

Ô nhỏ nhất có 50 quan sát. Không ô nào dưới 30 (đạt quy tắc E3). Để so sánh: ô nhỏ nhất trong Bảng 2 của bài báo chỉ có 12.

---

## 6. Kiểm tra chất lượng sau khi lọc

| Kiểm tra | Kết quả |
|---|---|
| Ô trống trong `quantity` | 0% |
| `promo_status` = NaN (ngày KM không giảm giá) | 2,5% số dòng; lượng bán trung bình 11,1 mỗi ngày, cao hơn cả ngày KM có giảm giá (6,8) và ngày không KM (3,8). **Nên tìm hiểu thêm** (có thể là KM trưng bày) |
| `holiday_status` = NaN (chính ngày lễ) | 0,9% |
| `regular_price` trống | 10,2% (mã hàng không có trong price_history và chưa từng có KM) |
| `lag_28` / `roll_mean_28` trống | 8,3% / 6,1% (các ngày đầu chuỗi) |
| `days_since_last_promo` trống | 33,6% (chưa từng có KM trước đó) |
| Cân bằng nhóm | Nhóm lớn nhất / nhỏ nhất ≈ 2 lần (quy tắc E1: không cắt bớt nhóm lớn) |
| Ô nhỏ nhất trong bảng 6 nhãn | 50 (≥ 30) |
| Thời gian chạy | Toàn bộ pipeline khoảng 2 phút. Đọc panel mất 0,4 giây (khoảng 1,5 GB RAM). Mô hình cây 300 cây train trong 12,6 giây trên 16 luồng CPU |

Các mô hình dạng cây (LightGBM, XGBoost, HistGradientBoosting) tự xử lý được giá trị trống. Với hồi quy tuyến tính hoặc mạng nơ-ron, cần điền giá trị trống bằng thống kê học từ tập train.

---

## 7. Dùng file nào cho việc gì

| Việc | File | Biến phụ thuộc | Biến giải thích / cách so sánh | Chia train/test? |
|---|---|---|---|---|
| Two-way ANOVA | `episodes.csv` | `log_post13` (bản phụ: `log_post47`) | `category` × `source`; câu hỏi được trả lời bởi hiệu ứng tương tác | Không |
| T-test | `episodes.csv` | `log_post13` | Trong từng `category`: KM thật vs đợt giả (Welch) | Không |
| Chi-square | `episodes.csv` (`source = promo`) | `dip_1_3` | `category`, `hol_status`, `promo_type` | Không |
| Logistic | `episodes.csv` | `dip_1_3` | `category`, `depth`, `duration`, `promo_type`, `log_price_ratio`, `store_id`, `start_month` | Không |
| Log-log elasticity | `episodes.csv` (`source = promo`) | `log_lift` | `log_price_ratio` × `category` | Không |
| Tái lập 6 nhãn của bài báo | `group_day.csv` | `qty_norm_mean` | `status6`, chạy riêng từng `group4`; so sánh cặp có hiệu chỉnh BH | Không |
| Dự báo nhu cầu (ML/DL) | `daily_panel.parquet` | `quantity` | Các feature ở mục 4.5 | **Có**: dùng cột `split` |
| Insight 1: lịch bổ sung hàng | `episodes.csv` + `group_day.csv` | `lift`, `log_post13`, `log_post47`; `status6` | Ma trận `dept_name` × giai đoạn và trạng thái lễ | Không |
| Insight 2: độ nhạy Thực phẩm vs Đồ tiêu dùng | `episodes.csv` (+ `group_day.csv` để đối chiếu với bài báo) | như ANOVA / t-test / elasticity | partial η², cỡ hiệu ứng | Không |
| Insight 3: mức giảm giá tối ưu | `daily_panel.parquet` + `episodes.csv` | `quantity` → `incr_rev` | Thay `depth` từ 0 đến 40% trong mô hình; đối chiếu với `incr_rev` thực tế | **Có** |

**Nguyên tắc khi phân tích:**

- Thống kê: luôn so KM thật với đợt giả; dùng Welch / HC3 / Games-Howell; hiệu chỉnh Benjamini–Hochberg; luôn báo cáo cỡ hiệu ứng và khoảng tin cậy 95% (quy tắc E4).
- ML: chỉ học tiền xử lý từ tập train; tune tham số trên valid; chỉ chạy test một lần ở cuối.

---

## 8. Hạn chế và lưu ý

1. **Lịch lễ Nga là giả thuyết.** Cần đối chiếu với mô tả của dataset trên Kaggle.
2. **Ngưỡng do nhóm tự chọn:** mật độ 0,2, cửa sổ 3 / 7 ngày, ngưỡng dip 0,9. Nên chạy kiểm tra độ nhạy, ví dụ mật độ ∈ {0,1; 0,3}, ngưỡng dip ∈ {0,8; 1,0}. Các tham số nằm ở đầu `src/prepare_data.py`.
3. **Đồ tiêu dùng nhỏ:** H1 + H2 chỉ chiếm khoảng 1,6% doanh thu. Mật độ bán thấp nên cửa sổ 3 ngày có nhiều số 0.
4. **Đợt giả chưa được ghép cặp** với đợt KM theo thời điểm hay mùa vụ. Có thể cải thiện bằng cách ghép cặp theo cùng tháng / cùng phân nhóm hàng, hoặc dùng mô hình difference-in-differences.
5. **Ít đợt KM rơi đúng trước / sau lễ** (mục 5.2). Phân tích theo thời điểm lễ nên dùng `group_day.csv`.
6. **Mức giảm do cửa hàng tự chọn**, nên độ co giãn đo được là tương quan, chưa chắc là nhân quả.
7. **Không có giá vốn**, nên Insight 3 chỉ tối ưu được doanh thu, không tối ưu được lợi nhuận.

---

## 9. Tái lập

```
python src/prepare_data.py
```

- Yêu cầu: Python 3, pandas, numpy, pyarrow.
- Đầu vào: `dataset/` (8 tệp CSV gốc, không bị sửa).
- Đầu ra: `data_model/` và `output/bao_cao_loc_du_lieu.md` (bảng log tự sinh).
- Seed ngẫu nhiên cho đợt giả là 42, nên mỗi lần chạy cho kết quả giống nhau.
