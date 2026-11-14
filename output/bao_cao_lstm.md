# Báo cáo mô hình LSTM dự báo nhu cầu

**Đề tài:** Phân tích doanh số trước, trong và sau khuyến mãi; dự báo nhu cầu mua hàng; đề xuất kinh doanh
**Dữ liệu:** `data_model/daily_panel.parquet` (bảng C: mã hàng × cửa hàng × ngày, 1.197.327 dòng, 4.063 cặp, 28/08/2022 – 26/09/2024)
**Script:** `src/train_lstm.py` · **Kết quả:** `output/lstm/` · **Ngày chạy:** 14/11/2026 (Kaggle, GPU Tesla T4)
**Phương pháp tuning (thống nhất cho mọi mô hình):** Manual Grid Search + Hold-out Validation + Early Stopping

---

## 1. Tóm tắt

| | LSTM | HistGradientBoosting | Trung bình 7 ngày (MA7) | Seasonal naive |
|---|---|---|---|---|
| **WAPE test** | **0,544** | 0,602 | 0,605 | 0,696 |
| WAPE valid | 0,508 | 0,566 | 0,582 | 0,691 |
| MAE test | **2,43** | 2,69 | 2,71 | 3,11 |
| RMSE test | 12,22 | 12,37 | **11,98** | 14,68 |
| Bias test (tổng dự báo / tổng thực tế − 1) | **−15,6%** | −3,8% | +0,1% | −0,3% |

- Cả hai mô hình học máy đều được tune bằng **lưới tham số đầy đủ**, chọn theo WAPE trên **tập hold-out valid** (04–06/2024). Mỗi cấu hình **dừng sớm** theo WAPE valid. Tập test (07–09/2024) chỉ được chạy **1 lần** cho cấu hình được chọn.
- LSTM **giảm WAPE khoảng 10% (tương đối)** so với mô hình cây và MA7, giảm 22% so với seasonal naive. LSTM tốt nhất theo WAPE ở **mọi nhóm hàng, mọi cửa hàng, mọi tầm dự báo 1–7 ngày, mọi giai đoạn trước / trong / sau KM**.
- Đổi lại, LSTM **dự báo thấp hơn thực tế khoảng 16% về tổng lượng** (−36% với Đồ tiêu dùng). Nguyên nhân: mô hình được tối ưu theo sai số tuyệt đối (khớp với WAPE), nên dự báo gần trung vị. Với chuỗi bán thưa, trung vị thường là 0 (mục 6).
- Lịch sử bán là nhóm feature quan trọng nhất. Tiếp theo là **lịch KM và mức giảm giá**, thông tin cửa hàng / sản phẩm, **trạng thái hậu KM**, thứ trong tuần.

---

## 2. Thiết lập bài toán

| Mục | Lựa chọn | Căn cứ |
|---|---|---|
| Cấp dự báo | Mã hàng × cửa hàng × ngày | Thống nhất theo file `STATISTIC + ML_DL.md` |
| Biến mục tiêu | `quantity` (lượng bán ròng theo ngày, ngày không bán = 0) | Mục 5 của `bao_cao_feature.md` |
| Tầm dự báo | 7 ngày tới | Các feature lag trong panel đã dịch ≥ 7 ngày, nên panel được thiết kế cho tầm tối đa 7 ngày |
| Chia dữ liệu (hold-out theo thời gian) | **train** ≤ 31/03/2024 · **valid** 01/04–30/06/2024 · **test** 01/07–26/09/2024 | Chia theo thời gian, không xáo trộn, như pipeline |
| Vai trò của valid | Chọn siêu tham số (grid search) **và** quyết định dừng sớm | |
| Vai trò của test | Chỉ chạy 1 lần cho cấu hình đã chọn, không dùng để chọn gì | |
| Cách đánh giá | Khối tuần không chồng lấn: origin cách nhau 7 ngày, mỗi ngày trong valid / test được dự báo đúng 1 lần (tầm 1–7) | 204.663 ngày-dòng valid; 177.713 ngày-dòng test (khớp đúng số dòng của panel) |
| Chỉ số | WAPE (chính, dùng để tune), MAE, RMSE, bias; tách theo ngày KM / không KM, giai đoạn KM, nhóm hàng, cửa hàng, tầm dự báo | Theo mục 5 của `bao_cao_feature.md` |

---

## 3. Kiến trúc và feature

**Kiến trúc:** encoder–decoder LSTM toàn cục (1 mô hình cho cả 4.063 chuỗi). Cấu hình được chọn có 128 đơn vị ẩn mỗi tầng, cửa sổ quá khứ 56 ngày, 221.533 tham số.

```
 L ngày quá khứ                                7 ngày tới
 [log1p(qty), cờ có dữ liệu,                   [KM đã lên lịch, giá,
  KM, giá, lịch, lễ, embedding tĩnh]            lịch, lễ, embedding tĩnh]
        │                                              │
   LSTM encoder ── trạng thái (h, c) ──────────▶  LSTM decoder
                                                       │
                                     MLP (+ nối tắt feature của ngày dự báo)
                                                       │
                          mức nền encoder + Δ  →  log1p(qtŷ)  →  qtŷ ≥ 0
```

**Feature (bám theo các gợi ý trong `STATISTIC + ML_DL.md`):**

| Nhóm | Feature | Gợi ý tương ứng |
|---|---|---|
| Lịch sử bán (chỉ encoder) | log1p(`quantity`), cờ có / không có dữ liệu | "độ trễ 7 ngày": LSTM đọc trực tiếp L ngày gần nhất nên đã chứa lag 1–L |
| KM đã lên lịch + mức giảm | `promo_disc`, `promo_nodisc` (KM không giảm giá), `depth`, **`depth × food`**, `price_ratio`, tiến độ đợt, `days_to_promo_end`, `promo_duration`, `promo_type` (5, 6, 10, 8, khác) | `promo_status`, `discount_depth`, tương tác mức giảm × nhóm hàng |
| Hậu KM | `days_since_promo` (≈ số ngày kể từ khi KM kết thúc), `never_promo`, `post_promo_1_7`, `post_promo_8_14` | `days_since_promotion_end`, trạng thái hậu KM |
| Mức giá | log giá thường (chuẩn hóa theo train), cờ thiếu giá | |
| Thứ trong tuần | one-hot `dow`, `is_weekend`, `is_friday` | `day_of_week`, `is_friday`, `is_weekend` |
| Tháng / ngày | sin/cos của tháng và ngày trong tháng | `month` |
| Lễ | `is_holiday`, trước lễ, sau lễ, lễ nhỏ, nghỉ Tết, `days_to_holiday`, `days_after_holiday` | `holiday_status`, `days_to_holiday` |
| Cơ cấu | `pheic` (giai đoạn COVID), `store4_period` (từ khi CH4 mở) | Thời điểm store bắt đầu có dữ liệu, thay đổi cơ cấu |
| Tĩnh (embedding) | `store_id`, `group4`, `dept_name`, `class_name`, `format` | `store_id`, `category` |

**Không dùng:** `revenue`, `qty_norm`, `density` (chứa thông tin tương lai), `phase`, `status6` (nhãn phân tích), `price_base`. Mọi phép chuẩn hóa chỉ học từ train.

---

## 4. Tuning: Manual Grid Search + Hold-out Validation + Early Stopping

### 4.1 Quy trình chung

1. Định nghĩa trước một **lưới đầy đủ** (tích Descartes) các giá trị siêu tham số.
2. Với **mỗi cấu hình**: học trên train, sau mỗi epoch (LSTM) hoặc mỗi 25 cây (HGB) thì đo WAPE trên valid. **Dừng sớm** khi WAPE valid không giảm thêm ít nhất 0,0001 trong 4 lần đo liên tiếp, rồi giữ lại trạng thái tốt nhất.
3. Chọn cấu hình có **WAPE valid thấp nhất**.
4. Chạy test **1 lần** cho cấu hình đó.

Early stopping của scikit-learn **không** được dùng, vì nó tách valid ngẫu nhiên từ train (rò rỉ thông tin theo thời gian) và theo dõi loss Poisson chứ không phải WAPE. Thay vào đó, HGB được thêm cây dần (warm start) và dừng theo WAPE trên tập valid theo thời gian. Mô hình học lại ở đúng số cây tốt nhất cho WAPE valid trùng khớp (0,5659), xác nhận quy trình tái lập được.

### 4.2 LSTM: 8 cấu hình

| Siêu tham số | Lưới | Cố định |
|---|---|---|
| Lookback L | 28, 56 ngày | Hàm mất mát MAE trên lượng gốc |
| Số đơn vị ẩn | 64, 128 | Dropout 0,1; batch 1.024; Adam |
| Learning rate | 0,001; 0,002 | 150.000 cửa sổ ngẫu nhiên / epoch (trong 816.119) |
| Early stopping | patience 4 epoch, tối đa 40 epoch, khôi phục trọng số epoch tốt nhất | LR giảm một nửa khi valid không cải thiện sau 1 epoch |

| Lookback | Hidden | LR | WAPE valid | Epoch tốt nhất / đã chạy | Dừng sớm |
|---|---|---|---|---|---|
| **56** | **128** | **0,002** | **0,5082** | 22 / 26 | ✓ |
| 56 | 128 | 0,001 | 0,5098 | 13 / 17 | ✓ |
| 28 | 64 | 0,001 | 0,5099 | 28 / 32 | ✓ |
| 28 | 128 | 0,002 | 0,5099 | 12 / 16 | ✓ |
| 56 | 64 | 0,002 | 0,5099 | 19 / 23 | ✓ |
| 28 | 128 | 0,001 | 0,5100 | 12 / 16 | ✓ |
| 28 | 64 | 0,002 | 0,5100 | 19 / 23 | ✓ |
| 56 | 64 | 0,001 | 0,5153 | 13 / 17 | ✓ |

- **Cả 8 cấu hình đều dừng sớm**, không cấu hình nào chạm giới hạn 40 epoch, tức là đều đã hội tụ trên valid.
- 7/8 cấu hình nằm trong khoảng 0,508–0,510: **mô hình không nhạy với siêu tham số trong vùng này**. Cấu hình được chọn chỉ tốt hơn các cấu hình còn lại khoảng 0,002 WAPE, chênh lệch nhỏ và có thể thay đổi theo seed. Theo đúng quy trình, vẫn chọn cấu hình có WAPE valid thấp nhất.
- Hàm mất mát được cố định là MAE trên lượng gốc vì WAPE = Σ|sai số| / Σ thực tế, nên MAE tối ưu đúng đại lượng được đánh giá. Ở lưới trước (hàm mất mát × lookback), MSE trên log1p kém hơn khoảng 3 điểm WAPE ở cả L = 28 và L = 56 (`output/lstm_v1_backup/variant_selection.csv`).

### 4.3 HistGradientBoosting (Poisson): 12 cấu hình

Lưới: learning rate {0,05; 0,1} × `max_leaf_nodes` {31, 63, 127} × `min_samples_leaf` {20, 100}. Số cây do early stopping quyết định (bước 25 cây, patience 4, tối đa 3.000).

| LR | Lá tối đa | Mẫu tối thiểu / lá | Số cây tốt nhất | WAPE valid |
|---|---|---|---|---|
| **0,05** | **127** | **100** | **950** | **0,5659** |
| 0,05 | 63 | 100 | 800 | 0,5660 |
| 0,05 | 31 | 100 | 1.100 | 0,5664 |
| 0,05 | 63 | 20 | 1.025 | 0,5695 |
| 0,05 | 31 | 20 | 400 | 0,5713 |
| 0,1 | 31 | 100 | 700 | 0,5726 |
| … | | | | … |
| 0,1 | 127 | 20 | 550 | 0,6487 |

- Cả 12 cấu hình đều dừng sớm. `min_samples_leaf = 100` luôn tốt hơn 20 (chuỗi bán thưa, cần lá lớn để tránh khớp nhiễu); learning rate 0,05 tốt hơn 0,1.
- So với cấu hình mặc định không tune (0,05 / 63 lá / 50 mẫu / 400 cây, cột `hgb_default`), tune chỉ cải thiện WAPE valid từ 0,568 xuống 0,566, và WAPE test từ 0,605 xuống 0,602.

Bảng đầy đủ: `variant_selection.csv`, `training_log.csv` (LSTM theo từng epoch), `hgb_grid_summary.csv`, `hgb_tuning.csv` (HGB theo từng bước 25 cây).

---

## 5. Kết quả trên test (01/07 – 26/09/2024)

Kiểm tra overfitting: WAPE của LSTM là 0,477 trên train, 0,508 trên valid, 0,544 trên test. Khoảng cách tăng dần theo thời gian ở mức vừa phải.

### 5.1 Theo trạng thái KM

| Ngày | Số dòng | WAPE LSTM | WAPE HGB | WAPE MA7 | WAPE seasonal naive |
|---|---|---|---|---|---|
| Không KM | 146.371 | **0,548** | 0,610 | 0,619 | 0,717 |
| KM có giảm giá | 24.468 | **0,539** | 0,592 | 0,580 | 0,666 |
| KM không giảm giá | 6.874 | **0,513** | 0,559 | 0,538 | 0,580 |

Ở ngày KM, các baseline dựa vào quá khứ (MA7, seasonal naive) dự báo thấp hơn thực tế 19–27% vì không biết có KM. LSTM và HGB nhận lịch KM nên không bị hụt riêng ở ngày KM: bias của LSTM ở ngày KM (−17%) tương đương ngày thường (−16%). Đây là bằng chứng trực tiếp rằng **đưa KM vào như một nhóm feature giúp dự báo tốt hơn**.

### 5.2 Theo giai đoạn quanh KM

| Giai đoạn | Số dòng | WAPE LSTM | WAPE HGB | WAPE MA7 | WAPE seasonal naive |
|---|---|---|---|---|---|
| 1–3 ngày trước KM | 4.889 | **0,625** | 0,674 | 0,717 | 0,825 |
| Trong KM | 23.725 | **0,533** | 0,586 | 0,574 | 0,660 |
| 1–3 ngày sau KM | 5.014 | **0,541** | 0,597 | 0,774 | 0,898 |
| 4–7 ngày sau KM | 6.323 | **0,526** | 0,611 | 0,819 | 1,069 |
| Ngày khác | 137.762 | **0,545** | 0,603 | 0,593 | 0,677 |

Ngay sau KM, các baseline dựa vào quá khứ sai rất nhiều (WAPE 0,77–1,07) vì chúng lấy lượng bán cao trong KM làm mức nền. LSTM giữ được sai số ngang ngày thường nhờ các feature hậu KM. Giai đoạn 1–3 ngày trước KM khó dự báo nhất với mọi mô hình.

### 5.3 Theo nhóm hàng

| Nhóm | Số dòng | WAPE LSTM | WAPE HGB | WAPE MA7 | Bias LSTM |
|---|---|---|---|---|---|
| **Thực phẩm** | 87.185 | **0,516** | 0,569 | 0,569 | −13% |
| F1 Rau quả tươi | 48.319 | **0,530** | 0,592 | 0,570 | −14% |
| F2 Thịt tươi, thịt chế biến | 38.866 | **0,483** | 0,513 | 0,569 | −9% |
| **Đồ tiêu dùng** | 90.528 | **0,747** | 0,844 | 0,870 | −36% |
| H1 Chăm sóc cá nhân, giấy | 71.786 | **0,722** | 0,809 | 0,832 | −33% |
| H2 Giặt giũ, tẩy rửa | 18.742 | **0,961** | 1,155 | 1,210 | −65% |

Đồ tiêu dùng khó dự báo hơn nhiều. Các chuỗi này bán thưa (trung bình 1,04 đơn vị/ngày, 57% số ngày bán 0; Thực phẩm: 8,04 đơn vị/ngày, 31%), nên WAPE cao với mọi mô hình. Lợi thế của LSTM ở đây lớn nhất (giảm khoảng 11–17% WAPE so với HGB), nhưng phải trả giá bằng bias âm lớn (mục 6).

### 5.4 Theo tầm dự báo

| Tầm (ngày) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| LSTM | **0,511** | **0,523** | **0,523** | **0,531** | **0,539** | **0,566** | **0,616** |
| HGB | 0,604 | 0,607 | 0,593 | 0,587 | 0,577 | 0,611 | 0,643 |
| MA7 | 0,552 | 0,586 | 0,577 | 0,591 | 0,606 | 0,638 | 0,686 |

Sai số tăng dần theo tầm dự báo, đúng kỳ vọng. HGB chỉ dùng lag ≥ 7 ngày nên lượng thông tin của nó chỉ bằng LSTM ở tầm 7. **Ở tầm 7, phép so sánh hoàn toàn công bằng, LSTM vẫn tốt hơn HGB (0,616 so với 0,643).**

### 5.5 Theo cửa hàng

| Cửa hàng | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| WAPE LSTM | **0,539** | **0,533** | **0,569** | **0,547** |
| WAPE HGB | 0,602 | 0,577 | 0,608 | 0,605 |
| MAE LSTM | **2,56** | **1,45** | **1,63** | **2,64** |
| MAE HGB | 2,86 | 1,57 | 1,74 | 2,92 |

LSTM tốt hơn ở cả 4 cửa hàng, kể cả CH4 (mới xuất hiện từ 12/2023).

---

## 6. Điểm yếu: dự báo thấp

| | Tổng thực tế | Tổng LSTM | Chênh LSTM | Tổng HGB | Chênh HGB |
|---|---|---|---|---|---|
| Toàn bộ test | 795.204 | 671.149 | −15,6% | 764.827 | −3,8% |
| Thực phẩm | 701.242 | 611.074 | −12,9% | 672.820 | −4,1% |
| Đồ tiêu dùng | 93.961 | 60.075 | −36,1% | 92.007 | −2,1% |

- Trung bình mỗi ngày toàn hệ thống (88 ngày test): thực tế 9.036, LSTM 7.627, HGB 8.691.
- **Nguyên nhân:** MAE dẫn mô hình về trung vị. Với chuỗi bán thưa (57% số ngày-dòng của Đồ tiêu dùng trong test bán 0, so với 31% ở Thực phẩm), trung vị thấp hơn trung bình nhiều. WAPE thưởng cho cách dự báo này, nhưng tổng dự báo bị hụt. Cấu hình lớn hơn được chọn qua grid search (hidden 128, L = 56) giảm WAPE nhưng làm bias âm sâu hơn (−10,4% ở cấu hình cũ 64 / 28): tối ưu WAPE và tối ưu tổng lượng là hai mục tiêu khác nhau.
- **Ý nghĩa thực tế:** nếu dùng để **đặt hàng / bổ sung kho** (cần tổng lượng chính xác), LSTM hiện tại sẽ đặt thiếu, nhất là Đồ tiêu dùng. Khi đó HGB (bias −4%) hợp lý hơn. Nếu cần sai số từng ngày nhỏ nhất thì LSTM là lựa chọn tốt nhất.
- **Cách khắc phục (chưa làm):** (a) dùng mất mát Poisson / Tweedie để dự báo trung bình; (b) dự báo phân vị (P50 cho sai số, P80–P90 cho tồn kho an toàn); (c) hiệu chỉnh hệ số theo nhóm hàng học từ valid.

---

## 7. Feature nào quan trọng

Cách đo: phá liên kết của 1 nhóm feature trên tập valid rồi xem WAPE tăng bao nhiêu (trung bình 3 lần, WAPE gốc 0,508). Nhóm theo chuỗi được hoán vị giữa các mẫu. Nhóm lịch được thay bằng lịch của một ngày ngẫu nhiên khác, vì trong cùng một khối đánh giá mọi chuỗi có cùng lịch.

| Nhóm feature | WAPE tăng thêm | Độ lệch chuẩn | Diễn giải |
|---|---|---|---|
| Lịch sử bán | **+0,950** | 0,016 | Không thể thiếu |
| **Lịch KM + mức giảm giá + loại KM** | **+0,140** | 0,006 | Nhóm biết trước quan trọng nhất |
| Tĩnh (cửa hàng, nhóm, ngành, class, format) | +0,084 | 0,004 | Phân biệt quy mô và hành vi chuỗi |
| **Hậu KM** (số ngày từ khi KM kết thúc) | **+0,058** | 0,013 | Rõ, khớp với kết quả mục 5.2 |
| **Thứ trong tuần / cuối tuần / thứ Sáu** | **+0,035** | 0,001 | Mùa vụ tuần rõ |
| Mức giá thường | +0,025 | 0,002 | |
| Lễ | +0,007 | 0,000 | Test (07–09) không có ngày lễ chính, nên hiệu ứng thấp là hợp lý |
| Tháng / ngày trong tháng | +0,006 | 0,001 | Test chỉ có 3 tháng hè nên ít biến động |
| Cơ cấu (COVID, CH4) | +0,002 | 0,000 | |

Đối chiếu với các gợi ý trong `STATISTIC + ML_DL.md`: tất cả nhóm được đề xuất đều có đóng góp dương. KM / mức giảm, hậu KM và thứ trong tuần là quan trọng nhất; tháng và lễ có tác dụng nhỏ. Mô hình được chọn (L = 56, hidden 128) dựa vào KM và hậu KM nhiều hơn rõ rệt so với cấu hình cũ (L = 28, hidden 64: +0,054 và +0,007). Lý do có thể là cửa sổ 56 ngày chứa trọn các đợt KM trước đó.

---

## 8. Mô phỏng mức giảm giá (phục vụ Insight 3)

Mô phỏng chạy trên 24.468 ngày KM có giảm giá trong test. Mỗi lần chỉ thay `depth` (và `price_ratio = 1 − depth`), giữ nguyên mọi feature khác. Doanh thu = lượng dự báo × giá thường × (1 − depth). Chỉ số = 100 tại mức giảm 20% (trung vị quan sát được).

| Mức giảm | 5% | 10% | 15% | **20%** | 25% | 30% | 35% | 40% |
|---|---|---|---|---|---|---|---|---|
| Lượng – Thực phẩm | 89 | 92 | 96 | 100 | 104 | 108 | 113 | 118 |
| Lượng – Đồ tiêu dùng | 84 | 89 | 94 | 100 | 106 | 112 | 119 | 126 |
| Doanh thu – Thực phẩm | **105** | 104 | 102 | 100 | 98 | 95 | 92 | 89 |
| Doanh thu – Đồ tiêu dùng | 98 | 99 | 100 | **100** | 100 | 99 | 98 | 96 |

- Mô hình học được rằng **giảm giá sâu thì lượng bán tăng**, nhưng tăng không đủ bù phần giá bị giảm. Độ co giãn ngầm định khoảng −0,6 (Thực phẩm) và −0,9 (Đồ tiêu dùng). **Doanh thu trong ngày KM không tăng khi giảm sâu hơn khoảng 20%.**
- **Phải đọc thận trọng:**
  1. Độ co giãn của LSTM thấp hơn ước lượng cấp đợt ở `bao_cao_feature.md` (−1,34 / −0,83), và thứ tự giữa hai nhóm bị đảo. Mô hình dự báo không được thiết kế để đo nhân quả. Mức giảm do cửa hàng tự chọn (nhiễu chọn), và hiệu ứng có thể bị "chia" sang các feature khác (loại KM, lịch sử bán trong các đợt trước).
  2. Mô phỏng chỉ tính **ngày trong KM**, chưa tính phần sụt hoặc duy trì sau KM.
  3. Không có giá vốn, nên chỉ nói được về doanh thu, không nói được về lợi nhuận.
  4. Không ngoại suy ra ngoài 5–40%.
  5. LSTM dự báo thấp (mục 6) nên mức tuyệt đối bị hụt. Chỉ nên đọc chỉ số tương đối.
- **Kết luận cho Insight 3:** kết quả LSTM ủng hộ hướng "mức giảm tối ưu theo doanh thu **không phải** mức sâu nhất". Nhưng mức tối ưu cụ thể nên lấy từ mô hình elasticity cấp đợt kết hợp `incr_rev`, dùng LSTM làm kiểm tra chéo.

---

## 9. Hạn chế và việc tiếp theo

1. **Bias âm** (mục 6): thử mất mát Poisson / Tweedie hoặc dự báo phân vị trước khi dùng cho bài toán tồn kho.
2. **Một lần hold-out, một seed:** chênh lệch WAPE valid giữa các cấu hình LSTM tốt nhất chỉ khoảng 0,002, nhỏ hơn mức dao động có thể có giữa các seed. Thứ hạng trong nhóm đầu không nên diễn giải quá mức.
3. **Lưới có giới hạn:** dropout, số tầng và kích thước batch được cố định. Hàm mất mát được cố định theo lập luận ở mục 4.2.
4. **Baseline cây** là HistGradientBoosting của scikit-learn và chỉ dùng lag ≥ 7. Một mô hình cây có lag 1–6 cho từng tầm có thể thu hẹp khoảng cách ở tầm ngắn.
5. **Tập test là mùa hè** (07–09/2024), không có ngày lễ chính. Chưa đánh giá được khả năng dự báo quanh Tết / 08/03.
6. **Chưa dùng kênh online** (lag `online.quantity` cho CH1, CH4) và chưa thử TFT như kế hoạch.

---

## 10. Tái lập

```
pip install torch                           # GPU nếu có; CPU vẫn chạy được nhưng chậm hơn nhiều
python src/train_lstm.py                    # ~90 phút trên Kaggle (T4 + 4 CPU): LSTM 10 phút, HGB ~80 phút
python src/train_lstm.py --importance-only  # chỉ tính lại mục 7 từ mô hình đã lưu
python src/train_lstm.py --smoke            # chạy thử khoảng 1–2 phút
```

Biến môi trường tùy chọn: `PANEL_PATH` (đường dẫn panel), `OUT_DIR` (thư mục kết quả). Mặc định đọc/ghi trong dự án. Kết quả GPU và CPU có thể lệch nhau ở chữ số thứ 4 (smoke: 0,5879 trên GPU so với 0,5881 trên CPU).

| File trong `output/lstm/` | Nội dung |
|---|---|
| `lstm_best.pt` | Trọng số LSTM được chọn, siêu tham số và danh sách feature |
| `variant_selection.csv`, `training_log.csv` | Grid search LSTM: kết quả từng cấu hình, log từng epoch |
| `hgb_grid_summary.csv`, `hgb_tuning.csv` | Grid search HGB: kết quả từng cấu hình, đường WAPE valid theo số cây |
| `metrics_*.csv`, `metrics.txt` | WAPE / MAE / RMSE / bias của các mô hình theo từng lát cắt |
| `test_predictions.parquet` | Dự báo từng dòng test của LSTM, HGB, MA7, seasonal naive |
| `feature_group_importance.csv` | Độ quan trọng nhóm feature (mục 7) |
| `depth_simulation.csv` | Mô phỏng mức giảm giá (mục 8) |
| `run.log`, `kaggle_kernel.log`, `summary.json` | Log đầy đủ và tóm tắt cấu hình |

Kết quả các lần chạy trước: `output/lstm_v1_backup/` (lưới loss × lookback, 20 epoch), `output/lstm_v2_backup/` (40 epoch, HGB chưa có early stopping).
