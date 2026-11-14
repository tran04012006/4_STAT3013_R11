# Bộ feature cho project và kết quả sơ bộ

*Đề tài: Phân tích doanh số trước, trong và sau khuyến mãi; dự báo nhu cầu; đề xuất kinh doanh. Dữ liệu: 8 tệp CSV trong `dataset/` (Kaggle `svizor/retail-sales-forecasting-data`). Tài liệu tham chiếu chính: Darbanian et al. (2025), "Timing Matters", Journal of Retailing and Consumer Services 85, 104317.*

Tất cả kết quả ở mục 6 là **sơ bộ**. Chúng được dùng để kiểm tra rằng bộ feature chạy được và đủ mẫu, chưa phải kết luận cuối cùng.

---

## 1. Ba bảng dữ liệu cho ba loại bài toán

| Bảng | 1 dòng là | Dùng cho | Trạng thái |
|---|---|---|---|
| **A. `episode_features.csv`** | 1 đợt KM của 1 mã hàng tại 1 cửa hàng | Two-way ANOVA, t-test, Chi-square, Logistic, Elasticity cấp đợt, Insight 1–3 | Đã dựng (184.822 đợt; 45.127 đợt đủ điều kiện) |
| **B. `placebo_episodes.csv`** | 1 đợt giả dài 14 ngày, **không có KM**, cùng cách tính với A | Nhóm đối chứng cho A | Đã dựng (188.603 đợt; 17.735 đợt đủ điều kiện) |
| **C. Panel ngày** | 1 ngày × 1 mã hàng × 1 cửa hàng | ML/DL dự báo nhu cầu, elasticity cấp ngày | Mới có đặc tả (mục 5), chưa dựng |

Tại sao cần bảng B: nếu chỉ so "sau KM" với "trước KM" thì không tách được phần sụt do khuyến mãi khỏi phần sụt tự nhiên (hồi quy về trung bình, sản phẩm đang dần ngừng bán). Mục 6 cho thấy phần sụt tự nhiên này **lớn hơn** phần sụt sau KM, nên thiếu nhóm đối chứng sẽ dẫn đến kết luận sai.

Script để tạo lại toàn bộ:

```
python src/build_episode_features.py   # tạo bảng A, B vào output/features/
python src/prelim_analysis.py          # in các kết quả ở mục 6
```

---

## 2. Các định nghĩa then chốt

| Khái niệm | Định nghĩa | Lý do |
|---|---|---|
| **Đợt KM** | Gộp các `doc_id` chồng lấn hoặc liền kề của cùng mã hàng – cửa hàng trong `discounts_history` (chỉ lấy ngày ≤ 26/09/2024) | 251.738 bộ doc × mã hàng × cửa hàng → 215.871 đợt sau khi gộp. Mỗi đợt thường dài 14 ngày |
| **Cửa sổ trước / sau** | `W = 14` ngày ngay trước ngày bắt đầu / ngay sau ngày kết thúc | Bằng độ dài trung vị của một đợt |
| **Đợt sạch** (`clean`) | Không có ngày KM nào khác của cùng mã hàng – cửa hàng trong cửa sổ trước và sau | 81% số đợt là sạch. Đợt không sạch thì "sau" của đợt này chính là "trước" của đợt kế |
| **Hàng bán đều** | Có bán ít nhất 7/14 ngày trong cửa sổ trước (`pre_days ≥ 7`) | Sales không ghi dòng cho ngày bán 0. Với hàng bán đều, ngày không có dòng được coi là 0 |
| **Đủ điều kiện** (`eligible`) | `clean` + hàng bán đều + thuộc nhóm `food` hoặc `household` | Mẫu dùng cho mọi kiểm định |
| **Nhóm hàng** (`group`) | `household` = 9 `dept_name` tiêu dùng phi thực phẩm (chăm sóc cá nhân, giấy, đồ gia dụng, tẩy rửa, giặt…); `other` = thuốc lá, hoa, đồ chơi, thức ăn thú cưng, nhóm kỹ thuật, nhóm "НЕ ИСПОЛЬЗОВАТЬ" (không sử dụng), mã hàng không có trong catalog; `food` = phần còn lại (thực phẩm và đồ uống) | Danh sách nằm trong hằng số `HOUSEHOLD` và `OTHER` ở đầu script, nhóm có thể sửa |
| **Ngày lễ** | Lịch lễ Nga: 01/01, 23/02, 08/03, 01/05, 09/05, 12/06, 04/11 | **Giả thuyết**: suy ra từ chữ Cyrillic và các đỉnh doanh số 30/12/2023, 07/03/2024. Cần đối chiếu với nguồn dataset |
| **Sụt giảm sau KM** (`dip`) | `post_ratio < 0,9`, tức lượng bán trung bình mỗi ngày sau KM thấp hơn trước KM trên 10% | Biến nhị phân cho Chi-square và Logistic. Biến liên tục tương ứng là `log_post_ratio` |
| **Quantity âm** | Cộng ròng theo ngày (hàng trả lại trừ vào lượng bán) | Không xóa dòng |

---

## 3. Từ điển feature (bảng A; bảng B có cùng cột trừ các cột về KM)

Chú thích cột "Dùng cho": **AN** = two-way ANOVA · **T** = t-test · **CHI** = Chi-square · **LR** = Logistic · **EL** = Elasticity · **ML** = ML/DL · **I1, I2, I3** = Insight 1, 2, 3.

### 3.1 Khóa và nhận diện

| Feature | Định nghĩa | Nguồn | Dùng cho |
|---|---|---|---|
| `item_id`, `store_id` | Khóa | sales, discounts | Mọi bảng (không đưa vào mô hình như biến số) |
| `episode` | Số thứ tự đợt KM của mã hàng – cửa hàng | tự tính | Khóa |
| `start`, `end` | Ngày đầu, ngày cuối đợt | discounts | Tạo các cửa sổ |
| `source` | `promo` / `placebo` | tự tính | AN (nhân tố), T, CHI |

### 3.2 Đặc trưng khuyến mãi

| Feature | Định nghĩa | Dùng cho |
|---|---|---|
| `duration`, `promo_days` | Độ dài đợt (ngày theo lịch / ngày có bản ghi KM) | LR, ML, I3 |
| `n_docs` | Số chứng từ được gộp vào đợt | Kiểm tra chất lượng |
| `price_before`, `price_promo` | Trung vị `sale_price_before_promo` / `sale_price_time_promo` | EL, I3 |
| `depth` | 1 − giá KM / giá trước KM (chỉ tính khi giá trước > 0) | LR, EL, ML, I3 |
| `zero_depth` | `depth == 0`: KM không giảm giá (17% số đợt; mã loại 20 gần như toàn bộ) | Lọc mẫu, LR |
| `promo_type` | Mã loại KM xuất hiện nhiều nhất trong đợt; `missing` nếu trống | LR, ML, I1 |
| `log_price_before` | ln(giá trước KM): mức giá của sản phẩm | LR, EL |
| `log_price_ratio` | ln(giá KM / giá trước KM), < 0 khi có giảm giá | **EL** (biến giải thích chính) |

### 3.3 Thời điểm và ngày lễ (nối với Darbanian et al., 2025)

| Feature | Định nghĩa | Dùng cho |
|---|---|---|
| `hol_status` | `holiday` (đợt KM chứa ngày lễ) > `pre_holiday` (chạm vào 7 ngày trước lễ) > `post_holiday` (chạm vào 7 ngày sau lễ) > `normal`; ưu tiên theo thứ tự này | AN (bản tái lập thiết kế 6 nhãn của paper), LR, I1 |
| `days_to_next_holiday` | Số ngày từ ngày bắt đầu đợt đến ngày lễ kế tiếp | LR, ML |
| `start_month`, `start_dow`, `year` | Tháng, thứ trong tuần, năm của ngày bắt đầu | Biến kiểm soát mùa vụ |
| `store4_period` | 1 nếu đợt bắt đầu từ 13/12/2023, khi cửa hàng 4 xuất hiện | Kiểm tra độ vững (thay cho tách mốc COVID như paper) |

### 3.4 Sản phẩm và cửa hàng

| Feature | Định nghĩa | Dùng cho |
|---|---|---|
| `group`, `is_food` | Nhóm hàng (mục 2) | **AN, T, CHI, LR, EL**, I2, I3 |
| `paper_group` | `fresh_fv` / `frozen_fv`: rau quả tươi / đông lạnh | Đối chiếu trực tiếp với paper |
| `dept_name`, `class_name`, `subclass_name` | 3 cấp phân loại hàng | Hiệu ứng cố định trong mô hình; ma trận ngành hàng × KM của I1 |
| `item_type`, `weight_volume`, `weight_netto` + cờ `*_missing` | Thuộc tính sản phẩm, thiếu nhiều (62–80%) | ML (giữ nguyên NaN cho mô hình cây) |
| `catalog_missing` | Mã hàng không có trong catalog | Lọc mẫu |
| `division`, `format`, `city`, `area` | Đặc trưng cửa hàng | Biến kiểm soát trong LR, EL, ML |
| (`fatness` đã bị loại) | Thiếu 96,7% và chỉ áp dụng cho một phần thực phẩm | — |

### 3.5 Kết quả bán hàng (biến phụ thuộc)

| Feature | Định nghĩa | Dùng cho |
|---|---|---|
| `pre_qty`, `during_qty`, `post_qty` | Lượng bán trung bình mỗi ngày trong cửa sổ trước / trong đợt / cửa sổ sau | Mô tả, I1 |
| `pre_rev`, `during_rev`, `post_rev` | Như trên nhưng tính theo `sum_total` | I3 |
| `pre_days`, `post_days` | Số ngày có bán trong cửa sổ trước / sau | Điều kiện chọn mẫu, LR |
| `lift` = during / pre | Mức tăng trong KM | I1, I2 |
| `log_lift` | ln((during + c) / (pre + c)), với c = 1/14 | **EL** (biến phụ thuộc), I2 |
| `post_ratio` = post / pre | Mức duy trì sau KM | I1, I2 |
| `log_post_ratio` | ln((post + c) / (pre + c)) | **AN, T** (biến phụ thuộc) |
| `dip` | `post_ratio < 0,9` | **CHI, LR** (biến phụ thuộc) |
| `incr_rev` | (during_rev − pre_rev) × duration + (post_rev − pre_rev) × 14 | **I3**: doanh thu tăng thêm thực sự |
| `arc_elasticity` | log_lift / log_price_ratio | EL (bản tính cho từng đợt) |

Cờ chọn mẫu: `clean`, `eligible`.

**Không dùng làm feature giải thích:** `sum_total` (gần bằng lượng × giá, sẽ rò rỉ thông tin vào biến phụ thuộc). `price_base` của chính ngày đang dự báo: vào ngày KM nó đã bằng giá KM (trung vị tỷ lệ = 1,00), có thể được tính ngược từ `sum_total / quantity`; khi dự báo nên dùng giá từ `discounts_history` / `price_history`. `actual_matrix` quá thưa (35 nghìn dòng) để suy ra mặt hàng nào đang bán. `markdowns` chỉ có 313 mã hàng, chỉ nên dùng để đối chiếu.

---

## 4. Ghép phương pháp với feature

| Phương pháp | Câu hỏi | Biến phụ thuộc | Nhân tố / biến giải thích | Mẫu | Ghi chú |
|---|---|---|---|---|---|
| **Two-way ANOVA** | Mức sụt sau KM có khác nhau giữa Thực phẩm và Hàng tiêu dùng không? | `log_post_ratio` | `group` × `source` (KM thật / đợt giả) | A + B, đủ điều kiện | **Tương tác `group:source`** mới trả lời câu hỏi (mức sụt *do KM* khác nhau giữa nhóm). Dùng sai số chuẩn HC3 vì phương sai không đều. Bản phụ: `group` × `hol_status` để tái lập thiết kế 6 nhãn của paper |
| **T-test** | Trong từng nhóm, doanh số sau KM có giảm có ý nghĩa không? | `log_post_ratio` | KM thật vs đợt giả (Welch) | A + B, theo từng nhóm | **Không dùng t-test 1 mẫu so với 0**, vì đợt giả cũng giảm (mục 6.2). Kèm Mann–Whitney làm kiểm tra độ vững và Cohen's d làm cỡ hiệu ứng |
| **Chi-square** *(đề xuất câu hỏi)* | Tỷ lệ đợt KM bị sụt có phụ thuộc vào nhóm hàng không? Có phụ thuộc vào `hol_status` / `promo_type` không? | `dip` | `group`; `hol_status`; `promo_type` | A | Báo cáo kèm Cramér's V, vì với mẫu lớn p-value gần như luôn rất nhỏ |
| **Logistic nhị phân** | "Là thực phẩm" có làm tăng rủi ro đợt KM bị sụt sau khi kết thúc không? | `dip` | `is_food`, `depth`, ln(`duration`), `log_price_before`, `pre_days`, `promo_type`, `hol_status`, `store_id`, `start_month` | A (bản chính thức: A + B, thêm `is_food × source`) | Báo cáo odds ratio + khoảng tin cậy 95%, sai số chuẩn vững |
| **Log-log elasticity** | Giảm giá 1% thì lượng bán mỗi nhóm tăng bao nhiêu %? | `log_lift` | `log_price_ratio` × `group` + `store_id` + `start_month` | A, `depth > 0` | Hệ số của `log_price_ratio` là độ co giãn. Bản cấp ngày: ln(q) ~ ln(giá) × group + hiệu ứng cố định mã hàng – cửa hàng + lịch (bảng C) |
| **ML/DL** | Dự báo lượng bán theo ngày | `quantity` (bảng C) | Mục 5 | C | Mô phỏng nhiều mức `depth` để phục vụ I3 |

---

## 5. Đặc tả bảng C: panel ngày cho ML/DL

Phạm vi: các cặp mã hàng – cửa hàng bán đều (ví dụ ≥ 50% số ngày trong giai đoạn đang hoạt động), hoặc gộp lên cấp `subclass_name` × cửa hàng × ngày để giảm số ngày bán 0.

| Nhóm | Feature | Biết trước tại thời điểm dự báo? |
|---|---|---|
| Biến phụ thuộc | `quantity` (cộng ròng theo ngày; ngày không có dòng = 0 với hàng bán đều); có thể dùng log1p | — |
| Lịch | thứ trong tuần, ngày trong tháng, tuần, tháng, cuối tuần, `is_holiday`, `days_to_holiday`, `days_after_holiday`, thời kỳ nghỉ Tết 01–08/01 | Có |
| KM đã lên kế hoạch | `promo_flag`, `depth`, `promo_type`, `day_in_promo`, `days_to_promo_end`, `days_since_last_promo`, `promo_duration` | Có (nhà bán lẻ tự lên lịch KM; đây là gap đã nêu từ bài Hierarchical) |
| Giá | giá thường (giá trước KM từ discounts / price_history), giá KM, `code` của price_history (nhóm 9999999999 riêng) | Có |
| Lịch sử bán | lag 7/14/28 ngày (≥ tầm dự báo); trung bình, độ lệch chuẩn, tổng trượt 7/28 ngày đã dịch ít nhất 1 ngày; lượng bán trung bình ở các đợt KM trước | Không, chỉ dùng quá khứ |
| Tĩnh | `dept/class/subclass_name`, `group`, `item_type`, trọng lượng + cờ thiếu, `store_id`, `format`, `area`, `city` | Có |
| Kênh online | lag của `online.quantity` (chỉ CH1, CH4) | Không, chỉ dùng quá khứ |

**Chia dữ liệu:** theo thời gian. Ví dụ train đến 31/03/2024, validation 04–06/2024, test 07–26/09/2024. Kiểm định chéo theo thời gian (rolling origin) để tune tham số, nối với gap "chưa tune" trong file Insight. Mọi phép điền thiếu / mã hóa chỉ được học từ tập train.
**Mô hình:** baseline seasonal naive (cùng thứ tuần trước) → LightGBM / XGBoost → LSTM hoặc TFT. Máy hiện **chưa cài `lightgbm`**.
**Đánh giá:** WAPE, RMSE, MAE; tách riêng sai số ngày có KM và ngày không KM, và theo `group`.
**Phục vụ I3:** cố định các feature khác, thay `depth` theo lưới 0–40% (khoảng quan sát được: trung vị 20%, P95 42%; không ngoại suy ra ngoài), chọn mức có `incr_rev` cao nhất. **Chỉ tối ưu được doanh thu**, vì dataset không có giá vốn. LLM chỉ nên dùng để diễn giải kết quả, không dùng để dự báo.

---

## 6. Kết quả sơ bộ

### 6.1 Quy mô mẫu (đợt đủ điều kiện)

| | Thực phẩm | Hàng tiêu dùng |
|---|---|---|
| Đợt KM thật | 43.615 (6.110 mã hàng) | 1.512 (391 mã hàng) |
| Đợt giả (đối chứng) | 16.792 | 943 |

`hol_status` của đợt KM: normal 25.636 · holiday 11.999 · pre_holiday 3.937 · post_holiday 3.555, nên **đủ mẫu để tái lập thiết kế của paper**. `promo_type`: loại 5 có 24.452 đợt, loại 6 có 16.468, loại 8 có 2.402; các loại còn lại dưới 1.000 đợt.

### 6.2 Mô tả

| | lift (trung vị) | post_ratio (trung vị) | Tỷ lệ dip | Trung bình log_post_ratio |
|---|---|---|---|---|
| Thực phẩm – KM | **1,42** | 0,92 | 47,4% | −0,170 |
| Thực phẩm – giả | 0,93 | 0,90 | 49,5% | −0,293 |
| Tiêu dùng – KM | **1,12** | 0,83 | 55,7% | −0,277 |
| Tiêu dùng – giả | 0,86 | 0,82 | 57,6% | −0,337 |

**Phát hiện quan trọng nhất:** ngay cả khi **không có KM**, lượng bán 14 ngày sau vẫn thấp hơn 14 ngày trước (đợt giả: −0,29 và −0,34 theo log). Nguyên nhân là chọn mẫu theo cửa sổ trước (hồi quy về trung bình) và vòng đời sản phẩm. Vì vậy so sánh sau/trước đơn thuần **phóng đại** hiện tượng sụt sau KM. Sau khi trừ đi nhóm đối chứng, các đợt KM **sụt ít hơn** mức nền.

### 6.3 Two-way ANOVA (`log_post_ratio ~ group × source`, HC3)

| Nguồn | F | p | partial η² |
|---|---|---|---|
| group | 34,16 | < 0,001 | 0,0005 |
| source | 261,98 | < 0,001 | 0,0042 |
| **group × source** | **3,54** | **0,060** | 0,0001 |

→ Chưa đủ bằng chứng (ở mức 0,05) rằng mức sụt *do KM* khác nhau giữa hai nhóm. Mọi cỡ hiệu ứng đều rất nhỏ (η² < 0,5%), tức phần lớn biến động nằm ở cấp từng sản phẩm.

### 6.4 T-test (KM thật vs đợt giả, Welch)

| Nhóm | Chênh lệch log (KM − giả) | Welch t | p | Cohen d | Mann–Whitney p |
|---|---|---|---|---|---|
| Thực phẩm | +0,123 (≈ +13%) | 16,19 | 1,2e-58 | 0,155 | 7,7e-22 |
| Tiêu dùng | +0,060 (≈ +6%) | 1,86 | 0,063 | 0,079 | 0,50 |

→ Thực phẩm: sau KM lượng bán **cao hơn** mức nền khoảng 13%, tức **không có sụt sau KM**, ngược lại còn có dấu hiệu duy trì. Hàng tiêu dùng: không khác nhóm đối chứng. Nếu dùng t-test 1 mẫu so với 0 (cách làm đơn thuần), cả hai nhóm đều "giảm có ý nghĩa" (p ≈ 0), và đó là **kết luận sai**.

### 6.5 Chi-square

- Nhóm hàng × dip (đợt KM thật): χ² = 39,89, p = 2,7e-10, **Cramér's V = 0,030**. Tỷ lệ dip của thực phẩm 47,4%, hàng tiêu dùng 55,7%.
- Nhưng ở đợt giả, khoảng cách gần như y hệt (49,5% so với 57,6%). → Khác biệt này là **đặc điểm sẵn có của nhóm hàng, không phải do KM**.

### 6.6 Logistic (`dip`, n = 44.983, pseudo R² = 0,009)

| Biến | OR | KTC 95% | p |
|---|---|---|---|
| is_food | **0,646** | 0,581 – 0,717 | 3e-16 |
| depth (trên mỗi 1,0 = 100 điểm %) | 3,79 (≈ 1,14 cho mỗi 10 điểm %) | 3,16 – 4,54 | 2,6e-47 |
| ln(duration) | 1,11 | 1,05 – 1,18 | 0,0004 |
| log_price_before | 1,10 | 1,07 – 1,12 | 7,5e-15 |
| pre_days | 1,01 | 1,005 – 1,021 | 0,002 |

→ Thực phẩm có odds bị sụt thấp hơn 35%. **Giảm giá càng sâu thì rủi ro sụt sau KM càng cao**, phù hợp với giả thuyết khách mua dồn. Pseudo R² rất thấp: mô hình phù hợp để giải thích, không phù hợp để dự báo. Theo mục 6.5, bản chính thức cần đưa đợt giả vào và dùng tương tác `is_food × source`.

### 6.7 Log-log elasticity (cấp đợt, n = 31.762 đợt có giảm giá)

| Nhóm | Độ co giãn | Diễn giải |
|---|---|---|
| Thực phẩm | **−1,34** (p ≈ 0) | Giảm giá 1% → lượng bán tăng khoảng 1,34% |
| Hàng tiêu dùng | **−0,83** | Giảm giá 1% → lượng bán tăng khoảng 0,83% |
| Chênh lệch | p = 0,029 | Có ý nghĩa ở mức 0,05 |

Trung vị arc elasticity từng đợt: thực phẩm −2,29, tiêu dùng −1,49.

### 6.8 Ý nghĩa cho các Insight

- **Insight 2** (thực phẩm vs hàng tiêu dùng): thực phẩm phản ứng mạnh hơn trong KM (lift 1,42 so với 1,12; độ co giãn −1,34 so với −0,83). Giả thuyết "hàng tiêu dùng sụt rõ hơn sau KM" **chưa được ủng hộ** sau khi có nhóm đối chứng (tương tác p = 0,06). Nên dùng partial η² từ mô hình ở mục 6.3 thay cho η² tính trên quantity thô.
- **Insight 1** (lịch bổ sung hàng): đủ mẫu để làm ma trận `dept_name` × `hol_status` / `promo_type`. Cần hiệu chỉnh đa kiểm định bằng Benjamini–Hochberg như paper.
- **Insight 3** (mức giảm tối ưu): đã có `incr_rev` và `depth`. Mức giảm sâu làm tăng rủi ro sụt (OR 1,14 cho mỗi 10 điểm %), nên mức tối ưu theo doanh thu tăng thêm thực sự có thể không phải mức giảm sâu nhất. Cần mô hình ở mục 5 để mô phỏng.

---

## 7. Hạn chế và việc tiếp theo

1. Đợt giả được chọn ngẫu nhiên theo ngày, chưa ghép cặp. Chỉ khoảng một nửa số mã hàng trùng với đợt KM thật (thực phẩm 3.282/6.110; tiêu dùng 163/391). Bước tiếp theo: ghép cặp đợt giả theo cùng mã hàng hoặc cùng `subclass` và cùng tháng, hoặc dùng difference-in-differences với hiệu ứng cố định mã hàng.
2. Ngưỡng "hàng bán đều" và ngưỡng dip (0,9) là lựa chọn của nhóm. Cần chạy độ nhạy với `W = 7`, `MIN_PRE_DAYS ∈ {3, 7, 10}`, ngưỡng dip ∈ {0,8; 0,9; 1,0}.
3. Lịch lễ Nga là giả thuyết, cần xác nhận với mô tả dataset trên Kaggle.
4. Nhóm hàng tiêu dùng nhỏ (1.512 đợt, 391 mã hàng; 2,1% doanh thu). Cần nêu rõ trong phần hạn chế.
5. KM có thể được chọn có chủ đích (cửa hàng giảm sâu cho những mặt hàng nhất định), nên độ co giãn đo được là tương quan, chưa chắc là nhân quả.
6. Chưa dựng bảng C và chưa huấn luyện ML/DL. Cần cài `lightgbm`.
