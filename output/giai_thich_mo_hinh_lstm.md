# Giải thích mô hình LSTM dự báo nhu cầu — dành cho người mới

File này giải thích **từ đầu** mọi thứ trong báo cáo `bao_cao_lstm.md`: bài toán là gì, dữ liệu gồm những biến nào, mô hình hoạt động ra sao, các thuật ngữ nghĩa là gì, và nên đọc các con số kết quả như thế nào. Không cần biết trước về thống kê hay học máy.

**Cách đọc:** đọc lần lượt từ mục 1 đến mục 9. Gặp thuật ngữ lạ thì tra **mục 10 (Từ điển thuật ngữ)** ở cuối file.

---

## 1. Bài toán là gì?

Hãy hình dung một chuỗi siêu thị có **4 cửa hàng**. Mỗi ngày, quản lý phải quyết định nhập bao nhiêu hàng cho từng mặt hàng:
- Nhập **quá ít**: hết hàng, mất khách.
- Nhập **quá nhiều**: tồn kho, hàng tươi bị hỏng.

Để quyết định tốt, cần biết trước **mỗi mặt hàng ở mỗi cửa hàng sẽ bán được bao nhiêu trong 7 ngày tới**. Đó chính là bài toán **dự báo nhu cầu**.

Ví dụ: "Táo ở cửa hàng 1, từ thứ Hai đến Chủ nhật tuần sau, mỗi ngày sẽ bán khoảng 12, 10, 11, 9, 15, 20, 18 kg."

Bài toán khó vì lượng bán thay đổi theo nhiều thứ: ngày trong tuần (cuối tuần đông hơn), ngày lễ, **khuyến mãi (KM)**, giá, mùa vụ… Mô hình phải học được các quy luật đó từ dữ liệu quá khứ.

**Mô hình LSTM** trong báo cáo là một mô hình học sâu (deep learning) được huấn luyện để làm việc này cho **4.063 cặp "mặt hàng × cửa hàng"** cùng lúc.

---

## 2. Dữ liệu trông như thế nào?

### 2.1 Một dòng dữ liệu

Bảng dữ liệu chính (`daily_panel.parquet`) có **1.197.327 dòng**. Mỗi dòng là:

> **1 mặt hàng × 1 cửa hàng × 1 ngày**

Ví dụ một dòng: *"Mã hàng 12345 (táo), cửa hàng 1, ngày 05/07/2024: bán 11,3 kg, không có KM, giá thường 150, là thứ Sáu…"*

- Dữ liệu kéo dài từ **28/08/2022 đến 26/09/2024** (761 ngày).
- **4.063 cặp mặt hàng × cửa hàng** thuộc 4 nhóm hàng: rau quả tươi, thịt, chăm sóc cá nhân / giấy, giặt giũ / tẩy rửa.

### 2.2 Biến cần dự báo: `quantity` (lượng bán)

| Đặc điểm | Giải thích |
|---|---|
| Ý nghĩa | Lượng hàng bán ra trong ngày, sau khi đã trừ hàng bị trả lại |
| Đơn vị | Cái / gói, hoặc **kg** với hàng cân (rau quả, thịt). Vì vậy có số lẻ như 0,632 |
| Giá trị 0 | Ngày đó mặt hàng có bán ở cửa hàng nhưng không ai mua |
| "Không có dữ liệu" | Ngày mặt hàng chưa được bán hoặc đã ngừng bán ở cửa hàng đó. Những ngày này **không được tính** khi đánh giá |
| Phân bố | Trung vị 0,63, trung bình 4,3, lớn nhất 988. Nghĩa là phần lớn các ngày bán rất ít, vài ngày bán rất nhiều |

### 2.3 Hai nhóm hàng lớn và 4 nhóm con

| Nhóm lớn | Mã | Nhóm con | Đặc điểm |
|---|---|---|---|
| **Thực phẩm** (food) | F1 | Rau quả tươi | Dễ hỏng, mua thường xuyên, bán nhiều (khoảng 8 đơn vị / ngày) |
| | F2 | Thịt tươi, thịt chế biến ướp lạnh | |
| **Đồ tiêu dùng** (household) | H1 | Chăm sóc cá nhân, giấy vệ sinh | Để được lâu, có thể mua tích trữ, bán **thưa** (khoảng 1 đơn vị / ngày, hơn một nửa số ngày bán 0) |
| | H2 | Giặt giũ, tẩy rửa | |

"Bán thưa" là một chi tiết rất quan trọng, sẽ quay lại ở mục 7.

---

## 3. Từ điển biến (feature) mà mô hình sử dụng

**Feature** (đặc trưng, biến đầu vào) là những thông tin mô hình được nhìn thấy để đoán lượng bán. Có thể hình dung đó là những gì một người quản lý giàu kinh nghiệm sẽ xem trước khi đặt hàng.

Điểm quan trọng: chỉ dùng thông tin **biết được tại thời điểm dự báo**. Ví dụ, lịch KM tuần sau đã được lên kế hoạch nên được dùng; còn doanh thu tuần sau thì chưa biết nên không được dùng.

### 3.1 Lịch sử bán hàng

| Biến | Ý nghĩa đời thường | Giá trị | Vì sao giúp dự báo |
|---|---|---|---|
| Lượng bán quá khứ, dạng `log1p(quantity)` | Lượng bán của mỗi ngày trong 56 ngày gần nhất | Số ≥ 0, đã qua phép biến đổi log1p (xem mục 10) | Tuần trước bán nhiều thì tuần sau thường cũng bán nhiều. Đây là thông tin quan trọng nhất |
| Cờ có dữ liệu | Ngày đó có số liệu hay không | 1 = có, 0 = không | Để mô hình phân biệt "bán 0" với "không có số liệu" |

### 3.2 Khuyến mãi (biết trước vì siêu thị tự lên lịch)

| Biến | Ý nghĩa đời thường | Giá trị | Ví dụ |
|---|---|---|---|
| `promo_status` | Trạng thái KM của ngày | **1** = KM có giảm giá · **0** = không KM · **trống** = có KM nhưng không giảm giá (có thể là KM trưng bày) | Trong mô hình được tách thành 2 cờ `promo_disc` (KM có giảm giá) và `promo_nodisc` (KM không giảm giá) |
| `depth` | **Mức giảm giá** = 1 − giá KM / giá trước KM | Từ 0 đến 0,9 (0 khi không giảm). Trung vị trong các ngày giảm giá khoảng 0,20 = **giảm 20%** | Giá 100 giảm còn 80 → depth = 0,2 |
| `depth × food` | Mức giảm giá, nhưng chỉ tính cho hàng thực phẩm (đồ tiêu dùng = 0) | 0 đến 0,9 | Cho phép mô hình học rằng thực phẩm và đồ tiêu dùng phản ứng khác nhau với cùng mức giảm |
| `price_ratio` | Tỷ lệ giá đang bán / giá thường | Thường từ 0,2 đến 1,5; bằng 1 khi không giảm | Giảm 20% → 0,8 |
| `promo_type` | Loại chương trình KM, là mã số trong dữ liệu gốc | 5, 6, 10, 8, hoặc "khác". Mã 5 phổ biến nhất | Dữ liệu gốc không giải thích từng mã nghĩa là gì, mô hình tự học ảnh hưởng của mỗi mã |
| Tiến độ đợt KM | Đang ở ngày thứ mấy của đợt KM, tính theo phần trăm | 0 đến 1 | Ngày 7 của đợt 14 ngày → 0,5 |
| `days_to_promo_end` | Còn bao nhiêu ngày nữa thì KM kết thúc | 0 đến 30 ngày (quy về 0–1) | Khách hay mua dồn ngày cuối |
| `promo_duration` | Đợt KM dài bao nhiêu ngày | 0 đến 60 ngày (quy về 0–1) | Đợt KM thường dài 14 ngày |

### 3.3 Hậu khuyến mãi (sau khi KM kết thúc)

| Biến | Ý nghĩa đời thường | Giá trị |
|---|---|---|
| `days_since_promo` | Đã bao nhiêu ngày kể từ lần KM gần nhất | 0 đến 60 ngày (quy về 0–1) |
| `never_promo` | Mặt hàng này chưa từng có KM | 1 = đúng, 0 = sai |
| `post_promo_1_7` | Đang ở trong 7 ngày đầu sau KM | 1 / 0 |
| `post_promo_8_14` | Đang ở ngày thứ 8–14 sau KM | 1 / 0 |

**Vì sao cần:** sau KM, khách có thể đã mua tích trữ nên lượng bán tạm giảm. Gọi là **sụt giảm sau KM** (post-promotion dip).

### 3.4 Giá

| Biến | Ý nghĩa | Giá trị |
|---|---|---|
| `log_reg_price` | Mức giá thường của mặt hàng, sau khi lấy log và chuẩn hóa (xem mục 10) | Số thực, 0 ≈ mức giá trung bình |
| Cờ thiếu giá | Không tìm được giá thường | 1 / 0 (khoảng 10% số dòng thiếu giá) |

Hàng đắt thường bán ít hơn hàng rẻ.

### 3.5 Lịch: thứ, ngày, tháng

| Biến | Ý nghĩa | Giá trị |
|---|---|---|
| `dow_0` … `dow_6` | Thứ trong tuần, mã hóa **one-hot** (xem mục 10) | 7 cột, đúng 1 cột bằng 1. `dow_0` = thứ Hai, …, `dow_6` = Chủ nhật |
| `is_weekend` | Có phải thứ Bảy / Chủ nhật | 1 / 0 |
| `is_friday` | Có phải thứ Sáu | 1 / 0 |
| `dom_sin`, `dom_cos` | Ngày trong tháng, mã hóa vòng tròn (xem mục 10) | −1 đến 1 |
| `mon_sin`, `mon_cos` | Tháng trong năm, mã hóa vòng tròn | −1 đến 1 |

### 3.6 Ngày lễ

Dữ liệu gốc dùng chữ Nga, nên nhóm **giả định** cửa hàng theo lịch lễ Nga. Đây là giả thuyết suy ra từ chữ Cyrillic và các đỉnh doanh số.

| Biến | Ý nghĩa | Giá trị |
|---|---|---|
| `is_holiday` | Chính ngày lễ lớn: 01/01, 23/02, 08/03 | 1 / 0 |
| Trước lễ (`hol_pre`) | Mấy ngày ngay trước lễ (27–31/12; 1–3 ngày trước 23/02 và 08/03) | 1 / 0 |
| Sau lễ (`hol_post`) | Mấy ngày ngay sau lễ (02–08/01; 1–3 ngày sau 23/02 và 08/03) | 1 / 0 |
| `minor_holiday` | Lễ nhỏ: 01/05, 09/05, 12/06, 04/11 | 1 / 0 |
| `ny_break` | Kỳ nghỉ Tết Dương lịch 01–08/01 | 1 / 0 |
| `days_to_holiday` | Còn bao nhiêu ngày đến lễ kế tiếp | 0 đến 30 (quy về 0–1) |
| `days_after_holiday` | Đã qua lễ bao nhiêu ngày | 0 đến 30 (quy về 0–1) |

### 3.7 Thay đổi cơ cấu

| Biến | Ý nghĩa | Giá trị |
|---|---|---|
| `pheic` | Giai đoạn COVID-19 còn được WHO coi là tình trạng khẩn cấp (đến 05/05/2023) | 1 / 0 |
| `store4_period` | Giai đoạn từ khi cửa hàng 4 mở (13/12/2023) | 1 / 0 |

Hành vi mua sắm có thể khác giữa các giai đoạn này.

### 3.8 Thông tin cố định (tĩnh) của mặt hàng và cửa hàng

| Biến | Ý nghĩa | Giá trị |
|---|---|---|
| `store_id` | Cửa hàng nào | 1, 2, 3, 4 |
| `group4` | Nhóm hàng con | F1, F2, H1, H2 |
| `dept_name` | Ngành hàng (cấp phân loại lớn) | 11 ngành, ví dụ trái cây, rau củ, thịt tươi, gia cầm… |
| `class_name` | Lớp hàng (cấp phân loại nhỏ hơn) | Nhiều giá trị |
| `format` | Mô hình cửa hàng | Format-1, MaxiEuro, Format-6, Format-7 express (tên lấy từ dữ liệu gốc) |

Các biến này không đổi theo ngày. Mô hình biến chúng thành số bằng kỹ thuật **embedding** (xem mục 10).

### 3.9 Những biến KHÔNG dùng, và lý do

| Biến | Lý do không dùng |
|---|---|
| `revenue` (doanh thu) | Doanh thu ≈ lượng × giá. Dùng nó chẳng khác nào biết trước đáp án (**rò rỉ dữ liệu**) |
| `qty_norm`, `density` | Được tính từ toàn bộ dữ liệu, kể cả tương lai |
| `phase`, `status6` | Nhãn tạo ra để phân tích, không phải thông tin dùng để dự báo |
| `price_base` | Vào ngày KM, giá này đã bằng giá KM và có thể tính ngược từ doanh thu |

---

## 4. Chia dữ liệu: học, ôn thi thử và thi thật

Giống như một học sinh, mô hình cần 3 bộ đề khác nhau:

| Tập | Thời gian | Giống như | Dùng để |
|---|---|---|---|
| **Train** (huấn luyện) | 28/08/2022 – 31/03/2024 | Sách bài tập có lời giải | Mô hình học quy luật |
| **Valid** (kiểm định) | 01/04 – 30/06/2024 | Đề thi thử | Chọn cấu hình tốt nhất và quyết định khi nào dừng học |
| **Test** (kiểm tra) | 01/07 – 26/09/2024 | Đề thi thật | Chấm điểm cuối cùng, **chỉ làm 1 lần** |

**Vì sao chia theo thời gian, không xáo trộn ngẫu nhiên?** Trong thực tế, ta luôn dùng quá khứ để đoán tương lai. Nếu xáo trộn, mô hình có thể "nhìn thấy" ngày 10/07 khi học rồi được hỏi về ngày 09/07, tức là gian lận.

**Vì sao test chỉ dùng 1 lần?** Nếu cứ sửa mô hình rồi xem lại điểm test, ta vô tình chỉnh mô hình cho "trúng tủ" đề thi thật, và điểm sẽ đẹp hơn thực tế.

### Cách chấm điểm theo khối tuần

Mô hình dự báo 7 ngày một lần. Trên valid và test, cứ mỗi 7 ngày ta đặt một **mốc dự báo** (origin): "đứng ở tối Chủ nhật, đoán thứ Hai đến Chủ nhật tới". Như vậy mỗi ngày trong valid / test được dự báo **đúng 1 lần**, ở **tầm** (horizon) từ 1 đến 7 ngày:
- Tầm 1 = đoán ngày mai.
- Tầm 7 = đoán 7 ngày sau.

---

## 5. Mô hình LSTM hoạt động thế nào?

### 5.1 Ý tưởng đời thường

Hãy tưởng tượng một quản lý kho rất giỏi làm việc theo 2 bước:
1. **Đọc lại sổ sách 56 ngày qua** của mặt hàng: bán bao nhiêu mỗi ngày, có KM không, giá bao nhiêu, hôm đó thứ mấy… Đọc xong, anh ta **ghi nhớ một bản tóm tắt** trong đầu, ví dụ "mặt hàng này bán đều khoảng 10 kg/ngày, cuối tuần tăng, cứ KM là tăng gấp rưỡi, sau KM thì hụt vài ngày".
2. **Nhìn lịch 7 ngày tới:** ngày nào là cuối tuần, ngày nào có KM, giảm bao nhiêu, có lễ không. Kết hợp với bản tóm tắt trong đầu để **đoán lượng bán từng ngày**.

LSTM làm đúng 2 bước đó:

| Bộ phận | Vai trò | Tương ứng |
|---|---|---|
| **Encoder** (bộ mã hóa) | Đọc tuần tự 56 ngày quá khứ, nén lại thành một "trí nhớ" | Bước 1: đọc sổ sách |
| **Decoder** (bộ giải mã) | Nhận "trí nhớ" đó cùng thông tin biết trước của 7 ngày tới, rồi đưa ra 7 dự báo | Bước 2: nhìn lịch và đoán |
| **Embedding tĩnh** | Cho mô hình biết đây là mặt hàng loại gì, ở cửa hàng nào | Kinh nghiệm về từng loại hàng / cửa hàng |
| **Mức nền + Δ** | Mô hình lấy mức bán trung bình gần đây làm mốc, rồi chỉ học **phần chênh lệch** so với mốc | Giúp học nhanh và ổn định hơn |

### 5.2 Vì sao gọi là LSTM?

**LSTM = Long Short-Term Memory** (bộ nhớ dài – ngắn hạn). Đây là một loại **mạng nơ-ron hồi quy**: đọc dữ liệu theo thứ tự thời gian, từng ngày một, và có các "cổng" quyết định **điều gì cần nhớ lâu, điều gì nên quên**. Ví dụ: nhớ lâu rằng "mặt hàng này thường tăng mạnh khi KM", nhưng quên một ngày bán đột biến không rõ lý do.

### 5.3 Một mô hình cho tất cả (mô hình toàn cục)

Thay vì xây 4.063 mô hình riêng cho từng cặp mặt hàng × cửa hàng, ta xây **1 mô hình chung** học từ tất cả. Lợi ích: mặt hàng ít dữ liệu được "học ké" quy luật từ các mặt hàng tương tự (ví dụ mọi loại rau đều bán chạy cuối tuần).

### 5.4 Mô hình học như thế nào?

1. Lấy ngẫu nhiên một **lô** (batch) gồm 1.024 ví dụ. Mỗi ví dụ là một **cửa sổ**: 56 ngày quá khứ + 7 ngày cần đoán của một cặp mặt hàng × cửa hàng.
2. Mô hình đoán 7 ngày, so với thực tế, tính **mức sai** (loss).
3. Điều chỉnh các **tham số** (221.533 con số bên trong mô hình) một chút theo hướng giảm mức sai. Bước điều chỉnh to hay nhỏ do **learning rate** quyết định.
4. Lặp lại. Mỗi **epoch** (vòng học) gồm 150.000 cửa sổ được chọn ngẫu nhiên trong tổng số 816.119 cửa sổ của tập train.
5. Sau mỗi epoch, làm "đề thi thử" (valid) để xem mô hình có tiến bộ không.

### 5.5 Các mốc so sánh (baseline)

Để biết LSTM có thực sự giỏi không, cần so với những cách đoán đơn giản mà ai cũng làm được:

| Baseline | Cách đoán | Ví dụ |
|---|---|---|
| **MA7** (trung bình trượt 7 ngày) | Lấy trung bình 7 ngày gần nhất, dùng cho cả 7 ngày tới | 7 ngày qua bán 8, 10, 9, 11, 15, 20, 18 → trung bình 13 → đoán mỗi ngày tới bán 13 |
| **Seasonal naive** (ngây thơ theo mùa) | Thứ Hai tới bán bằng thứ Hai tuần này, thứ Ba tới bằng thứ Ba tuần này… | Thứ Bảy tuần này bán 20 → đoán thứ Bảy tới bán 20 |

Nếu một mô hình phức tạp không thắng nổi 2 cách này thì không đáng dùng.

---

## 6. Tinh chỉnh mô hình (tuning)

### 6.1 Siêu tham số là gì?

**Tham số** là những con số mô hình tự học (221.533 con số). **Siêu tham số** là những lựa chọn **con người phải đặt trước** khi cho mô hình học, giống như chọn phương pháp học cho học sinh:

| Siêu tham số | Ý nghĩa đời thường | Giá trị đã thử |
|---|---|---|
| **Lookback** (L) | Mô hình được xem lại bao nhiêu ngày quá khứ | 28 hoặc 56 ngày |
| **Hidden units** (số đơn vị ẩn) | Độ lớn "bộ não", tức trí nhớ chứa được bao nhiêu thông tin | 64 hoặc 128 |
| **Learning rate** (tốc độ học) | Mỗi lần sửa sai thì sửa mạnh tay hay nhẹ tay | 0,001 hoặc 0,002 |
| Loss (hàm mất mát) | Cách tính "mức sai" để mô hình cố gắng giảm | Cố định: MAE (xem mục 7) |
| Dropout | Tỷ lệ ngẫu nhiên "tắt" bớt một phần bộ não khi học, để mô hình không học vẹt | Cố định: 0,1 (10%) |
| Batch | Số ví dụ học mỗi lượt | Cố định: 1.024 |

### 6.2 Phương pháp: Manual Grid Search + Hold-out Validation + Early Stopping

**Manual Grid Search (tìm kiếm theo lưới, đặt lưới thủ công):** liệt kê trước mọi tổ hợp siêu tham số rồi thử **tất cả**. Ở đây có 2 lookback × 2 hidden × 2 learning rate = **8 tổ hợp**. Giống như thử 8 công thức nấu ăn rồi chọn món ngon nhất.

**Hold-out Validation (kiểm định bằng tập giữ lại):** mỗi tổ hợp được chấm điểm trên tập valid, tức dữ liệu mô hình **không được học**. Tổ hợp có điểm valid tốt nhất được chọn.

**Early Stopping (dừng sớm):** học càng lâu không phải càng tốt. Đến một lúc mô hình bắt đầu **học vẹt** (overfitting): làm bài tập rất giỏi nhưng làm đề mới lại kém đi. Vì vậy:
- Sau mỗi epoch, chấm điểm trên valid.
- Nếu **4 epoch liên tiếp** (gọi là **patience** = 4) điểm valid không tốt hơn, thì dừng.
- Lấy lại phiên bản mô hình ở epoch có điểm valid tốt nhất.

Ví dụ: cấu hình được chọn đạt tốt nhất ở epoch 22, chạy thêm 4 epoch (23–26) không tiến bộ, nên dừng ở epoch 26 và dùng mô hình của epoch 22.

**Kết quả:** cả 8 tổ hợp đều dừng sớm. Tổ hợp được chọn là **lookback 56 ngày, 128 đơn vị ẩn, learning rate 0,002**, với WAPE valid = 0,508. Tuy nhiên 7/8 tổ hợp có điểm gần như bằng nhau (0,508–0,510), nghĩa là chọn tổ hợp nào trong số đó cũng cho kết quả tương tự.

---

## 7. Các chỉ số đánh giá: đọc con số thế nào?

### 7.1 Ví dụ nhỏ để hiểu công thức

Một mặt hàng, 3 ngày:

| Ngày | Thực tế | Dự báo | Sai số (dự báo − thực tế) | Sai số tuyệt đối |
|---|---|---|---|---|
| 1 | 10 | 8 | −2 | 2 |
| 2 | 0 | 1 | +1 | 1 |
| 3 | 5 | 5 | 0 | 0 |
| **Tổng** | **15** | **14** | **−1** | **3** |

| Chỉ số | Công thức | Kết quả ví dụ | Ý nghĩa |
|---|---|---|---|
| **WAPE** (chỉ số chính) | Tổng sai số tuyệt đối / tổng thực tế | 3 / 15 = **0,20** (20%) | Tổng lượng đoán sai bằng 20% tổng lượng bán. **Càng nhỏ càng tốt** |
| **MAE** | Trung bình sai số tuyệt đối | 3 / 3 = **1** | Trung bình mỗi ngày đoán lệch 1 đơn vị |
| **RMSE** | Căn bậc hai của trung bình bình phương sai số | √((4 + 1 + 0) / 3) ≈ **1,29** | Giống MAE nhưng **phạt nặng các lần sai lớn** |
| **Bias** (độ lệch) | Tổng dự báo / tổng thực tế − 1 | 14 / 15 − 1 = **−6,7%** | **Âm** = đoán thấp hơn thực tế; **dương** = đoán cao hơn. Càng gần 0 càng tốt |

### 7.2 Vì sao chọn WAPE làm chỉ số chính?

- Dễ hiểu: là phần trăm sai trên tổng lượng bán.
- Không bị hỏng khi có ngày bán 0. Chỉ số MAPE (sai số phần trăm từng ngày) sẽ chia cho 0 và vô nghĩa.
- Hàng bán nhiều có trọng số lớn hơn, hợp với thực tế kinh doanh.

### 7.3 Đọc kết quả chính

| | LSTM | MA7 | Seasonal naive |
|---|---|---|---|
| WAPE test | **0,544** | 0,605 | 0,696 |
| Bias test | −15,6% | +0,1% | −0,3% |

Cách đọc:
- **WAPE 0,544:** trên 3 tháng test, tổng lượng LSTM đoán sai bằng khoảng 54% tổng lượng bán thực tế. Con số nghe có vẻ cao, nhưng đây là dự báo **từng mặt hàng, từng cửa hàng, từng ngày**, mức chi tiết rất khó, nên ngay cả cách đoán đơn giản nhất cũng sai 60–70%.
- **LSTM tốt hơn MA7 khoảng 10%** (0,544 so với 0,605) và tốt hơn seasonal naive khoảng 22%.
- **Nhưng bias −15,6%:** tổng lượng LSTM dự báo thấp hơn thực tế khoảng 16%.

### 7.4 Vì sao LSTM vừa sai ít hơn lại vừa đoán thấp hơn?

Đây là điểm dễ gây nhầm lẫn nhất.

Mô hình được dạy giảm **sai số tuyệt đối** (MAE). Toán học cho thấy: để sai số tuyệt đối nhỏ nhất, nên đoán **trung vị** (giá trị ở giữa), không phải **trung bình**.

Ví dụ một mặt hàng tiêu dùng, 10 ngày bán lần lượt: 0, 0, 0, 0, 0, 0, 1, 2, 3, 4.
- Trung bình = 1; trung vị = 0.
- Đoán mỗi ngày 0: tổng sai tuyệt đối = 10. Đoán mỗi ngày 1: tổng sai tuyệt đối = 6 + 0 + 1 + 2 + 3 = 12.
- Vậy đoán 0 cho WAPE tốt hơn, **nhưng tổng dự báo = 0**, trong khi thực tế bán 10. Đoán thấp 100%.

Đồ tiêu dùng có hơn một nửa số ngày bán 0, nên LSTM đoán thấp nặng ở nhóm này (−36%).

**Hệ quả thực tế:**
- Muốn **sai số từng ngày nhỏ nhất**: LSTM tốt nhất.
- Muốn **đặt hàng cho đủ tổng lượng**: dùng nguyên dự báo LSTM sẽ đặt thiếu. Cần chỉnh lại, ví dụ đổi hàm mất mát hoặc nhân thêm hệ số.

### 7.5 Kiểm tra học vẹt (overfitting)

| Tập | WAPE LSTM |
|---|---|
| Train (bài tập đã học) | 0,477 |
| Valid (thi thử) | 0,508 |
| Test (thi thật) | 0,544 |

Điểm trên bài đã học tốt hơn trên bài mới là bình thường. Khoảng cách ở đây vừa phải nên mô hình **không học vẹt nghiêm trọng**. Test kém hơn valid một chút một phần vì test xa thời điểm huấn luyện hơn.

---

## 8. Đọc các phân tích sâu hơn trong báo cáo

### 8.1 Theo giai đoạn khuyến mãi (mục 5.2 của báo cáo)

| Nhãn | Nghĩa |
|---|---|
| `pre_1_3` | 1–3 ngày **trước** khi KM bắt đầu |
| `during` | **Trong** thời gian KM |
| `post_1_3` | 1–3 ngày **sau** khi KM kết thúc |
| `post_4_7` | 4–7 ngày sau khi KM kết thúc |
| `other` | Ngày bình thường, không gần đợt KM nào |

Phát hiện chính: ngay sau KM, MA7 và seasonal naive sai rất nặng (WAPE 0,77–1,07). Lý do là chúng lấy lượng bán cao trong KM làm mốc nên đoán quá cao. LSTM biết KM đã kết thúc nên vẫn đoán tốt (0,53–0,54).

### 8.2 Theo tầm dự báo (mục 5.4)

Tầm càng xa, sai số càng lớn: đoán ngày mai (0,511) dễ hơn đoán 7 ngày sau (0,616). Điều này giống dự báo thời tiết.

### 8.3 Độ quan trọng của nhóm biến (mục 7 của báo cáo)

**Cách đo (permutation importance, độ quan trọng bằng hoán vị):** muốn biết một nhóm biến quan trọng đến đâu, ta **xáo trộn ngẫu nhiên** nhóm biến đó giữa các mẫu, như gán lịch KM của mặt hàng này cho mặt hàng khác, rồi xem mô hình sai thêm bao nhiêu.
- Sai thêm nhiều: nhóm biến đó quan trọng.
- Gần như không đổi: mô hình không cần nhóm biến đó.

| Nhóm biến | WAPE tăng thêm khi bị xáo trộn | Đọc là |
|---|---|---|
| Lịch sử bán | +0,950 | Quan trọng nhất, gần như không thể thiếu |
| Lịch KM + mức giảm | +0,140 | Quan trọng thứ hai |
| Thông tin cửa hàng / mặt hàng | +0,084 | |
| Hậu KM | +0,058 | |
| Thứ trong tuần | +0,035 | |
| Lễ, tháng, cơ cấu | ≤ 0,007 | Ít tác dụng trong giai đoạn test, vì test là mùa hè không có lễ lớn |

Cột **"độ lệch chuẩn"** trong báo cáo cho biết kết quả dao động bao nhiêu giữa 3 lần xáo trộn. Nhỏ nghĩa là kết quả ổn định.

### 8.4 Mô phỏng mức giảm giá (mục 8 của báo cáo)

**Câu hỏi:** nếu giảm giá sâu hơn hoặc nông hơn thì bán được bao nhiêu, doanh thu ra sao?

**Cách làm:** lấy tất cả ngày KM có giảm giá trong test. Giữ nguyên mọi thông tin khác, chỉ đổi mức giảm thành 5%, 10%, …, 40%, rồi để mô hình dự báo lại.

**Cách đọc bảng "chỉ số":** lấy mức giảm 20% làm mốc = 100.
- "Lượng – Thực phẩm ở mức 40% = 118": giảm 40% thì bán được nhiều hơn 18% so với giảm 20%.
- "Doanh thu – Thực phẩm ở mức 40% = 89": nhưng doanh thu lại **thấp hơn 11%**, vì bán thêm không đủ bù phần giá bị giảm.

**Độ co giãn theo giá (price elasticity):** cho biết giá giảm 1% thì lượng bán tăng bao nhiêu %.
- Co giãn −0,6 (Thực phẩm theo LSTM): giảm giá 10% → bán thêm khoảng 6%.
- Nếu độ lớn nhỏ hơn 1 (như −0,6), giảm giá làm **doanh thu giảm**. Nếu lớn hơn 1 (ví dụ −1,5), giảm giá làm **doanh thu tăng**.

**Lưu ý quan trọng:** đây là mô phỏng từ một mô hình dự báo, **không phải thí nghiệm thật**. Mô hình học mối liên hệ trong dữ liệu, nhưng cửa hàng tự chọn mặt hàng nào giảm sâu, nên mối liên hệ đó chưa chắc là quan hệ nhân quả. Vì vậy chỉ dùng để tham khảo, cùng chiều với các phân tích thống kê khác.

---

## 9. Câu hỏi thường gặp

**Hỏi: WAPE 0,544 có phải là mô hình đúng 45,6%?**
Không hẳn. WAPE đo tổng lượng sai so với tổng lượng bán. Không nên đổi thành "độ chính xác %" vì dễ hiểu nhầm. Cách đọc đúng: "tổng sai lệch bằng khoảng 54% tổng lượng bán", và quan trọng hơn là **so với baseline**: LSTM sai ít hơn MA7 khoảng 10%.

**Hỏi: Sao không dự báo cho cả nhóm hàng hoặc cả cửa hàng để sai ít hơn?**
Gộp lại thì sai số nhỏ hơn nhiều, nhưng nhóm đã thống nhất dự báo ở cấp **mặt hàng × cửa hàng**, vì đó là cấp dùng được để đặt hàng.

**Hỏi: Vì sao chọn cấu hình có điểm valid tốt nhất, dù chỉ hơn 0,002?**
Vì phương pháp đã thống nhất là chọn theo điểm valid. Báo cáo ghi rõ khác biệt này rất nhỏ, nên không có nghĩa cấu hình đó vượt trội hẳn.

**Hỏi: Mô hình có dùng thông tin tương lai không?**
Chỉ dùng thông tin **biết trước được**: lịch, lễ, lịch KM đã lên kế hoạch, giá thường. Lượng bán tương lai và doanh thu không được dùng. Mọi phép chuẩn hóa chỉ tính từ tập train.

**Hỏi: Kết quả có chạy lại ra giống hệt không?**
Gần như giống. Có cố định "hạt giống ngẫu nhiên" (seed = 42). Chạy trên GPU và CPU có thể lệch ở chữ số thập phân thứ 4.

**Hỏi: Dùng mô hình này để đặt hàng ngay được không?**
Chưa nên dùng nguyên dự báo, vì mô hình đoán thấp khoảng 16% tổng lượng (mục 7.4). Cần hiệu chỉnh trước.

---

## 10. Từ điển thuật ngữ (A–Z)

| Thuật ngữ | Giải thích ngắn gọn |
|---|---|
| **Baseline** | Cách dự báo đơn giản dùng làm mốc so sánh (MA7, seasonal naive) |
| **Batch** (lô) | Nhóm ví dụ mô hình học cùng lúc trong một lượt (ở đây 1.024 ví dụ) |
| **Bias** (độ lệch) | Tổng dự báo cao hay thấp hơn tổng thực tế bao nhiêu %. Âm = đoán thấp |
| **Cửa sổ** (window) | Một đoạn dữ liệu liên tiếp: L ngày quá khứ + 7 ngày cần dự báo |
| **Decoder** | Phần mô hình đưa ra dự báo cho 7 ngày tới |
| **Deep learning** (học sâu) | Nhánh học máy dùng mạng nơ-ron nhiều tầng |
| **Depth** (mức giảm giá) | 1 − giá KM / giá thường. Ví dụ 0,2 = giảm 20% |
| **Dropout** | Tắt ngẫu nhiên một phần mạng khi học để tránh học vẹt |
| **Early stopping** (dừng sớm) | Ngừng học khi điểm valid không cải thiện nữa |
| **Elasticity** (độ co giãn) | % thay đổi lượng bán khi giá đổi 1% |
| **Embedding** | Cách biến nhãn chữ (cửa hàng 1, nhóm F1…) thành một dãy số mà mô hình tự học, sao cho các nhãn giống nhau có dãy số gần nhau |
| **Encoder** | Phần mô hình đọc dữ liệu quá khứ và nén thành "trí nhớ" |
| **Epoch** (vòng học) | Một lượt học qua một lượng dữ liệu nhất định (ở đây 150.000 cửa sổ), sau đó chấm điểm valid |
| **Feature** (biến đầu vào) | Thông tin mô hình dùng để dự báo |
| **GPU** | Card đồ họa, tính toán song song nên huấn luyện mạng nơ-ron nhanh hơn CPU nhiều lần. Ở đây dùng Tesla T4 trên Kaggle |
| **Grid search** (tìm theo lưới) | Thử tất cả tổ hợp siêu tham số đã liệt kê trước |
| **Hidden units** (đơn vị ẩn) | Kích thước "trí nhớ" của LSTM. Càng lớn càng chứa được nhiều thông tin, nhưng dễ học vẹt hơn |
| **Hold-out** | Giữ lại một phần dữ liệu không cho mô hình học, để chấm điểm |
| **Horizon** (tầm dự báo) | Dự báo cho bao nhiêu ngày sau mốc. Tầm 1 = ngày mai |
| **KM** | Khuyến mãi |
| **Learning rate** (tốc độ học) | Mức độ điều chỉnh tham số sau mỗi lần sai. Quá lớn thì học chập chờn, quá nhỏ thì học chậm |
| **log1p / expm1** | log1p(x) = ln(1 + x): nén các số lớn lại cho dễ học (0 → 0; 9 → 2,3; 99 → 4,6). expm1 là phép ngược để đổi về lượng bán thật |
| **Lookback** | Số ngày quá khứ mô hình được xem (28 hoặc 56) |
| **Loss** (hàm mất mát) | Công thức đo mức sai mà mô hình cố gắng giảm khi học |
| **LSTM** | Long Short-Term Memory: mạng nơ-ron đọc dữ liệu theo thời gian, có cơ chế chọn nhớ / quên |
| **MA7** | Trung bình 7 ngày gần nhất |
| **MAE** | Sai số tuyệt đối trung bình |
| **Median** (trung vị) | Giá trị đứng giữa khi xếp theo thứ tự |
| **One-hot** | Biến một biến nhiều loại thành nhiều cột 0/1. Ví dụ thứ trong tuần → 7 cột, chỉ cột của ngày đó bằng 1 |
| **Overfitting** (học vẹt) | Mô hình nhớ thuộc dữ liệu học nhưng đoán kém trên dữ liệu mới |
| **Origin** (mốc dự báo) | Thời điểm đứng để dự báo, chỉ biết dữ liệu đến ngày đó |
| **Parquet** | Định dạng file lưu bảng dữ liệu, gọn và đọc nhanh hơn CSV |
| **Patience** | Số epoch chờ thêm khi điểm valid không tiến bộ trước khi dừng (ở đây 4) |
| **Permutation importance** | Đo độ quan trọng của biến bằng cách xáo trộn biến đó rồi xem mô hình sai thêm bao nhiêu |
| **RMSE** | Căn bậc hai của trung bình bình phương sai số, phạt nặng lỗi lớn |
| **Seasonal naive** | Đoán bằng giá trị cùng thứ tuần trước |
| **Seed** (hạt giống ngẫu nhiên) | Con số cố định để các phép ngẫu nhiên lặp lại được |
| **Siêu tham số** (hyperparameter) | Lựa chọn con người đặt trước khi huấn luyện (lookback, hidden units, learning rate…) |
| **sin / cos (mã hóa vòng tròn)** | Biểu diễn tháng / ngày trên một vòng tròn, để mô hình hiểu tháng 12 nằm sát tháng 1, ngày 31 sát ngày 1 |
| **Chuẩn hóa** (standardize) | Trừ trung bình rồi chia độ lệch chuẩn, đưa số về cùng thang đo. Chỉ tính trên tập train |
| **Tham số** (parameter) | Các con số bên trong mô hình được tự học từ dữ liệu (ở đây 221.533 số) |
| **Test / Train / Valid** | Thi thật / bài tập / thi thử (mục 4) |
| **Trung bình** (mean) | Tổng chia số lượng |
| **WAPE** | Tổng sai số tuyệt đối / tổng thực tế. Chỉ số chính của báo cáo |

---

*File liên quan:* báo cáo kỹ thuật đầy đủ (tiếng Anh) ở `output/bao_cao_lstm.md`; định nghĩa gốc của các biến ở `output/bao_cao_pipeline_tien_xu_ly.md` và `output/bao_cao_feature.md`; mã nguồn ở `src/train_lstm.py`.
