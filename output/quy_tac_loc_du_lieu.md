# Quy tắc lọc dữ liệu

*Căn cứ: bộ quy tắc nhóm đề xuất, số liệu kiểm tra trực tiếp trên 8 tệp trong `dataset/`, và cách làm của bài baseline Darbanian et al. (2025), "Timing Matters", JRCS 85, 104317 (gọi tắt là "paper").*

Tài liệu này chỉ quy định **quy tắc**, chưa thực hiện lọc. Mọi con số dưới đây tính trên dữ liệu gốc.

---

## 1. Đánh giá từng quy tắc nhóm đề xuất

| # | Quy tắc đề xuất | Kết luận | Lý do và số liệu |
|---|---|---|---|
| 1 | Bỏ dòng có doanh số âm (trả hàng) | **Giữ** | Đúng như paper ("Negative sale values (returned items)… were removed"). Trong `sales.csv` có 1.659 dòng (0,022%) với quantity, sum_total hoặc price_base âm |
| 2 | Bỏ ngoại lệ (outliers) | **Sửa**: chỉ bỏ **lỗi dữ liệu**, không bỏ **đỉnh doanh số thật** | Áp quy tắc 3×IQR theo từng mã hàng – cửa hàng sẽ loại 1,4–2,2% số dòng. Trong đó 29–37% rơi vào **ngày có KM**, tức là đúng phần tín hiệu cần đo. Paper cũng không lọc ngoại lệ thống kê, chỉ bỏ những ngày bất thường cụ thể (xem #5). Chỉ bỏ 3.075 dòng có \|sum_total − quantity × price_base\| > 1 (lệch giữa các trường) |
| 3 | Bỏ dòng quantity hoặc doanh số = 0 hoặc thiếu | **Giữ**, gộp với #1 thành "giá trị ≤ 0" | Không có ô trống nào. Gộp #1 và #3 (quantity, sum_total hoặc price_base ≤ 0): **9.204 dòng (0,124%)** |
| 4 | Bỏ đơn bị hủy / hoàn trả | **Gộp vào #1** | Dataset không có mã đơn hàng hay trạng thái hủy. Dòng có giá trị âm là dấu hiệu duy nhất để nhận biết |
| 5 | Bỏ giai đoạn bất thường (dịch bệnh, thiên tai, đứt gãy chuỗi cung ứng) | **Sửa**: **không bỏ giai đoạn COVID**, chỉ bỏ các ngày bất thường cụ thể | Sales bắt đầu từ 28/08/2022. Nếu bỏ đến hết tình trạng khẩn cấp của WHO (05/05/2023) thì mất 25,2% số dòng, và không có giai đoạn nào trước dịch để so sánh. Paper giữ giai đoạn COVID, chỉ bỏ 13–14/03/2020 và tách trước/sau COVID làm kiểm tra độ vững. Trong dữ liệu không phát hiện giai đoạn bất thường kéo dài; các ngày bất thường đơn lẻ được xử lý ở #6 |
| 6 | Bỏ ngày cửa hàng đóng cửa / hoạt động không đủ giờ | **Giữ**, áp dụng theo số liệu | Ngày đóng cửa vốn không có dòng nào (CH3 ngày 16/10/2022). Ngày hoạt động không đủ giờ được phát hiện vì số dòng < 50% trung vị 29 ngày xung quanh: **CH3 ngày 22/10/2022** (112 dòng, bằng 10% mức thường) và **CH4 ngày 13/12/2023** (ngày đầu xuất hiện, 30 dòng, bằng 0,6% mức thường) |
| 7 | Chọn ngành hàng có KM thường xuyên và rõ ràng | **Giữ**, kèm ngưỡng định lượng | Ngưỡng: ≥ 100 đợt KM **và** ≥ 20% số mã hàng đang bán từng được KM. Kết quả: 111/158 ngành hàng đạt (102 thực phẩm, **đủ 9/9 ngành hàng tiêu dùng**), chiếm 91,4% số dòng sales. 13 ngành hàng không có đợt KM nào. Danh sách nhóm được chọn cuối cùng: **mục 6** |
| 8 | Lấy các ngày đang KM (During-Promo) | **Giữ** | Đợt KM = gộp các `doc_id` chồng lấn hoặc liền kề của cùng mã hàng – cửa hàng. Đợt có độ dài trung vị 14 ngày |
| 9 | Lấy cửa sổ trước KM 1–3 ngày; sau KM 1–3 ngày và 4–7 ngày | **Giữ**, thêm 2 điều kiện | (a) **Cửa sổ phải sạch**: không có KM khác của cùng mã hàng – cửa hàng trong 3 ngày trước và 7 ngày sau. Đạt 95,3% của 199.439 đợt (nếu cửa sổ 14/14 ngày thì chỉ còn 81,5%). (b) **Điền ngày bán 0**: sales không ghi dòng cho ngày không bán, nên cửa sổ 3 ngày rất dễ toàn số 0 giả. Chỉ điền 0 cho cặp mã hàng – cửa hàng bán đều (xem mục 2, bước C3) |
| 10 | Trạng thái KM 0/1 | **Giữ**, làm rõ định nghĩa | 1 = ngày nằm trong đợt KM có giảm giá thật (`depth > 0`). Ngày KM **không giảm giá** (giá trước = giá KM: 800.018 dòng trong discounts, chủ yếu mã loại 20 và một phần mã loại 5) không xếp vào 0 hay 1 mà **để trống** trong phân tích chính, để không làm bẩn nhóm "không KM" |
| 11 | Trạng thái lễ −1 / 0 / 1 | **Giữ**, thêm cách xử lý **chính ngày lễ** và **chọn ngày lễ theo số liệu** | Ở Áo, ngày lễ siêu thị đóng cửa nên paper không cần nhãn cho chính ngày lễ. Ở dataset này, ngày lễ vẫn có bán (01/01 chỉ bằng 0,51–0,55 mức thường), nên cần cờ riêng. Chỉ 3 ngày lễ có hiệu ứng rõ (mục 3) |

---

## 2. Bộ quy tắc cuối cùng (theo thứ tự thực hiện)

### A. Làm sạch: loại dòng lỗi

| Bước | Bảng | Quy tắc | Số dòng bị ảnh hưởng (dữ liệu gốc) | Nguồn căn cứ |
|---|---|---|---|---|
| A1 | Tất cả | Bỏ cột chỉ mục không tên (`Unnamed: 0`) | — | Báo cáo review |
| A2 | sales, online | Bỏ dòng có `quantity`, `sum_total` hoặc `price_base` ≤ 0 (trả hàng, hủy, dòng rỗng) | sales 9.204 (0,124%) | Paper; quy tắc #1, #3, #4 |
| A3 | sales | Bỏ dòng có \|sum_total − quantity × price_base\| > 1 | 3.075 (thêm, sau A2) | Lỗi dữ liệu; quy tắc #2 |
| A4 | discounts_history | Bỏ dòng có ngày > 26/09/2024 | 314.913 | Báo cáo review (dữ liệu kéo dài đến 2045) |
| A5 | discounts_history | Bỏ dòng có giá trước KM ≤ 0, hoặc giá KM > giá trước KM | ≈ 21 nghìn + ≈ 44 nghìn (trên dữ liệu gốc) | Không tính được mức giảm / mâu thuẫn |
| A6 | price_history, markdowns | Bỏ dòng trùng hoàn toàn | 18.641 / 268 | Báo cáo review |
| A7 | price_history | Bỏ dòng có `price` ≤ 0 (gồm code 9999999999) | ≈ 61,7 nghìn | Không phải giá bán |

### B. Phạm vi: chọn mẫu

| Bước | Quy tắc | Kết quả |
|---|---|---|
| B1 | Chỉ giữ mã hàng có trong catalog | Bỏ 0,49% số dòng sales |
| B2 | Chỉ giữ **11 ngành hàng thuộc 4 nhóm phân tích**: F1 Rau quả tươi, F2 Thịt tươi và thịt chế biến ướp lạnh (nhóm Thực phẩm); H1 Chăm sóc cá nhân và giấy vệ sinh, H2 Giặt giũ và tẩy rửa (nhóm Đồ tiêu dùng). Thêm cột `group4` (F1/F2/H1/H2) và `category` (food/household) | Mục 6. Theo phạm vi Thực phẩm – Đồ tiêu dùng trong file Insight |
| B3 | Các ngành hàng ở B2 đều đạt ngưỡng KM thường xuyên: ≥ 100 đợt KM **và** ≥ 20% số mã hàng từng được KM | Đây là tiêu chí đã dùng để chọn nhóm ở B2 (quy tắc #7). Trên toàn catalog có 111/158 ngành hàng đạt |
| B4 | Bỏ các ngày cửa hàng hoạt động bất thường: CH3 ngày 22/10/2022, CH4 ngày 13/12/2023. Cửa sổ nào chạm vào các ngày này (hoặc ngày CH3 đóng cửa 16/10/2022) thì coi như thiếu dữ liệu | Quy tắc #6 |
| B5 | Chỉ lấy `sales` (kênh cửa hàng). `online` được phân tích riêng, không cộng vào sales | Báo cáo review: hai kênh có quan hệ chưa rõ |
| B6 | **Không** lọc theo giai đoạn COVID. Thêm cờ `pheic` (ngày ≤ 05/05/2023) để kiểm tra độ vững, giống paper | Quy tắc #5 |

### C. Gán nhãn: không loại dòng

| Bước | Nhãn | Định nghĩa |
|---|---|---|
| C1 | `promo_status` | 1 = ngày nằm trong đợt KM có `depth > 0`; 0 = không có bản ghi KM nào; trống = ngày KM không giảm giá (`depth = 0`) |
| C2 | `holiday_status` | Xem mục 3 |
| C3 | Điền ngày bán 0 | Chỉ cho cặp mã hàng – cửa hàng **bán đều**: có bán ≥ **20%** số ngày giữa ngày bán đầu và cuối. Với các cặp còn lại, không điền 0. Không dùng ngưỡng 50%: đồ tiêu dùng có mật độ bán trung vị chỉ 0,12, nên ngưỡng 50% chỉ còn 686 đợt (H1) và 203 đợt (H2) (mục 6.4) |
| C4 | `status6` | Ghép như paper: `-10` trước lễ không KM, `-11` trước lễ có KM, `00`, `01`, `10`, `11` |
| C5 | `phase` | `pre_1_3`, `during`, `post_1_3`, `post_4_7`, `other` (theo quy tắc #9) |

### D. Cửa sổ quanh đợt KM

| Bước | Quy tắc |
|---|---|
| D1 | Cửa sổ: trước KM = 1–3 ngày trước ngày bắt đầu; trong KM = toàn bộ đợt; sau KM = 1–3 ngày và 4–7 ngày sau ngày kết thúc |
| D2 | Chỉ giữ đợt có **cửa sổ sạch**: không có ngày KM nào khác của cùng mã hàng – cửa hàng trong 3 ngày trước và 7 ngày sau (95,3% số đợt) |
| D3 | Toàn bộ cửa sổ phải nằm trong kỳ có dữ liệu của cửa hàng: CH1–3 từ 28/08/2022, CH4 từ 14/12/2023, đến hết 26/09/2024 |
| D4 | Chỉ giữ đợt của cặp mã hàng – cửa hàng bán đều (bước C3) |

---

## 3. Trạng thái ngày lễ

**Bằng chứng từ dữ liệu.** Tỷ lệ = lượng bán trong ngày / trung bình cùng thứ của 4 tuần trước. Chỉ tính CH1–3, vì CH4 xuất hiện muộn.

| Ngày lễ | −3 | −2 | −1 | Ngày lễ | +1 | +2 | +3 | Kết luận |
|---|---|---|---|---|---|---|---|---|
| 01/01/2023 | 1,59 | 1,66 | 1,51 | **0,55** | 0,72 | 0,78 | 0,80 | Hiệu ứng rất mạnh, kéo dài cả tuần |
| 01/01/2024 | 1,42 | 1,84 | 1,82 | **0,51** | 0,70 | 0,73 | 0,74 | Như trên |
| 23/02/2023 | 1,02 | 1,06 | **1,35** | 1,12 | 0,69 | 0,77 | 0,93 | Rõ ở ngày −1 và +1 |
| 23/02/2024 | 1,06 | 1,10 | **1,44** | 0,99 | 0,84 | 0,99 | 0,99 | Như trên |
| 08/03/2023 | 1,02 | 1,13 | **1,42** | 1,16 | 0,90 | 0,98 | 0,99 | Rõ ở ngày −1 |
| 08/03/2024 | 1,03 | 1,16 | **1,42** | 1,12 | 0,87 | 1,01 | 0,98 | Như trên |
| 09/05/2024 | 0,93 | 0,91 | 1,27 | 0,94 | 0,66 | 0,67 | 0,94 | Có hiệu ứng năm 2024, **không có** năm 2023 |
| 01/05, 12/06, 04/11 | | | | | | | | Không có hiệu ứng rõ (dao động 0,88–1,14) |

**Quy tắc gán nhãn:**

| Nhãn | Tết Dương lịch | 23/02, 08/03 |
|---|---|---|
| **−1** trước lễ | 27/12 – 31/12 (5 ngày, theo số liệu) | 1–3 ngày trước |
| **Ngày lễ** (cờ `is_holiday = 1`; không xếp vào −1, 0 hay 1, loại khỏi phân tích 6 nhãn) | 01/01 | Chính ngày lễ |
| **1** sau lễ | 02/01 – 08/01 (kỳ nghỉ Tết ở Nga) | 1–3 ngày sau |
| **0** ngày thường | Mọi ngày khác. Các ngày lễ 01/05, 09/05, 12/06, 04/11 cũng tính là 0 trong phân tích chính, kèm cờ `minor_holiday` để kiểm tra độ nhạy | |

Ghi chú:
- Việc dataset theo lịch Nga là suy luận từ chữ Cyrillic và các đỉnh doanh số ở bảng trên. Cần ghi rõ là giả thuyết trong báo cáo, hoặc đối chiếu với mô tả trên Kaggle.
- Paper không ghi độ dài cửa sổ trước/sau lễ. Ở đây độ dài được chọn theo số liệu, nên trong báo cáo cần nói rõ đây là điểm nhóm làm khác paper.

---

## 4. Những việc **không** làm, và lý do

| Không làm | Lý do |
|---|---|
| Lọc ngoại lệ thống kê theo IQR / z-score | Sẽ xóa đỉnh doanh số ngày KM và ngày lễ, tức chính đối tượng nghiên cứu. Nếu mô hình ML cần ổn định hơn, có thể chặn trần (winsorize) ở phân vị 99,5% theo từng mã hàng – cửa hàng, **chỉ trong tập train** |
| Bỏ giai đoạn COVID | Mất 25% dữ liệu và không có giai đoạn trước dịch để so sánh. Paper cũng giữ giai đoạn này |
| Lọc đơn hàng bị hủy riêng | Không có trường dữ liệu tương ứng |
| Điền quantity = 0 cho mọi ngày trống | Hàng bán không đều sẽ có hàng loạt số 0 giả; `actual_matrix` quá thưa nên không dùng được để biết hôm đó hàng có trên kệ hay không |
| Bỏ CH4 | Chỉ bỏ ngày đầu tiên (13/12/2023). Từ ngày thứ 2, số dòng mỗi ngày đã bình thường (trung vị 14 ngày đầu là 5.302, của 60 ngày sau đó là 5.007) |

---

## 5. Lưu ý khi triển khai

1. **Thứ tự:** làm A → B → C → D. Ghi lại số dòng trước và sau mỗi bước vào bảng log.
2. **Chuẩn hóa như paper:** sau khi lọc, chuẩn hóa min-max lượng bán **theo từng sản phẩm** để so được hàng tính theo kg và hàng tính theo cái, rồi lấy trung bình theo nhóm hàng. Min và max chỉ tính trên dữ liệu đã lọc.
3. **Cùng một mẫu cho mọi kiểm định:** ANOVA, t-test, chi-square, logistic và elasticity nên chạy trên cùng tập đợt KM đã qua bước D, để các kết quả so sánh được với nhau.
4. **Không sửa thư mục `dataset/`:** dữ liệu đã lọc ghi sang thư mục mới.

---

## 6. Chọn nhóm hàng: 2 nhóm Thực phẩm và 2 nhóm Đồ tiêu dùng

**Tương ứng với paper.** Paper chọn 2 nhóm hàng (Rau quả; Thịt cá), mỗi nhóm tách thành tươi và đông lạnh để so sánh hàng **dễ hỏng** với hàng **tích trữ được**. Đề tài của nhóm giữ nguyên ý tưởng đó, nhưng so sánh **Thực phẩm** (dễ hỏng, mua thường xuyên) với **Đồ tiêu dùng** (để lâu, tích trữ được). Vì vậy mỗi bên chọn 2 nhóm hàng.

### 6.1 Phân loại toàn bộ catalog

Catalog có **196 `dept_name`**, được xếp như sau. Danh sách đầy đủ kèm tên tiếng Việt và số liệu từng ngành hàng ở **mục 7**.

| Phân loại | Số ngành hàng | Ghi chú |
|---|---|---|
| Thực phẩm & đồ uống | 151 | Gồm cả đồ uống có cồn, đồ ăn làm sẵn |
| Đồ tiêu dùng | 9 | Hàng tiêu hao phi thực phẩm trong gia đình |
| Loại: phi thực phẩm khác | 12 | Thuốc lá, hoa, đồ chơi, văn phòng phẩm, hàng quầy thu ngân… |
| Loại: thú cưng | 6 | Là thức ăn nhưng không dành cho người, nên không thuộc cả hai phía của đề tài. Thức ăn cho mèo có KM rất nhiều (4.561 đợt) nên có thể giữ làm nhóm dự phòng nếu muốn mở rộng |
| Loại: kỹ thuật / không sử dụng / dịch vụ tại chỗ | 18 | Nhóm phụ trợ, nguyên liệu sản xuất tại chỗ, quầy cà phê, các ngành có chữ "НЕ ИСПОЛЬЗОВАТЬ" (không sử dụng), `ДОМАШНЯЯ КУХНЯ` (5 mã hàng nhưng chiếm 6,6% doanh thu, bất thường) |

### 6.2 Tiêu chí chọn

1. **Đặc tính đối lập rõ:** nhóm thực phẩm phải dễ hỏng, hạn dùng ngắn; nhóm đồ tiêu dùng phải để lâu và tích trữ được. Đây là cơ chế paper dùng để giải thích kết quả.
2. **KM thường xuyên và rõ ràng:** đạt quy tắc B3 (≥ 100 đợt KM, ≥ 20% số mã hàng có KM) và mức giảm trung vị > 0.
3. **Đủ mẫu:** mỗi nhóm có khoảng 1.500 đợt dùng được trở lên sau bước D (cửa sổ sạch, có giảm giá thật, mật độ bán ≥ 0,2).
4. **Đồng nhất trong nhóm:** các ngành hàng trong cùng nhóm có cách tiêu dùng giống nhau.
5. **Đối chiếu được với paper:** ưu tiên những nhóm paper cũng dùng.
6. **Cân bằng mẫu:** nhóm lớn nhất có không quá khoảng **2 lần** số đợt dùng được của nhóm nhỏ nhất. Nhóm nhỏ nhất luôn là H2, vì cả 9 ngành đồ tiêu dùng cộng lại chỉ có khoảng 5.200 đợt dùng được. Do đó việc cân bằng được thực hiện bằng cách chọn nhóm thực phẩm có quy mô tương ứng.

### 6.3 Kết quả chọn

| Phân loại | Nhóm | `dept_name` | Lý do chọn |
|---|---|---|---|
| **Thực phẩm** | **F1 Rau quả tươi** | `ФРУКТЫ` (trái cây), `ОВОЩИ` (rau củ) | Trùng với nhóm "Fresh Fruits and Vegetables" của paper nên đối chiếu trực tiếp được. Rất dễ hỏng, mua thường xuyên, chiếm doanh thu lớn nhất trong 4 nhóm (9,8%). Không đưa `ЗЕЛЕНЬ СВЕЖАЯ` (rau thơm) vào, vì chỉ 16% mã hàng từng được KM, không đạt B3 |
| | **F2 Thịt tươi và thịt chế biến ướp lạnh** | `СВЕЖЕЕ МЯСО` (thịt tươi), `ПТИЦА` (gia cầm), `КОЛБАСЫ ВАРЕНЫЕ` (xúc xích luộc), `СОСИСКИ.САРДЕЛЬКИ.ШПИКАЧКИ` (xúc xích nhỏ, xúc xích to) | Tương ứng nhóm "Fresh Meat and Fish" của paper. Hàng ướp lạnh, hạn dùng ngắn, mua thường xuyên (mật độ bán trung vị 0,60). Có 3.480 đợt dùng được, ngang với F1 và H1. 65% mã hàng có KM, mức giảm trung vị 19%. Không lấy xúc xích hun khói / xúc xích khô (để được lâu) và cá ướp lạnh (mỗi ngành dưới 260 đợt dùng được) |
| **Đồ tiêu dùng** | **H1 Chăm sóc cá nhân và giấy vệ sinh** | `ПЕРСОНАЛЬНЫЙ УХОД` (chăm sóc cá nhân), `БУМАЖНО-ВАТНАЯ ПРОДУКЦИЯ` (giấy vệ sinh, khăn giấy, bông) | Nhóm đồ tiêu dùng có nhiều đợt KM nhất (12.944 đợt). Giảm giá sâu (trung vị 27%). Giấy vệ sinh là ví dụ điển hình của hàng được tích trữ khi có KM |
| | **H2 Giặt giũ và tẩy rửa** | `СРЕДСТВА ДЛЯ СТИРКИ` (bột giặt, nước giặt), `СОПУТСТВУЮЩИЕ ТОВАРЫ ДЛЯ СТИРКИ` (nước xả, chất tẩy), `ЧИСТЯЩИЕ,МОЮЩИЕ СРЕДСТВА` (chất tẩy rửa, nước rửa chén) | Bột giặt là sản phẩm tích trữ kinh điển trong các nghiên cứu về KM (Johnson et al., 2013, có trong Bảng 1 của paper). 73% mã hàng có KM, giảm giá sâu nhất (trung vị 28%) |

### 6.4 Số liệu các nhóm (tính trên dữ liệu gốc, chỉ dòng có lượng và doanh số > 0)

| Nhóm | Mã hàng có bán | Dòng sales | % doanh thu | Mật độ bán trung vị | Đợt KM | % mã hàng có KM | Mức giảm trung vị | Đợt dùng được: mật độ ≥ 0 | ≥ 0,2 | ≥ 0,3 | ≥ 0,5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 Rau quả tươi | 643 | 233.780 | 9,83 | 0,53 | 5.554 | 58% | 11% | 4.212 | **3.592** | 3.218 | 2.386 |
| F2 Thịt tươi và thịt chế biến ướp lạnh | 568 | 185.563 | 6,71 | 0,60 | 4.618 | 65% | 19% | 3.808 | **3.480** | 3.257 | 2.707 |
| H1 Chăm sóc cá nhân và giấy vệ sinh | 2.168 | 263.257 | 1,27 | 0,12 | 12.944 | 58% | 27% | 10.711 | **3.351** | 1.939 | 686 |
| H2 Giặt giũ và tẩy rửa | 602 | 63.543 | 0,33 | 0,12 | 8.022 | 73% | 28% | 6.142 | **1.828** | 817 | 203 |
| *(dự phòng cho F2)* Sữa lên men truyền thống, sữa tươi, kem sữa | 410 | 303.056 | 3,07 | 0,82 | 4.382 | 74% | 14% | 3.480 | 3.380 | 3.315 | 3.112 |
| *(đã loại vì quá lớn)* Toàn bộ sữa (thêm sữa chua, tráng miệng sữa) | 848 | 578.398 | 4,27 | 0,76 | 11.402 | 78% | 17% | 9.961 | 9.840 | 9.699 | 8.998 |

Cách đọc bảng:
- "Mật độ bán" = số ngày có bán / số ngày từ lần bán đầu đến lần bán cuối của một cặp mã hàng – cửa hàng.
- "Đợt dùng được" = đợt có cửa sổ sạch (3 ngày trước, 7 ngày sau) và có giảm giá thật.

**Nhận xét:**
- **Cân bằng:** ở ngưỡng 0,2, 4 nhóm có 3.592 / 3.480 / 3.351 / 1.828 đợt. Nhóm lớn nhất gấp khoảng 2 lần nhóm nhỏ nhất (phiên bản trước dùng toàn bộ sữa cho F2 thì gấp 5,4 lần).
- Thực phẩm được bán gần như hằng ngày (mật độ 0,53–0,60), còn đồ tiêu dùng khoảng 8 ngày mới bán một lần (0,12). Đây chính là khác biệt về tần suất mua mà Insight 2 giả định, nên có thể đưa vào báo cáo như một kết quả mô tả.
- Vì khác biệt này, ngưỡng bán đều ở bước C3 được hạ từ 0,5 xuống **0,2**. Ở ngưỡng 0,2, cả 4 nhóm đều có từ 1.828 đợt trở lên.
- Với đồ tiêu dùng, cửa sổ 1–3 ngày sẽ có nhiều ngày bán 0. Nên bổ sung một bản phân tích tổng hợp theo **nhóm × cửa hàng × ngày**, giống cách paper lấy trung bình theo nhóm hàng, để kiểm tra độ vững.

### 6.5 Những ngành hàng không chọn, và lý do

| Ngành hàng | Lý do |
|---|---|
| `ТОВАРЫ ДЛЯ ДОМА` (đồ dùng gia đình) | Mức giảm trung vị 0%, chỉ 25% mã hàng có KM, nên KM không rõ ràng. Đây là đồ bền (dụng cụ), không phải hàng tiêu hao |
| `ТОВАРЫ ДЛЯ УБОРКИ` (dụng cụ dọn dẹp) | Mức giảm trung vị 0% |
| `ОСВЕЖИТЕЛИ,ИНСЕКТИЦИДЫ` (khử mùi, diệt côn trùng) | Thuốc diệt côn trùng bán theo mùa. Ít đợt KM (1.022 đợt) |
| `ТОВАРЫ ДЛЯ ЛИЧНОГО ПОЛЬЗОВАНИЯ` (đồ dùng cá nhân) | Nội dung không rõ; 815 mã hàng nhưng chỉ 14.115 dòng bán, rất thưa |
| Toàn bộ sữa (F2 của phiên bản trước) | Dễ hỏng và nhiều KM, nhưng có 9.840 đợt dùng được, gấp 5,4 lần H2, làm mất cân bằng. Nếu nhóm muốn giữ sữa thì dùng bản **dự phòng**: chỉ sữa lên men truyền thống + sữa tươi + kem sữa (3.380 đợt), bỏ `СОВРЕМЕННАЯ МОЛОЧНАЯ КАТЕГОРИЯ` |
| Phô mai | Hạn dùng dài hơn, nên tính dễ hỏng không rõ bằng. Lại có 5.840 đợt, quá lớn so với H2 |
| Cá tươi / cá ướp lạnh | Phần cá trong nhóm "Meat & Fish" của paper, nhưng mỗi ngành ở đây có dưới 260 đợt dùng được |
| Xúc xích hun khói / xúc xích khô | Để được lâu, không thuộc nhóm dễ hỏng |
| Bánh mì `ХЛЕБ` | Chỉ 21% mã hàng có KM, mức giảm trung vị 0% |
| Đồ ăn làm sẵn (salad, pizza, sushi, món chính) | Gần như không có KM giảm giá |
| Đồ uống, rượu bia, bánh kẹo | Không dễ hỏng, gần với hàng tích trữ, nên làm mờ sự đối lập Thực phẩm – Đồ tiêu dùng |

### 6.6 Hạn chế cần ghi trong báo cáo

- Nhóm H2 chỉ chiếm 0,33% doanh thu, và cả phía đồ tiêu dùng chiếm 1,6%. Kết quả về đồ tiêu dùng có độ chính xác thấp hơn phía thực phẩm.
- Bốn nhóm vẫn chưa cân bằng hoàn toàn (F1 gấp khoảng 2 lần H2). Đây cũng là hạn chế paper đã tự nêu. Khi so sánh, nên báo cáo cỡ hiệu ứng và khoảng tin cậy, không chỉ p-value. Cách xử lý: mục 6.7.

### 6.7 Chênh lệch số mẫu giữa các nhóm: paper xử lý thế nào, và nhóm nên làm gì

**Paper chênh lệch nhiều hơn của nhóm.** Bảng 2 của paper đếm số **ngày** theo từng nhóm hàng × trạng thái:

| Nhóm (paper) | 00 | 01 | −10 | −11 | 10 | 11 |
|---|---|---|---|---|---|---|
| Freezer Fruits & Veg | 1.019 | 58 | 309 | 19 | 325 | **12** |
| Fresh Fruits & Veg | 1.019 | 1.018 | 309 | 309 | 325 | 325 |
| Freezer Meat & Fish | 1.019 | 86 | 309 | 29 | 325 | 31 |
| Fresh Meat & Fish | 1.019 | 154 | 309 | 74 | 325 | 25 |

Ô nhỏ nhất của paper chỉ 12 ngày, so với ô lớn nhất 1.019 ngày (chênh khoảng **85 lần**). Ở nhóm bạn, theo số đợt KM, chênh lệch là 3.592 / 1.828 ≈ **2 lần** (phiên bản trước dùng toàn bộ sữa cho F2 là 5,4 lần).

**Paper đã xử lý như sau:**
1. **Không cắt bớt hay cân bằng mẫu**: giữ toàn bộ dữ liệu.
2. **Đơn vị phân tích là ngày, không phải đợt KM**: tổng hợp theo sản phẩm, chuẩn hóa min-max, rồi lấy trung bình theo nhóm hàng × ngày × trạng thái. Vì vậy mọi nhóm hàng có cùng số ngày, và chênh lệch chỉ còn nằm giữa các trạng thái.
3. **Kiểm định:** one-way ANOVA, rồi so sánh từng cặp có hiệu chỉnh Benjamini–Hochberg.
4. **Kiểm tra độ vững:** ANOVA riêng từng năm kèm η², mô hình hỗn hợp LMM với ICC, tách trước/sau COVID.
5. **Ghi vào hạn chế:** nhóm hàng tươi có nhiều dữ liệu KM hơn nhóm đông lạnh, nên ước lượng cho nhóm đông lạnh có thể thận trọng hơn thực tế. Tác giả đề xuất cho nghiên cứu sau: lấy mẫu phân tầng, thiết kế ghép cặp, hoặc gán trọng số.

**Áp cùng đơn vị của paper vào dữ liệu của nhóm:** đếm số nhóm × cửa hàng × ngày cho từng trạng thái, theo lịch lễ ở mục 3, bỏ chính ngày lễ.

| Nhóm | 00 | 01 | −10 | −11 | 10 | 11 |
|---|---|---|---|---|---|---|
| F1 Rau quả tươi | 2.381 | 2.357 | 77 | 61 | 91 | 51 |
| F2 Thịt tươi và thịt chế biến ướp lạnh | 2.381 | 2.372 | 77 | 76 | 90 | 91 |
| H1 Chăm sóc cá nhân và giấy vệ sinh | 2.382 | 2.035 | 77 | 68 | 91 | 76 |
| H2 Giặt giũ và tẩy rửa | 2.102 | 1.637 | 69 | 50 | 76 | 59 |

Ở đơn vị này, 4 nhóm gần như **cân bằng**. Ô nhỏ nhất là 50, lớn hơn ô nhỏ nhất của paper (12). Chênh lệch thực sự nằm ở các ô **trước/sau lễ**: chỉ có 22 ngày trước lễ và 26 ngày sau lễ mỗi cửa hàng trong 2 năm, trong khi paper có 309/325 ngày.

Ghi chú về con số 309 của paper: khoảng 5,5 năm × 52 tuần ≈ 286, cộng các ngày lễ thì xấp xỉ 309. Vì vậy nhiều khả năng paper **tính cả Chủ nhật** (ngày siêu thị Áo đóng cửa) là "ngày lễ". Đây là suy luận từ con số, paper không nói rõ. Dữ liệu của nhóm có bán hàng cả Chủ nhật nên không áp được cách này.

**Đủ mẫu chưa?** Với mức ý nghĩa 0,05 và power 80%, so H2 (1.828 đợt) với F1 (3.592 đợt) phát hiện được hiệu ứng nhỏ tới Cohen's d ≈ 0,08. Chênh lệch số mẫu làm khoảng tin cậy của nhóm nhỏ rộng hơn, nhưng **không làm lệch** ước lượng.

**Quy tắc bổ sung cho bước lọc:**

| Bước | Quy tắc |
|---|---|
| E1 | **Không cắt bớt (downsample)** nhóm lớn để bằng nhóm nhỏ. Paper không làm vậy, và việc này làm mất thông tin |
| E2 | Áp **cùng một bộ quy tắc A–D** cho cả 4 nhóm. Sau mỗi bước, ghi số mẫu theo nhóm × trạng thái thành một bảng giống Bảng 2 của paper, và đưa bảng cuối cùng vào báo cáo |
| E3 | Mỗi ô nhóm × trạng thái cần **≥ 30 quan sát** mới đưa vào kiểm định. Ô nào thiếu thì chỉ báo cáo mô tả, hoặc gộp −1/1 thành "quanh lễ" |
| E4 | Khi phân tích: dùng Welch t-test, Welch ANOVA hoặc sai số chuẩn HC3, so sánh cặp bằng Games-Howell, hiệu chỉnh đa kiểm định Benjamini–Hochberg như paper; luôn kèm cỡ hiệu ứng và khoảng tin cậy 95% |
| E5 | Kiểm tra độ vững theo đề xuất của paper: (a) **gán trọng số** để mỗi nhóm có tổng trọng số bằng nhau; (b) **lấy mẫu phân tầng**: rút ngẫu nhiên mỗi nhóm 1.800 đợt, lặp lại khoảng 200 lần, xem kết luận có giữ nguyên không. Đây chỉ là bản phụ, bản chính vẫn dùng toàn bộ mẫu |
| E6 | Kiểm tra cân bằng **bên trong** nhóm: báo cáo kết quả riêng cho từng `dept_name`, hoặc đưa `dept_name` vào mô hình làm hiệu ứng ngẫu nhiên. Lý do: H1 bị chi phối bởi ngành chăm sóc cá nhân (2.517/3.351 đợt); F2 có 2.217/3.480 đợt đến từ hai ngành xúc xích |

---

## 7. Phụ lục: phân loại toàn bộ 196 ngành hàng trong catalog

Cột "Nhóm chọn": F1/F2/H1/H2 = được chọn ở mục 6; "dự phòng" = nhóm dự phòng. Số liệu tính trên dữ liệu gốc: "Đợt KM" chỉ tính đến 26/09/2024; "% mã hàng có KM" = số mã hàng có KM / số mã hàng có bán.

### Thực phẩm & đồ uống (151 ngành hàng)

| dept_name | Tiếng Việt | Nhóm chọn | Mã hàng có bán | Dòng sales | Đợt KM | % mã hàng có KM |
|---|---|---|---|---|---|---|
| СОВРЕМЕННАЯ МОЛОЧНАЯ КАТЕГОРИЯ | Sữa chua, tráng miệng sữa (dòng sữa hiện đại) |  | 438 | 275.342 | 7.506 | 88% |
| ПИВО | Bia |  | 625 | 240.101 | 7.360 | 74% |
| СЛАДКИЕ МУЧНЫЕ ИЗДЕЛИЯ | Bánh quy, bánh ngọt đóng gói |  | 501 | 215.767 | 5.963 | 63% |
| ТРАДИЦИОННЫЕ МОЛОЧНЫЕ ПРОДУКТЫ | Sữa lên men truyền thống (kefir, smetana, tvorog) | dự phòng | 273 | 206.061 | 3.233 | 83% |
| СОКИ | Nước ép |  | 386 | 193.136 | 3.462 | 68% |
| ЛИМОНАДЫ | Nước ngọt có ga |  | 237 | 173.476 | 2.518 | 68% |
| КОНФЕТЫ | Kẹo |  | 497 | 172.650 | 5.020 | 67% |
| СЫРЫ ШТУЧНЫЕ | Phô mai đóng gói |  | 331 | 165.137 | 5.502 | 82% |
| ВИНО | Rượu vang |  | 1.346 | 140.359 | 11.370 | 81% |
| ХЛЕБ | Bánh mì |  | 269 | 137.612 | 426 | 21% |
| ДЕТСКОЕ ПИТАНИЕ | Thực phẩm trẻ em |  | 557 | 135.096 | 8.127 | 81% |
| ФРУКТЫ | Trái cây | F1 | 391 | 125.936 | 3.523 | 59% |
| ВОДА | Nước uống |  | 197 | 114.921 | 929 | 49% |
| ШОКОЛАД,ШОКОЛАДНАЯ ПАСТА | Sô-cô-la, bơ sô-cô-la |  | 280 | 109.639 | 6.524 | 85% |
| ОВОЩИ | Rau củ | F1 | 252 | 107.844 | 2.238 | 61% |
| СПЕЦИИ,ПРИПРАВА | Gia vị |  | 360 | 98.990 | 2.741 | 67% |
| ЧАЙ | Trà |  | 476 | 97.781 | 6.488 | 74% |
| ВОДКА,НАСТОЙКИ | Vodka, rượu ngâm |  | 356 | 88.173 | 3.422 | 72% |
| КОНСЕРВЫ ОВОЩНЫЕ | Rau củ đóng hộp |  | 370 | 87.696 | 3.502 | 67% |
| МОРОЖЕНОЕ И ЗАМОРОЖЕННЫЕ ДЕСЕРТЫ | Kem, tráng miệng đông lạnh |  | 209 | 87.459 | 2.478 | 79% |
| СЛАДОСТИ | Đồ ngọt |  | 249 | 83.213 | 1.630 | 60% |
| КОЛБАСЫ КОПЧЕНЫЕ.СЫРОВЯЛЕНЫЕ | Xúc xích hun khói, xúc xích khô |  | 248 | 82.827 | 2.293 | 61% |
| ЧИПСЫ | Khoai tây chiên |  | 164 | 82.357 | 2.670 | 66% |
| БЫСТРЫЙ ПЕРЕКУС | Đồ ăn nhẹ nhanh |  | 255 | 77.646 | 1.798 | 77% |
| КОФЕ | Cà phê |  | 349 | 76.293 | 6.584 | 81% |
| СЫРЫ ВЕСОВЫЕ | Phô mai bán cân |  | 140 | 71.683 | 2.520 | 84% |
| МОЛОКО | Sữa tươi | dự phòng | 91 | 71.192 | 1.065 | 71% |
| МАКАРОННЫЕ ИЗДЕЛИЯ | Mì ống, nui |  | 195 | 70.816 | 2.340 | 74% |
| МАЙОНЕЗ,СОУСЫ НА МАЙОНЕЗНОЙ ОСНОВЕ | Mayonnaise, sốt mayonnaise |  | 122 | 68.709 | 1.821 | 78% |
| КРУПЫ И ЗЕРНОВЫЕ | Ngũ cốc, hạt |  | 171 | 67.772 | 1.651 | 76% |
| ПРЕСЕРВЫ.ПАШТЕТЫ | Cá ngâm, pa-tê |  | 179 | 67.104 | 2.554 | 85% |
| ПОЛУФАБРИКАТЫ | Thực phẩm sơ chế |  | 172 | 60.332 | 1.937 | 71% |
| ЗАВТРАКИ | Ngũ cốc ăn sáng |  | 188 | 60.291 | 3.009 | 80% |
| ЭНЕРГЕТИКИ | Nước tăng lực |  | 63 | 59.208 | 1.069 | 78% |
| ДЕТСКИЕ МОЛОЧНЫЕ ПРОДУКТЫ | Sản phẩm sữa trẻ em |  | 124 | 57.043 | 938 | 65% |
| ПРОДУКЦИЯ БЫСТРОГО ПРИГОТОВЛЕНИЯ | Thực phẩm ăn liền |  | 107 | 56.435 | 442 | 56% |
| МЯСНЫЕ ДЕЛИКАТЕСЫ | Thịt nguội cao cấp |  | 196 | 55.616 | 1.106 | 46% |
| ПИРОЖНЫЕ | Bánh ngọt nhỏ |  | 145 | 54.435 | 136 | 17% |
| КОЛБАСЫ ВАРЕНЫЕ | Xúc xích luộc | F2 | 141 | 52.978 | 1.478 | 79% |
| ПТИЦА | Gia cầm | F2 | 187 | 52.571 | 1.406 | 70% |
| САЛАТЫ | Salad làm sẵn |  | 172 | 50.329 | 281 | 15% |
| ТОРТЫ | Bánh kem |  | 151 | 49.747 | 127 | 25% |
| ЗЕЛЕНЬ СВЕЖАЯ | Rau thơm tươi |  | 153 | 49.101 | 424 | 16% |
| НЕСЛАДКИЕ МУЧНЫЕ ИЗДЕЛИЯ | Bánh mặn đóng gói (bánh quy giòn, bánh mì khô) |  | 90 | 48.898 | 1.398 | 90% |
| ОРЕХИ | Các loại hạt |  | 120 | 48.273 | 1.140 | 56% |
| ВТОРЫЕ БЛЮДА | Món chính làm sẵn |  | 289 | 46.728 | 499 | 19% |
| ПИЦЦА | Pizza |  | 46 | 44.664 | 177 | 13% |
| СЛОЙКА | Bánh ngàn lớp |  | 75 | 44.511 | 255 | 31% |
| КЕТЧУПЫ,СОУСЫ НА ТОМАТНОЙ ОСНОВЕ | Tương cà, sốt cà chua |  | 137 | 43.467 | 1.744 | 72% |
| СОСИСКИ.САРДЕЛЬКИ.ШПИКАЧКИ | Xúc xích nhỏ, xúc xích to | F2 | 93 | 42.591 | 1.230 | 83% |
| КОНЬЯК | Cognac |  | 241 | 42.445 | 2.428 | 72% |
| ПИРОГИ | Bánh nướng nhân |  | 56 | 42.335 | 195 | 18% |
| СЕМЕЧКИ | Hạt hướng dương |  | 62 | 40.782 | 695 | 82% |
| КОНСЕРВЫ РЫБНЫЕ | Cá đóng hộp |  | 160 | 39.575 | 1.638 | 62% |
| СУХИЕ КОМПОНЕНТЫ | Nguyên liệu khô làm bánh |  | 170 | 39.084 | 244 | 37% |
| СУХАРИКИ | Bánh mì giòn (snack) |  | 53 | 38.877 | 742 | 66% |
| СВЕЖЕЕ МЯСО | Thịt tươi | F2 | 147 | 37.423 | 1.067 | 57% |
| ТОРТЫ И ПИРОЖНЫЕ | Bánh kem, bánh ngọt đóng gói |  | 92 | 34.739 | 1.040 | 77% |
| ПРОЧИЕ СОУСЫ | Các loại sốt khác |  | 120 | 32.628 | 906 | 64% |
| СУШИ | Sushi |  | 84 | 32.219 | 429 | 24% |
| НАБОРЫ КОНФЕТ | Hộp kẹo quà tặng |  | 165 | 30.751 | 1.376 | 62% |
| СДОБА | Bánh mì ngọt |  | 68 | 29.696 | 180 | 31% |
| МАСЛО И МАРГАРИН | Bơ, margarine |  | 58 | 29.614 | 736 | 81% |
| РАСТИТЕЛЬНЫЕ МОЛОЧНЫЕ ПРОДУКТЫ | Sữa thực vật |  | 93 | 29.504 | 801 | 73% |
| КРЕПКИЕ АЛКОГОЛЬНЫЕ НАПИТКИ | Rượu mạnh |  | 383 | 29.126 | 3.173 | 64% |
| РЫБКА К ПИВУ | Cá khô nhắm bia |  | 103 | 28.778 | 459 | 43% |
| СОЛЕНИЯ | Dưa muối |  | 64 | 28.488 | 626 | 77% |
| ПЕЛЬМЕНИ,МАНТЫ,ХИНКАЛИ | Sủi cảo đông lạnh |  | 86 | 26.402 | 1.065 | 85% |
| СЛИВКИ | Kem sữa (cream) | dự phòng | 46 | 25.803 | 391 | 65% |
| ПЕРВЫЕ БЛЮДА | Súp làm sẵn |  | 92 | 25.731 | 0 | 0% |
| ЧАЙ ХОЛОДНЫЙ | Trà đóng chai |  | 49 | 25.520 | 327 | 71% |
| СУХОФРУКТЫ | Trái cây sấy |  | 114 | 25.224 | 893 | 66% |
| МАСЛО ПОДСОЛНЕЧНОЕ | Dầu hướng dương |  | 45 | 25.133 | 688 | 69% |
| ДИЕТИЧЕСКОЕ ПИТАНИЕ | Thực phẩm ăn kiêng |  | 141 | 24.842 | 559 | 59% |
| ХОЛОДНЫЕ ЗАКУСКИ | Món khai vị nguội |  | 79 | 24.664 | 66 | 14% |
| КОНСЕРВЫ МЯСНЫЕ | Thịt đóng hộp |  | 90 | 23.762 | 816 | 54% |
| СЛАБОАЛКОГОЛЬНЫЕ НАПИТКИ | Đồ uống có cồn nhẹ |  | 153 | 23.072 | 1.073 | 76% |
| ПОНЧИКИ | Bánh donut |  | 37 | 22.102 | 2 | 3% |
| РЫБА СОЛЕНАЯ | Cá muối |  | 48 | 21.036 | 461 | 56% |
| РЫБА ХОЛОДНОГО КОПЧЕНИЯ | Cá hun khói lạnh |  | 75 | 20.712 | 627 | 43% |
| ИЗДЕЛИЯ ИЗ ТЕСТА | Sản phẩm từ bột nhào |  | 67 | 19.058 | 700 | 78% |
| ОВОЩИ ЗАМОРОЖЕННЫЕ | Rau củ đông lạnh |  | 62 | 18.968 | 654 | 76% |
| МУКА | Bột mì |  | 91 | 18.393 | 479 | 66% |
| КОНСЕРВЫ МОЛОЧНЫЕ | Sữa đặc, sữa đóng hộp |  | 27 | 18.350 | 373 | 67% |
| ЗОЖ | Thực phẩm lối sống lành mạnh |  | 130 | 17.868 | 381 | 48% |
| МАНГАЛ | Đồ nướng BBQ |  | 35 | 17.635 | 24 | 11% |
| ОСТРЫЕ СОУСЫ | Sốt cay |  | 61 | 17.612 | 240 | 54% |
| СНЕКИ | Snack |  | 26 | 17.081 | 241 | 62% |
| БУТЕРБРОДЫ | Bánh mì kẹp |  | 22 | 17.058 | 164 | 41% |
| ЯЙЦО | Trứng |  | 50 | 16.818 | 89 | 28% |
| ИКРА РЕЧНЫХ И МОРСКИХ РЫБ | Trứng cá sông, cá biển |  | 38 | 15.735 | 298 | 66% |
| СОУСЫ ДЛЯ ЭТНИЧЕСКОЙ КУХНИ И ПАСТЫ | Sốt món dân tộc, sốt mì Ý |  | 52 | 15.469 | 153 | 38% |
| ВАРЕНЬЕ,МЁД | Mứt, mật ong |  | 124 | 15.381 | 425 | 56% |
| ГРИЛЬ | Đồ nướng làm sẵn |  | 15 | 15.318 | 4 | 13% |
| САХАР | Đường |  | 29 | 14.743 | 81 | 28% |
| СОЛЬ | Muối |  | 40 | 13.796 | 71 | 32% |
| ШАУРМА | Shawarma |  | 7 | 11.692 | 0 | 0% |
| КВАС | Kvass |  | 31 | 11.579 | 95 | 48% |
| МЯСНЫЕ П/Ф | Thịt sơ chế |  | 35 | 9.896 | 395 | 89% |
| КРЕВЕТКИ | Tôm |  | 34 | 9.888 | 471 | 65% |
| БЕЛАЯ РЫБА МОРСКАЯ | Cá trắng biển |  | 41 | 9.743 | 197 | 63% |
| СПОРТИВНОЕ ПИТАНИЕ | Thực phẩm thể thao |  | 25 | 9.094 | 321 | 56% |
| МАСЛО ОЛИВКОВОЕ | Dầu ô liu |  | 98 | 8.963 | 917 | 67% |
| ГРИБЫ | Nấm |  | 32 | 8.502 | 424 | 72% |
| ЯГОДЫ ЗАМОРОЖЕННЫЕ | Quả mọng đông lạnh |  | 24 | 8.489 | 257 | 75% |
| ВОК | Món xào wok |  | 39 | 8.285 | 0 | 0% |
| ИКРА ЛОСОСЕВАЯ | Trứng cá hồi |  | 47 | 8.278 | 201 | 57% |
| КРАСНАЯ РЫБА ОХЛАЖДЕННАЯ | Cá đỏ ướp lạnh |  | 59 | 8.194 | 245 | 46% |
| СОКИ,МОРСЫ,НАПИТКИ | Nước ép, nước quả mọng |  | 23 | 7.832 | 0 | 0% |
| ЭТНИЧЕСКИЕ СНЭКИ И ВОДОРОСЛИ | Snack dân tộc, rong biển |  | 17 | 7.824 | 53 | 65% |
| КРАСНАЯ РЫБА | Cá đỏ |  | 38 | 7.752 | 236 | 39% |
| КОРЖИ И ТАРТАЛЕТКИ | Đế bánh, vỏ tart |  | 15 | 7.419 | 133 | 73% |
| ФРУКТОВЫЕ САЛАТЫ | Salad trái cây |  | 23 | 6.929 | 6 | 9% |
| КОНСЕРВЫ ФРУКТОВЫЕ,ЯГОДНЫЕ | Trái cây đóng hộp |  | 42 | 6.488 | 278 | 64% |
| УКСУС | Giấm |  | 33 | 6.458 | 51 | 21% |
| БЕЛАЯ РЫБА МОРСКАЯ ОХЛАЖДЕННАЯ | Cá trắng biển ướp lạnh |  | 28 | 6.436 | 269 | 57% |
| БУРГЕРЫ | Burger |  | 10 | 5.595 | 1 | 10% |
| КАКАО,ШОКОЛАД | Ca cao, sô-cô-la uống |  | 21 | 5.475 | 251 | 57% |
| ГОТОВЫЕ БЛЮДА | Món ăn sẵn |  | 16 | 5.279 | 204 | 81% |
| РЫБНАЯ КУЛИНАРИЯ | Món cá chế biến sẵn |  | 17 | 5.245 | 52 | 41% |
| МОРЕПРОДУКТЫ | Hải sản |  | 25 | 4.844 | 136 | 68% |
| МАСЛО ПРОЧЕЕ | Dầu ăn khác |  | 27 | 4.493 | 155 | 56% |
| ГОРЯЧИЕ БУТЕРБРОДЫ | Bánh mì kẹp nóng |  | 5 | 4.470 | 0 | 0% |
| КОНСЕРВЫ ГРИБНЫЕ | Nấm đóng hộp |  | 42 | 4.435 | 148 | 60% |
| ЭТНИЧЕСКИЕ МАКАРОННЫЕ ИЗДЕЛИЯ | Mì kiểu châu Á |  | 16 | 4.174 | 16 | 62% |
| ПРОДУКТЫ ИЗ СУРИМИ | Sản phẩm surimi |  | 6 | 3.821 | 72 | 67% |
| ПИЦЦА ПОЛУФАБРИКАТ | Pizza sơ chế |  | 5 | 3.662 | 0 | 0% |
| РЫБА ГОРЯЧЕГО КОПЧЕНИЯ | Cá hun khói nóng |  | 22 | 3.252 | 59 | 41% |
| СИРОПЫ | Si-rô |  | 27 | 3.211 | 55 | 63% |
| ТЕСТО | Bột nhào |  | 5 | 2.854 | 6 | 40% |
| РЫБНЫЕ ПОЛУФАБРИКАТЫ | Cá sơ chế |  | 10 | 2.162 | 49 | 50% |
| ФУНКЦИОНАЛЬНЫЕ НАПИТКИ | Đồ uống chức năng |  | 11 | 2.154 | 10 | 27% |
| РЫБНЫЕ ПОЛУФАБРИКАТЫ ОХЛАЖДЕННЫЕ | Cá sơ chế ướp lạnh |  | 15 | 1.481 | 7 | 27% |
| ГРИБЫ ЗАМОРОЖЕННЫЕ | Nấm đông lạnh |  | 11 | 1.247 | 108 | 55% |
| ИКРА БЕЛКОВАЯ | Trứng cá nhân tạo |  | 3 | 685 | 30 | 67% |
| РЫБА ЖИВАЯ | Cá sống |  | 4 | 620 | 1 | 25% |
| БЕЛАЯ РЫБА РЕЧНАЯ | Cá trắng sông |  | 8 | 601 | 9 | 38% |
| БЕЛАЯ РЫБА РЕЧНАЯ ОХЛАЖДЕННАЯ | Cá trắng sông ướp lạnh |  | 12 | 573 | 16 | 33% |
| ИКРА РЫБ ОСЕТРОВЫХ ПОРОД | Trứng cá tầm |  | 7 | 333 | 2 | 14% |
| МЯСНЫЕ СНЕКИ | Snack thịt |  | 1 | 234 | 0 | 0% |
| ЭТНИЧЕСКАЯ КОНСЕРВАЦИЯ | Đồ hộp món dân tộc |  | 1 | 101 | 0 | 0% |
| МЯСО,ПТИЦА ЗАМОРОЖЕННЫЕ | Thịt, gia cầm đông lạnh |  | 2 | 45 | 0 | 0% |
| ЛЕПИМ-ВАРИМ | Bộ tự làm sủi cảo |  | 1 | 7 | 0 | 0% |
| ДИАБЕТИЧЕСКИЕ ПРОДУКТЫ | Thực phẩm cho người tiểu đường |  | 0 | 0 | 0 |  |
| ДИЕТИЧЕСКИЕ ПРОДУКТЫ | Sản phẩm ăn kiêng |  | 0 | 0 | 0 |  |
| МОРОЖЕНОЕ | Kem |  | 0 | 0 | 0 |  |
| ОВОЩНЫЕ П/Ф | Rau sơ chế |  | 0 | 0 | 0 |  |
| ОРЕХИ,СУХОФРУКТЫ В ШОКОЛАДЕ | Hạt, trái cây sấy phủ sô-cô-la |  | 0 | 0 | 0 |  |
| МОРЕПРОДУКТЫ ОХЛАЖДЕННЫЕ | Hải sản ướp lạnh |  | 0 | 0 | 0 |  |
| МОРЕПРОДУКТЫ ЖИВЫЕ | Hải sản sống |  | 0 | 0 | 0 |  |
| СЫР КОПЧЕНЫЙ | Phô mai hun khói |  | 0 | 0 | 0 |  |

### Đồ tiêu dùng (9 ngành hàng)

| dept_name | Tiếng Việt | Nhóm chọn | Mã hàng có bán | Dòng sales | Đợt KM | % mã hàng có KM |
|---|---|---|---|---|---|---|
| ПЕРСОНАЛЬНЫЙ УХОД | Chăm sóc cá nhân | H1 | 1.885 | 171.993 | 12.240 | 65% |
| БУМАЖНО-ВАТНАЯ ПРОДУКЦИЯ | Giấy vệ sinh, khăn giấy, bông | H1 | 283 | 91.264 | 1.920 | 50% |
| ТОВАРЫ ДЛЯ ДОМА | Đồ dùng gia đình |  | 876 | 86.950 | 1.032 | 25% |
| ЧИСТЯЩИЕ,МОЮЩИЕ СРЕДСТВА | Chất tẩy rửa, nước rửa chén | H2 | 318 | 36.720 | 4.118 | 77% |
| ТОВАРЫ ДЛЯ УБОРКИ | Dụng cụ dọn dẹp |  | 204 | 33.483 | 826 | 54% |
| СРЕДСТВА ДЛЯ СТИРКИ | Bột giặt, nước giặt | H2 | 184 | 18.867 | 3.472 | 84% |
| ТОВАРЫ ДЛЯ ЛИЧНОГО ПОЛЬЗОВАНИЯ | Đồ dùng cá nhân |  | 815 | 14.115 | 1.154 | 45% |
| ОСВЕЖИТЕЛИ,ИНСЕКТИЦИДЫ | Chất khử mùi, thuốc diệt côn trùng |  | 158 | 11.652 | 1.022 | 79% |
| СОПУТСТВУЮЩИЕ ТОВАРЫ ДЛЯ СТИРКИ | Sản phẩm phụ trợ giặt (nước xả, tẩy) | H2 | 100 | 7.956 | 1.099 | 57% |

### Loại: phi thực phẩm khác (12 ngành hàng)

| dept_name | Tiếng Việt | Nhóm chọn | Mã hàng có bán | Dòng sales | Đợt KM | % mã hàng có KM |
|---|---|---|---|---|---|---|
| ТАБАЧНЫЕ ИЗДЕЛИЯ | Thuốc lá |  | 316 | 229.192 | 154 | 6% |
| ПРИКАССА | Hàng quầy thu ngân |  | 212 | 186.289 | 3.973 | 75% |
| ДЕТСКИЙ МИР | Đồ trẻ em, đồ chơi |  | 1.395 | 35.195 | 3.096 | 31% |
| ЦВЕТЫ | Hoa |  | 405 | 26.962 | 53 | 3% |
| ТОВАРЫ ДЛЯ ПРАЗДНИКА | Đồ trang trí lễ hội |  | 553 | 20.033 | 111 | 14% |
| ИНФОРМАЦИОННЫЕ ТОВАРЫ | Báo, tạp chí |  | 268 | 13.463 | 4 | 1% |
| АКСЕССУАРЫ ДЛЯ КУРЕНИЯ | Phụ kiện hút thuốc |  | 23 | 9.567 | 23 | 35% |
| СЕЗОННЫЙ АССОРТИМЕНТ | Hàng theo mùa |  | 256 | 8.730 | 702 | 58% |
| ТОВАРЫ ДЛЯ ШКОЛЫ И ОФИСА | Văn phòng phẩm |  | 146 | 7.140 | 17 | 1% |
| АВТОТОВАРЫ | Đồ dùng ô tô |  | 31 | 768 | 8 | 13% |
| НАРОДНЫЕ ПРОМЫСЛЫ | Đồ thủ công mỹ nghệ |  | 36 | 98 | 10 | 14% |
| ПОДАРОЧНЫЕ КОРЗИНЫ | Giỏ quà |  | 1 | 5 | 0 | 0% |

### Loại: thú cưng (6 ngành hàng)

| dept_name | Tiếng Việt | Nhóm chọn | Mã hàng có bán | Dòng sales | Đợt KM | % mã hàng có KM |
|---|---|---|---|---|---|---|
| КОРМА ДЛЯ КОШЕК | Thức ăn cho mèo |  | 219 | 120.697 | 4.561 | 96% |
| КОРМА ДЛЯ СОБАК | Thức ăn cho chó |  | 46 | 12.931 | 520 | 93% |
| ЛАКОМСТВА ДЛЯ ЖИВОТНЫХ | Đồ ăn vặt thú cưng |  | 33 | 10.186 | 329 | 91% |
| НАПОЛНИТЕЛИ | Cát vệ sinh thú cưng |  | 4 | 3.093 | 32 | 75% |
| ЗООТОВАРЫ | Đồ dùng thú cưng |  | 42 | 1.356 | 50 | 14% |
| КОРМА ДЛЯ ДРУГИХ ЖИВОТНЫХ | Thức ăn vật nuôi khác |  | 5 | 71 | 0 | 0% |

### Loại: kỹ thuật / không sử dụng / dịch vụ tại chỗ (18 ngành hàng)

| dept_name | Tiếng Việt | Nhóm chọn | Mã hàng có bán | Dòng sales | Đợt KM | % mã hàng có KM |
|---|---|---|---|---|---|---|
| КАФЕ | Quầy cà phê tại cửa hàng |  | 18 | 27.092 | 23 | 33% |
| ВСПОМОГАТЕЛЬНАЯ ГРУППА | Nhóm phụ trợ (kỹ thuật) |  | 83 | 14.843 | 0 | 0% |
| ДОМАШНЯЯ КУХНЯ | Món nhà làm (5 mã, doanh thu bất thường) |  | 5 | 3.182 | 0 | 0% |
| ИНГРЕДИЕНТЫ ДЛЯ ЭТНИЧЕСКОЙ КУХНИ | Nguyên liệu món dân tộc |  | 7 | 1.247 | 0 | 0% |
| ИНГРЕДИЕНТЫ ДЛЯ ПРИХОДА МЯСА | Nguyên liệu quầy thịt (kỹ thuật) |  | 24 | 938 | 267 | 100% |
| ГЛОБАЛЬНЫЙ КАТАЛОГ | Danh mục chung (kỹ thuật) |  | 7 | 310 | 25 | 86% |
| ИНГРЕДИЕНТЫ | Nguyên liệu sản xuất tại chỗ |  | 6 | 261 | 0 | 0% |
| ИНГРЕДИЕНТЫ ДЛЯ ПЕКАРНИ | Nguyên liệu lò bánh |  | 5 | 72 | 0 | 0% |
| РЫБА СВЕЖАЯ НЕ ИСПОЛЬЗОВАТЬ | Cá tươi (không sử dụng) |  | 4 | 40 | 4 | 25% |
| АКЦИЯ ЛОЯЛЬНОСТИ | Chương trình khách hàng thân thiết |  | 18 | 34 | 0 | 0% |
| АКЦИЯ ЛОЯЛЬНОСТИ (НЕ ИСПОЛЬЗОВАТЬ) | Khách hàng thân thiết (không sử dụng) |  | 1 | 5 | 0 | 0% |
| СДОБНЫЕ ИЗДЕЛИЯ(НЕ ИСПОЛЬЗОВАТЬ) | Bánh mì ngọt (không sử dụng) |  | 1 | 2 | 0 | 0% |
| КАФЕ ВОЛКОНСКИЙ | Quầy cà phê Volkonsky |  | 0 | 0 | 0 |  |
| КОЛБАСНАЯ ГАСТРОНОМИЯ НЕ ИСПОЛЬЗОВАТЬ | Xúc xích (không sử dụng) |  | 0 | 0 | 0 |  |
| РЫБНЫЕ П/Ф НЕ ИСПОЛЬЗОВАТЬ | Cá sơ chế (không sử dụng) |  | 0 | 0 | 0 |  |
| РЫБКИ К ПИВУ (НЕ ИСПОЛЬЗОВАТЬ) | Cá nhắm bia (không sử dụng) |  | 0 | 0 | 0 |  |
| ПОЛУФАБРИКАТЫ ЦПП | Bán thành phẩm bếp trung tâm |  | 0 | 0 | 0 |  |
| ТОРТЫ.ПИРОЖНЫЕ (НЕ ИСПОЛЬЗОВАТЬ) | Bánh kem (không sử dụng) |  | 0 | 0 | 0 |  |
