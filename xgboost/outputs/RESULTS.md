# Kết quả tuning XGBoost

Nguồn: `outputs/tuning_results.csv`, `outputs/train_log.txt` (xgboost 3.4.1, script `xgboost_train.py`).

**Mô hình được chọn: `tweedie1.2_rs02`**, với MSE trên tập valid là **76.11** (RMSE ≈ 8.72), MAE 2.437, WAPE 54.0%.
Checkpoint: `outputs/checkpoints/tweedie1.2_rs02.json`. Trên tập test, mô hình này đạt MAE 2.627, WAPE 58.7% (xem mục 6).

## 1. Dữ liệu và cách chia tập

| Tập   | Thời gian               | Số dòng                                          |
| ----- | ----------------------- | ------------------------------------------------ |
| Train | 2022-08-28 → 2024-03-31 | 814,951 (796,327 sau khi bỏ 18,624 dòng warm-up) |
| Valid | 2024-04-01 → 2024-06-30 | 204,663                                          |
| Test  | 2024-07-01 → 2024-09-26 | 177,713                                          |

- Tổng cộng 1,197,327 dòng và 55 đặc trưng, gồm 4 cửa hàng. Cửa hàng 1–3 có dữ liệu từ 2022-08-28, cửa hàng 4 chỉ có từ 2023-12-14.
- Tỉ lệ NaN của các đặc trưng mức bán lịch sử, ở train và valid:

| Đặc trưng           | Train | Valid |
| ------------------- | ----: | ----: |
| item_store_mean     | 2.72% | 1.35% |
| item_mean           | 1.77% | 0.89% |
| subclass_store_mean | 1.36% | 0.02% |

## 2. Cách tuning

- Có 20 thí nghiệm: 4 cấu hình mặc định (Tweedie với p = 1.2 / 1.4 / 1.6 và MSE), cộng 16 cấu hình chọn ngẫu nhiên (`rs00`–`rs15`).
- Mỗi thí nghiệm train trên tập train, tối đa 3000 vòng. Mô hình dừng sớm sau 100 vòng nếu RMSE trên valid không cải thiện.
- Mô hình được chọn là mô hình có MSE trên valid thấp nhất (dự báo được cắt về ≥ 0 trước khi tính). Mô hình không được train lại trên train+valid.
- Tổng thời gian train khoảng 12 phút (733 giây).

## 3. Kết quả tốt nhất của từng loại hàm mục tiêu

| Hàm mục tiêu       | Số lần chạy | Lần chạy tốt nhất | MSE valid |       MAE |      WAPE |
| ------------------ | ----------: | ----------------- | --------: | --------: | --------: |
| **Tweedie p=1.2**  |          11 | `tweedie1.2_rs02` | **76.11** | **2.437** | **54.0%** |
| MSE (squarederror) |           4 | `mse_rs04`        |     76.28 |     2.518 |     55.8% |
| Tweedie p=1.4      |           3 | `tweedie1.4_rs13` |     78.59 |     2.451 |     54.3% |
| Tweedie p=1.6      |           3 | `tweedie1.6_rs11` |     78.64 |     2.465 |     54.6% |

## 4. Toàn bộ kết quả (sắp theo MSE valid)

|   # | Thí nghiệm          |       MSE |   MAE |  WAPE | Vòng tốt nhất |   lr | depth | mcw | subsample | colsample |   λ | Thời gian (s) |
| --: | ------------------- | --------: | ----: | ----: | ------------: | ---: | ----: | --: | --------: | --------: | --: | ------------: |
|   1 | **tweedie1.2_rs02** | **76.11** | 2.437 | 54.0% |           596 | 0.03 |     6 |   5 |       0.7 |       1.0 | 5.0 |          73.4 |
|   2 | mse_rs04            |     76.28 | 2.518 | 55.8% |           191 | 0.03 |     8 |  50 |       0.8 |       0.6 | 0.5 |          34.2 |
|   3 | mse_rs15            |     76.45 | 2.523 | 55.9% |           179 | 0.05 |     6 |   5 |       1.0 |       1.0 | 1.0 |          22.9 |
|   4 | tweedie1.2_rs06     |     76.86 | 2.445 | 54.2% |           354 | 0.05 |     6 |  20 |       0.8 |       1.0 | 1.0 |          48.1 |
|   5 | mse_base            |     77.08 | 2.522 | 55.9% |           126 | 0.05 |     8 |  20 |       0.8 |       0.8 | 1.0 |          14.4 |
|   6 | tweedie1.2_rs03     |     77.19 | 2.456 | 54.4% |           161 |  0.1 |     6 |  50 |       1.0 |       1.0 | 5.0 |          30.1 |
|   7 | mse_rs05            |     77.35 | 2.526 | 56.0% |            98 | 0.05 |     8 |   5 |       0.7 |       0.8 | 0.5 |          26.0 |
|   8 | tweedie1.2_rs10     |     78.40 | 2.472 | 54.8% |           122 | 0.05 |     8 |  20 |       1.0 |       0.8 | 0.5 |          38.8 |
|   9 | tweedie1.4_rs13     |     78.59 | 2.451 | 54.3% |           212 | 0.05 |     8 |  20 |       1.0 |       1.0 | 5.0 |          46.5 |
|  10 | tweedie1.6_rs11     |     78.64 | 2.465 | 54.6% |           210 | 0.05 |     6 |  50 |       0.8 |       1.0 | 5.0 |          40.3 |
|  11 | tweedie1.2_base     |     78.65 | 2.473 | 54.8% |           122 | 0.05 |     8 |  20 |       0.8 |       0.8 | 1.0 |          36.1 |
|  12 | tweedie1.4_base     |     79.00 | 2.444 | 54.2% |           235 | 0.05 |     8 |  20 |       0.8 |       0.8 | 1.0 |          35.2 |
|  13 | tweedie1.2_rs07     |     79.22 | 2.505 | 55.5% |            41 |  0.1 |     8 |  50 |       0.7 |       0.8 | 0.5 |          20.5 |
|  14 | tweedie1.6_base     |     79.62 | 2.447 | 54.2% |           215 | 0.05 |     8 |  20 |       0.8 |       0.8 | 1.0 |          32.2 |
|  15 | tweedie1.4_rs14     |     80.57 | 2.480 | 54.9% |            53 |  0.1 |     8 |   5 |       0.7 |       0.6 | 1.0 |          24.8 |
|  16 | tweedie1.2_rs00     |     80.78 | 2.470 | 54.7% |           166 | 0.03 |    10 |  20 |       0.7 |       0.6 | 0.5 |          41.1 |
|  17 | tweedie1.2_rs09     |     81.23 | 2.481 | 55.0% |           159 | 0.03 |    10 |   5 |       0.8 |       0.6 | 0.5 |          67.0 |
|  18 | tweedie1.2_rs12     |     81.27 | 2.493 | 55.2% |            49 |  0.1 |    10 |   5 |       1.0 |       1.0 | 0.5 |          34.0 |
|  19 | tweedie1.2_rs01     |     82.15 | 2.503 | 55.5% |            44 |  0.1 |    10 |  50 |       0.7 |       1.0 | 1.0 |          33.0 |
|  20 | tweedie1.6_rs08     |     85.12 | 2.484 | 55.0% |            58 |  0.1 |    10 |  20 |       1.0 |       0.6 | 5.0 |          34.6 |

*mcw = `min_child_weight`, λ = `reg_lambda`.*

## 5. Nhận xét

1. **Tweedie 1.2 tốt nhất trên cả ba thước đo.** Về MSE, nó chỉ hơn mô hình MSE tốt nhất 0.17 (khoảng 0.2%). Nhưng WAPE thấp hơn khoảng 1.8 điểm phần trăm và MAE thấp hơn khoảng 0.08. Mọi mô hình dùng hàm mục tiêu MSE đều có WAPE 55.8–56.0%, cao hơn tất cả mô hình Tweedie.
2. **Cây sâu (`max_depth=10`) cho kết quả kém nhất.** Cả 5 lần chạy này có MSE từ 80.8 đến 85.1, nhiều khả năng do overfit.
3. **Learning rate 0.1 làm mô hình dừng quá sớm** (khoảng 40–60 vòng khi depth ≥ 8) và kết quả kém hơn. Các mô hình tốt dùng lr 0.03–0.05 với depth 6–8.
4. **Tăng hệ số Tweedie lên 1.4 hoặc 1.6 không cải thiện MSE.** Ở cấu hình mặc định, MSE tăng dần theo hệ số: 78.65 (1.2) → 79.00 (1.4) → 79.62 (1.6).
5. Sáu cấu hình đầu bảng có MSE sát nhau (76.1–77.4), nên kết quả chọn mô hình theo một tập valid duy nhất có thể dao động một chút.

## 6. Đánh giá chi tiết trên valid và test

Mục này so sánh **mô hình tốt nhất của mỗi hàm mục tiêu**: `mse_rs04`, `tweedie1.2_rs02`, `tweedie1.4_rs13`, `tweedie1.6_rs11`.
Các checkpoint được dùng nguyên, không train lại. Dự báo được cắt về ≥ 0 trước khi tính.
Script: `xgboost/evaluate_breakdown.py`. Số liệu đầy đủ: `outputs/breakdown_metrics.csv`.
Số liệu valid tính lại khớp với `tuning_results.csv`.

Số in **đậm** là kết quả tốt nhất trong cột (với bảng phase là tốt nhất trong hàng).

### 6.1. MAE, RMSE, WAPE tổng thể: mô hình nào dự báo chính xác hơn?

| Mô hình                             | Valid MAE | Valid RMSE | Valid WAPE |  Test MAE |  Test RMSE | Test WAPE |
| ----------------------------------- | --------: | ---------: | ---------: | --------: | ---------: | --------: |
| MSE (`mse_rs04`)                    |     2.518 |      8.734 |      55.8% |     2.690 | **11.789** |     60.1% |
| **Tweedie 1.2 (`tweedie1.2_rs02`)** | **2.438** |  **8.724** |  **54.0%** | **2.627** |     12.056 | **58.7%** |
| Tweedie 1.4 (`tweedie1.4_rs13`)     |     2.451 |      8.865 |      54.3% |     2.632 |     12.033 |     58.8% |
| Tweedie 1.6 (`tweedie1.6_rs11`)     |     2.465 |      8.868 |      54.6% |     2.633 |     11.902 |     58.8% |

- **Tweedie 1.2 chính xác nhất.** Nó có MAE và WAPE thấp nhất trên cả valid và test. Trên test, WAPE thấp hơn mô hình MSE 1.4 điểm phần trăm.
- **Riêng RMSE trên test, mô hình MSE tốt nhất** (11.79 so với 12.06). Hàm mục tiêu MSE phạt nặng sai số lớn, nên bắt các ngày bán đột biến tốt hơn một chút.
- Sai số trên test lớn hơn valid ở cả 4 mô hình: WAPE tăng khoảng 4.5 điểm phần trăm, RMSE tăng từ khoảng 8.7 lên khoảng 12. Giai đoạn tháng 7–9/2024 có nhiều ngày bán đột biến hơn.

### 6.2. WAPE theo Food / Household: mô hình nào phù hợp với từng ngành hàng?

| Mô hình         | Valid Food | Valid Household | Test Food | Test Household |
| --------------- | ---------: | --------------: | --------: | -------------: |
| MSE             |      52.2% |           87.5% |     56.6% |          86.1% |
| **Tweedie 1.2** |  **50.6%** |           84.3% | **55.4%** |          83.8% |
| **Tweedie 1.4** |      51.0% |       **83.6%** |     55.5% |      **83.5%** |
| Tweedie 1.6     |      51.2% |           84.3% |     55.5% |          83.8% |
| *Số dòng*       |  *109,289* |        *95,374* |  *87,185* |       *90,528* |

- **Food hợp với Tweedie 1.2**, tốt nhất trên cả valid và test.
- **Household hợp với Tweedie 1.4**, tốt nhất trên cả valid và test. Tuy nhiên nó chỉ hơn Tweedie 1.2 khoảng 0.3–0.7 điểm phần trăm, nên khác biệt nhỏ.
- **Household khó dự báo hơn food rất nhiều**: WAPE khoảng 84% so với khoảng 55%. Hàng household bán lẻ tẻ hơn, nhiều ngày bán 0 hơn. Hệ số Tweedie cao hơn (1.4) mô tả kiểu dữ liệu này tốt hơn một chút.
- Mô hình MSE kém nhất ở cả hai ngành hàng. Khoảng cách lớn nhất nằm ở household (86.1% so với 83.5% trên test).

### 6.3. MAE, WAPE theo phase khuyến mãi: mô hình nào dự báo tốt trong khuyến mãi?

Ý nghĩa các phase: `pre_1_3` = 1–3 ngày trước khuyến mãi, `during` = đang khuyến mãi, `post_1_3` / `post_4_7` = 1–3 / 4–7 ngày sau khuyến mãi, `other` = ngày thường.

| Phase      | Mô hình hiệu quả nhất | Test MAE | Test WAPE | Ghi chú                                                      |
| ---------- | --------------------- | -------: | --------: | ------------------------------------------------------------ |
| pre_1_3    | **Tweedie 1.2**       |    1.730 |    68.74% | Tốt nhất trên cả valid và test                               |
| **during** | **Tweedie 1.2**       |    2.857 |    56.50% | Tốt nhất trên cả valid và test                               |
| post_1_3   | Tweedie 1.6           |    2.762 |    58.60% | Ngang Tweedie 1.2 (58.62%). Trên valid, Tweedie 1.2 tốt nhất |
| post_4_7   | **Tweedie 1.2**       |    2.317 |    58.98% | Tốt nhất trên cả valid và test                               |
| other      | Tweedie 1.4           |    2.625 |    58.85% | Ngang Tweedie 1.2 (58.93%). Trên valid, Tweedie 1.2 tốt nhất |

*Chọn theo kết quả trên test. Số liệu chi tiết của cả 4 mô hình nằm trong `outputs/breakdown_metrics.csv`.*

- **Trong khuyến mãi (`during`), Tweedie 1.2 tốt nhất cả MAE và WAPE**, trên cả valid và test. Trên test, WAPE thấp hơn mô hình MSE 1.9 điểm phần trăm.
- **Sau khuyến mãi, lợi thế của Tweedie so với MSE lớn nhất.** Ở `post_4_7` trên test, WAPE là 59.0% so với 62.2%. Mô hình MSE có xu hướng dự báo "dư âm" của khuyến mãi kém hơn.
- Ở `post_1_3` và `other` trên test, Tweedie 1.6 hoặc 1.4 nhỉnh hơn Tweedie 1.2, nhưng chỉ chênh dưới 0.1 điểm phần trăm, có thể coi là ngang nhau.
- `pre_1_3` có WAPE cao nhất (khoảng 69% trên test), nhưng mẫu nhỏ (khoảng 4.9 nghìn dòng). Dự báo ngay trước khuyến mãi khó vì người mua có thể chờ đến đợt giảm giá.

### 6.4. Tổng kết

| Câu hỏi                                  | Trả lời                                                                                            |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Mô hình nào dự báo chính xác hơn?        | **Tweedie 1.2**: MAE và WAPE thấp nhất trên cả valid và test. Riêng RMSE trên test thì MSE tốt hơn |
| Mô hình nào hợp với food?                | **Tweedie 1.2** (test WAPE 55.4%)                                                                  |
| Mô hình nào hợp với household?           | **Tweedie 1.4** (test WAPE 83.5%), chỉ nhỉnh hơn Tweedie 1.2 (83.8%)                               |
| Mô hình nào dự báo tốt trong khuyến mãi? | **Tweedie 1.2** (test `during`: MAE 2.857, WAPE 56.5%), và cũng tốt nhất giai đoạn sau khuyến mãi  |

## 7. Việc tiếp theo

- So với một mô hình nền đơn giản (ví dụ trung bình bán của từng mặt hàng ở từng cửa hàng) để biết XGBoost hơn được bao nhiêu, vì WAPE khoảng 54% vẫn còn cao.
