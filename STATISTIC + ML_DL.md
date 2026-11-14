

|  | Câu hỏi cần trả lời |
| :---- | :---- |
| Statistic: Two-way ANOVA  | Post-Promotion Dip có khác nhau giữa hai nhóm (food và household) không?  Khác biệt giữa hai nhóm  (food và household) có liên quan riêng đến khuyến mãi không?  |
| Statistic: **paired t-test**.   | Lượng bán trong KM có khác trước KM không?  Lượng bán sau KM có thấp hơn trước KM không?  |
| ML /DL  | TẤT CẢ MODEL ĐỀU THỐNG NHẤT QUYẾT ĐỊNH DỰ BÁO Ở CẤP Sản phẩm × cửa hàng (nếu được) Cần xem xét `store_id`, thời điểm mà các store bắt đầu có dữ liệu và các thay đổi cơ cấu  Nên xem xét thông tin khuyến mãi như một nhóm feature đầu vào khi dự báo nhu cầu mua hàng.  có thể thử feature như `days_since_promotion_end` hoặc trạng thái hậu khuyến mãi,  Nên xem xét `category`, `promo_status` và `discount_depth` Có thể thử bổ sung các biến thể hiện thời gian kể từ khi chương trình khuyến mãi kết thúc để dự báo lượng bán. Nên thử sử dụng `discount_depth` như một feature khi dự báo nhu cầu mua hàng. Có thể thử các tương tác giữa mức chiết khấu (discount depth: giá giảm bao nhiêu so với giá bán thông thường) và nhóm sản phẩm. Nên thử các feature `day_of_week`, `is_friday`, `is_weekend` và độ trễ 7 ngày để cải thiện dự báo. Có thể thử các feature `month`, `holiday_status` và `days_to_holiday`. |

