# Hướng dẫn Xây dựng Features (Feature Engineering) cho Mô hình VRNN

Tài liệu này giải thích chi tiết các đặc trưng (features) được sử dụng để huấn luyện mô hình VRNN trong nghiên cứu tâm lý thị trường chứng khoán Việt Nam. Toàn bộ logic được triển khai trong class `FeatureBuilder` tại file `utils/build_features.py`.

## 1. Tổng quan quy trình (Pipeline)

Quy trình xử lý dữ liệu bao gồm 4 bước chính:

1.  **Tải và làm sạch dữ liệu (`load_data`)**: Chuyển đổi định dạng ngày tháng, xử lý các ký tự đặc biệt trong giá (như dấu phẩy) và khối lượng (như 'M' cho triệu, 'K' cho nghìn).
2.  **Tính toán chỉ số (`build_features`)**: Tạo ra các đặc trưng kỹ thuật từ dữ liệu giá (Open, High, Low, Price/Close) và khối lượng.
3.  **Chuẩn hóa (`normalize`)**: Đưa các giá trị về cùng một thang đo (Z-score normalization) để mô hình học tập hiệu quả hơn: $x' = \frac{x - \mu}{\sigma}$.
4.  **Chuyển đổi sang Tensor (`to_tensor`)**: Định dạng lại dữ liệu để đưa vào PyTorch.

---

## 2. Chi tiết các nhóm Features

### A. Nhóm Động Lực Học Giá (Price Dynamics)
Các chỉ số này mô tả sự thay đổi và cấu trúc của nến giá trong ngày.

*   **`log_return`**: Lợi nhuận logarit ($\ln(P_t) - \ln(P_{t-1})$). Chỉ số này giúp dữ liệu có tính ổn định (stationary) tốt hơn so với giá gốc.
*   **`oc_return`**: Lợi nhuận nội ngày (Open-to-Close). Phản ánh tâm lý thị trường từ khi mở cửa đến lúc đóng cửa.
*   **`hl_range`**: Biên độ nến (High-to-Low) so với giá. Cho thấy mức độ dao động tối đa trong phiên.
*   **`upper_shadow`**: Độ dài bóng nến trên so với giá. Thể hiện áp lực bán tại vùng giá cao.
*   **`lower_shadow`**: Độ dài bóng nến dưới so với giá. Thể hiện lực cầu bắt đáy tại vùng giá thấp.

### B. Nhóm Biến Động (Volatility)
Đo lường mức độ rủi ro và "sự hoảng loạn" hoặc "hưng phấn" của thị trường.

*   **`abs_return`**: Giá trị tuyệt đối của log return. Một thước đo đơn giản cho biến động tức thời.
*   **`volatility_10` / `volatility_20`**: Độ lệch chuẩn của lợi nhuận trong 10 và 20 phiên gần nhất.
*   **`range_vol`**: Trung bình động 10 phiên của biên độ High-Low.

### C. Nhóm Động Lượng và Xu Hướng (Momentum & Trend)
Xác định hướng đi và sức mạnh của xu hướng giá.

*   **`momentum_5`**: Lợi nhuận lũy kế trong 5 phiên gần nhất.
*   **`ma20`**: Đường trung bình động 20 phiên.
*   **`ma_gap`**: Khoảng cách giữa giá hiện tại và đường MA20. Khi gap quá lớn, giá thường có xu hướng đảo chiều về vùng trung bình.
*   **`trend_strength`**: Hiệu số giữa MA5 và MA20. Giá trị dương lớn thể hiện xu hướng tăng mạnh.

### D. Nhóm Khối Lượng (Volume)
Phản ánh mức độ quan tâm của nhà đầu tư và tính thanh khoản.

*   **`volume`**: Khối lượng giao dịch gốc đã qua xử lý đơn vị.
*   **`volume_change`**: Phần trăm thay đổi khối lượng so với phiên trước.
*   **`volume_z`**: Chỉ số Z-score của khối lượng trong 20 phiên. Giúp xác định các phiên đột biến khối lượng (Breakout).

---

## 3. Cách sử dụng

Bạn có thể sử dụng `FeatureBuilder` thông qua file `utils/main.py`:

```python
from .build_features import FeatureBuilder

# 1. Khởi tạo
builder = FeatureBuilder()

# 2. Xây dựng toàn bộ pipeline (từ CSV -> Feature DataFrame)
df, feature_df, mean, std = builder.build("data/FPT_History.csv")

# 3. Lưu kết quả
builder.save_features(feature_df, "data/processed_features.csv")
```

---
*Tài liệu này hỗ trợ cho Luận văn Tốt nghiệp (KLTN) - Nghiên cứu tâm lý thị trường chứng khoán Việt Nam sử dụng VRNN.*
