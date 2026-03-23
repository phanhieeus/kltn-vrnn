# Hướng Dẫn Sử Dụng DuckLake (DuckDB Local)

Tài liệu này cung cấp hướng dẫn chi tiết cách thức sử dụng thư viện `DuckLake` trong dự án. Bản hiện tại của `DuckLake` đã được tối ưu hóa rành riêng cho Local DuckDB, đóng vai trò như một kho lưu trữ Lakehouse nhỏ gọn với khả năng hỗ trợ giao dịch ACID, schema evolution (diễn tiến lược đồ cấu trúc) và Time Travel (du hành thời gian phiên bản).

---

## 1. Khởi tạo kết nối

Có hai cách để khởi tạo DuckLake trong dự án:

### Khởi tạo trực tiếp (Single Instance)
Dùng cho mã chạy phân tích, Jupyter Notebook hoặc script đồng bộ chạy một lần:

```python
from ducklake.ducklake import create_local_ducklake

# Khởi tạo instance Local DuckDB trong thư mục "lakehouse_data"
# Tham số data_path bắt buộc khai báo.
lake = create_local_ducklake(
    data_path="./lakehouse_data", 
    db_file="my_catalog.duckdb" # Tùy chọn: tự động tạo datetime.duckdb nếu bỏ trống
)

# Thiết lập kết nối
lake.connect()

# Làm việc với database...

# Luôn nhớ đóng kết nối khi kết thúc
lake.close()
```

Kể cả khi dùng single instance, có thể sử dụng biểu thức `with` (context manager) để tiện lợi hơn:
```python
with create_local_ducklake(data_path="./lakehouse_data") as lake:
    tables = lake.list_tables()
    print("Các bảng đang có:", tables)
```

### Khởi tạo quản lý kết nối an toàn với Luồng (Singleton)
Dùng trong ứng dụng Web hoặc Background Workers cần chia sẻ connection (tránh trường hợp truy cập đồng thời chạm quyền File/Lock của DuckDB):

```python
from ducklake.singleton_manager import DuckLakeSingleton

# Sử dụng context manager chuyên dụng từ Singleton an toàn Thread:
with DuckLakeSingleton.get_connection(data_path="./lakehouse_data", db_file="my_catalog.duckdb") as lake:
    print(lake.list_tables())
    
# Nhận trực tiếp object
lake_instance = DuckLakeSingleton.get_local_ducklake(data_path="./lakehouse_data")

# Nếu cần reset hoặc đóng dọn toàn bộ connections
DuckLakeSingleton.close_all()
```

---

## 2. Thao Tác Bảng (Table Operations)

### Tạo bảng

Bạn chỉ cần liệt kê cấu trúc kiểu dữ liệu của DuckDB (VARCHAR, INTEGER, DOUBLE, BIGINT,...):

```python
schema = {
    "id": "INTEGER",
    "symbol": "VARCHAR",
    "price": "DOUBLE",
    "timestamp": "TIMESTAMP"
}

# Tham số partition_by hỗ trợ nhóm dữ liệu vật lý trên ổ cứng 
lake.create_table(
    table_name="stock_prices",
    schema=schema,
    partition_by=["symbol"]
)
```

### Chèn và cập nhật DataFrame (Polars)

DuckLake được thiết kế để tích hợp cực sâu cùng `Polars DataFrame` để xử lý mảng và memory-efficient.

```python
import polars as pl
from datetime import datetime

df = pl.DataFrame({
    "id": [1, 2],
    "symbol": ["FPT", "VNM"],
    "price": [130.5, 65.2],
    "timestamp": [datetime.now(), datetime.now()]
})

# Chèn thêm vào DB (Append) - mặc định
lake.insert_dataframe(df, "stock_prices", if_exists="append")
# Hoặc hàm viết tắt tương tự
lake.append_dataframe(df, "stock_prices")

# Xóa trắng bảng cũ và thay thế bằng Dataframe mới
lake.replace_dataframe(df, "stock_prices")
```

---

## 3. Truy vấn Dữ Liệu (Query)

Hai hàm phổ biến là `execute_query` (thực thi không cần trả về) và `query` (chạy và trả về kết quả).

```python
# Insert thông qua SQL Statement truyền thống không lấy kết quả trả về
lake.execute_query("INSERT INTO stock_prices VALUES (3, 'VIC', 45.0, now())")

# Truy vấn trả về List (dạng Tuple Python thuần)
results = lake.query("SELECT * FROM stock_prices WHERE price > 50")
print(results)

# Truy vấn và convert tự động ra thẳng Polars Dataframe
df_results = lake.query("SELECT * FROM stock_prices", fetch_df=True)
print(df_results.head())
```

---

## 4. Quản Lý Tính Năng Nâng Cao của Lakehouse

Lakehouse cung cấp một số tính năng đặc thù (ACID/Snapshots/Time Travel).

### Dọn dẹp & Tối ưu hóa Database

- **Loại bỏ File Rác (Cleanup)**: Nếu bạn sửa/xóa nhiều dữ liệu, file cứng vẫn chiếm dung lượng. Hàm sau sẽ gọi dọn dẹp các files bị treo hoặc không thuộc Snapshot nào.
    ```python
    lake.cleanup_old_files()
    ```

- **Gộp File (Merge Adjacent Files)**: Nếu thực hiện quá nhiều đợt Insert siêu bé (vài rows), duckdb sẽ sinh ra chục file parquet nhỏ. Dùng script sau để gom chúng thành chunk to, tăng tốc độ truy vấn ở lần load sau.
    ```python
    lake.merge_adjacent_files("stock_prices")
    ```

- **Lược bỏ Dữ Liệu Bị Trùng Lặp (Deduplicate)**: 
    ```python
    # Bắt buộc khai báo trường ID (id_column) làm cơ sở căn chuẩn để xóa duplicate row
    # Cơ chế: hệ thống sẽ giữ lại ROW được tạo gần nhất (timestamp mới nhất) và xóa các ROW có cùng ID cũ hơn.
    lake.deduplicate("stock_prices", id_column="id")
    ```

### Hệ thống Snapshot (Phiên bản & Time Travel)

Ducklake ngầm lưu lại toàn bộ các Snapshot mọi thay đổi trong CSDL của bạn giống như hàm commit ở Github.

```python
# Trích xuất Metadata Bảng để biết Snapshot, Schema cụ thể:
info = lake.get_ducklake_table_info("stock_prices")
print("Số Snapshot hiện lên:", len(info["snapshots"])) 

# Lấy Version Mới Nhất
current_ver = lake.get_current_version()
print(f"Bảng hiện đang ở Version: {current_ver}")

# Xem thử CÁC DÒNG NÀO vừa được CHÈN MỚI vào CSDL giữa Version 1 và Version 3:
inserted_rows = lake.get_table_insertions("stock_prices", from_version=1, to_version=3)

# Hoặc xem các dòng NÀO BỊ XÓA (Deleted)
deleted_rows = lake.get_table_deletions("stock_prices", from_version=1, to_version=3)

# Xóa bớt các Snapshot đã cũ (tránh tốn bộ nhớ)
lake.expire_snapshots(older_than="7d") # Xóa mọi phiên bản cũ hơn 7 ngày
lake.expire_snapshots(versions=[1, 2, 3]) # Xóa đích danh ID
```
