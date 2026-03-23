# Hướng Dẫn Crawl Dữ Liệu (Playwright & Proxies)

Tài liệu này giải thích cơ chế hoạt động và cách sử dụng script `crawl_investing.py` để lấy dữ liệu HTML thô từ các trang web cấu trúc phức tạp hoặc có hệ thống chống bot (như investing.com).

---

## 1. Tổng Quan

Script `data/crawl_data/crawl_investing.py` sử dụng thư viện **Playwright** thay vì `requests` truyền thống bởi vì:
- Các trang web tài chính thường dùng **JavaScript để render** nội dung (Dynamic DOM).
- Các trang này sử dụng các hệ thống phát hiện bot (Anti-Bot detection như Cloudflare, Incapsula).
- Cần cơ chế Stealth (ẩn thân) ngụy trang như một người dùng trình duyệt thật.

Hiện tại, mục tiêu chính của script là **fetch (tải)** toàn bộ mã nguồn HTML sau khi trang tải xong mà chưa bóc tách (parse) dữ liệu cụ thể. Khi đã có HTML, bạn có thể phân tích bằng `BeautifulSoup` hoặc thư viện tương tự.

---

## 2. Yêu Cầu Chẩn Bị (Prerequisites)

Để trình duyệt có thể đổi IP liên tục trốn block, bạn cần:

1. **Danh sách Proxy**: Đã khai báo tại `constants/proxies.py`. Script sẽ random một proxy mỗi lần chạy qua hàm `get_random_proxy()`.
2. **Biến môi trường (Environment Variables)**: Nếu Proxy của bạn yêu cầu xác thực, hãy cấu hình tài khoản ở file `.env` tại thư mục gốc của dự án:
   ```env
   PROXY_USERNAME=username_cua_ban
   PROXY_PASSWORD=mat_khau_cua_ban
   ```

*(Lưu ý: Nếu proxy IP không cần xác thực, bạn có thể để trống hai biến này).*

---

## 3. Cơ Chế Hoạt Động ("Stealth Mode")

Hàm `fetch_investing_page(url)` thực hiện các bước sau:

1. **Khởi tạo Proxy**: Gắn cấu hình proxy lấy từ file `constants/proxies.py` vào trình duyệt Chromium.
2. **Tắt cờ Automation**: Khởi tạo Chromium với một loạt các `args` (vd: `--disable-blink-features=AutomationControlled`) nhằm qua mặt hệ thống kiểm tra bot mặc định.
3. **Mô phỏng User-Agent**: Định danh trình duyệt như Firefox/Chrome thông thường trên HĐH Windows, cùng kích thước màn hình phổ biến (`1366x768`) và Locale chuẩn (`en-US`).
4. **Xóa cờ Webdriver**: Bơm đoạn JavaScript nhỏ `Object.defineProperty(navigator, 'webdriver', { get: () => undefined });` vào trang trước khi tải, chặn Web xử lý nhận diện lệnh tự động (`navigator.webdriver == true`).
5. **Cơ chế Retry (Thử lại)**: Hàm sẽ kiên nhẫn thử tải trang tối đa 3 lần (`max_retries = 3`), chờ sau mỗi lần rớt mạng hoặc Time-out (Giới hạn timeout khá xa, lên đến 60s cho load xong HTML DOM cứng).
6. **Trả về kết quả**: Gọi `page.content()` lấy toàn bộ mã nguồn HTML hoàn chỉnh giao lại cho Python.

---

## 4. Cách Sử Dụng Tập Lệnh

Bạn có thể dễ dàng gọi hàm này ở trong luồng lấy dữ liệu (Data Pipeline) hoặc file chạy phân tích độc lập. 

Ví dụ:

```python
from data.crawl_data.crawl_investing import fetch_investing_page
from bs4 import BeautifulSoup

# 1. URL mục tiêu bạn muốn lấy data
target_url = "https://www.investing.com/indices/vn-historical-data"

# 2. Lấy HTML thô (raw)
html_content = fetch_investing_page(target_url)

# 3. Kiểm tra xem có lấy được không
if html_content:
    print(f"Thành công! Lấy được mã nguồn dài {len(html_content)} ký tự.")
    
    # 4. Truyền sang BeautifulSoup để trích xuất Data bảng/dòng
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Ở đây tùy cấu trúc DOM web bạn quan sát được để bóc tách:
    # table = soup.find("table", {"class": "historicalTbl"})
    # for row in table.find_all("tr"):
    #      ... xử lý lưu vô bảng
else:
    print("Thất bại. Có thể do lỗi mạng, bị chặn hoặc sai Proxy.")
```

---

## 5. Các Chú Ý & Nâng Cấp Tương Lai

- **Đồng bộ với DuckDB**: Khi bạn đã bóc tách dữ liệu ra dạng Table/JSON/Dictionary thành công trong tương lai, bạn có thể đẩy thẳng chúng vào Polars DataFrame và gọi `lake.insert_dataframe()` (từ thư viện `ducklake` của bạn) để lưu trữ vĩnh viễn trong CSDL Lakehouse.
- **Xử lý Cookie Banners**: Phần cứng (Pop-up Accept Cookie) trên trang thường cản trở việc nhấp nút/Load elements. Hiện tại trong code có một đoạn bị comment: `page.click("text=Accept All Cookies")`. Bạn có thể bỏ comment và tùy chỉnh bộ lọc (text=...) theo trang thực tế.
- **Trích xuất thông qua JSON Network (Nâng cao)**: Đôi khi các trang như Investing không chứa bảng thẳng trên HTML, mà Fetch bằng AJAX ngầm. Nếu vậy, sau này thay vì bắt html `page.content()`, ta sẽ bắt request API của nó qua module `page.on("response", handler)`.
