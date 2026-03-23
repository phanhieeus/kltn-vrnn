import os
import sys
import asyncio
import polars as pl
from datetime import datetime
import logging

# Thêm workspace root vào sys.path để import thư viện nội bộ
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from data.crawl_data.crawl_investing import crawl_fpt_news_async
from data.crawl_data.extract_investing import extract_news_from_html_content
from ducklake.singleton_manager import DuckLakeSingleton

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

async def run_pipeline():
    logger.info("=== BẮT ĐẦU PIPELINE FPT NEWS (REAL-TIME STREAMING) ===")
    
    start_page = 1
    end_page = 405
    
    # Cấu hình đường dẫn Lakehouse
    duckdb_data_path = os.path.join(workspace_root, "lakehouse_data")
    os.makedirs(duckdb_data_path, exist_ok=True)
    db_file_path = os.path.join(duckdb_data_path, "investing_catalog.duckdb")
    
    # Trackers chống trùng lặp dữ liệu trên RAM (không cần Deduplicate DB sinh file rác)
    seen_bronze_files = set()
    seen_silver_urls = set()
    
    db_lock = asyncio.Lock()
    
    # 1. Mở Connection suốt phiên chạy
    with DuckLakeSingleton.get_connection(data_path=duckdb_data_path, db_file=db_file_path) as lake:
        
        # Tắt auto-checkpoint ngăn Windows khóa file WAL ngầm
        try:
            lake.execute_query("PRAGMA wal_autocheckpoint='1GB';")
        except Exception:
            pass
        
        # 2. Khởi tạo Tables với tính năng PARTITION BY năm/tháng
        logger.info("Đang khởi tạo Bronze và Silver Tables (Partition: Year, Month)...")
        
        bronze_schema = {
            "source_file": "VARCHAR",
            "raw_html": "VARCHAR",
            "crawled_at": "TIMESTAMP",
            "crawl_year": "INTEGER",
            "crawl_month": "INTEGER"
        }
        lake.create_table("fpt_news_bronze", schema=bronze_schema, partition_by=["crawl_year", "crawl_month"])
        
        silver_schema = {
            "title": "VARCHAR",
            "url": "VARCHAR",
            "publish_time": "VARCHAR",
            "datetime": "VARCHAR",
            "provider": "VARCHAR",
            "description": "VARCHAR",
            "source_file": "VARCHAR",
            "extracted_at": "TIMESTAMP",
            "news_year": "INTEGER",
            "news_month": "INTEGER"
        }
        lake.create_table("fpt_news_silver", schema=silver_schema, partition_by=["news_year", "news_month"])
        
        # 3. Định nghĩa Webhook Callback - Hứng data trực tiếp tại RAM khi tải xong từng trang
        async def on_page_fetched(page_num, html_content):
            now = datetime.now()
            source_file = f"page_{page_num}.html"
            
            # ---------------- BRONZE TIER ----------------
            bronze_records = []
            if source_file not in seen_bronze_files:
                seen_bronze_files.add(source_file)
                bronze_records.append({
                    "source_file": source_file,
                    "raw_html": html_content,
                    "crawled_at": now,
                    "crawl_year": now.year,
                    "crawl_month": now.month
                })
            
            # ---------------- SILVER TIER ----------------
            silver_records = []
            extracted_items = extract_news_from_html_content(html_content)
            
            for item in extracted_items:
                url = item.get("url", "")
                
                # Quét lọc trùng (Đảm bảo tin tức không bị lặp khi chuyển trang)
                if not url or url in seen_silver_urls:
                    continue
                    
                seen_silver_urls.add(url)
                
                item.pop('raw_html', None) # Lọc bỏ rác
                item['extracted_at'] = now
                item['source_file'] = source_file
                
                # Cắt năm/tháng để làm khóa Partition
                dt_str = item.get("datetime", "")
                news_year, news_month = now.year, now.month # Fallback mặc định
                if dt_str:
                    try:
                        # Ví dụ parse str: "2026-03-24 14:30:00"
                        dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                        news_year = dt_obj.year
                        news_month = dt_obj.month
                    except Exception:
                        pass
                
                item['news_year'] = news_year
                item['news_month'] = news_month
                silver_records.append(item)
            
            # 4. Ghi trực tiếp mẻ dữ liệu vào Database
            async with db_lock:
                try:
                    if bronze_records:
                        lake.insert_dataframe(pl.DataFrame(bronze_records), "fpt_news_bronze", if_exists="append")
                    if silver_records:
                        lake.insert_dataframe(pl.DataFrame(silver_records), "fpt_news_silver", if_exists="append")
                    
                    logger.info(f"==> Đã lưu Trực Tiếp Trang {page_num} vào DB:  1 Bronze | {len(silver_records)} Silver News")
                except Exception as e:
                    logger.error(f"Lỗi khi nạp DB ở Trang {page_num}: {e}")

        # 5. Khởi động vòng lặp Asyncio với Callback streaming
        logger.info(f"Bắt đầu CRAWL STREAMING ({start_page}->{end_page}) - Lấy xong lưu thẳng, KHÔNG XÀI FILE Ổ CỨNG!")
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
        await crawl_fpt_news_async(
            start_page=start_page, 
            end_page=end_page, 
            concurrency_limit=5, 
            on_page_fetched=on_page_fetched
        )
        
        # Kết toán Database
        logger.info("Đóng Connection để Duck/Lake xả các Partitions xuống ổ đĩa...")
        
        # 6. Ép DuckDB xuất dữ liệu đã cấu trúc thành file Parquet vật lý phân vùng
        logger.info("Bắt đầu thao tác COPY TO Parquet Files theo phân vùng Thời gian...")
        
        bronze_dir = os.path.join(duckdb_data_path, "main", "fpt_news_bronze").replace("\\", "/")
        silver_dir = os.path.join(duckdb_data_path, "main", "fpt_news_silver").replace("\\", "/")
        
        lake.execute_query(f"COPY fpt_news_bronze TO '{bronze_dir}' (FORMAT PARQUET, PARTITION_BY (crawl_year, crawl_month), OVERWRITE_OR_IGNORE);")
        lake.execute_query(f"COPY fpt_news_silver TO '{silver_dir}' (FORMAT PARQUET, PARTITION_BY (news_year, news_month), OVERWRITE_OR_IGNORE);")
        
        logger.info(f"Đã xuất Parquet thành công vào {bronze_dir} và {silver_dir}.")

    logger.info(f"=== PIPELINE HOÀN TẤT! Tổng ghi nhận toàn chu kỳ: {len(seen_bronze_files)} Bronze, {len(seen_silver_urls)} News Silver duy nhất ===")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
