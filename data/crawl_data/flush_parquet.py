import os
import sys
import logging

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from ducklake.singleton_manager import DuckLakeSingleton

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def export_data_to_parquet():
    duckdb_data_path = os.path.join(workspace_root, "lakehouse_data")
    db_file_path = os.path.join(duckdb_data_path, "investing_catalog.duckdb")
    
    logger.info("Đang trích xuất dữ liệu từ DuckDB Catalog ra file Parquet có phân vùng...")
    
    with DuckLakeSingleton.get_connection(data_path=duckdb_data_path, db_file=db_file_path) as lake:
        
        # Đường dẫn thư mục xuất
        bronze_dir = os.path.join(duckdb_data_path, "main", "fpt_news_bronze").replace("\\", "/")
        silver_dir = os.path.join(duckdb_data_path, "main", "fpt_news_silver").replace("\\", "/")
        
        # Dùng lệnh COPY chuẩn của DuckDB để rải file Parquet theo cấu trúc Hive Partition
        try:
            lake.execute_query(f"COPY fpt_news_bronze TO '{bronze_dir}' (FORMAT PARQUET, PARTITION_BY (crawl_year, crawl_month), OVERWRITE_OR_IGNORE);")
            logger.info(f"Đã xuất thành công Bronze Parquet tại: {bronze_dir}")
        except Exception as e:
            logger.error(f"Bronze hiện tại chưa có dữ liệu hoặc gặp lỗi: {e}")
            
        try:
            lake.execute_query(f"COPY fpt_news_silver TO '{silver_dir}' (FORMAT PARQUET, PARTITION_BY (news_year, news_month), OVERWRITE_OR_IGNORE);")
            logger.info(f"Đã xuất thành công Silver Parquet tại: {silver_dir}")
        except Exception as e:
            logger.error(f"Silver hiện tại chưa có dữ liệu hoặc gặp lỗi: {e}")
            
    logger.info("Hoàn tất quy trình ấn định Parquet.")

if __name__ == "__main__":
    export_data_to_parquet()
