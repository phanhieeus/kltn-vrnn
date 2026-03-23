import os
import glob
import json
import logging
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def extract_news_from_html_content(html_content: str) -> list:
    """
    Trích xuất danh sách tin tức từ chuỗi HTML thô (không cần lưu file).
    """
    news_items = []
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Tìm thẻ ul chứa danh sách tin tức
    news_list_ul = soup.find('ul', {'data-test': 'news-list'})
    
    if not news_list_ul:
        logger.warning(f"Không tìm thấy <ul data-test='news-list'> trong chuỗi HTML.")
        return news_items
    
    # Duyệt qua các thẻ con (thường là <li>)
    list_items = news_list_ul.find_all('li', recursive=False)
    
    for item in list_items:
        try:
            # Lấy Title và URL thông qua data-test="article-title-link"
            title_tag = item.find('a', {'data-test': 'article-title-link'})
            if not title_tag:
                continue
                
            url = title_tag.get('href', '')
            if url and not url.startswith('http'):
                url = 'https://vn.investing.com' + url
                
            title = title_tag.get_text(strip=True)
            
            # Lấy Ngày tháng phát hành từ thẻ time
            time_tag = item.find('time', {'data-test': 'article-publish-date'})
            publish_time = time_tag.get_text(strip=True) if time_tag else ''
            datetime_val = time_tag.get('datetime', '') if time_tag else ''
            
            # Lấy Nhà cung cấp (Tác giả)
            provider_tag = item.find('a', {'data-test': 'article-provider-link'})
            if not provider_tag:
                # Fallback, có đôi khi Nguồn không có thẻ a mà là span
                provider_tag = item.find('span', {'data-test': 'article-provider-name'})
            provider = provider_tag.get_text(strip=True) if provider_tag else ''
            
            # Trích xuất đoạn mô tả (snippet/summary)
            description = ""
            desc_tag = item.find('p', {'data-test': 'article-description'})
            if desc_tag:
                description = desc_tag.get_text(strip=True)

            news_dict = {
                "title": title,
                "url": url,
                "publish_time": publish_time,
                "datetime": datetime_val,
                "provider": provider,
                "description": description,
                "raw_html": str(item) # Lưu lại HTML thô của node này để đối chiếu tương lai
            }
            
            # Lọc bỏ những dòng không phải tin tức thật (quảng cáo kẹp giữa)
            if title and url:
                news_items.append(news_dict)
                
        except Exception as e:
            logger.warning(f"Lỗi khi trích xuất tin tức: {e}")
            continue
            
    return news_items


def extract_news_from_html(html_file_path: str) -> list:
    """Hàm wrapper hỗ trợ đọc từ file cũ."""
    with open(html_file_path, 'r', encoding='utf-8') as f:
        return extract_news_from_html_content(f.read())

def process_all_crawled_html():
    """Đọc toàn bộ file HTML trong thư mục raw_html/fpt_news và tổng hợp."""
    raw_html_dir = os.path.join(os.path.dirname(__file__), 'raw_html', 'fpt_news')
    output_dir = os.path.join(os.path.dirname(__file__), 'extracted_data')
    os.makedirs(output_dir, exist_ok=True)
    
    all_news = []
    
    html_files = glob.glob(os.path.join(raw_html_dir, "*.html"))
    logger.info(f"Tìm thấy {len(html_files)} files HTML chờ trích xuất.")
    
    for file_path in html_files:
        logger.info(f"Đang xử lý: {os.path.basename(file_path)}")
        page_news = extract_news_from_html(file_path)
        all_news.extend(page_news)
        
    logger.info(f"Đã trích xuất tổng cộng {len(all_news)} bản tin từ {len(html_files)} trang.")
    
    # Xóa duplicate (do trang tin tức cập nhật real-time có thể bị trùng page trước/page sau)
    unique_news = []
    seen_urls = set()
    for news in all_news:
        if news['url'] not in seen_urls:
            unique_news.append(news)
            seen_urls.add(news['url'])
            
    logger.info(f"Sau khi lọc trùng: còn lại {len(unique_news)} bản tin duy nhất.")
    
    # ------------------ LƯU DỮ LIỆU THÀNH JSON ------------------
    output_file = os.path.join(output_dir, "fpt_news_extracted.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_news, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Đã lưu kết quả JSON vào: {output_file}")
    
    # Khuyến nghị: Trong tương lai bạn có thể đẩy list dict "unique_news" này vào DuckLake!

if __name__ == "__main__":
    process_all_crawled_html()
