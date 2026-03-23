import asyncio
from playwright.async_api import async_playwright, Browser, BrowserContext
from dotenv import load_dotenv
import os
import sys
import random
import logging

# Add the workspace root to Python path for imports
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from constants.proxies import get_random_proxy

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def create_browser_context(browser: Browser) -> BrowserContext:
    """Tạo một context mới từ trình duyệt với cấu hình stealth và proxy ngẫu nhiên."""
    proxy_username = os.getenv("PROXY_USERNAME")
    proxy_password = os.getenv("PROXY_PASSWORD")
    proxy_host = get_random_proxy()
    
    proxy_config = None
    if proxy_host:
        proxy_config = {"server": f"http://{proxy_host}"}
        if proxy_username and proxy_password:
            proxy_config["username"] = proxy_username
            proxy_config["password"] = proxy_password
            
    context = await browser.new_context(
        proxy=proxy_config,
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        viewport={'width': 1366, 'height': 768},
        locale='en-US',
        timezone_id='America/New_York'
    )
    
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
    """)
    return context


async def process_page(browser: Browser, page_num: int, semaphore: asyncio.Semaphore, output_dir: str, on_page_fetched=None):
    """Xử lý tải 1 page cụ thể lồng trong kiến trúc Async."""
    base_url = "https://vn.investing.com/equities/fpt-corp-news"
    url = base_url if page_num == 1 else f"{base_url}/{page_num}"
    max_retries = 3
    success = False

    async with semaphore:
        attempt = 0
        while not success:
            attempt += 1
            context = None
            try:
                # Mỗi lần thử sẽ tạo một BrowserContext mới để đổi Proxy
                context = await create_browser_context(browser)
                page = await context.new_page()
                page.set_default_timeout(60000)
                
                # Delay ngẫu nhiên để không bị dính còi chống DDOS
                await asyncio.sleep(random.uniform(2.0, 5.0))
                
                logger.info(f"Đang fetch trang {page_num} (Thử lại {attempt}): {url}")
                response = await page.goto(url, wait_until="domcontentloaded", timeout=60000)

                if response and response.status == 200:
                    await page.wait_for_timeout(2000) # Đợi JS xử lý
                    logger.info(f"Tải thành công trang {page_num}")
                    
                    html_content = await page.content()
                    
                    if on_page_fetched:
                        # Async callback support (pass logic outside)
                        if asyncio.iscoroutinefunction(on_page_fetched):
                            await on_page_fetched(page_num, html_content)
                        else:
                            on_page_fetched(page_num, html_content)
                    else:
                        output_file = os.path.join(output_dir, f"page_{page_num}.html")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.info(f"Đã lưu trang {page_num} vào {output_file} ({len(html_content)} bytes)")
                        
                    success = True
                    break
                else:
                    status_code = response.status if response else 'None'
                    logger.warning(f"Lỗi tải trang {page_num} (Lần {attempt}): Status {status_code}")
                    await asyncio.sleep(3)

            except Exception as e:
                logger.warning(f"Ngoại lệ ở trang {page_num} (Lần {attempt}): {str(e)}")
                await asyncio.sleep(3)
            finally:
                if context:
                    try:
                        await context.close()
                    except:
                        pass


async def crawl_fpt_news_async(start_page: int = 1, end_page: int = 405, concurrency_limit: int = 5, on_page_fetched=None):
    """
    Crawl HTML nội dung bằng thiết kế Asyncio, chạy multi-task để tăng tốc độ tối đa.
    """
    output_dir = os.path.join(os.path.dirname(__file__), 'raw_html', 'fpt_news')
    os.makedirs(output_dir, exist_ok=True)

    async with async_playwright() as p:
        # Launch duy nhất MỘT trình duyệt, nhưng sẽ spawn nhiều tabs/contexts
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-zygote',
                '--disable-gpu',
                '--disable-blink-features=AutomationControlled'
            ]
        )
        
        # Kiểm soát số lượng tab/request chạy đồng thời
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        tasks = []
        for page_num in range(start_page, end_page + 1):
            task = asyncio.create_task(process_page(browser, page_num, semaphore, output_dir, on_page_fetched))
            tasks.append(task)
            
        logger.info(f"Đã lập lịch tải {len(tasks)} trang (Max chạy song song {concurrency_limit} luồng)...")
        
        # Chờ tất cả hoàn thành
        await asyncio.gather(*tasks)
        
        await browser.close()


if __name__ == "__main__":
    logger.info("BẮT ĐẦU CRAWL ASYNC DỮ LIỆU FPT CORP NEWS")
    # Sử dụng ProactorEventLoopPolicy để hỗ trợ khởi tạo subprocess trên Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    # Chạy thử 10 trang với 5 trang đồng thời
    # Khi quen có thể đổi thành start_page=1, end_page=405
    asyncio.run(crawl_fpt_news_async(start_page=1, end_page=10, concurrency_limit=5))
    logger.info("HOÀN THÀNH QUÁ TRÌNH CRAWL")