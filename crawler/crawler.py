import asyncio
import logging
from urllib.parse import urlparse, urljoin
import httpx
from backend.config import settings
from backend.parser.parser import DocParser

logger = logging.getLogger("crawler")
logging.basicConfig(level=logging.INFO)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

class CrawlCoordinator:
    def __init__(self):
        self.is_crawling = False
        self.pages_crawled = []
        self.crawled_content = []  # Stores parsed pages: [{"url", "title", "blocks"}]
        
        # Local queue & visited set
        self.local_queue = asyncio.Queue()
        self.local_visited = set()
        self.max_pages = 50
        self.limit_domain = True
        self.base_domain = ""
        self.start_url = ""

    def reset(self):
        self.is_crawling = False
        self.pages_crawled = []
        self.crawled_content = []
        self.local_visited = set()
        # Empty local queue
        while not self.local_queue.empty():
            try:
                self.local_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def get_queue_size(self) -> int:
        return self.local_queue.qsize()

    async def push_to_queue(self, url: str, depth: int):
        if url not in self.local_visited:
            logger.debug(f"[DEBUG] Enqueuing URL for crawl: {url} at depth {depth}")
            await self.local_queue.put((url, depth))
        else:
            logger.debug(f"[DEBUG] Skipping enqueue for already-visited URL: {url}")

    async def pop_from_queue(self) -> tuple[str, int]:
        if not self.local_queue.empty():
            item = await self.local_queue.get()
            logger.debug(f"[DEBUG] Dequeued item for worker: {item[0]} (depth {item[1]})")
            return item
        return None

    def mark_visited(self, url: str) -> bool:
        if url not in self.local_visited:
            self.local_visited.add(url)
            logger.debug(f"[DEBUG] Marked URL as visited ({len(self.local_visited)} total visited): {url}")
            return True
        return False

    async def crawl_page(self, client: httpx.AsyncClient, url: str, depth: int):
        """Downloads a single page, parses content, and queues outgoing links."""
        if len(self.pages_crawled) >= self.max_pages:
            logger.debug(f"[DEBUG] Max pages limit reached ({self.max_pages}). Skipping {url}")
            return
            
        if not self.mark_visited(url):
            return
            
        logger.info(f"Crawling URL: {url} at depth {depth}")

        try:
            logger.debug(f"[DEBUG] Sending HTTP GET to {url}...")
            response = await client.get(url, headers=BROWSER_HEADERS, timeout=15.0, follow_redirects=True)
            logger.debug(f"[DEBUG] Received response from {url}: status={response.status_code}, content_type={response.headers.get('content-type', '')}")

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                logger.warning(f"Rate limited by {url}. Retrying after {retry_after}s...")
                await asyncio.sleep(retry_after)
                response = await client.get(url, headers=BROWSER_HEADERS, timeout=15.0, follow_redirects=True)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: Status code {response.status_code}")
                return

            # Check content type is HTML
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                logger.info(f"Skipping non-HTML page {url} with type {content_type}")
                return

            html_content = response.text

        except httpx.HTTPError as e:
            logger.error(f"HTTP error crawling {url}: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error crawling {url}: {e}")
            return

        
        try:
            parsed_doc = DocParser.parse_document(html_content, url)
            self.crawled_content.append(parsed_doc)
            self.pages_crawled.append(url)
            logger.debug(f"[DEBUG] Successfully indexed parsed content for {url} ({len(self.pages_crawled)}/{self.max_pages} crawled)")
            
            if depth < 3: 
                outgoing_links = DocParser.extract_links(html_content, url, limit_domain=self.limit_domain)
                logger.info(f"Found {len(outgoing_links)} outgoing links on {url}")
                for link in outgoing_links:
                    await self.push_to_queue(link, depth + 1)
        except Exception as e:
            logger.error(f"Error parsing content from {url}: {e}")

    async def run_crawl(self, start_url: str, max_pages: int = 50, limit_domain: bool = True):
        """Starts concurrent BFS crawl using a worker-pool for maximum speed."""
        self.reset()
        self.is_crawling = True
        self.max_pages = max_pages
        self.limit_domain = True
        self.start_url = start_url

        # Establish base domain restriction
        parsed_start = urlparse(start_url)
        self.base_domain = parsed_start.netloc

        logger.info(f"Starting concurrent BFS crawl from {start_url} (Max Pages: {max_pages})")
        logger.debug(f"[DEBUG] Base domain restriction set to: {self.base_domain}")

        # Enqueue start URL at depth 0
        await self.push_to_queue(start_url, 0)

        # Semaphore limits concurrent requests (8 = fast but polite)
        CONCURRENCY = 8
        semaphore = asyncio.Semaphore(CONCURRENCY)
        active_tasks: set[asyncio.Task] = set()

        async def crawl_with_semaphore(client, url, depth):
            async with semaphore:
                await self.crawl_page(client, url, depth)

        async with httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            timeout=httpx.Timeout(15.0)
        ) as client:
            while self.is_crawling:
                done = {t for t in active_tasks if t.done()}
                active_tasks -= done

                if len(self.pages_crawled) >= self.max_pages:
                    logger.debug(f"[DEBUG] Worker pool loop ending: max pages target hit ({len(self.pages_crawled)}/{self.max_pages})")
                    break

                while len(active_tasks) < CONCURRENCY:
                    queue_item = await self.pop_from_queue()
                    if not queue_item:
                        break  
                    url, depth = queue_item
                   
                    if url in self.local_visited:
                        continue
                    task = asyncio.create_task(crawl_with_semaphore(client, url, depth))
                    active_tasks.add(task)

                if not active_tasks:
                    logger.debug("[DEBUG] Worker pool loop ending: no active or queued tasks remaining.")
                    break

                await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

            if active_tasks:
                logger.debug(f"[DEBUG] Awaiting remaining {len(active_tasks)} active tasks to finish...")
                await asyncio.gather(*active_tasks, return_exceptions=True)

        self.is_crawling = False
        logger.info(f"Crawl completed. Crawled {len(self.pages_crawled)} pages.")

# Global crawler instance
crawler_coordinator = CrawlCoordinator()
