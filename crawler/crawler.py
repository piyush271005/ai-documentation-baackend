import asyncio
import logging
from urllib.parse import urlparse
import httpx
from backend.config import settings
from backend.parser.parser import DocParser

logger = logging.getLogger("crawler")
logging.basicConfig(level=logging.INFO)

class CrawlCoordinator:
    """Manages crawl state, queue, and visited lists using in-memory queues."""
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
        """Pushes (url, depth) to queue."""
        if url not in self.local_visited:
            await self.local_queue.put((url, depth))

    async def pop_from_queue(self) -> tuple[str, int]:
        """Pops (url, depth) from queue. Returns None if empty."""
        if not self.local_queue.empty():
            return await self.local_queue.get()
        return None

    def mark_visited(self, url: str) -> bool:
        """Marks a URL as visited. Returns True if it was NOT already visited."""
        if url not in self.local_visited:
            self.local_visited.add(url)
            return True
        return False

    async def crawl_page(self, client: httpx.AsyncClient, url: str, depth: int):
        """Downloads a single page, parses content, and queues outgoing links."""
        if len(self.pages_crawled) >= self.max_pages:
            return
            
        if not self.mark_visited(url):
            return
            
        logger.info(f"Crawling URL: {url} at depth {depth}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        
        try:
            response = await client.get(url, headers=headers, timeout=10.0, follow_redirects=True)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: Status code {response.status_code}")
                return
            
            print(response)
                
            # Double check content type is HTML
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                logger.info(f"Skipping non-HTML page {url} with type {content_type}")
                return
                
            html_content = response.text
            
            # Parse page content
            parsed_doc = DocParser.parse_document(html_content, url)
            self.crawled_content.append(parsed_doc)
            self.pages_crawled.append(url)
            
            # Extract links and queue them for next depth level
            if depth < 3:  # Max BFS depth of 3 to avoid infinite crawl
                outgoing_links = DocParser.extract_links(html_content, url, limit_domain=self.limit_domain)
                for link in outgoing_links:
                    await self.push_to_queue(link, depth + 1)
                    
        except httpx.HTTPError as e:
            logger.error(f"HTTP error crawling {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error crawling {url}: {e}")

    async def run_crawl(self, start_url: str, max_pages: int = 50, limit_domain: bool = True):
        """Starts BFS crawl execution loop."""
        self.reset()
        self.is_crawling = True
        self.max_pages = max_pages
        self.limit_domain = limit_domain
        self.start_url = start_url
        
        # Establish base domain restriction
        parsed_start = urlparse(start_url)
        self.base_domain = parsed_start.netloc
        
        logger.info(f"Starting BFS crawl from {start_url} (Max Pages: {max_pages})")
        
        # Enqueue start URL at depth 0
        await self.push_to_queue(start_url, 0)
        
        async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)) as client:
            while self.is_crawling and len(self.pages_crawled) < self.max_pages:
                queue_item = await self.pop_from_queue()
                if not queue_item:
                    # Queue is empty, crawl completed
                    break
                    
                url, depth = queue_item
                # Execute single crawl task
                await self.crawl_page(client, url, depth)
                # Polite crawling: add a tiny delay between requests
                await asyncio.sleep(0.5)
                
        self.is_crawling = False
        logger.info(f"Crawl completed. Crawled {len(self.pages_crawled)} pages.")

# Global crawler instance
crawler_coordinator = CrawlCoordinator()
