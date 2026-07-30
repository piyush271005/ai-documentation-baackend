import asyncio
import logging
import xml.etree.ElementTree as ET
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

    async def _fetch_sitemap_urls(self, client: httpx.AsyncClient, base_url: str) -> list[str]:
        """
        Attempt to discover article URLs via sitemap.xml / sitemap_index.xml / robots.txt.
        Returns a list of same-domain URLs found in the sitemap.
        """
        parsed = urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        base_parts = parsed.netloc.split(".")
        root_domain = ".".join(base_parts[-2:]) if len(base_parts) >= 2 else parsed.netloc

        sitemap_urls_to_try = []

        # 1. Check robots.txt for Sitemap: directives
        try:
            robots_resp = await client.get(f"{origin}/robots.txt", headers=BROWSER_HEADERS, timeout=10.0)
            if robots_resp.status_code == 200:
                for line in robots_resp.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sm_url = line.split(":", 1)[1].strip()
                        sitemap_urls_to_try.append(sm_url)
                        logger.info(f"Found sitemap in robots.txt: {sm_url}")
        except Exception:
            pass

        # 2. Common fallback sitemap locations
        for path in ["/sitemap.xml", "/sitemap_index.xml", "/news-sitemap.xml"]:
            sitemap_urls_to_try.append(f"{origin}{path}")

        article_urls = []
        seen_sitemaps = set()
        NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

        async def parse_sitemap(sm_url: str, depth: int = 0):
            if sm_url in seen_sitemaps or depth > 2:
                return
            seen_sitemaps.add(sm_url)
            try:
                resp = await client.get(sm_url, headers=BROWSER_HEADERS, timeout=10.0, follow_redirects=True)
                if resp.status_code != 200:
                    return
                ct = resp.headers.get("content-type", "")
                if "xml" not in ct and "text" not in ct:
                    return
                root = ET.fromstring(resp.text)
                # Sitemap index — recurse into child sitemaps
                for sitemap_tag in root.findall(f"{NS}sitemap"):
                    loc = sitemap_tag.findtext(f"{NS}loc")
                    if loc:
                        await parse_sitemap(loc.strip(), depth + 1)
                # Regular sitemap — collect <url><loc> entries
                for url_tag in root.findall(f"{NS}url"):
                    if len(article_urls) >= self.max_pages * 3:
                        break
                    loc = url_tag.findtext(f"{NS}loc")
                    if loc:
                        loc = loc.strip().rstrip("/")
                        p = urlparse(loc)
                        # Only keep same-domain URLs
                        if p.netloc == parsed.netloc or p.netloc.endswith("." + root_domain):
                            article_urls.append(loc)
            except Exception as e:
                logger.debug(f"Sitemap parse error for {sm_url}: {e}")

        for sm_url in sitemap_urls_to_try:
            if len(article_urls) >= self.max_pages * 3:
                break
            await parse_sitemap(sm_url)

        logger.info(f"Sitemap discovery found {len(article_urls)} URLs for {base_url}")
        return article_urls

    async def crawl_page(self, client: httpx.AsyncClient, url: str, depth: int):
        """Downloads a single page, parses content, and queues outgoing links."""
        if len(self.pages_crawled) >= self.max_pages:
            return
            
        if not self.mark_visited(url):
            return
            
        logger.info(f"Crawling URL: {url} at depth {depth}")

        try:
            response = await client.get(url, headers=BROWSER_HEADERS, timeout=15.0, follow_redirects=True)

            # Handle rate-limiting: wait and retry once
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
        # Parse page content
        try:
            parsed_doc = DocParser.parse_document(html_content, url)
            self.crawled_content.append(parsed_doc)
            self.pages_crawled.append(url)
            
            # Extract links and queue them for next depth level
            if depth < 3:  # Max BFS depth of 3 to avoid infinite crawl
                outgoing_links = DocParser.extract_links(html_content, url, limit_domain=self.limit_domain)
                logger.info(f"Found {len(outgoing_links)} outgoing links on {url}")
                for link in outgoing_links:
                    await self.push_to_queue(link, depth + 1)
                
                # If zero links found on the start page, this is likely a JS-rendered site.
                # Fall back to sitemap discovery to seed the queue.
                if len(outgoing_links) == 0 and depth == 0 and hasattr(self, '_client_ref'):
                    logger.warning(f"No links found on start page {url} — likely JS-rendered. Trying sitemap discovery...")
                    sitemap_urls = await self._fetch_sitemap_urls(self._client_ref, url)
                    for sm_url in sitemap_urls[:self.max_pages * 3]:  # Queue up to 3x max_pages candidates
                        await self.push_to_queue(sm_url, 1)
                    logger.info(f"Seeded {len(sitemap_urls)} URLs from sitemap into BFS queue")
        except Exception as e:
            logger.error(f"Error parsing content from {url}: {e}")

    async def run_crawl(self, start_url: str, max_pages: int = 50, limit_domain: bool = True):
        """Starts concurrent BFS crawl using a worker-pool for maximum speed."""
        self.reset()
        self.is_crawling = True
        self.max_pages = max_pages
        self.limit_domain = limit_domain
        self.start_url = start_url

        # Establish base domain restriction
        parsed_start = urlparse(start_url)
        self.base_domain = parsed_start.netloc

        logger.info(f"Starting concurrent BFS crawl from {start_url} (Max Pages: {max_pages})")

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
            # Store client reference so crawl_page can use it for sitemap fallback
            self._client_ref = client

            # Proactive sitemap seed: try to discover article URLs before starting BFS.
            # This helps JS-heavy sites where the homepage has zero crawlable links.
            logger.info("Pre-checking sitemap for article URL discovery...")
            sitemap_urls = await self._fetch_sitemap_urls(client, start_url)
            if sitemap_urls:
                for sm_url in sitemap_urls[:max_pages * 3]:
                    await self.push_to_queue(sm_url, 1)
                logger.info(f"Pre-seeded {len(sitemap_urls)} sitemap URLs into BFS queue")

            while self.is_crawling:
                # Drain finished tasks
                done = {t for t in active_tasks if t.done()}
                active_tasks -= done

                # Stop if we've hit the page limit
                if len(self.pages_crawled) >= self.max_pages:
                    break

                # Fill up to CONCURRENCY tasks from the queue
                while len(active_tasks) < CONCURRENCY:
                    queue_item = await self.pop_from_queue()
                    if not queue_item:
                        break  # Nothing queued right now
                    url, depth = queue_item
                    # Skip already-visited URLs early
                    if url in self.local_visited:
                        continue
                    task = asyncio.create_task(crawl_with_semaphore(client, url, depth))
                    active_tasks.add(task)

                if not active_tasks:
                    # Nothing in queue and no active tasks = done
                    break

                # Wait for at least one task to finish before refilling
                await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

            # Wait for all remaining tasks to finish
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

        self.is_crawling = False
        logger.info(f"Crawl completed. Crawled {len(self.pages_crawled)} pages.")

# Global crawler instance
crawler_coordinator = CrawlCoordinator()
