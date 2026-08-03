
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger("parser")

class DocParser:
    @staticmethod
    def clean_html(html_content: str) -> BeautifulSoup:
        logger.debug(f"[DEBUG] Cleaning HTML content (raw length: {len(html_content)} chars)...")
        soup = BeautifulSoup(html_content, "html.parser")
        
        for element in soup(["script", "style", "iframe", "noscript", "svg", "form"]):
            element.decompose()
            
        selectors_to_remove = [
            "nav", "footer", "header", "aside",
            ".nav", ".footer", ".header", ".sidebar", ".navigation", ".menu",
            ".toc", "#toc", ".ad-container", ".ads", ".promo", ".search-bar",
            ".breadcrumbs", ".edit-page-link"
        ]
        
        for selector in selectors_to_remove:
            for element in soup.select(selector):
                element.decompose()
                
        return soup

    @classmethod
    def extract_links(cls, html_content: str, base_url: str, limit_domain: bool = True) -> list[str]:
        soup = BeautifulSoup(html_content, "html.parser")
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc 

        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
           
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)
            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}".rstrip("/")

            if parsed_url.scheme not in ("http", "https"):
                continue

            path_lower = parsed_url.path.lower()
            if any(path_lower.endswith(ext) for ext in (
                ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
                ".pdf", ".zip", ".mp4", ".mp3", ".css", ".js"
            )):
                continue

            if limit_domain:
                link_domain = parsed_url.netloc
                if link_domain == base_domain:
                    links.append(clean_url)
            else:
                links.append(clean_url)

        unique_links = list(dict.fromkeys(links))
        logger.debug(f"[DEBUG] Extracted {len(unique_links)} unique links from {base_url} (limit_domain={limit_domain})")
        return unique_links

    @classmethod
    def parse_document(cls, html_content: str, url: str) -> dict:
        logger.debug(f"[DEBUG] Parsing document for URL: {url}")
        soup = cls.clean_html(html_content)
        
        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else ""
        if not title:
            h1_tag = soup.find("h1")
            title = h1_tag.get_text().strip() if h1_tag else "Untitled Documentation Page"
            
        title = title.strip()
        content_blocks = []
        current_header = None
        
        body = soup.find("body") or soup
        
        for element in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
            text = element.get_text().strip()
            if not text:
                continue
                
            tag_name = element.name
            
            if tag_name in ("h1", "h2", "h3", "h4"):
                current_header = text
                content_blocks.append({
                    "type": "heading",
                    "text": text,
                    "level": int(tag_name[1]),
                    "header": current_header
                })
            elif tag_name == "p":
                if len(text) > 10:
                    content_blocks.append({
                        "type": "paragraph",
                        "text": text,
                        "header": current_header
                    })
            elif tag_name == "li":
                content_blocks.append({
                    "type": "list_item",
                    "text": f"• {text}",
                    "header": current_header
                })
            elif tag_name == "pre":
                content_blocks.append({
                    "type": "code_block",
                    "text": text,
                    "header": current_header
                })
                
        logger.debug(f"[DEBUG] Document parsed successfully: Title='{title}', Total Blocks={len(content_blocks)}")
        return {
            "url": url,
            "title": title,
            "blocks": content_blocks
        }
