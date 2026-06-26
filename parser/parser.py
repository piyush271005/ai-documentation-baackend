import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

class DocParser:
    @staticmethod
    def clean_html(html_content: str) -> BeautifulSoup:
        """Parse HTML and strip common non-content tags."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script, style, and iframe tags
        for element in soup(["script", "style", "iframe", "noscript", "svg", "form"]):
            element.decompose()
            
        # Remove common navigation, header, and footer tags/classes
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
        """Extract all valid hyperlinks from the page belonging to the same domain."""
        soup = cls.clean_html(html_content)
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            # Join relative paths
            absolute_url = urljoin(base_url, href)
            # Remove url fragments (#section-1) and query params to avoid duplicate page crawls
            parsed_url = urlparse(absolute_url)
            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            
            # Keep links only if they are HTTP/HTTPS
            if parsed_url.scheme not in ("http", "https"):
                continue
                
            if limit_domain:
                if parsed_url.netloc == base_domain:
                    links.append(clean_url)
            else:
                links.append(clean_url)
                
        # Return unique links, preserving order
        return list(dict.fromkeys(links))

    @classmethod
    def parse_document(cls, html_content: str, url: str) -> dict:
        """
        Extract title and hierarchical text blocks (headings + paragraphs) from the page.
        """
        soup = cls.clean_html(html_content)
        
        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else ""
        if not title:
            h1_tag = soup.find("h1")
            title = h1_tag.get_text().strip() if h1_tag else "Untitled Documentation Page"
            
        # Clean title of common suffixes
        title = re.sub(r"\s*\|\s*.*$", "", title)
        title = re.sub(r"\s*-\s*.*$", "", title)
        title = title.strip()
        
        # Extract content structure
        # Traverse body tags to keep hierarchy (H1, H2, H3, P, UL/OL list items)
        content_blocks = []
        current_header = None
        
        body = soup.find("body") or soup
        
        for element in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
            # Check if element is inside code block
            # If it's a code tag, but not inside a pre, we keep it as inline
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
                # Only keep paragraphs with substantial text
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
                # Code blocks
                content_blocks.append({
                    "type": "code_block",
                    "text": text,
                    "header": current_header
                })
                
        return {
            "url": url,
            "title": title,
            "blocks": content_blocks
        }
