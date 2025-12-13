import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import List, Dict


class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 50, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.max_pages = max_pages
        self.timeout = timeout

        self.visited_urls = set()
        self.queue = deque([self.base_url])
        self.domain = urlparse(self.base_url).netloc

    def is_valid_url(self, url: str) -> bool:
        """
        Allow only same-domain links with http/https.
        """
        parsed = urlparse(url)
        return (
            parsed.netloc == self.domain
            and parsed.scheme in ("http", "https")
        )

    def fetch_page(self, url: str) -> str:
        """
        Download HTML content of a webpage.
        """
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")
            return ""

    def extract_text(self, html: str) -> str:
        """
        Convert HTML → clean readable text.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove useless tags
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator=" ", strip=True)
        return text

    def extract_links(self, base_url: str, html: str) -> List[str]:
        """
        Extract all valid hyperlinks.
        """
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            full_url = urljoin(base_url, href)

            if self.is_valid_url(full_url):
                links.append(full_url)

        return links

    def crawl(self) -> List[Dict[str, str]]:
        """
        Crawl website and return list of:
        { "url": ..., "text": ... }
        """
        crawled_data = []

        while self.queue and len(self.visited_urls) < self.max_pages:
            current_url = self.queue.popleft()

            if current_url in self.visited_urls:
                continue

            print(f"[CRAWLING] {current_url}")
            html = self.fetch_page(current_url)
            if not html:
                continue

            text = self.extract_text(html)

            self.visited_urls.add(current_url)

            # FIXED OUTPUT FORMAT
            crawled_data.append({
                "url": current_url,
                "text": text
            })

            # enqueue links
            links = self.extract_links(current_url, html)
            for link in links:
                if link not in self.visited_urls:
                    self.queue.append(link)

        print(f"\n✅ Crawling finished. Total pages crawled: {len(crawled_data)}")
        return crawled_data


# Manual test
if __name__ == "__main__":
    test_url = "https://example.com"
    crawler = WebCrawler(base_url=test_url, max_pages=5)
    data = crawler.crawl()

    for page in data:
        print("\nURL:", page["url"])
        print("Text length:", len(page["text"]))
