from bs4 import BeautifulSoup
from typing import List, Dict


class TextExtractor:
    def __init__(self):
        pass

    def clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style"]):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())

        return text

    # ✅ REQUIRED FOR FASTAPI /INGEST
    def extract_text(self, html: str) -> str:
        return self.clean_html(html)

    # ✅ OPTIONAL FOR BATCH MODE
    def extract_texts(self, crawled_pages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        extracted = []

        for page in crawled_pages:
            url = page.get("url")
            html = page.get("html", "")
            text = self.clean_html(html)

            if text:
                extracted.append({
                    "url": url,
                    "text": text
                })

        return extracted
