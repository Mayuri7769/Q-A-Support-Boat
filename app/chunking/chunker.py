from typing import List, Dict

class Chunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap

        return chunks

    def chunk_pages(self, extracted_pages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Input:  [{"url":..., "text":...}]
        Output: [{"url":..., "chunk_id":..., "text":...}]
        """
        all_chunks = []
        global_id = 0   # Ensures chunk IDs remain unique across pages

        for page in extracted_pages:
            url = page.get("url")
            text = page.get("text", "")

            page_chunks = self.chunk_text(text)

            for chunk in page_chunks:
                all_chunks.append({
                    "url": url,
                    "chunk_id": global_id,
                    "text": chunk   # FIX: always use "text", not "chunk"
                })
                global_id += 1

        return all_chunks


# Test
if __name__ == "__main__":
    example_pages = [
        {"url": "https://example.com", "text": "Lorem ipsum dolor sit amet, " * 50}
    ]

    chunker = Chunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_pages(example_pages)

    for c in chunks:
        print(f"\nURL: {c['url']}, Chunk ID: {c['chunk_id']}")
        print("Chunk:", c['text'])
