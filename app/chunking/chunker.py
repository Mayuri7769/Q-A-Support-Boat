from typing import List, Dict

class Chunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        :param chunk_size: number of characters per chunk
        :param overlap: number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Break a single text into chunks with overlap.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap  # move start with overlap

        return chunks

    def chunk_pages(self, extracted_pages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Accepts list of extracted pages: [{"url":..., "text":...}]
        Returns list of chunks with metadata:
        [{"url":..., "chunk_id":..., "chunk":...}]
        """
        all_chunks = []

        for page in extracted_pages:
            url = page.get("url")
            text = page.get("text", "")
            page_chunks = self.chunk_text(text)

            for i, chunk in enumerate(page_chunks):
                all_chunks.append({
                    "url": url,
                    "chunk_id": i,
                    "chunk": chunk
                })

        return all_chunks


# ✅ Manual test
if __name__ == "__main__":
    example_pages = [
        {"url": "https://example.com", "text": "Lorem ipsum dolor sit amet, " * 50}
    ]

    chunker = Chunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_pages(example_pages)

    for c in chunks:
        print(f"\nURL: {c['url']}, Chunk ID: {c['chunk_id']}")
        print("Chunk text:", c['chunk'])
