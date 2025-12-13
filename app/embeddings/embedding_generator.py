from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("✅ Loading FREE embedding model...")
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single query
        """
        embedding = self.model.encode(text)
        return embedding.tolist()   # 🔑 RETURN LIST[float]

    def generate_embeddings_for_chunks(self, chunks) -> List[List[float]]:
        """
        Generate embeddings ONLY (no metadata)
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts)

        return embeddings.tolist()  # 🔑 LIST OF VECTORS



# ✅ Test run
if __name__ == "__main__":
    generator = EmbeddingGenerator()

    example_chunks = [
        {"url": "test.com", "chunk_id": 1, "text": "This is a test sentence"},
        {"url": "test.com", "chunk_id": 2, "text": "This is another test sentence"}
    ]

    embeddings = generator.generate_embeddings_for_chunks(example_chunks)

    print("✅ Embeddings generated successfully!")
    print("Vector length:", len(embeddings[0]["embedding"]))
