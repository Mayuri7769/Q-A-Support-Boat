from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("✅ Loading FREE embedding model...")
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str):
        embedding = self.model.encode(text)
        return np.array(embedding)

    def generate_embeddings_for_chunks(self, chunks):
        embeddings = []
        for chunk in chunks:
            emb = self.generate_embedding(chunk["text"])
            embeddings.append({
                "url": chunk["url"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"], 
                "embedding": emb
            })
        return embeddings


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
