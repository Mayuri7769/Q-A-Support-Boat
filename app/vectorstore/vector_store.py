import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, dim=384, index_path="faiss.index", meta_path="metadata.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(index_path) and os.path.exists(meta_path):
            print("✅ Loading existing FAISS index...")
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            print("✅ Creating new FAISS index...")
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []

    def add_embeddings(self, chunks, embeddings):
        """
        chunks: list of dicts -> {'url':..., 'chunk_id':..., 'chunk':...}
        embeddings: list of lists (vectors)
        """
        for chunk, emb in zip(chunks, embeddings):

            # Validate embedding shape
            if not isinstance(emb, (list, np.ndarray)):
                raise TypeError(f"❌ Invalid embedding type: {type(emb)}")

            vector = np.array([emb], dtype="float32")

            if vector.shape[1] != self.dim:
                raise ValueError(
                    f"❌ Embedding dimension mismatch: expected {self.dim}, got {vector.shape[1]}"
                )

            self.index.add(vector)
            # store metadata in correct format for Gemini
        self.metadata.append({
            "url": chunk["url"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["chunk"]
        })

        # Save FAISS + metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print("✅ Embeddings stored in FAISS")

    def search(self, query_embedding, top_k=5):
        vector = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results
