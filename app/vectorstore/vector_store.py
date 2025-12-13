import faiss
import numpy as np
import pickle
import os
from typing import List, Dict


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

    def add_embeddings(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        chunks: [{'url', 'chunk_id', 'text'}]
        embeddings: list of vectors
        """

        if len(chunks) != len(embeddings):
            raise ValueError("❌ chunks and embeddings length mismatch")

        vectors = np.array(embeddings, dtype="float32")

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"❌ Embedding dimension mismatch: expected {self.dim}, got {vectors.shape[1]}"
            )

        # ✅ Add ALL vectors at once
        self.index.add(vectors)

        # ✅ Metadata MUST align with FAISS indices
        for chunk in chunks:
            self.metadata.append({
                "url": chunk["url"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"]   # 🔑 STANDARD KEY
            })

        # ✅ Persist index + metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"✅ Stored {len(chunks)} embeddings")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        vector = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(vector, top_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results
