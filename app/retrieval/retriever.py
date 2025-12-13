from typing import List, Dict

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        retrieved_chunks = []

        for item in results:
            if "text" not in item:
                continue

            retrieved_chunks.append({
                "url": item.get("url", ""),
                "chunk_id": item.get("chunk_id", -1),
                "text": item["text"]
            })

        return retrieved_chunks
