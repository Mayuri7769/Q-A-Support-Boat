class Retriever:
    def __init__(self, store):
        self.store = store

    def retrieve(self, query_embedding, top_k=3):
        results = self.store.search(query_embedding, top_k)
        return results
