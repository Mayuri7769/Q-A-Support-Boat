from ..embeddings.embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

generator = EmbeddingGenerator()
store = VectorStore()

chunks = [
    {"url": "site1.com", "chunk_id": 1, "chunk": "Python is a programming language"},
    {"url": "site1.com", "chunk_id": 2, "chunk": "Machine learning uses data"},
    {"url": "site1.com", "chunk_id": 3, "chunk": "FastAPI is used for APIs"},
]

# Extract text for embeddings
texts = [c["chunk"] for c in chunks]
embeddings = generator.generate_embeddings_for_chunks(texts)

# Add embeddings to store
store.add_embeddings(chunks, embeddings)

# Search
query = "How to build APIs?"
query_embedding = generator.generate_embedding(query)

results = store.search(query_embedding)

print("\n🔍 Search Results:")
for r in results:
    print(f"- {r['text']}")
