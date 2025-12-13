import os
from app.retrieval.retriever import Retriever
import logging
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.vectorstore.vector_store import VectorStore
from app.llm.gemini_client import GeminiClient
import uvicorn

from dotenv import load_dotenv
load_dotenv()
from app.crawling.crawler import WebCrawler


from app.chunking.chunker import Chunker

start_url = os.getenv("START_URL", "https://example.com")



chunker = Chunker(chunk_size=500)  # Adjust chunk size as needed


# -----------------------------
# FastAPI app instance
# -----------------------------
app = FastAPI(title="RAG Support Bot")

# -----------------------------
# Pydantic model for query
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

# -----------------------------
# Initialize RAG components
# -----------------------------
generator = EmbeddingGenerator()
store = VectorStore()  # Make sure this loads your FAISS index
retriever = Retriever(store)


# Initialize Gemini client
gemini = GeminiClient(model_name=os.getenv("LLM_MODEL"))




class IngestRequest(BaseModel):
    urls: List[str]



@app.post("/ingest")
def ingest_data(request: IngestRequest):
    print("📥 /ingest endpoint hit!")

    try:
        total_pages = 0
        total_chunks = 0

        for url in request.urls:
            # 1️⃣ Crawl site → returns {"url", "text"}
            crawler = WebCrawler(base_url=url, max_pages=10)
            pages = crawler.crawl()

            total_pages += len(pages)

            # 2️⃣ Chunk pages (expects "text")
            chunks = chunker.chunk_pages(pages)

            if not chunks:
                continue

            # 3️⃣ Generate embeddings
            embeddings = generator.generate_embeddings_for_chunks(chunks)

            # 4️⃣ Store in vector DB
            store.add_embeddings(chunks, embeddings)

            total_chunks += len(chunks)

        return {
            "message": "Ingestion completed successfully",
            "pages_crawled": total_pages,
            "total_chunks": total_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/query")
def query_bot(request: QueryRequest):
    print("❓ /query endpoint hit!")

    try:
        question = request.question
        print("✅ Question:", question)

        # 1. Create embedding for question
        query_embedding = generator.generate_embedding(question)
        print("✅ Query embedding generated")

        # 2. Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(query_embedding, top_k=request.top_k)
        print("✅ Retrieved Chunks:", len(retrieved_chunks))

        # 3. Clean results for JSON (REMOVE numpy embeddings)
        for item in retrieved_chunks:
            if "embedding" in item:
                del item["embedding"]

        # 4. Generate final answer
        answer = gemini.generate_answer(question, retrieved_chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", 8000)))