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

from app.extraction.text_extractor import TextExtractor
from app.chunking.chunker import Chunker

start_url = os.getenv("START_URL", "https://example.com")
crawler = WebCrawler(
    base_url=start_url,
    max_pages=int(os.getenv("MAX_PAGES", 50))
)

text_extractor = TextExtractor()
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

import requests



@app.post("/ingest")
def ingest_data(request: IngestRequest):
    print("📥 /ingest endpoint hit!")
    try:
        pages_crawled = 0
        total_chunks = 0

        for url in request.urls:
            # 1️⃣ Fetch HTML
            response = requests.get(url, timeout=15)
            html = response.text

            # 2️⃣ Extract clean text
            extracted_pages = text_extractor.extract_texts([
                {"url": url, "html": html}
            ])

            # 3️⃣ Chunk pages correctly ✅
            chunks = chunker.chunk_pages(extracted_pages)

            # 4️⃣ Generate embeddings using CHUNK TEXT ✅
            texts = [c["chunk"] for c in chunks]
            embeddings = generator.generate_embeddings_for_chunks(chunks)

            # 5️⃣ Store as DICTIONARY chunks ✅
            store.add_embeddings(chunks, embeddings)

            pages_crawled += 1
            total_chunks += len(chunks)

        return {
            "message": "Ingestion completed successfully",
            "pages_crawled": pages_crawled,
            "total_chunks": total_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/query")
def query_bot(request: QueryRequest):
    print("❓ /query endpoint hit!")
    try:

        print("✅ Question:", request.question)
        # 1. Create embedding for question
        query_embedding = generator.generate_embedding(request.question)
        print("✅ Query embedding generated")

        # 2. Retrieve relevant chunks using Retriever
        results = retriever.retrieve(query_embedding, top_k=request.top_k)
        print("✅ Retrieved Results TYPE:", type(results))
        print("✅ First result TYPE:", type(results[0]))
        print("✅ First result VALUE:", results[0])

        # 3. Generate answer using Gemini
        answer = gemini.generate_answer(request.question, results)

        return {
            "answer": answer,
            "sources": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", 8000)))