# RAG Q & A Support Bot

An end-to-end **Retrieval Augmented Generation (RAG)** system built with **FastAPI**, **FAISS**, **SentenceTransformers**, and **Gemini**.  
It crawls websites, extracts and cleans text, chunks and embeds content, retrieves relevant context, and generates grounded answers with strict hallucination control.

---

## Features

- **Web Crawling**: Automatically fetch pages from a target website.  
- **Text Extraction**: Clean and normalize HTML content.  
- **Chunking**: Split text into overlapping chunks for better retrieval.  
- **Embeddings**: Generate vector embeddings for each chunk using `SentenceTransformers`.  
- **Vector Search**: Store and search chunks in **FAISS** vector database.  
- **Answer Generation**: Generate answers grounded in retrieved content using **Gemini**.  
- **API Endpoints**:
  - `POST /ingest` – Crawl URLs and store embeddings.
  - `POST /query` – Ask a question and get answers from crawled content.


## Setup & Usage

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt

2. Create a .env file with your settings (API_HOST, API_PORT, START_URL, LLM_MODEL, etc.).
3. Important: FAISS index (faiss.index) and metadata (metadata.pkl) are generated locally during ingestion.
They are not included in the repo. Run the ingestion endpoint first to create them:

# Start FastAPI server
python main.py

POST /ingest
Content-Type: application/json

{
    "urls": ["https://example.com"]
}

4. Once ingestion completes, you can query your bot:
POST /query
Content-Type: application/json

{
    "question": "What is this site about?",
    "top_k": 3
}
