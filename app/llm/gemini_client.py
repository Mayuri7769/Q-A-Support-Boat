import os
import google.generativeai as genai
from google.generativeai import GenerativeModel


class GeminiClient:
    def __init__(self, model_name="gemini-2.0-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file")

        # Configure Google GenAI
        genai.configure(api_key=api_key)

        # Load model (NEW API)
        self.model = GenerativeModel(model_name)


    def generate_answer(self, question: str, retrieved_chunks: list):
        """
        retrieved_chunks = list of dicts:
          { "url": "...", "chunk_id": ..., "text": "..." }
        """

        # Build RAG context properly
        context_text = "\n\n---\n".join([
            f"Source: {c['url']}\nContent:\n{c['text']}"
            for c in retrieved_chunks
        ])

        assert all("text" in c for c in retrieved_chunks)


        prompt = f"""
You are a helpful AI assistant. 
Answer ONLY using the context below. If the answer is not in the context, say "I don't know."

Context:
{context_text}

Question: {question}

Answer:
"""

        # New Gemini API call
        response = self.model.generate_content(prompt)

        return response.text
    



# Manual Test
if __name__ == "__main__":
    gemini = GeminiClient()

    test_chunks = [
        {"url": "site1.com", "chunk_id": 1, "text": "Python is a programming language."},
        {"url": "site1.com", "chunk_id": 2, "text": "It is used for AI and web development."}
    ]

    print(gemini.generate_answer("What is Python used for?", test_chunks))