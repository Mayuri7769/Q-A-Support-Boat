import os
import google.generativeai as genai


class GeminiClient:
    def __init__(self, model_name="gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file")

        # Configure the API
        genai.configure(api_key=api_key)

        # Load model
        self.model_name = model_name
      

    def generate_answer(self, question: str, retrieved_chunks: list):
        """
        retrieved_chunks = list of dicts:
          { "url": "...", "text": "...", "chunk_id": ... }
        """

        # Build clean context text
        context_text = "\n\n".join([f"URL: {c['url']}\nTEXT: {c['text']}" for c in retrieved_chunks])
        prompt = f"You are a helpful AI assistant.\nAnswer using ONLY the context below.\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"

        

        # Generate answer
        response = genai.generate_text(
            model=self.model_name,
            prompt=prompt,
            temperature=0.0,
            max_output_tokens=350
        )

        return response.text
    
    '''  response = genai.Models.generate_content(
            model=self.model_name,
            contents=[{"type": "text", "text": prompt}],
            generation_config={
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 350
            }
        )
        '''




# Manual Test
if __name__ == "__main__":
    gemini = GeminiClient()

    test_chunks = [
        {"url": "site1.com", "chunk_id": 1, "text": "Python is a programming language."},
        {"url": "site1.com", "chunk_id": 2, "text": "It is used for AI and web development."}
    ]

    print(
        gemini.generate_answer("What is Python used for?", test_chunks)
    )
