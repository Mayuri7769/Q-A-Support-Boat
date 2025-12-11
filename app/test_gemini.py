import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # This loads .env into os.environ
print("API key loaded:", os.getenv("GEMINI_API_KEY"))

genai.configure(api_key="GEMINI_API_KEY")

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "Write a short poem about stars.",
    generation_config={
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 200
    }
)

print(response.text)
